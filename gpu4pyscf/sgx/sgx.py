# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Seminumerical exchange for the SCF methods (COSX; PySCF's sgx with dfj): the
exchange matrix on a grid, J unchanged (the J engine, or density fitting).

    >>> mf = dft.RKS(mol, xc='b3lyp').COSX()
    >>> mf.kernel()

An approximation: with the default grid (level 1 of PySCF's ORCA-like SGX
grids, overlap fitted) the energy of caffeine moves by about 1e-5 Eh.
scf.two_level uses it for the iterations of an exact model.
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.sgx import sgx_jk as cpu_sgx_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.gen_grid import arg_group_grids
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.scf import jk
from gpu4pyscf.sgx import sgx_jk

__all__ = ['SGX', 'cosx']

# The partition of the SGX grids (PySCF's becke_lko, VXCgen_grid_lko): Becke's
# with the distance of two atoms R replaced by f(R) = RCUT (1 - exp(-sum_m
# (R/RCUT)^m / m)), m = 1..MC
LKO_RCUT = 5.0
LKO_MC = 12

def _lko_sat(r):
    '''f(R) of the LKO partition and its derivative f'(R)'''
    x = r / LKO_RCUT
    tot = np.zeros_like(r)
    dtot = np.zeros_like(r)
    xm = np.ones_like(r)
    for m in range(1, LKO_MC + 1):
        dtot += xm
        xm = xm * x
        tot += xm / m
    return LKO_RCUT * (1 - np.exp(-tot)), np.exp(-tot) * dtot

_LKO_DW_SRC = r'''
#define NATM %(natm)d
extern "C" __global__
void lko_dw_dot(double *out, const double *coords, const double *weights, const double *v,
                const int *atm_idx, const double *atm_coords, const double *a_ij, const double *da_ij,
                const double *radii, const int ng)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    double acc[NATM*3];
    for (int k = 0; k < NATM*3; k++) {
        acc[k] = 0;
    }
    const int A = g < ng ? atm_idx[g] : -1;
    const double wv = (A >= 0) ? weights[g] * v[g] : 0.;
    if (A >= 0 && wv != 0) {
        const double rx = coords[g], ry = coords[ng+g], rz = coords[2*ng+g];
        double d[NATM], p[NATM];
        for (int i = 0; i < NATM; i++) {
            const double dx = rx - atm_coords[i*3+0];
            const double dy = ry - atm_coords[i*3+1];
            const double dz = rz - atm_coords[i*3+2];
            d[i] = sqrt(dx*dx + dy*dy + dz*dz);
            p[i] = 1.;
        }
        for (int i = 0; i < NATM; i++) {
        for (int j = 0; j < i; j++) {
            double s = (d[i] - d[j]) * a_ij[i*NATM+j];
            s = fmax(-1., fmin(1., s));
            if (radii != NULL) {
                s += radii[i*NATM+j] * (1 - s*s);
            }
            s = (3 - s*s) * s * .5;
            s = (3 - s*s) * s * .5;
            s = ((3 - s*s) * s * .5) * .5;
            p[i] *= .5 - s;
            p[j] *= .5 + s;
        } }
        double z = 0;
        for (int i = 0; i < NATM; i++) {
            z += p[i];
        }
        const double invz = 1. / z;
        for (int i = 0; i < NATM; i++) {
        for (int j = 0; j < i; j++) {
            const double a = a_ij[i*NATM+j];
            double s = (d[i] - d[j]) * a;
            s = fmax(-1., fmin(1., s));
            double ds = (s == -1. || s == 1.) ? 0. : 1.;
            if (radii != NULL) {
                const double fac = radii[i*NATM+j];
                ds -= 2 * fac * s;
                s += fac * (1 - s*s);
            }
            ds *= 1.5 * (1 - s*s);
            s = (3 - s*s) * s * .5;
            ds *= 1.5 * (1 - s*s);
            s = (3 - s*s) * s * .5;
            ds *= .75 * (1 - s*s);
            s = ((3 - s*s) * s * .5) * .5;
            if (s <= -.5 || s >= .5 || ds == 0) {
                continue;
            }
            // sum_B c_B d ln p_B of this pair, c_B = w v (delta_BA - p_B / Z)
            const double ci = wv * ((i == A ? 1. : 0.) - p[i] * invz);
            const double cj = wv * ((j == A ? 1. : 0.) - p[j] * invz);
            const double t = cj * ds / (.5 + s) - ci * ds / (.5 - s);
            // the factor 1/f(R_ij) of mu_ij
            const double ta = t * (d[i] - d[j]);
            for (int x = 0; x < 3; x++) {
                const double e = ta * da_ij[(i*NATM+j)*3+x];
                acc[i*3+x] += e;
                acc[j*3+x] -= e;
            }
            // the distances of the point, which moves with atom A
            const double tb = t * a;
            if (i != A) {
                const double f = tb / d[i];
                const double ux = f * (rx - atm_coords[i*3+0]);
                const double uy = f * (ry - atm_coords[i*3+1]);
                const double uz = f * (rz - atm_coords[i*3+2]);
                acc[i*3+0] -= ux; acc[i*3+1] -= uy; acc[i*3+2] -= uz;
                acc[A*3+0] += ux; acc[A*3+1] += uy; acc[A*3+2] += uz;
            }
            if (j != A) {
                const double f = tb / d[j];
                const double ux = f * (rx - atm_coords[j*3+0]);
                const double uy = f * (ry - atm_coords[j*3+1]);
                const double uz = f * (rz - atm_coords[j*3+2]);
                acc[j*3+0] += ux; acc[j*3+1] += uy; acc[j*3+2] += uz;
                acc[A*3+0] -= ux; acc[A*3+1] -= uy; acc[A*3+2] -= uz;
            }
        } }
    }
    for (int k = 0; k < NATM*3; k++) {
        double val = acc[k];
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, off);
        }
        if ((threadIdx.x & 31) == 0 && val != 0) {
            atomicAdd(out + k, val);
        }
    }
}
'''
_lko_kernels = {}

def lko_weight_derivs_dot(mol, coords, weights, atm_idx, v, radii_adjust=None, atomic_radii=None):
    '''sum_g dw[A,x,g] v[g], [natm,3]: the derivatives of the weights w of the
    points of an LKO-partitioned grid (PySCF's becke_lko) with respect to the
    positions of the atoms, each point moving with its atom atm_idx[g] (-1:
    padding). coords [3,ngrids], weights, atm_idx and v on the device.

    PySCF's VXCgen_grid_lko_deriv takes the derivative of 1/f(R) as
    f'(R)/f(R)^3, which is that of 1/R; f'(R)/(f(R)^2 R) here.'''
    natm = mol.natm
    rc = mol.atom_coords()
    rij = rc[:,None,:] - rc[None,:,:]
    r = np.sqrt((rij**2).sum(axis=2))
    np.fill_diagonal(r, 1.)
    f, fp = _lko_sat(r)
    a = 1. / f
    da = (-fp / (f**2 * r))[:,:,None] * rij
    radii = None
    if callable(radii_adjust) and atomic_radii is not None:
        fadj = radii_adjust(mol, atomic_radii)
        radii = cp.asarray(np.array([[fadj(i, j, 0) for j in range(natm)] for i in range(natm)]))
    if natm not in _lko_kernels:
        _lko_kernels[natm] = cp.RawKernel(_LKO_DW_SRC % {'natm': natm}, 'lko_dw_dot')
    out = cp.zeros(natm * 3)
    ng = weights.size
    nthreads = 128
    _lko_kernels[natm](((ng + nthreads - 1) // nthreads,), (nthreads,),
                       (out, cp.ascontiguousarray(coords, dtype=np.float64),
                        cp.ascontiguousarray(weights, dtype=np.float64),
                        cp.ascontiguousarray(v, dtype=np.float64),
                        cp.ascontiguousarray(atm_idx, dtype=np.int32),
                        cp.asarray(rc), cp.asarray(a), cp.asarray(da), radii, np.int32(ng)))
    return out.reshape(natm, 3).get()


def _operator(mol, omega, lr_factor, sr_factor):
    '''(omega, c_full, c_lr) of c_full / r + c_lr erf(omega r) / r for the
    arguments of jk.get_k'''
    omega, lr, sr = jk._check_rsh_factors(mol, omega, lr_factor, sr_factor)
    if omega == 0:
        return 0., float(sr), 0.
    return abs(omega), float(sr), float(lr - sr)


class SGX(lib.StreamObject):
    '''The seminumerical exchange of a molecule.

    Attributes:
        grids_level : int
            Level of the grid of PySCF's sgx module. Default 1.
        use_opt_grids : bool
            Its ORCA-like grids. Default True.
        grids_thrd : float
            Grid points are left out where every weighted AO is below this.
        fit_ovlp : bool
            The AOs on the grid are corrected so that the overlap matrix
            integrated on the grid is exact, in the symmetric form of PySCF
            (to first order in the grid's error of the overlap). Default True.
        kernel_dtype : 'double' or 'float'
            Precision of the integrals and of the products with the AO values.

    Every shell pair is evaluated on every block of grid points.
    '''

    _keys = {'mol', 'grids_level', 'use_opt_grids', 'grids_thrd', 'fit_ovlp',
             'kernel_dtype', 'blksize', 'grids', 'max_memory'}

    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.grids_level = 1
        self.use_opt_grids = True
        self.grids_thrd = 1e-10
        self.fit_ovlp = True
        self.kernel_dtype = 'double'
        self.blksize = sgx_jk.BLKSIZE
        self.grids = None
        self._coords = None
        self._weights = None
        self._proj = None
        self._sorted_mol = None
        self._gdftopt = None
        self._ao_inv = None
        self._pair_data = {}
        self._dev = {}
        self._built = None
        # for the gradient: the overlap matrix on the grid, the points of the
        # device arrays in grids, the pair data of the derivative integrals
        self._sn = None
        self._grid_idx = None
        self._atm_idx = None
        self._deriv_pair_data = {}
        self._dev_deriv = {}

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('grids_level = %s', self.grids_level)
        log.info('use_opt_grids = %s', self.use_opt_grids)
        log.info('fit_ovlp = %s', self.fit_ovlp)
        log.info('kernel_dtype = %s', self.kernel_dtype)
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.grids = None
        self._coords = None
        self._weights = None
        self._proj = None
        self._sorted_mol = None
        self._gdftopt = None
        self._ao_inv = None
        self._pair_data = {}
        self._dev = {}
        self._built = None
        self._sn = None
        self._grid_idx = None
        self._atm_idx = None
        self._deriv_pair_data = {}
        self._dev_deriv = {}
        return self

    @property
    def size(self):
        return 0 if self._weights is None else len(self._weights)

    def _settings(self):
        return (self.mol, self.grids_level, self.use_opt_grids, self.grids_thrd, self.fit_ovlp)

    def build(self):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        mol = self.mol
        self._built = self._settings()
        self.grids = grids = cpu_sgx_jk.get_gridss(
            mol, self.grids_level, self.grids_thrd, self.use_opt_grids)
        keep = grids.weights != 0
        coords = cp.asarray(grids.coords[keep])
        weights = cp.asarray(grids.weights[keep])
        # points near each other next to each other: the kernels take blocks
        idx = arg_group_grids(mol, coords)
        self._coords = cp.ascontiguousarray(coords[idx].T)
        self._weights = cp.ascontiguousarray(weights[idx])
        self._grid_idx = np.nonzero(keep)[0][cp.asnumpy(idx)]
        self._atm_idx = None
        self._gdftopt = opt = numint._GDFTOpt.from_mol(mol)
        self._ao_inv = cp.asarray(np.argsort(opt._ao_idx))
        self._sorted_mol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=0.3)
        self._pair_data = {}
        self._dev = {}
        self._deriv_pair_data = {}
        self._dev_deriv = {}
        self._proj = None
        self._sn = None
        if self.fit_ovlp:
            nao = mol.nao
            sn = cp.zeros((nao, nao))
            for c0, c1 in self._batches():
                ao = self._eval_ao(c0, c1)
                sn += (ao * self._weights[c0:c1]).dot(ao.T)
            s = asarray(mol.intor_symmetric('int1e_ovlp'))
            proj = cp.linalg.solve(sn, s)
            self._proj = .5 * (proj + cp.eye(nao))
            self._sn = sn
        log.info('SGX grid: level %d, %d points', self.grids_level, self.size)
        log.timer('SGX grid', *t0)
        return self

    def _batches(self, grad=False):
        '''Ranges of grid points: the AO values and the two [AOs, points]
        arrays of a density at a time, in a fifth of the free memory (grad: the
        arrays of get_k_grad, in two fifths)'''
        ngrids = self.size
        nao = self.mol.nao
        nao_c = int(self._sorted_mol.p_ao_loc[-1])
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        if grad:
            # AO values and gradients, f, g, h, dg and a product; the arrays
            # of the derivative shells (about 3.5 times the Cartesian functions)
            per_point = 8 * (13 * nao + 13 * nao_c)
            share = .4
        else:
            per_point = 8 * (3 * nao + 3 * nao_c)
            share = .2
        n = int(free * share / per_point) // self.blksize * self.blksize
        n = max(n, self.blksize)
        return [(c0, min(c0 + n, ngrids)) for c0 in range(0, ngrids, n)]

    def _eval_ao(self, c0, c1, deriv=0):
        '''AO values [nao,points] in the order of mol (deriv=1: with their
        gradients, [4,nao,points])'''
        opt = self._gdftopt
        coords = self._coords[:,c0:c1].T
        ao = numint.eval_ao(opt._sorted_mol, coords, deriv, gdftopt=opt, transpose=False)
        if deriv == 0:
            return ao[self._ao_inv]
        return ao[:,self._ao_inv]

    def _device_pairs(self, omega, c_full, c_lr):
        key = (omega, c_full, c_lr, self.kernel_dtype)
        if key not in self._dev:
            ops = (omega, c_full, c_lr)
            if ops not in self._pair_data:
                self._pair_data[ops] = sgx_jk.pair_data(self._sorted_mol, omega, c_full, c_lr)
            self._dev[key] = sgx_jk._DevicePairs(
                self._pair_data[ops], omega, c_full, c_lr, self.kernel_dtype)
        return self._dev[key]

    def _device_deriv_pairs(self, omega, c_full, c_lr):
        '''The pair data of the derivative integrals (sgx_jk.deriv_pair_data),
        in the precision of the kernels'''
        ops = (omega, c_full, c_lr)
        key = ops + (self.kernel_dtype,)
        if key not in self._dev_deriv:
            if ops not in self._deriv_pair_data:
                self._deriv_pair_data[ops] = sgx_jk.deriv_pair_data(
                    self._sorted_mol, omega, c_full, c_lr)
            self._dev_deriv[key] = sgx_jk._DeviceDerivPairs(
                self._deriv_pair_data[ops], omega, c_full, c_lr, self.kernel_dtype)
        return self._dev_deriv[key]

    def _dweights_dot(self, v):
        '''sum_g dw[A,x,g] v[g], [natm,3]: dw the derivatives of the weights of
        the points (v in the order of the device arrays) with respect to the
        positions of the atoms, each point moving with its atom
        (lko_weight_derivs_dot)'''
        grids = self.grids
        if self._atm_idx is None:
            self._atm_idx = cp.asarray(grids.atm_idx[self._grid_idx], dtype=np.int32)
        return lko_weight_derivs_dot(self.mol, self._coords, self._weights, self._atm_idx, v,
                                     grids.radii_adjust, grids.atomic_radii)

    def get_k_grad(self, dm, omega=None, lr_factor=None, sr_factor=None, grid_response=True):
        '''The derivatives of sum_n tr(D_n K[D_n]) with respect to the positions
        of the atoms at fixed density matrices D_n (dm: [nao,nao] or
        [n,nao,nao], symmetric), K of get_k: [natm,3] on the host.

        With X the AO values on the grid, P the overlap fit and f_g = D P^T X_g,

            tr(D K[D]) = sum_g w_g f_g^T A_g f_g,

        and the derivative has the terms of the AOs' centers and of the
        points (each moves with its atom) in X and in the integrals A_g, of
        the weights, and of the overlap fit (the overlap matrix on the grid
        and the analytical one). The integrals' derivatives come from
        sgx_jk.contract_ip1. grid_response=False leaves out the motion of the
        points and the derivatives of the weights: not the derivative of the
        energy then.

        The integrals and their derivatives in kernel_dtype (as get_k), the
        rest in double precision.
        '''
        if grid_response and not self.use_opt_grids:
            raise NotImplementedError('the grid response of SGX grids without use_opt_grids')
        if self._coords is None or self._built != self._settings():
            self.build()
        log = logger.new_logger(self)
        t0 = log.init_timer()
        mol = self.mol
        smol = self._sorted_mol
        natm = mol.natm
        omega, c_full, c_lr = _operator(mol, omega, lr_factor, sr_factor)
        dev = self._device_pairs(omega, c_full, c_lr)
        devd = self._device_deriv_pairs(omega, c_full, c_lr)
        real = dev.real
        dm = asarray(dm)
        nao = dm.shape[-1]
        dms = dm.reshape(-1, nao, nao)
        proj = self._proj
        if proj is not None:
            pdms = cp.einsum('ki,xij->xkj', proj, dms)
        else:
            pdms = dms
        ngrids = self.size

        per_ao = cp.zeros((3, nao))
        per_point = cp.zeros((3, ngrids))
        edens = cp.zeros(ngrids)
        kd = cp.zeros((nao, nao))
        batches = self._batches(grad=True)
        for c0, c1 in batches:
            ao = self._eval_ao(c0, c1, 1)
            weights = self._weights[c0:c1]
            coords = self._coords[:,c0:c1]
            for n in range(len(dms)):
                f = pdms[n].T.dot(ao[0])
                fc = cp.ascontiguousarray(smol.apply_C_dot(f), dtype=real)
                g = smol.apply_CT_dot(cp.asarray(
                    sgx_jk.contract(dev, fc, coords, blksize=self.blksize), dtype=np.float64))
                dgc = cp.asarray(sgx_jk.contract_ip1(devd, fc, coords, blksize=self.blksize),
                                 dtype=np.float64)
                fc = None
                edens[c0:c1] += (f * g).sum(axis=0)
                # K D of the exchange matrix before the overlap fit
                kd += (ao[0] * weights).dot(g.T).dot(dms[n])
                h = pdms[n].dot(g)
                g = None
                for x in range(3):
                    t = f * smol.apply_CT_dot(dgc[x])
                    t -= h * ao[1+x]
                    t *= 2 * weights
                    per_ao[x] += t.sum(axis=1)
                    per_point[x,c0:c1] += t.sum(axis=0)
                f = h = dgc = t = None
            ao = None
        t1 = log.timer_debug1('SGX K gradient: integrals', *t0)

        if proj is not None:
            # the overlap fit P = (S_n^-1 S + 1) / 2: with C = S_n^-1 K D,
            #     2 tr(dP^T K D) = sum_ab dS[a,b] C[a,b] - sum_ab dS_n[a,b] (C Q^T)[a,b],   Q = S_n^-1 S
            c = cp.linalg.solve(self._sn, kd)
            q = 2 * proj - cp.eye(nao)
            c2 = c.dot(q.T)
            c2 = c2 + c2.T
            sdens = cp.zeros(ngrids)
            for c0, c1 in batches:
                ao = self._eval_ao(c0, c1, 1)
                weights = self._weights[c0:c1]
                cx = c2.dot(ao[0])
                sdens[c0:c1] = .5 * (ao[0] * cx).sum(axis=0)
                cx *= weights
                for x in range(3):
                    t = ao[1+x] * cx
                    per_ao[x] += t.sum(axis=1)
                    per_point[x,c0:c1] += t.sum(axis=0)
                ao = cx = t = None
            edens -= sdens
            ipovlp = asarray(mol.intor('int1e_ipovlp', comp=3))
            per_ao -= (ipovlp * (c + c.T)).sum(axis=2)
            t1 = log.timer_debug1('SGX K gradient: overlap fit', *t1)

        de = np.zeros((natm, 3))
        per_ao = per_ao.get()
        for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:,2:]):
            de[ia] = per_ao[:,p0:p1].sum(axis=1)
        if grid_response:
            de += self._dweights_dot(edens)
            atm_idx = self._atm_idx.get()
            per_point = per_point.get()
            for x in range(3):
                de[:,x] -= np.bincount(atm_idx, weights=per_point[x], minlength=natm)
            log.timer_debug1('SGX K gradient: grid response', *t1)
        log.timer_debug1('SGX K gradient', *t0)
        return de

    def get_k(self, dm, hermi=1, omega=None, lr_factor=None, sr_factor=None):
        '''The exchange matrix of the operator
        lr_factor erf(omega r)/r + sr_factor erfc(omega r)/r (jk.get_k)'''
        if self._coords is None or self._built != self._settings():
            self.build()
        log = logger.new_logger(self)
        t0 = log.init_timer()
        mol = self.mol
        smol = self._sorted_mol
        omega, c_full, c_lr = _operator(mol, omega, lr_factor, sr_factor)
        dev = self._device_pairs(omega, c_full, c_lr)
        real = dev.real
        dm = asarray(dm)
        nao = dm.shape[-1]
        dms = dm.reshape(-1, nao, nao)
        if self._proj is not None:
            dms = cp.einsum('ki,xij->xkj', self._proj, dms)
        vk = cp.zeros_like(dms)
        single = real == np.float32
        for c0, c1 in self._batches():
            ao = self._eval_ao(c0, c1)
            weights = self._weights[c0:c1]
            coords = self._coords[:,c0:c1]
            if single:
                # the products with the AO values in the kernels' precision
                ao = ao.astype(np.float32)
                weights = weights.astype(np.float32)
            for n, d in enumerate(dms):
                f = d.astype(real).T.dot(ao)
                f *= weights
                fc = smol.apply_C_dot(cp.asarray(f, dtype=np.float64))
                f = None
                fc = cp.ascontiguousarray(fc, dtype=real)
                gc = sgx_jk.contract(dev, fc, coords, blksize=self.blksize)
                fc = None
                g = smol.apply_CT_dot(cp.asarray(gc, dtype=np.float64))
                gc = None
                vk[n] += ao.dot(cp.asarray(g, dtype=real).T)
                g = None
        if self._proj is not None:
            vk = cp.einsum('ik,xij->xkj', self._proj, vk)
        if hermi == 1:
            vk = (vk + vk.transpose(0, 2, 1)) * .5
        log.timer_debug1('SGX vk', *t0)
        return vk.reshape(dm.shape)


class _COSX:
    '''Mixin of the SCF methods: K from with_sgx'''

    _keys = {'with_sgx'}

    # The exchange of a range-separated hybrid in one pass of the kernels,
    # with the operator hyb / r + (alpha - hyb) erf(omega r) / r, where the
    # potential (and the density-fitted SCF's) and the gradient choose by this
    range_separated_mode = 'mix_inside_kernel'

    def __init__(self, mf, with_sgx):
        self.__dict__.update(mf.__dict__)
        self.with_sgx = with_sgx

    def get_k(self, mol=None, dm=None, hermi=1, omega=None, lr_factor=None, sr_factor=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        sgx = self.with_sgx
        if sgx.mol is not mol:
            sgx.reset(mol)
        vk = sgx.get_k(dm, hermi, omega, lr_factor, sr_factor)
        if not isinstance(dm, cp.ndarray):
            vk = vk.get()
        return vk

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None, lr_factor=None, sr_factor=None):
        from gpu4pyscf.df.df_jk import _DFHF
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vj = vk = None
        if with_j:
            # J of the method below this mixin: the fitted one (its get_j
            # goes through get_jk), or the J engine's
            if isinstance(self, _DFHF):
                vj = super().get_jk(mol, dm, hermi, True, False, omega)[0]
            else:
                vj = super().get_j(mol, dm, hermi, omega)
        if with_k:
            vk = self.get_k(mol, dm, hermi, omega, lr_factor, sr_factor)
        return vj, vk

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_sgx.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        super().reset(mol)
        self.with_sgx.reset(mol)
        return self

    def undo_cosx(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _COSX))
        del obj.with_sgx
        return obj

    def Gradients(self):
        '''The nuclear gradients of this model: those of the method below the
        mixin (with its J, solvent, dispersion), the exchange terms from
        with_sgx (SGX.get_k_grad)'''
        g = super().Gradients()
        return lib.set_class(_COSXGradients(g), (_COSXGradients, g.__class__))

    def nuc_grad_method(self):
        return self.Gradients()

    def Hessian(self):
        raise NotImplementedError('Hessians with seminumerical exchange')

    def to_cpu(self):
        raise NotImplementedError


class _COSXGradients:
    '''Mixin of the gradient object of a _COSX method: the exchange terms of
    jk_energy_per_atom from the seminumerical exchange of base.with_sgx, J
    from the gradient object below (the J engine, or density fitting).

    Attributes:
        sgx_grid_response : bool
            With the derivatives of the grid points and weights of the
            exchange (the gradient of the energy). Default True.
    '''

    _keys = {'sgx_grid_response'}

    # grad.rks._jk_energy_per_atom: the exchange of a range-separated hybrid
    # in one call, with both ranges (one pass of the kernels)
    _k_mixed_range = True

    def __init__(self, mf_grad):
        self.__dict__.update(mf_grad.__dict__)
        self.sgx_grid_response = True

    def undo_cosx(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _COSXGradients))
        del obj.sgx_grid_response
        return obj

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, omega=None,
                           lr_factor=None, sr_factor=None, hermi=0, verbose=None):
        '''j_factor * dE_J / dR - k_factor * dE_K / dR as the gradient objects
        below compute it (E_K = tr(D K[D]) / 4 of one density matrix, the sum
        of tr(D_s K[D_s]) / 2 of the two of unrestricted orbitals), K of
        with_sgx for the operator of omega, lr_factor and sr_factor'''
        mf = self.base
        if dm is None:
            dm = mf.make_rdm1()
        if hermi == 2:
            raise NotImplementedError('the exchange gradient of antisymmetric densities')
        de = 0
        if j_factor != 0:
            if omega or lr_factor is not None or sr_factor is not None:
                raise NotImplementedError('J of a range-separated operator')
            de = super().jk_energy_per_atom(dm, j_factor, 0, hermi=hermi, verbose=verbose)
        if k_factor != 0:
            dm = asarray(dm)
            nao = dm.shape[-1]
            dms = dm.reshape(-1, nao, nao)
            assert len(dms) <= 2
            dms = (dms + dms.transpose(0, 2, 1)) * .5
            fac = .25 if len(dms) == 1 else .5
            sgx = mf.with_sgx
            if sgx.mol is not mf.mol:
                sgx.reset(mf.mol)
            ek = sgx.get_k_grad(dms, omega, lr_factor, sr_factor,
                                grid_response=self.sgx_grid_response)
            ek *= -k_factor * fac
            if isinstance(de, cp.ndarray):
                ek = cp.asarray(ek)
            de = de + ek
        return de


def cosx(mf, with_sgx=None):
    '''The SCF method mf with its exchange evaluated seminumerically'''
    if isinstance(mf, _COSX):
        if with_sgx is not None:
            mf = mf.copy()
            mf.with_sgx = with_sgx
        return mf
    if with_sgx is None:
        with_sgx = SGX(mf.mol)
        with_sgx.verbose = mf.verbose
        with_sgx.stdout = mf.stdout
    return lib.set_class(_COSX(mf, with_sgx), (_COSX, mf.__class__))
