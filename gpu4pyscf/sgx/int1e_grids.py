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
The integrals of AO pairs with charges at points, (mu nu|g), contracted with
a density matrix or with the charges, without the integrals being stored:

    v[g] = sum_{mu nu} D[mu,nu] (mu nu|g)        (Int1eGridsOpt.density_contracted)
    V[mu,nu] = sum_g c[g] (mu nu|g)              (Int1eGridsOpt.charge_contracted)

and the integrals of each point between two sets of orbitals,

    T[g,p,i] = sum_{mu nu} C[mu,p] (mu nu|g) C'[nu,i]     (Int1eGridsOpt.orbital_contracted)

from which the two contractions of any number of densities C x C'^T + c.c.
and of sets of charges are matrix products (the response of a continuum
solvent in the CPHF equations),

for point charges, or for normalized Gaussian charges of an exponent per
point (the surface charges of a continuum solvent). These are the contractions
of gpu4pyscf.gto.int3c1e.int1e_grids with dm and with charges, on the kernels
of gpu4pyscf.sgx.kernels (Obara-Saika, generated per class of shell pairs):
a primitive pair on a point costs the Boys functions and the vertical
recurrence to [e|0], and one multiply-add per Cartesian power e with

    H[e] = sum_{ab} w(ab,e) D[a,b]      or into      M[e] = sum_g c[g] [e|0]_g,

the horizontal recurrence, (a|b) = sum_e w(ab,e) [e|0], being applied to the
density block of a shell pair before the kernels, or to M after them.
'''

import math
import numpy as np
import cupy as cp
import cupyx
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.sgx import kernels, sgx_jk
from gpu4pyscf.sgx.kernels import cart_tuples

__all__ = ['Int1eGridsOpt', 'density_contracted', 'charge_contracted']

# The horizontal recurrence of the integrals of the shell pairs of a class on
# a batch of points, m[g,pair,e] = [e|0]_g, into the matrices of the points:
# dense[g, index[pair,ab]] = sum_e w[pair,ab,e] m[g,pair,e]
_HRR_SCATTER = cp.RawKernel(r'''
extern "C" __global__
void sgx_hrr_scatter(double* __restrict__ dense, const double* __restrict__ m,
        const double* __restrict__ w, const long long* __restrict__ index,
        const int ng, const int npairs, const int nab, const int ne, const long long n2)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)ng * npairs * nab) {
        return;
    }
    const int ab = (int)(t % nab);
    const long long gp = t / nab;
    const int ip = (int)(gp % npairs);
    const long long g = gp / npairs;
    const double* __restrict__ mm = m + (g * npairs + ip) * ne;
    const double* __restrict__ ww = w + ((long long)ip * nab + ab) * ne;
    double acc = 0;
    for (int e = 0; e < ne; ++e) {
        acc += ww[e] * mm[e];
    }
    dense[g * n2 + index[(long long)ip * nab + ab]] = acc;
}
''', 'sgx_hrr_scatter')

# threads of a block: grid points for the potential, primitive pairs for the moments
BLKSIZE = 128
# threads a launch should have at least (a slice of the other index each)
MIN_THREADS = 32768
# what a thread's slice should hold at least
MIN_SLICE = 32


def _hrr_weights(li, lj, ab):
    '''w[pair, a nj + b, e]: (a|b) = sum_e w [e|0] for the Cartesian functions
    a of the first shell, b of the second, and the powers e of degree
    li..li+lj at the center of the first; from x_B = x_A + AB.'''
    A = cart_tuples(li)
    B = cart_tuples(lj)
    targets = [e for n in range(li, li+lj+1) for e in cart_tuples(n)]
    pos = {e: n for n, e in enumerate(targets)}
    ks = [k for n in range(lj+1) for k in cart_tuples(n)]
    nj = len(B)
    w = np.zeros((len(ab), len(A) * nj, len(targets)))
    for ia, a in enumerate(A):
        for ib, b in enumerate(B):
            for k in ks:
                if all(x <= y for x, y in zip(k, b)):
                    coef = math.comb(b[0], k[0]) * math.comb(b[1], k[1]) * math.comb(b[2], k[2])
                    mono = (ab[:,0]**(b[0]-k[0]) * ab[:,1]**(b[1]-k[1]) * ab[:,2]**(b[2]-k[2]))
                    e = (a[0]+k[0], a[1]+k[1], a[2]+k[2])
                    w[:,ia*nj+ib,pos[e]] += coef * mono
    return w


class Int1eGridsOpt:
    '''The data of a molecule for the two contractions.

    A shell pair is left out where a bound of its integrals is below pair_tol,
    a primitive pair below prim_tol (sgx_jk.pair_data).
    '''

    def __init__(self, mol, pair_tol=1e-15, prim_tol=1e-17):
        self.mol = mol
        self.sorted_mol = smol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=0.3)
        self.nao_cart = nao = int(smol.p_ao_loc[-1])
        self.classes = []
        self._prims = {}
        for (li, lj), pc in sorted(sgx_jk.pair_data(smol, None, 1., 0., pair_tol, prim_tol).items()):
            ni = (li + 1) * (li + 2) // 2
            nj = (lj + 1) * (lj + 2) // 2
            nprim = pc.pairs[:,3] - pc.pairs[:,2]
            rows = pc.pairs[:,0].astype(np.int64)[:,None,None] + np.arange(ni)[None,:,None]
            cols = pc.pairs[:,1].astype(np.int64)[:,None,None] + np.arange(nj)[None,None,:]
            self.classes.append(dict(
                li=li, lj=lj, npairs=pc.npairs, nprims=len(pc.prims),
                prims=pc.prims, pairs=pc.pairs,
                prim_pair=cp.asarray(np.repeat(np.arange(pc.npairs, dtype=np.int32), nprim)),
                # the elements of the pairs' blocks in the [nao,nao] matrix of the sorted mol
                index=cp.asarray((rows * nao + cols).reshape(pc.npairs, ni * nj)),
                w=cp.asarray(_hrr_weights(li, lj, pc.ab))))

    def _device_prims(self, n, dtype):
        key = (n, dtype)
        if key not in self._prims:
            real = np.float64 if dtype == 'double' else np.float32
            self._prims[key] = cp.asarray(self.classes[n]['prims'].astype(real))
        return self._prims[key]

    @staticmethod
    def _grid(coords, exponents, real):
        coords = cp.asarray(coords)
        ngrids = len(coords)
        grid = cp.ascontiguousarray(coords.T, dtype=real)
        if exponents is None:
            return ngrids, grid, cp.zeros(1, dtype=real), False
        exponents = cp.asarray(exponents, dtype=np.float64)
        gexp = cp.empty((2, ngrids), dtype=real)
        gexp[0] = exponents
        gexp[1] = cp.sqrt(exponents)
        return ngrids, grid, gexp, True

    def density_contracted(self, dm, coords, exponents=None, dtype='double'):
        '''v[g] = sum_{mu nu} dm[mu,nu] (mu nu|g) for the points coords [n,3],
        with point charges, or normalized Gaussian charges of the given
        exponents [n]. dm [nao,nao], or [m,nao,nao] for v [m,n].'''
        real = np.float64 if dtype == 'double' else np.float32
        smol = self.sorted_mol
        dm = cp.asarray(dm)
        if dm.ndim == 3:
            return cp.stack([self.density_contracted(d, coords, exponents, dtype) for d in dm])
        ngrids, grid, gexp, gauss = self._grid(coords, exponents, real)
        dmc = smol.apply_C_mat_CT(dm)
        # both triangles on the pairs i >= j (the diagonal pairs count half)
        dmc = (dmc + dmc.T).ravel()
        nblk = (ngrids + BLKSIZE - 1) // BLKSIZE
        out = cp.zeros(ngrids)
        for n, c in enumerate(self.classes):
            L = c['li'] + c['lj']
            nprims = c['nprims']
            hd = cp.einsum('pme,pm->pe', c['w'], dmc[c['index']])
            hd = cp.ascontiguousarray(hd, dtype=real)
            kern = kernels.get_grid_kernel('v', c['li'], c['lj'], gauss, dtype)
            boys = kernels.boys_table(L, dtype, kernels.GRID_BOYS_ORDER[dtype],
                                      kernels.grid_boys_tmax(L, dtype))
            prims = self._device_prims(n, dtype)
            nslice = max(1, min(nprims // MIN_SLICE, -(-MIN_THREADS // ngrids)))
            # primitive pairs per launch, by their work
            step = max(nslice, int(sgx_jk._LAUNCH_WORK / (ngrids * sgx_jk._class_cost(L))))
            for k0 in range(0, nprims, step):
                k1 = min(k0 + step, nprims)
                ns = max(1, min(nslice, (k1 - k0) // MIN_SLICE))
                v = cp.zeros((ns, ngrids), dtype=real)
                kern((nblk, ns), (BLKSIZE,),
                     (v, grid, gexp, prims, c['prim_pair'], hd, boys,
                      np.int32(k0), np.int32(k1), np.int32(ngrids), np.int32(ngrids)))
                out += v.sum(axis=0, dtype=np.float64)
        return out

    def charge_contracted(self, charges, coords, exponents=None, dtype='double'):
        '''V[mu,nu] = sum_g charges[g] (mu nu|g) for the points coords [n,3],
        with point charges, or normalized Gaussian charges of the given
        exponents [n]. charges [n], or [m,n] for V [m,nao,nao].'''
        real = np.float64 if dtype == 'double' else np.float32
        smol = self.sorted_mol
        charges = cp.asarray(charges)
        if charges.ndim == 2:
            return cp.stack([self.charge_contracted(c, coords, exponents, dtype) for c in charges])
        ngrids, grid, gexp, gauss = self._grid(coords, exponents, real)
        charges = cp.ascontiguousarray(charges, dtype=real)
        nao = self.nao_cart
        out = cp.zeros(nao * nao)
        for n, c in enumerate(self.classes):
            L = c['li'] + c['lj']
            nprims = c['nprims']
            ne = c['w'].shape[2]
            kern = kernels.get_grid_kernel('m', c['li'], c['lj'], gauss, dtype)
            boys = kernels.boys_table(L, dtype, kernels.GRID_BOYS_ORDER[dtype],
                                      kernels.grid_boys_tmax(L, dtype))
            prims = self._device_prims(n, dtype)
            nblk = (nprims + BLKSIZE - 1) // BLKSIZE
            nslice = max(1, min(ngrids // MIN_SLICE, -(-MIN_THREADS // nprims)))
            m = cp.zeros((nslice, nprims, ne), dtype=real)
            # grid points per launch, by their work
            step = max(nslice, int(sgx_jk._LAUNCH_WORK / (nprims * sgx_jk._class_cost(L))))
            for g0 in range(0, ngrids, step):
                g1 = min(g0 + step, ngrids)
                kern((nblk, nslice), (BLKSIZE,),
                     (m, grid, gexp, charges, prims, boys,
                      np.int32(nprims), np.int32(g0), np.int32(g1), np.int32(ngrids)))
            # the primitive pairs of each shell pair, then the horizontal recurrence
            mp = cp.zeros((c['npairs'], ne))
            cupyx.scatter_add(mp, c['prim_pair'], m.sum(axis=0, dtype=np.float64))
            out[c['index']] = cp.einsum('pme,pe->pm', c['w'], mp)
        out = out.reshape(nao, nao)
        out = smol.apply_CT_mat_C(out + out.T)
        # symmetric to the last bit, as the blocks it was made of
        return (out + out.T) * .5


    def orbital_contracted(self, mo_left, mo_right, coords, exponents=None, out=None,
                           max_memory=None):
        '''T[g,p,i] = sum_{mu nu} mo_left[mu,p] (mu nu|g) mo_right[nu,i] for the
        points coords [n,3], with point charges, or normalized Gaussian
        charges of the given exponents [n]: [n, columns of mo_left, columns of
        mo_right].

        The integrals of a batch of points are made as matrices and
        transformed; max_memory (bytes, a fifth of the free device memory by
        default) bounds the batch. out may be a NumPy array: the batches are
        then moved off the device as they are made.'''
        from gpu4pyscf.lib.cupy_helper import get_avail_mem
        smol = self.sorted_mol
        ngrids, grid, gexp, gauss = self._grid(coords, exponents, np.float64)
        nao = self.nao_cart
        cl = cp.ascontiguousarray(smol.apply_C_dot(cp.asarray(mo_left, dtype=np.float64)))
        cr = cp.ascontiguousarray(smol.apply_C_dot(cp.asarray(mo_right, dtype=np.float64)))
        nl, nr = cl.shape[1], cr.shape[1]
        if out is None:
            out = cp.empty((ngrids, nl, nr))
        assert out.shape == (ngrids, nl, nr)
        if max_memory is None:
            max_memory = get_avail_mem() * .2
        # of a point: the matrix and its transpose, the two transformed ones
        per_point = (2 * nao * nao + nao * nr + nl * nr) * 8
        step = max(1, min(ngrids, 1024, int(max_memory / per_point)))
        for g0 in range(0, ngrids, step):
            g1 = min(g0 + step, ngrids)
            ng = g1 - g0
            dense = cp.zeros((ng, nao * nao))
            for n, c in enumerate(self.classes):
                L = c['li'] + c['lj']
                npairs = c['npairs']
                nab, ne = c['w'].shape[1:]
                kern = kernels.get_grid_kernel('p', c['li'], c['lj'], gauss, 'double')
                boys = kernels.boys_table(L, 'double', kernels.GRID_BOYS_ORDER['double'],
                                          kernels.grid_boys_tmax(L, 'double'))
                prims = self._device_prims(n, 'double')
                if 'pairs_dev' not in c:
                    c['pairs_dev'] = cp.asarray(c['pairs'], dtype=np.int32)
                m = cp.empty((ng, npairs, ne))
                nblk = (npairs + BLKSIZE - 1) // BLKSIZE
                # points per launch, by their work
                sub = max(1, int(sgx_jk._LAUNCH_WORK / (c['nprims'] * sgx_jk._class_cost(L))))
                for h0 in range(0, ng, sub):
                    h1 = min(h0 + sub, ng)
                    kern((nblk, h1 - h0), (BLKSIZE,),
                         (m[h0:h1], grid, gexp, c['pairs_dev'], prims, boys,
                          np.int32(npairs), np.int32(g0 + h0), np.int32(ngrids)))
                total = ng * npairs * nab
                _HRR_SCATTER(((total + 255) // 256,), (256,),
                             (dense, m, c['w'], c['index'], np.int32(ng), np.int32(npairs),
                              np.int32(nab), np.int32(ne), np.int64(nao * nao)))
                m = None
            dense = dense.reshape(ng, nao, nao)
            # both triangles from the pairs i >= j (the diagonal pairs count half)
            dense = dense + dense.transpose(0, 2, 1)
            t = cp.matmul(dense, cr)
            dense = None
            t = cp.tensordot(t, cl, axes=(1, 0)).transpose(0, 2, 1)
            if isinstance(out, cp.ndarray):
                out[g0:g1] = t
            else:
                out[g0:g1] = t.get()
            t = None
        return out


def density_contracted(mol, coords, dm, exponents=None, opt=None, dtype='double'):
    '''sum_{mu nu} dm[mu,nu] (mu nu|g): int3c1e.int1e_grids(mol, coords,
    charge_exponents=exponents, dm=dm)'''
    if opt is None:
        opt = Int1eGridsOpt(mol)
    return opt.density_contracted(dm, coords, exponents, dtype)

def charge_contracted(mol, coords, charges, exponents=None, opt=None, dtype='double'):
    '''sum_g charges[g] (mu nu|g): int3c1e.int1e_grids(mol, coords,
    charge_exponents=exponents, charges=charges)'''
    if opt is None:
        opt = Int1eGridsOpt(mol)
    return opt.charge_contracted(charges, coords, exponents, dtype)
