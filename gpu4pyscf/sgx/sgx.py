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
        self._gdftopt = opt = numint._GDFTOpt.from_mol(mol)
        self._ao_inv = cp.asarray(np.argsort(opt._ao_idx))
        self._sorted_mol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=0.3)
        self._pair_data = {}
        self._dev = {}
        self._proj = None
        if self.fit_ovlp:
            nao = mol.nao
            sn = cp.zeros((nao, nao))
            for c0, c1 in self._batches():
                ao = self._eval_ao(c0, c1)
                sn += (ao * self._weights[c0:c1]).dot(ao.T)
            s = asarray(mol.intor_symmetric('int1e_ovlp'))
            proj = cp.linalg.solve(sn, s)
            self._proj = .5 * (proj + cp.eye(nao))
        log.info('SGX grid: level %d, %d points', self.grids_level, self.size)
        log.timer('SGX grid', *t0)
        return self

    def _batches(self):
        '''Ranges of grid points: the AO values and the two [AOs, points]
        arrays of a density at a time, in a fifth of the free memory'''
        ngrids = self.size
        nao = self.mol.nao
        nao_c = int(self._sorted_mol.p_ao_loc[-1])
        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        per_point = 8 * (3 * nao + 3 * nao_c)
        n = int(free * .2 / per_point) // self.blksize * self.blksize
        n = max(n, self.blksize)
        return [(c0, min(c0 + n, ngrids)) for c0 in range(0, ngrids, n)]

    def _eval_ao(self, c0, c1):
        '''AO values [nao,points] in the order of mol'''
        opt = self._gdftopt
        coords = self._coords[:,c0:c1].T
        ao = numint.eval_ao(opt._sorted_mol, coords, 0, gdftopt=opt, transpose=False)
        return ao[self._ao_inv]

    def _device_pairs(self, omega, c_full, c_lr):
        key = (omega, c_full, c_lr, self.kernel_dtype)
        if key not in self._dev:
            ops = (omega, c_full, c_lr)
            if ops not in self._pair_data:
                self._pair_data[ops] = sgx_jk.pair_data(self._sorted_mol, omega, c_full, c_lr)
            self._dev[key] = sgx_jk._DevicePairs(
                self._pair_data[ops], omega, c_full, c_lr, self.kernel_dtype)
        return self._dev[key]

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

    def nuc_grad_method(self):
        raise NotImplementedError('nuclear gradients with seminumerical exchange')

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError('Hessians with seminumerical exchange')

    def to_cpu(self):
        raise NotImplementedError


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
