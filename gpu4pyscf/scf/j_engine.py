# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
J engine using McMurchie-Davidson algorithm
'''

import ctypes
import math
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF, PTR_COORD
from pyscf.scf import _vhf
from gpu4pyscf.lib.cupy_helper import (
    load_library, condense, dist_matrix, transpose_sum, hermi_triu, asarray,
    ndarray)
from gpu4pyscf.__config__ import num_devices, shm_size
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.scf import jk
from gpu4pyscf.scf.jk import _nearest_power2
from gpu4pyscf.gto.mole import SortedGTO, _scale_sp_ctr_coeff

__all__ = [
    'get_j',
]

PTR_BAS_COORD = 7
LMAX = 4
SHM_SIZE = shm_size - 1024
THREADS = 256
# The least blocks per SM a launch of the generic MD J kernel gets (twice that for
# diagonal classes): its bra tile is cut down to reach it (_fill_sms_tilex), as the
# unrolled kernels choose theirs at run time.
J_BLOCKS_PER_SM = 64

libvhf_md = load_library('libgvhf_md')
libvhf_md.MD_build_j.restype = ctypes.c_int
libvhf_md.dm_to_Rt.restype = ctypes.c_int
libvhf_md.Rt_to_dm.restype = ctypes.c_int

def get_j(mol, dm, hermi=1, vhfopt=None, verbose=None):
    r'''Compute Coulomb (J) matrix.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Mole object that defines the basis set and molecular geometry.
    dm : ndarray or sequence of ndarray
        Density matrix or a sequence of density matrices. If multiple density
        matrices are provided, the corresponding J and K matrices are computed
        for each input density matrix.
    hermi : int, optional
        Symmetry of the density matrix:
        - ``0``: no symmetry
        - ``1``: Hermitian
        - ``2``: anti-Hermitian
    vhfopt : gpu4ypscf.scf.j_engine._VHFOpt object, optional
        Handler for J matrix computation.

    Returns
    -------
    vj : ndarray or sequence of ndarray
        Exchange matrices. The returned object has the same shape as the input
        density matrices.
    '''
    log = logger.new_logger(mol, verbose)
    cput0 = log.init_timer()
    if vhfopt is None:
        vhfopt = _VHFOpt(mol).build()

    nao_orig = mol.nao
    dm = cp.asarray(dm, order='C')
    dms = dm.reshape(-1,nao_orig,nao_orig)
    dms = vhfopt.apply_coeff_C_mat_CT(dms)
    if hermi != 1:
        dms = transpose_sum(dms)
        dms *= .5

    vj = vhfopt.get_j(dms, log)
    #:vj = cp.einsum('pi,npq,qj->nij', vhfopt.coeff, cp.asarray(vj), vhfopt.coeff)
    vj = vhfopt.apply_coeff_CT_mat_C(vj)
    vj = vj.reshape(dm.shape)
    log.timer('vj', *cput0)
    return vj

class _VHFOpt(jk._VHFOpt):
    def build(self, group_size=None, verbose=None):
        log = logger.new_logger(self.mol, verbose)
        cput0 = log.init_timer()
        mol = self.sorted_mol = SortedGTO.from_mol(
            self.mol, decontract=True, diffuse_cutoff=1e200)

        # very high angular momentum basis are processed on CPU
        uniq_l_ctr = mol.uniq_l_ctr
        l_ctr_counts = mol.l_ctr_counts
        lmax = uniq_l_ctr[:,0].max()
        nbas_by_l = [l_ctr_counts[uniq_l_ctr[:,0]==l].sum() for l in range(lmax+1)]
        l_slices = np.append(0, np.cumsum(nbas_by_l))
        if lmax > LMAX:
            self.h_shls = l_slices[LMAX+1:].tolist()
        else:
            self.h_shls = []

        self.bas_pair_cache = _cache_q_cond_and_non0pairs(
            mol, self.rys_envs, self.direct_scf_tol)
        log.timer('Initialize q_cond', *cput0)
        return self

    def get_j(self, dms, verbose):
        log = logger.new_logger(self.mol, verbose)
        mol = self.sorted_mol
        if callable(dms):
            dms = dms()
        ao_loc = mol.ao_loc
        n_dm, nao = dms.shape[:2]
        assert dms.ndim == 3 and nao == ao_loc[-1]
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_cutoff = math.log(self.direct_scf_tol)

        l_counts = np.bincount(mol._bas[:,ANG_OF])[:LMAX+1]
        n_groups = len(l_counts)
        l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
        l_symb = lib.param.ANGULAR
        pair_mappings = _make_pair_qd_cond(mol, self.bas_pair_cache, dm_cond)
        dm_cond = None

        pair_lst = []
        task_offsets = {} # the pair_loc offsets for each ij pair
        p0 = p1 = 0
        ij_tasks = ((i, j) for i in range(n_groups) for j in range(i+1))
        for key in ij_tasks:
            pair_ij_mapping = pair_mappings[key][0]
            pair_lst.append(pair_ij_mapping)
            p0, p1 = p1, p1 + pair_ij_mapping.size
            task_offsets[key] = p0
        pair_mapping_size = p1
        pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)

        ls = cp.asarray(mol._bas[:,ANG_OF], dtype=np.int32)
        ll = ls[:,None] + ls
        ll = ll.ravel()[pair_lst] # drops the pairs that do not contribute to integrals
        xyz_size = (ll+1)*(ll+2)*(ll+3)//6
        pair_loc = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32), xyz_size.ravel()), dtype=np.int32)
        xyz_size = ls = ll = None

        _env = _scale_sp_ctr_coeff(mol)

        dm_xyz_size = int(pair_loc[-1])
        log.debug1('dm_xyz_size = %s, nao = %s, pair_mapping_size = %s',
                   dm_xyz_size, nao, pair_mapping_size)

        if self.h_shls:
            dm_xyz = _dm_to_Rt(self.sorted_mol, dms, pair_lst, pair_loc)
        else:
            dm_xyz = _dm_to_Rt(self.sorted_mol, dms, pair_lst, pair_loc, self.rys_envs)

        tasks = []
        for i in range(n_groups):
            for j in range(i+1):
                for k in range(i+1):
                    for l in range(k+1):
                        if i == k and j < l: continue
                        tasks.append((i,j,k,l))
        schemes = {t: _md_j_engine_quartets_scheme(t, n_dm=n_dm) for t in tasks}
        tasks = iter(tasks)

        def proc(dm_xyz):
            device_id = cp.cuda.device.get_device_id()
            stream = cp.cuda.stream.get_current_stream()
            log = logger.new_logger(self.mol, verbose)
            t1 = log.init_timer()

            dm_xyz = asarray(dm_xyz) # transfer to current device
            vj_xyz = cp.zeros_like(dm_xyz)

            _pair_mappings = pair_mappings
            if device_id > 0: # Ensure the precomputation avail on each device
                _pair_mappings = {k: [cp.asarray(x) for x in v]
                                  for k, v in pair_mappings.items()}
            _pair_loc_gpu = cp.asarray(pair_loc)

            timing_collection = jk._TimingCollector(log.timer_debug1)
            kern_counts = 0
            kern = libvhf_md.MD_build_j
            n_sm = cp.cuda.Device().attributes['MultiProcessorCount']
            rys_envs = self.rys_envs

            for task in tasks:
                i, j, k, l = task
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_mapping, q_cond_ij, qd_ij = _pair_mappings[i,j]
                pair_kl_mapping, q_cond_kl, qd_kl = _pair_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                pair_ij_loc = _pair_loc_gpu[task_offsets[i,j]:]
                pair_kl_loc = _pair_loc_gpu[task_offsets[k,l]:]
                scheme = _fill_sms_tilex(schemes[task], npairs_ij, npairs_kl,
                                         (i, j) == (k, l), n_sm)
                err = kern(
                    ctypes.cast(vj_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
                    ctypes.byref(rys_envs), (ctypes.c_int*6)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)

                llll = f'({l_symb[i]}{l_symb[j]}|{l_symb[k]}{l_symb[l]})'
                if err != 0:
                    raise RuntimeError(f'MD_build_j kernel for {llll} failed')
                kern_counts += 1

                if log.verbose >= logger.DEBUG1:
                    ntasks = npairs_ij * npairs_kl
                    msg = f'processing {llll}, scheme={scheme} tasks ~= {ntasks}'
                    t1 = timing_collection.collect(llll, t1, msg)
                if num_devices > 1:
                    stream.synchronize()
            return vj_xyz, kern_counts, timing_collection

        results = multi_gpu.run(proc, args=(dm_xyz,), non_blocking=True)
        if log.verbose >= logger.DEBUG1:
            log.debug1('kernel launches %d', sum(x[1] for x in results))
            jk._TimingCollector.summary(log.debug1, (x[2] for x in results))

        vj_xyz = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        vj_xyz = vj_xyz.get()

        h_shls = self.h_shls
        if h_shls:
            vj = _Rt_to_dm(self.sorted_mol, vj_xyz, pair_lst, pair_loc)
            vj *= 2.
            mol = self.sorted_mol
            log.debug3('Integrals for %s functions on CPU',
                       lib.param.ANGULAR[LMAX+1])
            scripts = 'ji->s2kl'
            shls_excludes = [0, h_shls[0]] * 4
            vs_h = _vhf.direct_mapdm('int2e_cart', 's8', scripts,
                                     dms.get(), 1, mol._atm, mol._bas, mol._env,
                                     shls_excludes=shls_excludes)
            vj1 = asarray(vs_h)
            vj += hermi_triu(vj1)
        else:
            vj = _Rt_to_dm(self.sorted_mol, vj_xyz, pair_lst, pair_loc, self.rys_envs)
            vj *= 2.
        return vj

    def get_ej_ip1(self, dm, verbose=None):
        '''Nuclear derivatives of the Coulomb energy 1/2 Tr(D J[D]) for a density
        matrix in the AO basis, an array of shape (natm, 3).

        The derivative of a pair's charge distribution with respect to one of its
        centers, d/dA_x (G_i G_j) = 2 a G_{i+1_x} G_j - i_x G_{i-1_x} G_j, is a
        Hermite expansion one order above the pair's. MD_build_j_ip1 gives every
        pair's Hermite potential to that order (the J engine with the densities in a
        layout of one more order per pair, the top components zero), and
        Rt_ip1_dot_dm contracts them with the pairs' derivative densities per atom.
        '''
        log = logger.new_logger(self.mol, verbose)
        cput0 = log.init_timer()
        mol = self.sorted_mol
        assert not self.h_shls
        dm = cp.asarray(dm)
        dms = self.apply_coeff_C_mat_CT(transpose_sum(dm[None].copy()) * .5)
        ao_loc = mol.ao_loc
        dm_cond = cp.log(condense('absmax', dms, ao_loc) + 1e-300).astype(np.float32)
        log_cutoff = math.log(self.direct_scf_tol)

        l_counts = np.bincount(mol._bas[:,ANG_OF])[:LMAX+1]
        n_groups = len(l_counts)
        l_ctr_bas_loc = np.cumsum(np.append(0, l_counts))
        pair_mappings = _make_pair_qd_cond(mol, self.bas_pair_cache, dm_cond)
        dm_cond = None

        pair_lst = []
        task_offsets = {}
        p1 = 0
        for key in ((i, j) for i in range(n_groups) for j in range(i+1)):
            pair_lst.append(pair_mappings[key][0])
            task_offsets[key] = p1
            p1 += pair_mappings[key][0].size
        pair_lst = cp.asarray(cp.hstack(pair_lst), dtype=np.int32)
        ls = cp.asarray(mol._bas[:,ANG_OF], dtype=np.int32)
        ll = (ls[:,None] + ls).ravel()[pair_lst]
        pair_loc = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32),
                                       (ll+1)*(ll+2)*(ll+3)//6), dtype=np.int32)
        # one Hermite order more per pair
        pair_loc1 = cp.cumsum(cp.append(cp.zeros(1, dtype=np.int32),
                                        (ll+2)*(ll+3)*(ll+4)//6), dtype=np.int32)
        ls = ll = None
        dm_xyz = _dm_to_Rt(mol, dms, pair_lst, pair_loc, self.rys_envs)[0]
        dm1_size = int(pair_loc1[-1])
        dm1 = cp.zeros(dm1_size)
        for (i, j), p0 in task_offsets.items():
            n = pair_mappings[i,j][0].size
            if n == 0:
                continue
            a0, a1 = int(pair_loc[p0]), int(pair_loc[p0+n])
            b0, b1 = int(pair_loc1[p0]), int(pair_loc1[p0+n])
            dm1[b0:b1].reshape(n, -1)[:,_herm_in_next_order(i+j)] = dm_xyz[a0:a1].reshape(n, -1)
        dm_xyz = None

        tasks = [(i, j, k, l)
                 for i in range(n_groups) for j in range(i+1)
                 for k in range(i+1) for l in range(k+1)
                 if not (i == k and j < l)]
        schemes = {t: _md_j_engine_quartets_scheme((t[0]+1, t[1], t[2]+1, t[3]))
                   for t in tasks}
        tasks = iter(tasks)
        _env = _scale_sp_ctr_coeff(mol)

        def proc(dm1):
            device_id = cp.cuda.device.get_device_id()
            dm1 = asarray(dm1)
            vj1 = cp.zeros_like(dm1)
            _pair_mappings = pair_mappings
            if device_id > 0:
                _pair_mappings = {k: [cp.asarray(x) for x in v]
                                  for k, v in pair_mappings.items()}
            _pair_loc1 = cp.asarray(pair_loc1)
            kern = libvhf_md.MD_build_j_ip1
            n_sm = cp.cuda.Device().attributes['MultiProcessorCount']
            for task in tasks:
                i, j, k, l = task
                pair_ij_mapping, q_cond_ij, qd_ij = _pair_mappings[i,j]
                pair_kl_mapping, q_cond_kl, qd_kl = _pair_mappings[k,l]
                npairs_ij = pair_ij_mapping.size
                npairs_kl = pair_kl_mapping.size
                if npairs_ij == 0 or npairs_kl == 0:
                    continue
                shls_slice = l_ctr_bas_loc[[i, i+1, j, j+1, k, k+1, l, l+1]]
                pair_ij_loc = _pair_loc1[task_offsets[i,j]:]
                pair_kl_loc = _pair_loc1[task_offsets[k,l]:]
                scheme = _fill_sms_tilex(schemes[task], npairs_ij, npairs_kl,
                                         (i, j) == (k, l), n_sm)
                err = kern(
                    ctypes.cast(vj1.data.ptr, ctypes.c_void_p),
                    ctypes.cast(dm1.data.ptr, ctypes.c_void_p),
                    ctypes.c_int(1), ctypes.c_int(dm1_size),
                    ctypes.byref(self.rys_envs), (ctypes.c_int*6)(*scheme),
                    (ctypes.c_int*8)(*shls_slice),
                    ctypes.c_int(npairs_ij), ctypes.c_int(npairs_kl),
                    ctypes.cast(pair_ij_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_mapping.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_ij_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(pair_kl_loc.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(qd_kl.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_ij.data.ptr, ctypes.c_void_p),
                    ctypes.cast(q_cond_kl.data.ptr, ctypes.c_void_p),
                    ctypes.c_float(log_cutoff),
                    mol._atm.ctypes, ctypes.c_int(mol.natm),
                    mol._bas.ctypes, ctypes.c_int(mol.nbas), _env.ctypes)
                if err != 0:
                    raise RuntimeError(f'MD_build_j_ip1 kernel for {task} failed')
            if num_devices > 1:
                cp.cuda.stream.get_current_stream().synchronize()
            return vj1

        results = multi_gpu.run(proc, args=(dm1,), non_blocking=True)
        vj1 = multi_gpu.array_reduce(results, inplace=True)
        ej = cp.zeros((mol.natm, 3))
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)  # held until the kernel has run
        err = libvhf_md.Rt_ip1_dot_dm(
            ctypes.cast(ej.data.ptr, ctypes.c_void_p),
            ctypes.cast(vj1.data.ptr, ctypes.c_void_p),
            ctypes.cast(dms.data.ptr, ctypes.c_void_p),
            ctypes.byref(self.rys_envs),
            ctypes.cast(pair_lst.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_loc1.data.ptr, ctypes.c_void_p),
            ctypes.c_int(pair_lst.size),
            ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('Rt_ip1_dot_dm kernel failed')
        # E_J = 2 sum_P D^P.V^P over the unique pairs, and the derivative of either
        # density in it
        ej *= 4.
        log.timer_debug1('J engine gradient', *cput0)
        return ej.get()

def ej_ip1_available():
    '''Whether the library has the J engine's gradient (exp/prim-screen builds)'''
    return hasattr(libvhf_md, 'MD_build_j_ip1') and hasattr(libvhf_md, 'Rt_ip1_dot_dm')

def _herm_in_next_order(l):
    '''The positions of the Hermite components of order l among those of order l+1'''
    nxt = [(t, u, v) for t in range(l+2) for u in range(l+2-t) for v in range(l+2-t-u)]
    return cp.asarray([nxt.index((t, u, v)) for t in range(l+1)
                       for u in range(l+1-t) for v in range(l+1-t-u)])

def _cache_q_cond_and_non0pairs(mol, rys_envs, precision):
    assert all(mol.uniq_l_ctr[:,1] == 1)
    bas_pair_cache = jk._cache_q_cond_and_non0pairs(mol, rys_envs, precision,
                                                    tile=1)
    uniq_l = mol.uniq_l_ctr[:,0]
    lmax = min(uniq_l.max(), LMAX)
    # The pairs for MD-J must include all pairs between angular moments.
    # bas_pair_cache are l_ctr-indexed. The missing (li,lj) entries are filled
    # zero-length orbital-pairs.
    padding = (cp.zeros(0, dtype=np.uint32), cp.zeros(0, dtype=np.float32))
    out = {(li, lj): padding for li in range(lmax+1) for lj in range(li+1)}
    for i, j in bas_pair_cache:
        li = uniq_l[i]
        lj = uniq_l[j]
        pair_ij, q_cond, s_cond = bas_pair_cache[i, j]
        idx = cp.argsort(q_cond)[::-1]
        out[li, lj] = pair_ij[idx], q_cond[idx]
    return out

def _make_pair_qd_cond(mol, bas_pair_cache, dm_cond):
    dm_cond = dm_cond.ravel()
    out = {}
    for k, (pair, q_cond) in bas_pair_cache.items():
        if len(pair) != 0:
            sub_q = q_cond + dm_cond[pair]
            qd_batch_max = _make_tile_max_hierarchy(sub_q)
        else:
            qd_batch_max = q_cond
        out[k] = pair, q_cond, qd_batch_max
    return out

def _make_tile_max_hierarchy(sub_q):
    size_aligned = (sub_q.size+31) & 0xffffffe0 # 32-element aligned
    offset2 = size_aligned
    offset4 = offset2 + size_aligned // 2
    offset8 = offset4 + size_aligned // 4
    offset16 = offset8 + size_aligned // 8
    offset32 = offset16 + size_aligned // 16
    tile_max = cp.full(offset32+size_aligned//32, -700., dtype=np.float32)
    tile_max[:sub_q.size] = sub_q.ravel()
    tile1_max = tile_max[:offset2]
    tile2_max = tile1_max.reshape(-1,2).max(axis=1, out=tile_max[offset2:offset4])
    tile4_max = tile2_max.reshape(-1,2).max(axis=1, out=tile_max[offset4:offset8])
    tile8_max = tile4_max.reshape(-1,2).max(axis=1, out=tile_max[offset8:offset16])
    tile16_max = tile8_max.reshape(-1,2).max(axis=1, out=tile_max[offset16:offset32])
    tile32_max = tile16_max.reshape(-1,2).max(axis=1, out=tile_max[offset32:]) # noqa
    return tile_max

VJ_IJ_REGISTERS = 11
MULTI_VJ_IJ_REGISTERS = 8
RT_TMP_REGISTERS = 31
RT2_IDX_CACHE_SIZE = 35 * 56
def _fill_sms_tilex(scheme, npairs_ij, npairs_kl, diagonal, n_sm):
    '''The scheme with the generic kernel's bra tile (scheme[3], in batches of
    threads_ij pairs) cut down so that the launch has at least J_BLOCKS_PER_SM
    blocks per SM, twice that for diagonal classes (half of their blocks lie above
    the diagonal). The tile sets neither shared memory nor the screening, which is
    per batch (qd_ij_max), so the integrals are the same; only the order of the
    atomic adds changes. Large tiles leave SMs idle on small systems and the
    tail of dense first tiles on large ones.
    '''
    threads_ij, threads_kl, gout_stride, tilex, tiley, buflen = scheme
    nbatch = (npairs_ij + threads_ij - 1) // threads_ij
    grid_y = (npairs_kl + threads_kl * tiley - 1) // (threads_kl * tiley)
    grid_x = (nbatch + tilex - 1) // tilex
    want = J_BLOCKS_PER_SM * n_sm * (2 if diagonal else 1)
    if grid_x * grid_y < want:
        grid_x = min(nbatch, (want + grid_y - 1) // grid_y)
        tilex = (nbatch + grid_x - 1) // grid_x
    return threads_ij, threads_kl, gout_stride, tilex, tiley, buflen

def _md_j_engine_quartets_scheme(ls, shm_size=SHM_SIZE, n_dm=1):
    vj_ij_registers = VJ_IJ_REGISTERS
    if n_dm > 1:
        vj_ij_registers = MULTI_VJ_IJ_REGISTERS
        n_dm = 4

    li, lj, lk, ll = ls
    order = li + lj + lk + ll
    lij = li + lj
    lkl = lk + ll
    nf3ij = (lij+1)*(lij+2)*(lij+3)//6
    nf3kl = (lkl+1)*(lkl+2)*(lkl+3)//6
    Rt_size = (order+1)*(order+2)*(order+3)//6
    gout_stride_min = max(
        _nearest_power2(int((nf3ij+vj_ij_registers-1) / vj_ij_registers), False),
        _nearest_power2(int((Rt_size+RT_TMP_REGISTERS-1) / RT_TMP_REGISTERS), False))

    unit = order+1 + Rt_size
    #counts = shm_size // ((unit+gout_stride_min-1)//gout_stride_min*8)
    counts = shm_size // (unit*8)
    threads = THREADS
    if counts * gout_stride_min >= threads:
        nsq = threads // gout_stride_min
    else:
        nsq = _nearest_power2(counts)
    kl = _nearest_power2(int(nsq**.5))
    ij = nsq // kl

    cache_Rt2_idx = nf3ij * nf3kl <= RT2_IDX_CACHE_SIZE
    if cache_Rt2_idx:
        shm_size -= nf3ij * nf3kl * 2

    tilex = 32
    # Guess number of batches for kl indices
    tiley = (shm_size//8 - nsq*unit - (ij*4+ij*nf3ij*n_dm)) // (kl*4+kl*nf3kl*n_dm)
    tiley = min(tilex, tiley)
    tiley = tiley // 4 * 4
    if tiley < 4:
        tiley = 4
    if li == lk and lj == ll:
        tilex = tiley
    # vj_cache reuses the space which was used by ij*4+ij*nf3ij*n_dm+nsq*unit
    vj_cache_reserve = n_dm * threads - nsq*unit
    cache_size = max(ij*4 + ij*nf3ij*n_dm, vj_cache_reserve) + kl*tiley*4 + kl*nf3kl*tiley*n_dm
    while (nsq * unit + cache_size) * 8 > shm_size:
        nsq //= 2
        assert nsq >= 1
        kl = _nearest_power2(int(nsq**.5))
        ij = nsq // kl
        cache_size = max(ij*4 + ij*nf3ij*n_dm, vj_cache_reserve) + kl*tiley*4 + kl*nf3kl*tiley*n_dm
    gout_stride = threads // nsq
    buflen = (nsq * unit + cache_size) * 8
    if cache_Rt2_idx:
        buflen += nf3ij * nf3kl * 2
    return ij, kl, gout_stride, tilex, tiley, buflen

def _dm_to_Rt(mol, dm, shl_pair_idx, pair_loc, envs=None, out=None):
    ao_loc = mol.ao_loc
    nao = ao_loc[-1]
    dm_ndim = dm.ndim
    dm = dm.reshape(-1,nao,nao)
    n_dm = len(dm)
    dm_xyz_size = int(pair_loc[-1])
    out = ndarray((n_dm, dm_xyz_size), buffer=out)

    if envs is None:
        # on CPU
        dm = cp.asnumpy(dm)
        dm_xyz = np.empty((n_dm, dm_xyz_size))
        pair_loc = np.asarray(cp.asnumpy(pair_loc), dtype=np.int32)
        shl_pair_idx = np.asarray(cp.asnumpy(shl_pair_idx), dtype=np.int32)
        # Must use this modified _env to ensure the consistency with GPU kernel
        # In this _env, normalization coefficients for s and p funcitons are scaled.
        _env = _scale_sp_ctr_coeff(mol)
        libvhf_md.Et_dot_dm(
            dm_xyz.ctypes, dm.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc.ctypes,
            shl_pair_idx.ctypes, ctypes.c_int(len(shl_pair_idx)),
            ctypes.c_int(mol.nbas), mol._bas.ctypes, _env.ctypes)
        out.set(dm_xyz)
    else:
        dm = cp.asarray(dm)
        pair_loc = cp.asarray(pair_loc, dtype=np.int32)
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)
        shl_pair_idx = cp.asarray(shl_pair_idx, dtype=np.int32)
        err = libvhf_md.dm_to_Rt(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.byref(envs),
            ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(shl_pair_idx)),
            ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('dm_to_Rt kernel failed')

    if dm_ndim == 2:
        out = out[0]
    return out

def _Rt_to_dm(mol, dm_xyz, shl_pair_idx, pair_loc, envs=None, out=None):
    ao_loc = mol.ao_loc
    nao = ao_loc[-1]
    dm_ndim = dm_xyz.ndim
    dm_xyz_size = int(pair_loc[-1])
    dm_xyz = dm_xyz.reshape(-1,dm_xyz_size)
    n_dm = len(dm_xyz)
    out = ndarray((n_dm, nao, nao), buffer=out)
    out[:] = 0.

    if envs is None:
        # on CPU
        dm = np.zeros((n_dm, nao, nao))
        dm_xyz = cp.asnumpy(dm_xyz)
        pair_loc = np.asarray(cp.asnumpy(pair_loc), dtype=np.int32)
        shl_pair_idx = np.asarray(cp.asnumpy(shl_pair_idx), dtype=np.int32)
        _env = _scale_sp_ctr_coeff(mol)
        libvhf_md.jengine_dot_Et(
            dm.ctypes, dm_xyz.ctypes,
            ctypes.c_int(n_dm), ctypes.c_int(dm_xyz_size),
            ao_loc.ctypes, pair_loc.ctypes,
            shl_pair_idx.ctypes, ctypes.c_int(len(shl_pair_idx)),
            ctypes.c_int(mol.nbas), mol._bas.ctypes, _env.ctypes)
        out.set(dm)
    else:
        dm_xyz = cp.asarray(dm_xyz)
        pair_loc = cp.asarray(pair_loc, dtype=np.int32)
        ao_loc = cp.asarray(ao_loc, dtype=np.int32)
        shl_pair_idx = cp.asarray(shl_pair_idx, dtype=np.int32)
        err = libvhf_md.Rt_to_dm(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm_xyz.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.byref(envs),
            ctypes.cast(shl_pair_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(shl_pair_idx)),
            ctypes.cast(ao_loc.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('Rt_to_dm kernel failed')

    out = transpose_sum(out)
    if dm_ndim == 1:
        out = out[0]
    return out
