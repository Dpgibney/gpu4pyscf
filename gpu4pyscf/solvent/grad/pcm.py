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
Gradient of PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import cupy
import ctypes
from cupyx.scipy.special import erf
from pyscf import lib
from pyscf import gto
from pyscf.grad import rhf as rhf_grad
from pyscf.data.elements import is_ghost_atom
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.solvent.pcm import (PI, switch_h, libsolvent, left_multiply_S, left_multiply_D,
                                   has_intermediates_for,
                                   atom_coords_without_ghost, atom_charges_without_ghost, natm_without_ghost)
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from gpu4pyscf.grad.rhf import GradientsBase
from pyscf import lib as pyscf_lib

def gradient_nonghost_to_all(mol, grad):
    natm = grad.shape[0]
    if mol.natm == natm:
        return grad

    natm_nonghost = natm_without_ghost(mol)
    assert grad.shape == (natm_nonghost, 3)

    if isinstance(grad, cupy.ndarray):
        grad_full = cupy.zeros((mol.natm, 3))
    else:
        assert isinstance(grad, numpy.ndarray)
        grad_full = numpy.zeros((mol.natm, 3))
    nonghost_mask = [not is_ghost_atom(element) for element in mol.elements]
    grad_full[nonghost_mask, :] = grad
    return grad_full

def grad_switch_h(x):
    ''' first derivative of h(x)'''
    dy = 30.0*x**2 - 60.0*x**3 + 30.0*x**4
    dy[x<0] = 0.0
    dy[x>1] = 0.0
    return dy

def get_dF_dA(surface, with_dA = True, surface_discretization_method = "SWIG"):
    '''
    J. Chem. Phys. 133, 244111 (2010), Appendix C
    '''
    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    area        = surface['area']
    if surface_discretization_method.upper() == "SWIG":
        R_in_J = surface['R_in_J']
        R_sw_J = surface['R_sw_J']
    elif surface_discretization_method.upper() == "ISWIG":
        charge_exp = surface['charge_exp']
        R_J = surface['R_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]
    dF = cupy.zeros([ngrids, natom, 3])
    if with_dA:
        dA = cupy.zeros([ngrids, natom, 3])
    else:
        dA = None

    for ia in range(natom):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        ri_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        riJ = cupy.linalg.norm(ri_rJ, axis=-1)
        ri_rJ[:,ia,:] = 0.0

        if surface_discretization_method.upper() == "SWIG":
            diJ = (riJ - R_in_J) / R_sw_J
            diJ[:,ia] = 1.0
            diJ[diJ < 1e-8] = 0.0
            ri_rJ[diJ < 1e-8] = 0.0

            fiJ = switch_h(diJ)
            dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
            dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ
        elif surface_discretization_method.upper() == "ISWIG":
            xi = charge_exp[p0:p1]
            erf_input_p = xi[:, None] * (R_J[None, :] + riJ)
            erf_input_m = xi[:, None] * (R_J[None, :] - riJ)
            fiJ = 1 - 0.5 * (erf(erf_input_m) + erf(erf_input_p))
            fiJ[:,ia] = 1.0

            dfiJ = 1/numpy.sqrt(numpy.pi) * xi[:, None] * \
                   (cupy.exp(-erf_input_m**2) - cupy.exp(-erf_input_p**2)) / (fiJ * riJ)
            dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ

            dfiJ[:, ia, :] = -cupy.sum(dfiJ, axis=1) # Translation invariance for diagonal terms
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        Fi = switch_fun[p0:p1]
        if with_dA:
            Ai = area[p0:p1]

        # grids response
        Fi = cupy.expand_dims(Fi, axis=-1)
        dFi_grid = cupy.sum(dfiJ, axis=1)

        dF[p0:p1,ia,:] += Fi * dFi_grid
        if with_dA:
            Ai = cupy.expand_dims(Ai, axis=-1)
            dA[p0:p1,ia,:] += Ai * dFi_grid

        # atom response
        Fi = cupy.expand_dims(Fi, axis=-2)
        dF[p0:p1,:,:] -= Fi * dfiJ
        if with_dA:
            Ai = cupy.expand_dims(Ai, axis=-2)
            dA[p0:p1,:,:] -= Ai * dfiJ
    dF = dF.transpose([2,0,1])
    if with_dA:
        dA = dA.transpose([2,0,1])
    return dF, dA

def get_dD_dS_slow(surface, with_S=True, with_D=False):
    '''
    derivative of D and S w.r.t grids, partial_i D_ij = -partial_j D_ij
    S is symmetric, D is not
    '''
    grid_coords = surface['grid_coords']
    exponents   = surface['charge_exp']
    norm_vec    = surface['norm_vec']

    xi_i, xi_j = cupy.meshgrid(exponents, exponents, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    ri_rj = cupy.expand_dims(grid_coords, axis=1) - grid_coords
    rij = cupy.linalg.norm(ri_rj, axis=-1)
    xi_r_ij = xi_ij * rij
    cupy.fill_diagonal(rij, 1)
    xi_i = xi_j = None

    dS_dr = -(erf(xi_r_ij) - 2.0*xi_r_ij/PI**0.5*cupy.exp(-xi_r_ij**2))/rij**2
    cupy.fill_diagonal(dS_dr, 0)

    dS_dr= cupy.expand_dims(dS_dr, axis=-1)
    drij = ri_rj/cupy.expand_dims(rij, axis=-1)
    dS = dS_dr * drij

    dD = None
    if with_D:
        nj_rij = cupy.sum(ri_rj * norm_vec, axis=-1)
        dD_dri = 4.0*xi_r_ij**2 * xi_ij / PI**0.5 * cupy.exp(-xi_r_ij**2) * nj_rij / rij**3
        cupy.fill_diagonal(dD_dri, 0.0)

        rij = cupy.expand_dims(rij, axis=-1)
        nj_rij = cupy.expand_dims(nj_rij, axis=-1)
        nj = cupy.expand_dims(norm_vec, axis=0)
        dD_dri = cupy.expand_dims(dD_dri, axis=-1)

        dD = dD_dri * drij + dS_dr * (-nj/rij + 3.0*nj_rij/rij**2 * drij)
        dD_dri = None
    dD = dD.transpose([2,0,1])
    dS = dS.transpose([2,0,1])
    return dD, dS

def get_dD_dS(surface, with_S=True, with_D=False, stream=None):
    ''' Derivatives of D matrix and S matrix (offdiagonals only)
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    norm_vec    = surface['norm_vec']
    n = charge_exp.shape[0]
    dS = cupy.empty([3,n,n])
    dD = None
    dS_ptr = ctypes.cast(dS.data.ptr, ctypes.c_void_p)
    dD_ptr = pyscf_lib.c_null_ptr()
    if with_D:
        dD = cupy.empty([3,n,n])
        dD_ptr = ctypes.cast(dD.data.ptr, ctypes.c_void_p)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_dd_ds(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        dD_ptr, dS_ptr,
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(norm_vec.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM dD and dS matrices.')
    return dD, dS

def left_multiply_dS_offdiagonal(surface, right_vector, transpose = False, stream = None):
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    n = charge_exp.shape[0]
    assert right_vector.size == n
    output = cupy.empty([3,n], order = "C")
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_left_multiply_ds(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(output.data.ptr, ctypes.c_void_p),
        ctypes.cast(right_vector.data.ptr, ctypes.c_void_p),
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in left_multiply_dS_offdiagonal')
    if transpose:
        # S is symmetric and Sij depends only on ri and rj
        output *= -1
    return output.T

def left_multiply_dD(surface, right_vector, transpose = False, stream = None):
    grid_coords = surface['grid_coords']
    charge_exp  = surface['charge_exp']
    norm_vec    = surface['norm_vec']
    n = charge_exp.shape[0]
    assert right_vector.size == n
    assert type(transpose) is bool
    output = cupy.empty([3,n], order = "C")
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_left_multiply_dd(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(output.data.ptr, ctypes.c_void_p),
        ctypes.cast(right_vector.data.ptr, ctypes.c_void_p),
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.cast(norm_vec.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n),
        ctypes.c_bool(transpose),
    )
    if err != 0:
        raise RuntimeError('Failed in left_multiply_dD')
    return output.T

def get_dSii(surface, dF):
    ''' Derivative of S matrix (diagonal only)
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    dSii_dF = -charge_exp * (2.0/PI)**0.5 / switch_fun**2
    dSii = dSii_dF[:,None] * dF
    return dSii

def contract_dSii(surface, contract_vector, surface_discretization_method = "SWIG"):
    '''
        Returns contract('g,dgA->Ad', contract_vector, dSii)
        The main purpose of this function is to save memory, as dSii requires 3*natm*ngrids storage.
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    n = charge_exp.shape[0]
    assert contract_vector.shape == (n,)

    dSii_dF = -numpy.sqrt(2.0/PI) * charge_exp / switch_fun**2
    contract_vector *= dSii_dF
    dSii_dF = None

    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    if surface_discretization_method.upper() == "SWIG":
        R_in_J = surface['R_in_J']
        R_sw_J = surface['R_sw_J']
    elif surface_discretization_method.upper() == "ISWIG":
        charge_exp = surface['charge_exp']
        R_J = surface['R_J']

    natom = atom_coords.shape[0]
    de_dSii_term = cupy.zeros([natom, 3])

    for ia in range(natom):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        ri_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        riJ = cupy.linalg.norm(ri_rJ, axis=-1)
        ri_rJ[:,ia,:] = 0.0

        if surface_discretization_method.upper() == "SWIG":
            diJ = (riJ - R_in_J) / R_sw_J
            diJ[:,ia] = 1.0
            diJ[diJ < 1e-8] = 0.0
            ri_rJ[diJ < 1e-8] = 0.0

            fiJ = switch_h(diJ)
            dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
            dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ
        elif surface_discretization_method.upper() == "ISWIG":
            xi = charge_exp[p0:p1]
            erf_input_p = xi[:, None] * (R_J[None, :] + riJ)
            erf_input_m = xi[:, None] * (R_J[None, :] - riJ)
            fiJ = 1 - 0.5 * (erf(erf_input_m) + erf(erf_input_p))
            fiJ[:,ia] = 1.0

            dfiJ = 1/numpy.sqrt(numpy.pi) * xi[:, None] * \
                   (cupy.exp(-erf_input_m**2) - cupy.exp(-erf_input_p**2)) / (fiJ * riJ)
            dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ

            dfiJ[:, ia, :] = -cupy.sum(dfiJ, axis=1) # Translation invariance for diagonal terms
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        Fi = switch_fun[p0:p1]

        # grids response
        Fi = cupy.expand_dims(Fi, axis=-1)
        dFi_grid = cupy.sum(dfiJ, axis=1)
        # dF[p0:p1,ia,:] += Fi * dFi_grid
        de_dSii_term[ia,:] += contract("gd,g->d", Fi * dFi_grid, contract_vector[p0:p1])

        # atom response
        Fi = cupy.expand_dims(Fi, axis=-2)
        # dF[p0:p1,:,:] -= Fi * dfiJ
        de_dSii_term -= contract("gAd,g->Ad", Fi * dfiJ, contract_vector[p0:p1])

    return de_dSii_term

def contract_dA(surface, contract_vector, surface_discretization_method = "SWIG"):
    '''
        Returns contract('g,dgA->Ad', contract_vector, dA)
        The main purpose of this function is to save memory, as dA requires 3*natm*ngrids storage.
        If not in short of memory, it is preferable to compute dA together with dF (part of dSii) using get_dF_dA().
    '''
    charge_exp  = surface['charge_exp']
    n = charge_exp.shape[0]
    assert contract_vector.shape == (n,)

    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    area        = surface['area']
    if surface_discretization_method.upper() == "SWIG":
        R_in_J = surface['R_in_J']
        R_sw_J = surface['R_sw_J']
    elif surface_discretization_method.upper() == "ISWIG":
        charge_exp = surface['charge_exp']
        R_J = surface['R_J']

    natom = atom_coords.shape[0]
    de_dA_term = cupy.zeros([natom, 3])

    for ia in range(natom):
        p0,p1 = surface['gslice_by_atom'][ia]
        coords = grid_coords[p0:p1]
        ri_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        riJ = cupy.linalg.norm(ri_rJ, axis=-1)
        ri_rJ[:,ia,:] = 0.0

        if surface_discretization_method.upper() == "SWIG":
            diJ = (riJ - R_in_J) / R_sw_J
            diJ[:,ia] = 1.0
            diJ[diJ < 1e-8] = 0.0
            ri_rJ[diJ < 1e-8] = 0.0

            fiJ = switch_h(diJ)
            dfiJ = grad_switch_h(diJ) / (fiJ * riJ * R_sw_J)
            dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ
        elif surface_discretization_method.upper() == "ISWIG":
            xi = charge_exp[p0:p1]
            erf_input_p = xi[:, None] * (R_J[None, :] + riJ)
            erf_input_m = xi[:, None] * (R_J[None, :] - riJ)
            fiJ = 1 - 0.5 * (erf(erf_input_m) + erf(erf_input_p))
            fiJ[:,ia] = 1.0

            dfiJ = 1/numpy.sqrt(numpy.pi) * xi[:, None] * \
                   (cupy.exp(-erf_input_m**2) - cupy.exp(-erf_input_p**2)) / (fiJ * riJ)
            dfiJ = cupy.expand_dims(dfiJ, axis=-1) * ri_rJ

            dfiJ[:, ia, :] = -cupy.sum(dfiJ, axis=1) # Translation invariance for diagonal terms
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        Ai = area[p0:p1]

        # grids response
        Ai = cupy.expand_dims(Ai, axis=-1)
        dFi_grid = cupy.sum(dfiJ, axis=1)
        # dA[p0:p1,ia,:] += Ai * dFi_grid
        de_dA_term[ia,:] += contract("gd,g->d", Ai * dFi_grid, contract_vector[p0:p1])

        # atom response
        Ai = cupy.expand_dims(Ai, axis=-2)
        # dA[p0:p1,:,:] -= Ai * dfiJ
        de_dA_term -= contract("gAd,g->Ad", Ai * dfiJ, contract_vector[p0:p1])

    return de_dA_term

def grad_nuc(pcmobj, dm, q_sym = None):
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)

    mol = pcmobj.mol
    if q_sym is None:
        q_sym = pcmobj._intermediates['q_sym'].get()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords'].get()
    exponents    = pcmobj.surface['charge_exp'].get()

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        # Note: The q_sym computed above actually use frozen_dm0 as input, so it's actually q_sym_right
        q_sym_left, _ = pcmobj._get_qsym(dm, with_nuc = True)
        q_sym_left = q_sym_left.get()
        q_sym += q_sym_left

    atom_coords = atom_coords_without_ghost(mol)
    atom_charges = numpy.asarray(atom_charges_without_ghost(mol), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')

    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)

    dv_g = numpy.einsum('g,xng->nx', q_sym, v_ng_ip1)
    de = -numpy.einsum('nx,n->nx', dv_g, atom_charges)

    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)

    dv_g = numpy.einsum('n,xgn->gx', atom_charges, v_ng_ip1)
    dv_g = numpy.einsum('gx,g->gx', dv_g, q_sym)

    de -= numpy.asarray([numpy.sum(dv_g[p0:p1], axis=0) for p0,p1 in gridslice])
    de = gradient_nonghost_to_all(mol, de)
    t1 = log.timer_debug1('grad nuc', *t1)
    return de

def _int3c2e_opt_for_charges(pcmobj):
    '''The 3-center integral data of the molecule's basis with the surface
    charges of pcmobj as auxiliary functions, one normalized s Gaussian and
    one center each'''
    from gpu4pyscf.df.int3c2e_bdiv import Int3c2eOpt
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    return Int3c2eOpt(pcmobj.mol, fakemol).build(1e-14)

def _grad_qv_int3c2e(pcmobj, dm, q_sym, int3c2e_opt):
    '''grad_qv's two terms for a symmetric density matrix from the kernel of
    the density-fitted Coulomb gradient (sum_ejk_int3c2e_ip1), with the
    surface charges as its auxiliary vector: sum_ij,g D_ij q_g d(ij|g)/dR for
    the centers of the basis functions, per atom, and, by translational
    invariance, for those of the charges, per charge. One pass of the Rys
    3-center kernels instead of two passes of the int3c1e ones, the first of
    which also builds a [3,nao,nao] matrix: 2.5-3 times faster for caffeine.
    '''
    from gpu4pyscf.df.int3c2e_bdiv import int3c2e_scheme, libvhf_rys
    from gpu4pyscf.gto.mole import _split_l_ctr_groups
    from gpu4pyscf.lib.cupy_helper import transpose_sum
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    gridslice = pcmobj.surface['gslice_by_atom']
    ngrids = pcmobj.surface['grid_coords'].shape[0]

    dm = mol.apply_C_mat_CT(cupy.asarray(dm))
    dm = transpose_sum(dm, inplace=True)
    dm *= .5
    auxvec = auxmol.C_dot_mat(cupy.asarray(q_sym))
    naux = len(auxvec)

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(gout_width=54, deriv=(1,0,0))
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    # Split the charges into small batches for load balance
    _, aux_counts = _split_l_ctr_groups(
        auxmol.uniq_l_ctr, auxmol.l_ctr_counts, [16]*(laux+1))
    ksh_offsets_cpu = numpy.append(0, numpy.cumsum(aux_counts))
    ksh_offsets_gpu = cupy.asarray(ksh_offsets_cpu+mol.nbas, dtype=numpy.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    natm = pcmobj.mol.natm
    # per atom of the basis functions, and per charge (each its own center)
    ej = cupy.zeros((natm, 3))
    ej_grids = cupy.zeros((ngrids, 3))
    err = libvhf_rys.sum_ejk_int3c2e_ip1(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_grids.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1),
        ctypes.c_double(0.), ctypes.c_double(1.), ctypes.c_double(1.),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(int(shm_size_max)),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        pyscf_lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(naux), ctypes.c_int(natm))
    if err != 0:
        raise RuntimeError('sum_ejk_int3c2e_ip1 failed')
    dq = cupy.asarray([cupy.sum(ej_grids[p0:p1], axis=0) for p0,p1 in gridslice])
    dq = gradient_nonghost_to_all(pcmobj.mol, dq)
    # the functions' derivatives of the kernel are those of both functions of
    # each pair of the lower triangle, and the surface charges' potential
    # enters the energy with the sign of the electrons' charge
    return -2.0 * (ej + dq)

def _is_symmetric(dm):
    return (dm.ndim == 2 and dm.shape[0] == dm.shape[1]
            and float(abs(dm - dm.T).max()) <= 1e-12 * max(1., float(abs(dm).max())))

def grad_qv(pcmobj, dm, q_sym = None):
    '''
    contributions due to integrals
    '''
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    gridslice   = pcmobj.surface['gslice_by_atom']
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    if q_sym is None:
        q_sym = pcmobj._intermediates['q_sym']

    from gpu4pyscf.df.int3c2e_bdiv import LMAX
    frozen_dm0 = pcmobj.frozen_dm0_for_finite_difference_without_response
    dm = cupy.asarray(dm)
    if (mol._bas[:,gto.ANG_OF].max() <= LMAX and q_sym.ndim == 1 and _is_symmetric(dm)
            and (frozen_dm0 is None or _is_symmetric(cupy.asarray(frozen_dm0)))):
        # One pass of the Rys 3-center kernels (symmetric densities; a
        # density that is not symmetric takes the int3c1e kernels below, whose
        # first term is the derivative of the bra functions alone)
        int3c2e_opt = _int3c2e_opt_for_charges(pcmobj)
        de = _grad_qv_int3c2e(pcmobj, dm, q_sym, int3c2e_opt)
        if frozen_dm0 is not None:
            # Note: The q_sym computed above actually use frozen_dm0 as input, so it's actually q_sym_right
            q_sym_left, _ = pcmobj._get_qsym(dm, with_nuc = True)
            de += _grad_qv_int3c2e(pcmobj, frozen_dm0, q_sym_left, int3c2e_opt)
        t1 = log.timer_debug1('grad qv', *t1)
        return de.get()

    intopt = int3c1e.VHFOpt(mol)
    intopt.build(1e-14, aosym=False)
    dvj = int1e_grids_ip1(mol, grid_coords, dm = dm, charges = q_sym,
                          direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                          intopt=intopt)
    dq  = int1e_grids_ip2(mol, grid_coords, dm = dm, charges = q_sym,
                          direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                          intopt=intopt)

    if frozen_dm0 is not None:
        # Note: The q_sym computed above actually use frozen_dm0 as input, so it's actually q_sym_right
        q_sym_left, _ = pcmobj._get_qsym(dm, with_nuc = True)
        dvj += int1e_grids_ip1(mol, grid_coords, dm = frozen_dm0, charges = q_sym_left,
                              direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                              intopt=intopt)
        dq  += int1e_grids_ip2(mol, grid_coords, dm = frozen_dm0, charges = q_sym_left,
                              direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                              intopt=intopt)

    aoslice = mol.aoslice_by_atom()
    dvj = 2.0 * cupy.asarray([cupy.sum(dvj[:,p0:p1], axis=1) for p0,p1 in aoslice[:,2:]])
    dq = cupy.asarray([cupy.sum(dq[:,p0:p1], axis=1) for p0,p1 in gridslice])
    dq = gradient_nonghost_to_all(mol, dq)
    de = dq + dvj
    t1 = log.timer_debug1('grad qv', *t1)
    return de.get()

def grad_solver(pcmobj, dm, v_grids = None, v_grids_l = None, q = None):
    '''
    dE = 0.5*v* d(K^-1 R) *v + q*dv
    v^T* d(K^-1 R)v = v^T*K^-1(dR - dK K^-1R)v = v^T K^-1(dR v - dK q)
    '''
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = log.init_timer()
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    if v_grids is None:
        v_grids = pcmobj._intermediates['v_grids']
    if v_grids_l is None:
        v_grids_l = pcmobj._intermediates['v_grids']
    if q is None:
        q = pcmobj._intermediates['q']
    f_epsilon = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        # D, S and their derivatives act on vectors straight from the surface
        # data, whether or not dense matrices exist: the [3,n,n] arrays of the
        # derivatives, 6 n^2 numbers, are never formed.
        A = pcmobj._intermediates['A']
        def DT_dot(x):
            return left_multiply_D(pcmobj.surface, x, transpose = True)
        def S_dot(x):
            return left_multiply_S(pcmobj.surface, x)

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        # Note: The v_grids computed above actually use frozen_dm0 as input, so it's actually v_grids_right
        v_grids_l = pcmobj._get_vgrids(dm, with_nuc = True)[0]

        # TODO: In the case where left and right dm are not the same,
        #       we indeed need to compute the derivative of 0.5 * (K^-1 R + R^T (K^-1)^T).
        #       This is not the same as the derivative of K^-1 R, if IEFPCM or SSVPE is used.
        #       However the difference is too small, and in the use case of computing polarizability derivative,
        #       we are not able to observe the difference.
        #       If there are other use cases where the error is more significant,
        #       we probably need to fix this.
        #       This is the only term affected by this problem in energy and gradient calculation.
        #       In hessian calculation, similar problems occur in energy 2nd derivative and Fock derivative terms.

    vK_1 = pcmobj.left_solve_K(v_grids_l, K_transpose = True)

    natm = natm_without_ghost(pcmobj.mol)
    de = cupy.zeros([natm, 3])
    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        # dR = 0, dK = dS
        de_dS  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, q, stream = None)
        de_dS -= 0.5 * q.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, vK_1, transpose = True, stream = None)
        de -= cupy.asarray([cupy.sum(de_dS[p0:p1], axis=0) for p0,p1 in gridslice])

        de -= 0.5 * contract_dSii(pcmobj.surface, vK_1 * q, pcmobj.surface_discretization_method)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * vK_1.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, Av, stream = None)
        de_dR -= 0.5*fac * Av.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dR  = cupy.asarray([cupy.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dS0  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, q, stream = None)
        de_dS0 -= 0.5 * q.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dS0  = cupy.asarray([cupy.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = DT_dot(vK_1)
        vK_1_DA = vK_1_D * A
        de_dS1  = 0.5 * vK_1_DA.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, q, stream = None)
        de_dS1 -= 0.5 * q.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, vK_1_DA, transpose = True, stream = None)
        de_dS1  = cupy.asarray([cupy.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])

        Sq = S_dot(q)
        ASq = A * Sq
        de_dD  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, ASq, stream = None)
        de_dD -= 0.5 * ASq.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dD  = cupy.asarray([cupy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dK = de_dS0 - fac * (de_dD + de_dS1)
        de_dK += 0.5 * contract_dSii(pcmobj.surface, vK_1 * q - fac * vK_1_DA * q, pcmobj.surface_discretization_method)

        de += de_dR - de_dK
        de += 0.5*fac * contract_dA(pcmobj.surface, vK_1_D * v_grids + vK_1_D * Sq,
                                    pcmobj.surface_discretization_method) # First term from de_dR, second from de_dK

    elif pcmobj.method.upper() in [ 'SS(V)PE' ]:
        # dR = f_eps/(2*pi) * (dD*A + D*dA),
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        fac = f_epsilon/(2.0*PI)

        Av = A*v_grids
        de_dR  = 0.5*fac * vK_1.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, Av, stream = None)
        de_dR -= 0.5*fac * Av.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dR  = cupy.asarray([cupy.sum(de_dR[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dS0  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, q, stream = None)
        de_dS0 -= 0.5 * q.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dS0  = cupy.asarray([cupy.sum(de_dS0[p0:p1], axis=0) for p0,p1 in gridslice])

        vK_1_D = DT_dot(vK_1)
        vK_1_DA = vK_1_D * A
        de_dS1  = 0.5 * vK_1_DA.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, q, stream = None)
        de_dS1 -= 0.5 * q.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, vK_1_DA, transpose = True, stream = None)
        de_dS1  = cupy.asarray([cupy.sum(de_dS1[p0:p1], axis=0) for p0,p1 in gridslice])

        DT_q = DT_dot(q)
        ADT_q = A * DT_q
        de_dS1_T  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, ADT_q, stream = None)
        de_dS1_T -= 0.5 * ADT_q.reshape(-1, 1) * left_multiply_dS_offdiagonal(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dS1_T  = cupy.asarray([cupy.sum(de_dS1_T[p0:p1], axis=0) for p0,p1 in gridslice])

        Sq = S_dot(q)
        ASq = A * Sq
        de_dD  = 0.5 * vK_1.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, ASq, stream = None)
        de_dD -= 0.5 * ASq.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, vK_1, transpose = True, stream = None)
        de_dD  = cupy.asarray([cupy.sum(de_dD[p0:p1], axis=0) for p0,p1 in gridslice])

        # S is symmetric
        vK_1_S = S_dot(vK_1)
        vK_1_SA = vK_1_S * A
        # Attention on the transposed order here
        de_dD_T  = 0.5 * q.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, vK_1_SA, stream = None)
        de_dD_T -= 0.5 * vK_1_SA.reshape(-1, 1) * left_multiply_dD(pcmobj.surface, q, transpose = True, stream = None)
        de_dD_T  = cupy.asarray([cupy.sum(de_dD_T[p0:p1], axis=0) for p0,p1 in gridslice])

        de_dK = de_dS0 - 0.5*fac * (de_dD + de_dS1 + de_dD_T + de_dS1_T)
        de_dK += 0.5 * contract_dSii(pcmobj.surface, vK_1 * q - 0.5*fac * (vK_1_DA * q + vK_1 * ADT_q), pcmobj.surface_discretization_method)

        de += de_dR - de_dK
        de += 0.5*fac * contract_dA(pcmobj.surface, vK_1_D * v_grids + 0.5 * (vK_1_D * Sq + vK_1_S * DT_q),
                                    pcmobj.surface_discretization_method) # First term from de_dR, second from de_dK

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    if pcmobj.frozen_dm0_for_finite_difference_without_response is not None:
        # Refer to the comments in gpu4pyscf/solvent/pcm.py::_get_vind()
        de *= 2

    de = gradient_nonghost_to_all(mol, de)

    t1 = log.timer_debug1('grad solver', *t1)
    return de.get()

def make_grad_object(base_method):
    '''Create nuclear gradients object with solvent contributions for the given
    solvent-attached method based on its gradients method in vaccum
    '''
    if isinstance(base_method, GradientsBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a gradient object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    # create the Gradients in vacuum. Cannot call super().Gradients() here
    # because other dynamic corrections might be applied to the base_method.
    # Calling super().Gradients might discard these corrections.
    vac_grad = base_method.undo_solvent().Gradients()
    # The base method for vac_grad discards the with_solvent. Change its base to
    # the solvent-attached base method
    vac_grad.base = base_method
    name = with_solvent.__class__.__name__ + vac_grad.__class__.__name__
    return lib.set_class(WithSolventGrad(vac_grad),
                         (WithSolventGrad, vac_grad.__class__), name)

class WithSolventGrad:
    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventGrad, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        from pyscf.solvent.grad import pcm  # type: ignore
        return self.base.to_cpu().PCM().Gradients()

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        logger.debug(self, 'Compute gradients from solvents')
        self.de_solvent = self.base.with_solvent.grad(dm)
        logger.debug(self, 'Compute gradients from solutes')
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent

        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s (+%s) gradients ---------------',
                        self.base.__class__.__name__,
                        self.base.with_solvent.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')
        return self.de

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
