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
Hessian of PCM family solvent model
'''
# pylint: disable=C0103

import numpy
import cupy
import ctypes
from cupyx.scipy.special import erf
from pyscf import lib, gto
from gpu4pyscf import scf
from gpu4pyscf.solvent.pcm import PI, switch_h, libsolvent, natm_without_ghost, has_intermediates_for
from gpu4pyscf.solvent.grad.pcm import grad_qv, get_dD_dS, get_dF_dA, get_dSii, grad_switch_h
from gpu4pyscf.solvent.grad.pcm import left_multiply_dS_offdiagonal
from gpu4pyscf.lib import logger
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.gto.int3c1e_ipip import int1e_grids_ipip1, int1e_grids_ipvip1, int1e_grids_ipip2, int1e_grids_ip1ip2
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.hessian.rhf import HessianBase, _ao2mo
from gpu4pyscf.lib import utils
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem

def gradgrad_switch_h(x):
    ''' 2nd derivative of h(x) '''
    ddy = 60.0*x - 180.0*x**2 + 120.0*x**3
    ddy[x<0] = 0.0
    ddy[x>1] = 0.0
    return ddy

def get_d2F_d2A(surface, surface_discretization_method = "SWIG"):
    '''
    Notations adopted from
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
    d2F = cupy.zeros([ngrids, natom, natom, 3, 3])
    d2A = cupy.zeros([ngrids, natom, natom, 3, 3])

    for i_grid_atom in range(natom):
        p0,p1 = surface['gslice_by_atom'][i_grid_atom]
        coords = grid_coords[p0:p1]
        si_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        norm_si_rJ = cupy.linalg.norm(si_rJ, axis=-1)
        si_rJ[:,i_grid_atom,:] = 0.0

        if surface_discretization_method.upper() == "SWIG":
            diJ = (norm_si_rJ - R_in_J) / R_sw_J
            diJ[:,i_grid_atom] = 1.0
            diJ[diJ < 1e-8] = 0.0
            si_rJ[diJ < 1e-8] = 0.0

            fiJ = switch_h(diJ)
            dfiJ = grad_switch_h(diJ)

            fiJK = fiJ[:, :, cupy.newaxis] * fiJ[:, cupy.newaxis, :]
            dfiJK = dfiJ[:, :, cupy.newaxis] * dfiJ[:, cupy.newaxis, :]
            R_sw_JK = R_sw_J[:, cupy.newaxis] * R_sw_J[cupy.newaxis, :]
            norm_si_rJK = norm_si_rJ[:, :, cupy.newaxis] * norm_si_rJ[:, cupy.newaxis, :]
            terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK * R_sw_JK)
            si_rJK = si_rJ[:, :, cupy.newaxis, :, cupy.newaxis] * si_rJ[:, cupy.newaxis, :, cupy.newaxis, :]
            d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, cupy.newaxis, cupy.newaxis] * si_rJK

            d2fiJ = gradgrad_switch_h(diJ)
            terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2 * R_sw_J) - dfiJ / (norm_si_rJ**3)
            si_rJJ = si_rJ[:, :, :, cupy.newaxis] * si_rJ[:, :, cupy.newaxis, :]
            d2fiJK_diagonal = cupy.einsum('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
            d2fiJK_diagonal += cupy.einsum('qA,dD->qAdD', dfiJ / norm_si_rJ, cupy.eye(3))
            d2fiJK_diagonal /= (fiJ * R_sw_J)[:, :, cupy.newaxis, cupy.newaxis]

            d2fiJK = d2fiJK_offdiagonal
            for i_atom in range(natom):
                d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]
        elif surface_discretization_method.upper() == "ISWIG":
            xi = charge_exp[p0:p1]
            erf_input_p = xi[:, None] * (R_J[None, :] + norm_si_rJ)
            erf_input_m = xi[:, None] * (R_J[None, :] - norm_si_rJ)
            fiJ = 1 - 0.5 * (erf(erf_input_p) + erf(erf_input_m))
            # fiJ[:,i_grid_atom] = 1.0
            dfiJ = 1/numpy.sqrt(numpy.pi) * xi[:, None] * (cupy.exp(-erf_input_m**2) - cupy.exp(-erf_input_p**2))
            ### It is necessary to zero out i \in I term in dfiJ, because the second term of d2fiJK_diagonal
            ### is not zeroed out otherwise.
            dfiJ[:,i_grid_atom] = 0

            fiJK = fiJ[:, :, cupy.newaxis] * fiJ[:, cupy.newaxis, :]
            dfiJK = dfiJ[:, :, cupy.newaxis] * dfiJ[:, cupy.newaxis, :]
            norm_si_rJK = norm_si_rJ[:, :, cupy.newaxis] * norm_si_rJ[:, cupy.newaxis, :]
            terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK)
            si_rJK = si_rJ[:, :, cupy.newaxis, :, cupy.newaxis] * si_rJ[:, cupy.newaxis, :, cupy.newaxis, :]
            d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, cupy.newaxis, cupy.newaxis] * si_rJK

            d2fiJ = 2.0/cupy.sqrt(cupy.pi) * (xi**2)[:, None] * (
                + erf_input_m * cupy.exp(-erf_input_m**2)
                + erf_input_p * cupy.exp(-erf_input_p**2)
            )
            # d2fiJ[:,i_grid_atom] = 0.0

            terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2) - dfiJ / (norm_si_rJ**3)
            si_rJJ = si_rJ[:, :, :, cupy.newaxis] * si_rJ[:, :, cupy.newaxis, :]
            d2fiJK_diagonal = cupy.einsum('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
            d2fiJK_diagonal += cupy.einsum('qA,dD->qAdD', dfiJ / norm_si_rJ, cupy.eye(3))
            d2fiJK_diagonal /= fiJ[:, :, cupy.newaxis, cupy.newaxis]

            d2fiJK = d2fiJK_offdiagonal
            for i_atom in range(natom):
                d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]

        d2F[p0:p1, :, :, :, :] += contract('q,qABdD->qABdD', Fi, d2fiJK)
        d2A[p0:p1, :, :, :, :] += contract('q,qABdD->qABdD', Ai, d2fiJK)

        d2fiJK_grid_atom_offdiagonal = -cupy.einsum('qABdD->qAdD', d2fiJK)
        d2F[p0:p1, i_grid_atom, :, :, :] = contract('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2F[p0:p1, :, i_grid_atom, :, :] = contract('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal)
        d2A[p0:p1, i_grid_atom, :, :, :] = contract('q,qAdD->qAdD', Ai, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2A[p0:p1, :, i_grid_atom, :, :] = contract('q,qAdD->qAdD', Ai, d2fiJK_grid_atom_offdiagonal)

        d2fiJK_grid_atom_diagonal = -cupy.einsum('qAdD->qdD', d2fiJK_grid_atom_offdiagonal)
        d2F[p0:p1, i_grid_atom, i_grid_atom, :, :] = contract('q,qdD->qdD', Fi, d2fiJK_grid_atom_diagonal)
        d2A[p0:p1, i_grid_atom, i_grid_atom, :, :] = contract('q,qdD->qdD', Ai, d2fiJK_grid_atom_diagonal)

    d2F = d2F.transpose(1,2,3,4,0)
    d2A = d2A.transpose(1,2,3,4,0)
    return d2F, d2A

def get_d2Sii(surface, dF, d2F, stream=None):
    ''' Second derivative of S matrix (diagonal only)
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    ngrids = switch_fun.shape[0]
    dF = dF.transpose(2,0,1)
    natm = dF.shape[0]
    assert dF.shape == (natm, 3, ngrids)

    # dF_dF = dF[:, cupy.newaxis, :, cupy.newaxis, :] * dF[cupy.newaxis, :, cupy.newaxis, :, :]
    # dF_dF_over_F3 = dF_dF * (1.0/(switch_fun**3))
    # d2F_over_F2 = d2F * (1.0/(switch_fun**2))
    # d2Sii = 2 * dF_dF_over_F3 - d2F_over_F2
    # d2Sii = (2.0/PI)**0.5 * (d2Sii * charge_exp)

    dF = dF.flatten()   # Make sure the underlying data order is the same as shape shows
    d2F = d2F.flatten() # Make sure the underlying data order is the same as shape shows
    d2Sii = cupy.empty((natm, natm, 3, 3, ngrids), dtype=cupy.float64)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_d2f_to_d2sii(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(switch_fun.data.ptr, ctypes.c_void_p),
        ctypes.cast(dF.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2F.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.cast(d2Sii.data.ptr, ctypes.c_void_p),
        ctypes.c_int(natm),
        ctypes.c_int(ngrids),
    )
    if err != 0:
        raise RuntimeError('Failed in converting PCM d2F to d2Sii.')
    return d2Sii

def contract_d2S_diagonal(surface, dF, prefactor, surface_discretization_method = "SWIG"):
    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    charge_exp = surface['charge_exp']
    if surface_discretization_method.upper() == "SWIG":
        R_in_J = surface['R_in_J']
        R_sw_J = surface['R_sw_J']
    elif surface_discretization_method.upper() == "ISWIG":
        R_J = surface['R_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]

    assert prefactor.shape == (ngrids,)
    dF = dF.transpose(2,0,1)
    assert dF.shape == (natom, 3, ngrids)

    d2Sii = cupy.zeros((natom, natom, 3, 3))

    for i_grid_atom in range(natom):
        p0,p1 = surface['gslice_by_atom'][i_grid_atom]
        coords = grid_coords[p0:p1]
        si_rJ = cupy.expand_dims(coords, axis=1) - atom_coords
        norm_si_rJ = cupy.linalg.norm(si_rJ, axis=-1)
        si_rJ[:,i_grid_atom,:] = 0.0

        if surface_discretization_method.upper() == "SWIG":
            diJ = (norm_si_rJ - R_in_J) / R_sw_J
            diJ[:,i_grid_atom] = 1.0
            diJ[diJ < 1e-8] = 0.0
            si_rJ[diJ < 1e-8] = 0.0

            fiJ = switch_h(diJ)
            dfiJ = grad_switch_h(diJ)

            fiJK = fiJ[:, :, cupy.newaxis] * fiJ[:, cupy.newaxis, :]
            dfiJK = dfiJ[:, :, cupy.newaxis] * dfiJ[:, cupy.newaxis, :]
            R_sw_JK = R_sw_J[:, cupy.newaxis] * R_sw_J[cupy.newaxis, :]
            norm_si_rJK = norm_si_rJ[:, :, cupy.newaxis] * norm_si_rJ[:, cupy.newaxis, :]
            terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK * R_sw_JK)
            si_rJK = si_rJ[:, :, cupy.newaxis, :, cupy.newaxis] * si_rJ[:, cupy.newaxis, :, cupy.newaxis, :]
            d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, cupy.newaxis, cupy.newaxis] * si_rJK

            d2fiJ = gradgrad_switch_h(diJ)
            terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2 * R_sw_J) - dfiJ / (norm_si_rJ**3)
            si_rJJ = si_rJ[:, :, :, cupy.newaxis] * si_rJ[:, :, cupy.newaxis, :]
            d2fiJK_diagonal = cupy.einsum('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
            d2fiJK_diagonal += cupy.einsum('qA,dD->qAdD', dfiJ / norm_si_rJ, cupy.eye(3))
            d2fiJK_diagonal /= (fiJ * R_sw_J)[:, :, cupy.newaxis, cupy.newaxis]

            d2fiJK = d2fiJK_offdiagonal
            for i_atom in range(natom):
                d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]
        elif surface_discretization_method.upper() == "ISWIG":
            xi = charge_exp[p0:p1]
            erf_input_p = xi[:, None] * (R_J[None, :] + norm_si_rJ)
            erf_input_m = xi[:, None] * (R_J[None, :] - norm_si_rJ)
            fiJ = 1 - 0.5 * (erf(erf_input_p) + erf(erf_input_m))
            # fiJ[:,i_grid_atom] = 1.0
            dfiJ = 1/numpy.sqrt(numpy.pi) * xi[:, None] * (cupy.exp(-erf_input_m**2) - cupy.exp(-erf_input_p**2))
            ### It is necessary to zero out i \in I term in dfiJ, because the second term of d2fiJK_diagonal
            ### is not zeroed out otherwise.
            dfiJ[:,i_grid_atom] = 0

            fiJK = fiJ[:, :, cupy.newaxis] * fiJ[:, cupy.newaxis, :]
            dfiJK = dfiJ[:, :, cupy.newaxis] * dfiJ[:, cupy.newaxis, :]
            norm_si_rJK = norm_si_rJ[:, :, cupy.newaxis] * norm_si_rJ[:, cupy.newaxis, :]
            terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK)
            si_rJK = si_rJ[:, :, cupy.newaxis, :, cupy.newaxis] * si_rJ[:, cupy.newaxis, :, cupy.newaxis, :]
            d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, cupy.newaxis, cupy.newaxis] * si_rJK

            d2fiJ = 2.0/cupy.sqrt(cupy.pi) * (xi**2)[:, None] * (
                + erf_input_m * cupy.exp(-erf_input_m**2)
                + erf_input_p * cupy.exp(-erf_input_p**2)
            )
            # d2fiJ[:,i_grid_atom] = 0.0

            terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2) - dfiJ / (norm_si_rJ**3)
            si_rJJ = si_rJ[:, :, :, cupy.newaxis] * si_rJ[:, :, cupy.newaxis, :]
            d2fiJK_diagonal = cupy.einsum('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
            d2fiJK_diagonal += cupy.einsum('qA,dD->qAdD', dfiJ / norm_si_rJ, cupy.eye(3))
            d2fiJK_diagonal /= fiJ[:, :, cupy.newaxis, cupy.newaxis]

            d2fiJK = d2fiJK_offdiagonal
            for i_atom in range(natom):
                d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        Fi = switch_fun[p0:p1]

        d2F = contract('q,qABdD->qABdD', Fi, d2fiJK)

        d2fiJK_grid_atom_offdiagonal = -cupy.einsum('qABdD->qAdD', d2fiJK)
        d2F[:, i_grid_atom, :, :, :] = contract('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2F[:, :, i_grid_atom, :, :] = contract('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal)

        d2fiJK_grid_atom_diagonal = -cupy.einsum('qAdD->qdD', d2fiJK_grid_atom_offdiagonal)
        d2F[:, i_grid_atom, i_grid_atom, :, :] = contract('q,qdD->qdD', Fi, d2fiJK_grid_atom_diagonal)

        del d2fiJK, d2fiJK_grid_atom_offdiagonal, d2fiJK_grid_atom_diagonal

        d2F_prefactor = charge_exp[p0:p1] / Fi**2 * prefactor[p0:p1]
        d2Sii -= cupy.einsum("qABdD->ABdD", d2F_prefactor[:,None,None,None,None] * d2F)
        del d2F

        dF_prefactor = d2F_prefactor / Fi
        dFi = dF[:,:,p0:p1]
        d2Sii += 2 * cupy.einsum("Adq,BDq->ABdD", dFi * dF_prefactor, dFi)

        del d2F_prefactor, dF_prefactor

    d2Sii *= numpy.sqrt(2 / numpy.pi)

    return d2Sii

def get_d2D_d2S(surface, with_S=True, with_D=False, stream=None):
    ''' Second derivatives of D matrix and S matrix (offdiagonals only)
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    norm_vec    = surface['norm_vec']
    n = charge_exp.shape[0]
    d2S = cupy.empty([3,3,n,n])
    d2D = None
    d2S_ptr = ctypes.cast(d2S.data.ptr, ctypes.c_void_p)
    d2D_ptr = lib.c_null_ptr()
    if with_D:
        d2D = cupy.empty([3,3,n,n])
        d2D_ptr = ctypes.cast(d2D.data.ptr, ctypes.c_void_p)
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_d2d_d2s(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        d2D_ptr, d2S_ptr,
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(norm_vec.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n)
    )
    if err != 0:
        raise RuntimeError('Failed in generating PCM d2D and d2S matrices.')
    return d2D, d2S

def contract_d2S_offdiagonal(surface, left_vector, right_vector, gridslice, stream = None):
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    n = charge_exp.shape[0]
    assert left_vector.size == n
    assert right_vector.size == n

    gridslice = cupy.asarray(gridslice, dtype = cupy.int32, order = "C")
    natm = gridslice.shape[0]
    sandwiched_d2S = cupy.zeros([3,3,natm,natm], dtype = cupy.float64, order = "C")
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    err = libsolvent.pcm_contract_d2s_offdiagonal(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(sandwiched_d2S.data.ptr, ctypes.c_void_p),
        ctypes.cast(left_vector.data.ptr, ctypes.c_void_p),
        ctypes.cast(right_vector.data.ptr, ctypes.c_void_p),
        ctypes.cast(gridslice.data.ptr, ctypes.c_void_p),
        ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n),
        ctypes.c_int(natm),
    )
    if err != 0:
        raise RuntimeError('Failed in left_multiply_dS_offdiagonal')

    reordered_d2S = cupy.zeros([natm,natm,3,3])
    for i_atom in range(natm):
        for j_atom in range(natm):
            reordered_d2S[i_atom, i_atom, :, :] += sandwiched_d2S[:, :, i_atom, j_atom]
            reordered_d2S[j_atom, j_atom, :, :] += sandwiched_d2S[:, :, i_atom, j_atom]
            reordered_d2S[i_atom, j_atom, :, :] -= sandwiched_d2S[:, :, i_atom, j_atom]
            reordered_d2S[j_atom, i_atom, :, :] -= sandwiched_d2S[:, :, i_atom, j_atom]
    return reordered_d2S

def analytical_hess_nuc(pcmobj, dm, verbose=None):
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()

    q_sym        = pcmobj._intermediates['q_sym'].get()
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords'].get()
    exponents    = pcmobj.surface['charge_exp'].get()

    ngrids = q_sym.shape[0]

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    d2e_from_d2I = numpy.zeros([mol.natm, mol.natm, 3, 3])

    int2c2e_ip1ip2 = mol._add_suffix('int2c2e_ip1ip2')
    d2I_dAdC = gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dAdC = d2I_dAdC.reshape(3, 3, mol.natm, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[:, i_atom, :, :] += numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
        d2e_from_d2I[i_atom, :, :, :] += numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
    del d2I_dAdC

    # # Some explanations here:
    # # Why can we use the ip1ip2 here? Because of the translational invariance
    # # $\frac{\partial^2 I_{AC}}{\partial A^2} + \frac{\partial^2 I_{AC}}{\partial A \partial C} = 0$
    # # Why not using the ipip1 here? Because the nuclei, a point charge, is handled as a Gaussian charge with exponent = 1e16
    # # This causes severe numerical problem in function int2c2e_ip1ip2, and make the main diagonal of hessian garbage.
    # d2I_dA2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol_nuc, fakemol)
    d2I_dA2 = -gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    del int2c2e_ip1ip2
    d2I_dA2 = d2I_dA2 @ q_sym
    d2I_dA2 = d2I_dA2.reshape(3, 3, mol.natm)
    for i_atom in range(mol.natm):
        d2e_from_d2I[i_atom, i_atom, :, :] += atom_charges[i_atom] * d2I_dA2[:, :, i_atom]
    del d2I_dA2

    int2c2e_ipip1 = mol._add_suffix('int2c2e_ipip1')
    d2I_dC2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol, fakemol_nuc)
    del int2c2e_ipip1
    d2I_dC2 = d2I_dC2 @ atom_charges
    d2I_dC2 = d2I_dC2.reshape(3, 3, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]
    del d2I_dC2

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm), intopt_derivative)

    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1_on_nuclei = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    v_ng_ip1_on_nuclei = cupy.asarray(v_ng_ip1_on_nuclei)
    v_ng_ip1_on_grid = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)
    v_ng_ip1_on_grid = cupy.asarray(v_ng_ip1_on_grid)
    atom_charges = cupy.asarray(atom_charges)

    d2e_from_dIdq = cupy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            dv_g = cupy.einsum('g,xng->nx', dqdx[i_atom, i_xyz, :], v_ng_ip1_on_nuclei)
            de_grad_nuc = -cupy.einsum('nx,n->nx', dv_g, atom_charges)
            dv_g = cupy.einsum('n,xgn->gx', atom_charges, v_ng_ip1_on_grid)
            dv_g = cupy.einsum('gx,g->gx', dv_g, dqdx[i_atom, i_xyz, :])
            de_grad_nuc -= cupy.asarray([cupy.sum(dv_g[p0:p1], axis=0) for p0,p1 in gridslice])

            d2e_from_dIdq[i_atom, :, i_xyz, :] = de_grad_nuc
            del dv_g, de_grad_nuc
    del v_ng_ip1_on_nuclei, v_ng_ip1_on_grid
    d2e_from_dIdq = d2e_from_dIdq.get()

    d2e = d2e_from_d2I - d2e_from_dIdq

    t1 = log.timer_debug1('solvent hessian d(dVnuc/dx * q)/dx contribution', *t1)
    return d2e

def analytical_hess_qv(pcmobj, dm, verbose=None):
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()

    gridslice   = pcmobj.surface['gslice_by_atom']
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    q_sym       = pcmobj._intermediates['q_sym']

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    # fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    # intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    # intopt.build(1e-14, diag_block_with_triu=True, aosym=False)

    d2e_from_d2I = cupy.zeros([mol.natm, mol.natm, 3, 3])

    # d2I_dA2 = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip1', direct_scf_tol=1e-14)
    # d2I_dA2 = cupy.einsum('dijq,q->dij', d2I_dA2, q_sym)
    # d2I_dA2 = d2I_dA2.reshape([3, 3, nao, nao])
    d2I_dA2 = int1e_grids_ipip1(mol, grid_coords, charges = q_sym, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in range(mol.natm):
        p0,p1 = aoslice[i_atom, 2:]
        d2e_from_d2I[i_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dA2[:, :, p0:p1, :])
        d2e_from_d2I[i_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dA2[:, :, p0:p1, :].transpose(0,1,3,2))
    d2I_dA2 = None

    # d2I_dAdB = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipvip1', direct_scf_tol=1e-14)
    # d2I_dAdB = cupy.einsum('dijq,q->dij', d2I_dAdB, q_sym)
    # d2I_dAdB = d2I_dAdB.reshape([3, 3, nao, nao])
    d2I_dAdB = int1e_grids_ipvip1(mol, grid_coords, charges = q_sym, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in range(mol.natm):
        pi0,pi1 = aoslice[i_atom, 2:]
        for j_atom in range(mol.natm):
            pj0,pj1 = aoslice[j_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[pi0:pi1, pj0:pj1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1])
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[pj0:pj1, pi0:pi1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1].transpose(0,1,3,2))
    d2I_dAdB = None

    for j_atom in range(mol.natm):
        g0,g1 = gridslice[j_atom]
        if g0 == g1:
            continue
        # d2I_dAdC = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ip1ip2', direct_scf_tol=1e-14)
        # d2I_dAdC = cupy.einsum('dijq,q->dij', d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
        # d2I_dAdC = d2I_dAdC.reshape([3, 3, nao, nao])
        d2I_dAdC = int1e_grids_ip1ip2(mol, grid_coords[g0:g1, :], charges = q_sym[g0:g1], intopt = intopt_derivative, charge_exponents = charge_exp[g0:g1]**2)

        for i_atom in range(mol.natm):
            p0,p1 = aoslice[i_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :])
            d2e_from_d2I[i_atom, j_atom, :, :] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(0,1,3,2))

            d2e_from_d2I[j_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,2,3))
            d2e_from_d2I[j_atom, i_atom, :, :] += contract('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,3,2))
    d2I_dAdC = None

    # d2I_dC2 = int3c2e.get_int3c2e_general(mol, fakemol, ip_type='ipip2', direct_scf_tol=1e-14)
    # d2I_dC2 = cupy.einsum('dijq,ij->dq', d2I_dC2, dm)
    # d2I_dC2 = d2I_dC2.reshape([3, 3, ngrids])
    d2I_dC2 = int1e_grids_ipip2(mol, grid_coords, dm = dm, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        if g0 == g1:
            continue
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]
    d2I_dC2 = None

    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm), intopt_derivative)

    rho_q_dorbital = int1e_grids_ip1(mol, grid_coords, dm = dm,
                                     direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                                     intopt=intopt_derivative)
    rho_q_dq = int1e_grids_ip2(mol, grid_coords, dm = dm,
                               direct_scf_tol = 1e-14, charge_exponents = charge_exp**2,
                               intopt=intopt_derivative)

    d2e_from_dIdq = cupy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            dq  = rho_q_dq * dqdx[i_atom, i_xyz, :]
            dq = cupy.asarray([cupy.sum(dq[:,p0:p1], axis=1) for p0,p1 in gridslice])
            dvj = 2.0 * cupy.einsum("Adq,q->Ad", rho_q_dorbital, dqdx[i_atom, i_xyz, :])

            d2e_from_dIdq[i_atom, :, i_xyz, :] = dq + dvj

    d2e = d2e_from_d2I + d2e_from_dIdq
    d2e *= -1
    d2e = d2e.get()

    t1 = log.timer_debug1('solvent hessian d(dI/dx * q)/dx contribution', *t1)
    return d2e

def einsum_ij_Adj_Adi_inverseK(pcmobj, Adj_term, K_transpose = False):
    nA, nd, nj = Adj_term.shape
    # return cupy.einsum('ij,Adj->Adi', cupy.linalg.inv(K), Adj_term)
    # return cupy.linalg.solve(K, Adj_term.reshape(nA * nd, nj).T).T.reshape(nA, nd, nj)
    return pcmobj.left_solve_K(Adj_term.reshape(nA * nd, nj).T, K_transpose = K_transpose).T.reshape(nA, nd, nj)
def einsum_Adi_ij_Adj_inverseK(Adi_term, pcmobj, K_transpose = False):
    # return cupy.einsum('Adi,ij->Adj', Adi_term, cupy.linalg.inv(K))
    return einsum_ij_Adj_Adi_inverseK(pcmobj, Adi_term, K_transpose = not K_transpose)

def get_dS_dot_q(dS, dSii, q, atmlst, gridslice):
    output = contract('diA,i->Adi', dSii[:,:,atmlst], q)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dS[:,g0:g1,:] @ q
        output[i_atom, :, :] -= dS[:,:,g0:g1] @ q[g0:g1]
    return output
# S is symmetric
get_dST_dot_q = get_dS_dot_q

def get_dS_dot_q_lowmem(dSii, surface, q, atmlst, gridslice, stream = None):
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    ngrids = charge_exp.shape[0]
    assert q.size == ngrids

    output = contract('diA,i->Adi', dSii[:,:,atmlst], q) # (len(atmlst), 3, ngrids)
    dSq = left_multiply_dS_offdiagonal(surface, q, stream = stream).T # (3, ngrids)
    qdS = cupy.empty([3, ngrids], dtype = cupy.float64, order = "C")

    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        if g0 == g1:
            continue

        err = libsolvent.pcm_left_multiply_ds_one_atom(
            ctypes.cast(stream.ptr, ctypes.c_void_p),
            ctypes.cast(qdS.data.ptr, ctypes.c_void_p),
            ctypes.cast(q.data.ptr, ctypes.c_void_p),
            ctypes.cast(grid_coords.data.ptr, ctypes.c_void_p),
            ctypes.cast(charge_exp.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids),
            ctypes.c_int(g0),
            ctypes.c_int(g1),
        )
        if err != 0:
            raise RuntimeError('Failed in left_multiply_dS_offdiagonal_one_atom')

        output[i_atom, :, g0:g1] += dSq[:, g0:g1]
        output[i_atom, :, :] -= qdS

    return output
# S is symmetric
get_dST_dot_q_lowmem = get_dS_dot_q_lowmem

def get_dA_dot_q(dA, q, atmlst):
    return contract('diA,i->Adi', dA[:,:,atmlst], q)

def get_dD_dot_q(dD, q, atmlst, gridslice, ngrids):
    output = cupy.zeros([len(atmlst), 3, ngrids])
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dD[:,g0:g1,:] @ q
        output[i_atom, :, :] -= dD[:,:,g0:g1] @ q[g0:g1]
    return output
def get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids):
    return get_dD_dot_q(-dD.transpose(0,2,1), q, atmlst, gridslice, ngrids)

_SURFACE_KERNEL_CODE = r'''
#define SQRT_PI 1.7724538509055160272981674833411

// out[a,d,i] = sum_{j in the grid points of atom a} dD[d,i,j] q[j], or with
// transpose dD[d,j,i] q[j]; dD[d,i,j] is the derivative of D_ij with respect
// to the position of point i (the off-diagonal elements, as _pcm_left_multiply_dD
// of lib/solvent/pcm.cu has them, with the normal vector of point i for the
// transpose).
extern "C" __global__
void pcm_dD_dot_q_by_atom(double* __restrict__ out, const double* __restrict__ q,
        const double* __restrict__ coords, const double* __restrict__ charge_exp,
        const double* __restrict__ norm_vec, const int* __restrict__ gridslice,
        const int n, const int transpose)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = blockIdx.y;
    if (i >= n) {
        return;
    }
    const int j0 = gridslice[2*a];
    const int j1 = gridslice[2*a+1];
    const double xi = coords[3*i];
    const double yi = coords[3*i+1];
    const double zi = coords[3*i+2];
    const double ei = charge_exp[i];
    const double nxi = norm_vec[3*i];
    const double nyi = norm_vec[3*i+1];
    const double nzi = norm_vec[3*i+2];

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (int j = j0; j < j1; ++j) {
        const double ej = charge_exp[j];
        const double xi_ij = ei * ej * rsqrt(ei*ei + ej*ej);
        const double dx = xi - coords[3*j];
        const double dy = yi - coords[3*j+1];
        const double dz = zi - coords[3*j+2];
        const double rij = sqrt(dx*dx + dy*dy + dz*dz);

        const double xi_r_ij = xi_ij * rij;
        const double xi2_r2_ij = xi_r_ij * xi_r_ij;
        const double rij_1 = (rij < 1e-14) ? 0.0 : 1.0/rij; // This handles i == j case
        const double rij_2 = rij_1 * rij_1;
        const double exp_term = exp(-xi2_r2_ij);

        const double dS_dr = -(erf(xi_r_ij) - (2.0/SQRT_PI) * xi_r_ij * exp_term) * rij_2;
        const double dx_rij = dx * rij_1;
        const double dy_rij = dy * rij_1;
        const double dz_rij = dz * rij_1;

        const double nxj = transpose ? nxi : norm_vec[3*j];
        const double nyj = transpose ? nyi : norm_vec[3*j+1];
        const double nzj = transpose ? nzi : norm_vec[3*j+2];
        const double nj_rij = dx*nxj + dy*nyj + dz*nzj;
        const double rij_3 = rij_1 * rij_2;
        const double xi3_r3_ij = xi2_r2_ij * xi_ij;
        const double dD_dri = (4.0/SQRT_PI) * xi3_r3_ij * exp_term * nj_rij * rij_3;

        const double nj_rij_over_rij2 = 3.0 * nj_rij * rij_2;
        const double qj = q[j];
        sum_x += (dD_dri * dx_rij + dS_dr * (-nxj * rij_1 + nj_rij_over_rij2 * dx_rij)) * qj;
        sum_y += (dD_dri * dy_rij + dS_dr * (-nyj * rij_1 + nj_rij_over_rij2 * dy_rij)) * qj;
        sum_z += (dD_dri * dz_rij + dS_dr * (-nzj * rij_1 + nj_rij_over_rij2 * dz_rij)) * qj;
    }
    out[((size_t)a * 3    ) * n + i] = sum_x;
    out[((size_t)a * 3 + 1) * n + i] = sum_y;
    out[((size_t)a * 3 + 2) * n + i] = sum_z;
}

// out[c,a,i] = left[i] sum_{j in the grid points of atom a} d2D[c,i,j] right[j],
// c = 3 x + y: the second derivatives of D_ij with respect to the position of
// point i (the off-diagonal elements, as _pcm_d2D_d2S of lib/solvent/pcm.cu
// has them).
extern "C" __global__
void pcm_d2D_sandwich_by_atom(double* __restrict__ out, const double* __restrict__ left,
        const double* __restrict__ right, const double* __restrict__ coords,
        const double* __restrict__ charge_exp, const double* __restrict__ norm_vec,
        const int* __restrict__ gridslice, const int n, const int natm)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = blockIdx.y;
    if (i >= n) {
        return;
    }
    const int j0 = gridslice[2*a];
    const int j1 = gridslice[2*a+1];
    const double xi = coords[3*i];
    const double yi = coords[3*i+1];
    const double zi = coords[3*i+2];
    const double ei = charge_exp[i];

    double s[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int j = j0; j < j1; ++j) {
        const double ej = charge_exp[j];
        const double eij = ei * ej * rsqrt(ei*ei + ej*ej);

        const double dx = xi - coords[3*j];
        const double dy = yi - coords[3*j+1];
        const double dz = zi - coords[3*j+2];
        const double rij = sqrt(dx*dx + dy*dy + dz*dz);
        const double rij_1 = (i != j) ? (1.0 / rij) : 0.0; // This guarantees that if i == j, all matrix elements = 0
        const double rij_2 = rij_1 * rij_1;
        const double rij_3 = rij_2 * rij_1;
        const double rij_4 = rij_2 * rij_2;
        const double rij_5 = rij_2 * rij_3;
        const double eij2 = eij * eij;

        const double eij_rij = eij * rij;
        const double erf_eij_rij = erf(eij_rij);
        const double exp_minus_eij2_rij2 = exp(-eij_rij * eij_rij);
        const double two_eij_over_sqrt_pi = 2.0 * eij / SQRT_PI;
        const double two_eij_over_sqrt_pi_exp_minus_eij2_rij2 = exp_minus_eij2_rij2 * two_eij_over_sqrt_pi;

        const double S_direct_product_prefactor = -two_eij_over_sqrt_pi_exp_minus_eij2_rij2 * (3 * rij_4 + 2 * eij2 * rij_2)
                                                  + 3 * rij_5 * erf_eij_rij;

        const double nxj = norm_vec[3*j];
        const double nyj = norm_vec[3*j+1];
        const double nzj = norm_vec[3*j+2];
        const double nj_rij = dx * nxj + dy * nyj + dz * nzj;

        const double eij4 = eij2 * eij2;
        const double rij_6 = rij_4 * rij_2;
        const double rij_7 = rij_4 * rij_3;

        const double D_direct_product_prefactor = (-two_eij_over_sqrt_pi_exp_minus_eij2_rij2 * (15 * rij_6 + 10 * eij2 * rij_4 + 4 * eij4 * rij_2)
                                                   + 15 * rij_7 * erf_eij_rij) * nj_rij;
        const double w = right[j];
        s[0] += w * (D_direct_product_prefactor * dx * dx - S_direct_product_prefactor * (dx * nxj + dx * nxj + nj_rij));
        s[1] += w * (D_direct_product_prefactor * dx * dy - S_direct_product_prefactor * (dy * nxj + dx * nyj));
        s[2] += w * (D_direct_product_prefactor * dx * dz - S_direct_product_prefactor * (dz * nxj + dx * nzj));
        s[3] += w * (D_direct_product_prefactor * dy * dx - S_direct_product_prefactor * (dx * nyj + dy * nxj));
        s[4] += w * (D_direct_product_prefactor * dy * dy - S_direct_product_prefactor * (dy * nyj + dy * nyj + nj_rij));
        s[5] += w * (D_direct_product_prefactor * dy * dz - S_direct_product_prefactor * (dz * nyj + dy * nzj));
        s[6] += w * (D_direct_product_prefactor * dz * dx - S_direct_product_prefactor * (dx * nzj + dz * nxj));
        s[7] += w * (D_direct_product_prefactor * dz * dy - S_direct_product_prefactor * (dy * nzj + dz * nyj));
        s[8] += w * (D_direct_product_prefactor * dz * dz - S_direct_product_prefactor * (dz * nzj + dz * nzj + nj_rij));
    }
    const double l = left[i];
    for (int c = 0; c < 9; ++c) {
        out[((size_t)c * natm + a) * n + i] = l * s[c];
    }
}
'''

_surface_kernels = {}
def _surface_kernel(name):
    if name not in _surface_kernels:
        _surface_kernels[name] = cupy.RawKernel(_SURFACE_KERNEL_CODE, name, options=('-std=c++17',))
    return _surface_kernels[name]

def _dD_dot_q_by_atom(surface, q, gridslice, transpose=False):
    '''[natm,3,ngrids]: sum_j dD[d,i,j] q[j] over the grid points j of each
    atom (dD[d,j,i] q[j] with transpose), dD the derivative of D with respect
    to the position of its first point, evaluated from the surface data'''
    grid_coords = cupy.ascontiguousarray(surface['grid_coords'])
    charge_exp = cupy.ascontiguousarray(surface['charge_exp'])
    norm_vec = cupy.ascontiguousarray(surface['norm_vec'])
    n = charge_exp.shape[0]
    assert q.size == n
    gs = cupy.asarray(gridslice, dtype=cupy.int32, order='C')
    natm = gs.shape[0]
    out = cupy.empty((natm, 3, n))
    nthread = 128
    _surface_kernel('pcm_dD_dot_q_by_atom')(
        ((n + nthread - 1) // nthread, natm), (nthread,),
        (out, cupy.ascontiguousarray(q, dtype=cupy.float64), grid_coords, charge_exp,
         norm_vec, gs, numpy.int32(n), numpy.int32(bool(transpose))))
    return out

def get_dD_dot_q_lowmem(surface, q, atmlst, gridslice):
    '''get_dD_dot_q without the matrix dD [3,ngrids,ngrids]'''
    natm = len(gridslice)
    assert list(atmlst) == list(range(natm))
    by_atom = _dD_dot_q_by_atom(surface, q, gridslice)
    dDq = by_atom.sum(axis=0)
    output = -by_atom
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dDq[:, g0:g1]
    return output

def get_dDT_dot_q_lowmem(surface, q, atmlst, gridslice):
    '''get_dDT_dot_q without the matrix dD [3,ngrids,ngrids]'''
    natm = len(gridslice)
    assert list(atmlst) == list(range(natm))
    output = _dD_dot_q_by_atom(surface, q, gridslice, transpose=True)
    dDTq = output.sum(axis=0)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] -= dDTq[:, g0:g1]
    return output

def contract_d2D_offdiagonal(surface, left_vector, right_vector, gridslice, transpose=False):
    '''get_v_dot_d2D_dot_q (get_v_dot_d2DT_dot_q with transpose) without the
    matrix d2D [3,3,ngrids,ngrids]'''
    if transpose:
        # v d2D^T q = q d2D v, with the atoms of the two points exchanged
        left_vector, right_vector = right_vector, left_vector
    grid_coords = cupy.ascontiguousarray(surface['grid_coords'])
    charge_exp = cupy.ascontiguousarray(surface['charge_exp'])
    norm_vec = cupy.ascontiguousarray(surface['norm_vec'])
    n = charge_exp.shape[0]
    assert left_vector.size == n
    assert right_vector.size == n
    gs = cupy.asarray(gridslice, dtype=cupy.int32, order='C')
    natm = gs.shape[0]
    tmp = cupy.empty((9, natm, n))
    nthread = 128
    _surface_kernel('pcm_d2D_sandwich_by_atom')(
        ((n + nthread - 1) // nthread, natm), (nthread,),
        (tmp, cupy.ascontiguousarray(left_vector, dtype=cupy.float64),
         cupy.ascontiguousarray(right_vector, dtype=cupy.float64), grid_coords, charge_exp,
         norm_vec, gs, numpy.int32(n), numpy.int32(natm)))
    # [atom of the left point, atom of the right point, 3, 3]
    sandwiched = cupy.stack([tmp[:, :, g0:g1].sum(axis=2) for g0,g1 in gridslice])
    sandwiched = sandwiched.transpose(0,2,1).reshape(natm, natm, 3, 3)
    if transpose:
        sandwiched = sandwiched.transpose(1,0,2,3)
    output = -(sandwiched + sandwiched.transpose(1,0,2,3))
    idx = cupy.arange(natm)
    output[idx, idx] += sandwiched.sum(axis=1) + sandwiched.sum(axis=0)
    return output

def get_v_dot_d2S_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice):
    output = d2Sii @ (v_left * q_right)
    for i_atom in range(natom):
        gi0,gi1 = gridslice[i_atom]
        for j_atom in range(natom):
            gj0,gj1 = gridslice[j_atom]
            d2S_atom_ij = contract('q,dDq->dD', v_left[gi0:gi1], d2S[:,:,gi0:gi1,gj0:gj1] @ q_right[gj0:gj1])
            output[i_atom, i_atom, :, :] += d2S_atom_ij
            output[j_atom, j_atom, :, :] += d2S_atom_ij
            output[i_atom, j_atom, :, :] -= d2S_atom_ij
            output[j_atom, i_atom, :, :] -= d2S_atom_ij
    return output
get_v_dot_d2ST_dot_q = get_v_dot_d2S_dot_q

def get_v_dot_d2A_dot_q(d2A, v_left, q_right):
    return d2A @ (v_left * q_right)

def get_v_dot_d2D_dot_q(d2D, v_left, q_right, natom, gridslice):
    output = cupy.zeros([natom, natom, 3, 3])
    for i_atom in range(natom):
        gi0,gi1 = gridslice[i_atom]
        for j_atom in range(natom):
            gj0,gj1 = gridslice[j_atom]
            d2D_atom_ij = contract('q,dDq->dD', v_left[gi0:gi1], d2D[:,:,gi0:gi1,gj0:gj1] @ q_right[gj0:gj1])
            output[i_atom, i_atom, :, :] += d2D_atom_ij
            output[j_atom, j_atom, :, :] += d2D_atom_ij
            output[i_atom, j_atom, :, :] -= d2D_atom_ij
            output[j_atom, i_atom, :, :] -= d2D_atom_ij
    return output
def get_v_dot_d2DT_dot_q(d2D, v_left, q_right, natom, gridslice):
    return get_v_dot_d2D_dot_q(d2D.transpose(0,1,3,2), v_left, q_right, natom, gridslice)

# The part of the available memory that the matrices of the derivatives of D
# and S may take (dD, dS [3,ngrids,ngrids]; d2D, d2S [3,3,ngrids,ngrids]) in
# the IEF-PCM and SMD terms; beyond it their products with vectors are taken
# from the surface data (0: always)
DENSE_DERIVATIVE_MEMORY_FRACTION = .8

def _dDT_dot_q(dD, q, atmlst, gridslice, ngrids):
    '''get_dDT_dot_q without the copy of -dD^T that it makes (as large as dD)'''
    output = cupy.zeros([len(atmlst), 3, ngrids])
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        if g0 == g1:
            continue
        output[i_atom, :, g0:g1] -= contract('dji,j->di', dD[:,:,g0:g1], q)
        output[i_atom, :, :] += contract('dji,j->di', dD[:,g0:g1,:], q[g0:g1])
    return output

def _first_derivative_products(pcmobj, dSii, atmlst, gridslice):
    '''get_dS_dot_q, get_dD_dot_q and get_dDT_dot_q of the surface of pcmobj as
    functions of the vector: from the matrices dS and dD [3,ngrids,ngrids]
    where the two fit in memory, from the surface data otherwise.'''
    surface = pcmobj.surface
    ngrids = surface['charge_exp'].shape[0]
    required = 2 * 3 * ngrids**2 * 8
    dD = dS = None
    if required <= get_avail_mem() * DENSE_DERIVATIVE_MEMORY_FRACTION:
        try:
            dD, dS = get_dD_dS(surface, with_D=True, with_S=True)
        except cupy.cuda.memory.OutOfMemoryError:
            dD = dS = None
    if dD is None:
        def dS_dot(x):
            return get_dS_dot_q_lowmem(dSii, surface, x, atmlst, gridslice)
        def dD_dot(x):
            return get_dD_dot_q_lowmem(surface, x, atmlst, gridslice)
        def dDT_dot(x):
            return get_dDT_dot_q_lowmem(surface, x, atmlst, gridslice)
    else:
        def dS_dot(x):
            return get_dS_dot_q(dS, dSii, x, atmlst, gridslice)
        def dD_dot(x):
            return get_dD_dot_q(dD, x, atmlst, gridslice, ngrids)
        def dDT_dot(x):
            return _dDT_dot_q(dD, x, atmlst, gridslice, ngrids)
    return dS_dot, dD_dot, dDT_dot

def _second_derivative_sandwiches(pcmobj, dF, d2F, natom, gridslice):
    '''get_v_dot_d2S_dot_q and get_v_dot_d2D_dot_q of the surface of pcmobj as
    functions of the two vectors: from the matrices d2S and d2D
    [3,3,ngrids,ngrids] where the two fit in memory, from the surface data
    otherwise.'''
    surface = pcmobj.surface
    ngrids = surface['charge_exp'].shape[0]
    required = (2 * 9 * ngrids**2 + 9 * ngrids * natom**2) * 8
    d2D = d2S = d2Sii = None
    if required <= get_avail_mem() * DENSE_DERIVATIVE_MEMORY_FRACTION:
        try:
            d2Sii = get_d2Sii(surface, dF, d2F)
            d2D, d2S = get_d2D_d2S(surface, with_D=True, with_S=True)
        except cupy.cuda.memory.OutOfMemoryError:
            d2D = d2S = d2Sii = None
    if d2D is None:
        method = pcmobj.surface_discretization_method
        def d2S_sandwich(v, q):
            out = contract_d2S_offdiagonal(surface, v, q, gridslice)
            out += contract_d2S_diagonal(surface, dF, v * q, method)
            return out
        def d2D_sandwich(v, q):
            return contract_d2D_offdiagonal(surface, v, q, gridslice)
    else:
        def d2S_sandwich(v, q):
            return get_v_dot_d2S_dot_q(d2S, d2Sii, v, q, natom, gridslice)
        def d2D_sandwich(v, q):
            return get_v_dot_d2D_dot_q(d2D, v, q, natom, gridslice)
    return d2S_sandwich, d2D_sandwich

def analytical_hess_solver(pcmobj, dm, verbose=None):
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(mol, verbose)
    t1 = log.init_timer()

    natom = mol.natm
    atmlst = range(natom) # Attention: This cannot be split

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    q            = pcmobj._intermediates['q']
    f_epsilon    = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        S = pcmobj._intermediates['S']

    ngrids = q.shape[0]

    vK_1 = pcmobj.left_solve_K(v_grids, K_transpose = True)

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        dF, _ = get_dF_dA(pcmobj.surface, with_dA = False, surface_discretization_method = pcmobj.surface_discretization_method)
        dSii = get_dSii(pcmobj.surface, dF)

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.8) # Don't use too much gpu memory
        dS_required_memory = 3 * ngrids * ngrids * cupy.dtype(cupy.float64).itemsize

        # dR = 0, dK = dS
        # d(S-1 R) = - S-1 dS S-1 R
        # d2(S-1 R) = (S-1 dS S-1 dS S-1 R) + (S-1 dS S-1 dS S-1 R) - (S-1 d2S S-1 R)
        if dS_required_memory <= available_gpu_memory:
            _, dS = get_dD_dS(pcmobj.surface, with_D=False, with_S=True)
            dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
            S_1_dSdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dSdx_dot_q)
            del dSdx_dot_q
            VS_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
            del dS
        else:
            dSdx_dot_q = get_dS_dot_q_lowmem(dSii, pcmobj.surface, q, atmlst, gridslice)
            S_1_dSdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dSdx_dot_q)
            del dSdx_dot_q
            VS_1_dot_dSdx = get_dST_dot_q_lowmem(dSii, pcmobj.surface, vK_1, atmlst, gridslice)

        del dSii
        d2e_from_d2KR = contract('Adi,BDi->ABdD', VS_1_dot_dSdx, S_1_dSdx_dot_q) * 2

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.8) # Don't use too much gpu memory
        d2S_required_memory = (9 * ngrids * ngrids + 9 * ngrids * natom**2) * cupy.dtype(cupy.float64).itemsize

        if d2S_required_memory <= available_gpu_memory:
            _, d2S = get_d2D_d2S(pcmobj.surface, with_D=False, with_S=True)
            d2F, _ = get_d2F_d2A(pcmobj.surface, pcmobj.surface_discretization_method)
            d2Sii = get_d2Sii(pcmobj.surface, dF, d2F)
            del d2F
            v_d2S_q = get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
            del d2S, d2Sii
        else:
            v_d2S_q = contract_d2S_offdiagonal(pcmobj.surface, vK_1, q, gridslice)
            v_d2S_q += contract_d2S_diagonal(pcmobj.surface, dF, vK_1 * q, pcmobj.surface_discretization_method)
        del dF
        d2e_from_d2KR -= v_d2S_q

        dK_1Rv = -S_1_dSdx_dot_q
        R = -f_epsilon
        dvK_1R = -einsum_Adi_ij_Adj_inverseK(VS_1_dot_dSdx, pcmobj) * R

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dSii = get_dSii(pcmobj.surface, dF)
        # The products of dS, dD and dD^T with a vector, of the matrices or of
        # the surface data. D A, A S and R are applied as D and S on scaled
        # vectors (three matrices of ngrids^2 less).
        dS_dot, dD_dot, dDT_dot = _first_derivative_products(pcmobj, dSii, atmlst, gridslice)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(2*pi) * (d2D*A*S + D*d2A*S + D*A*d2S + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS)
        # The terms showing up twice on equation above (dD*dA + dD*dA for example) refer to dD/dx * dA/dy + dD/dy * dA/dx,
        # since D is not symmetric, they are not the same.

        # d(K-1 R) = - K-1 dK K-1 R + K-1 dR
        # d2(K-1 R) = (K-1 dK K-1 dK K-1 R) + (K-1 dK K-1 dK K-1 R) - (K-1 d2K K-1 R) - (K-1 dK K-1 dR)
        #             - (K-1 dK K-1 dR) + (K-1 d2R)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = dS_dot(q)
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * contract('ij,Adj->Adi', D, dSdx_dot_q * A)
        Sq = S @ q
        dAdx_dot_Sq = get_dA_dot_q(dA, Sq, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)
        ASq = A * Sq
        dDdx_dot_ASq = dD_dot(ASq)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq
        dDdx_dot_ASq = None

        K_1_dot_dKdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = dS_dot(vK_1)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = dDT_dot(vK_1)
        vK_1_dot_dKdx -= f_eps_over_2pi * contract('ij,Adj->Adi', S.T, vK_1_dot_dDdx * A)
        vK_1D = D.T @ vK_1
        vK_1D_dot_dAdx = get_dA_dot_q(dA, vK_1D, atmlst)
        vK_1_dot_dKdx -= f_eps_over_2pi * contract('ij,Adj->Adi', S.T, vK_1D_dot_dAdx)
        vK_1DA = A * vK_1D
        vK_1DA_dot_dSdx = dS_dot(vK_1DA)
        dS_dot = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_2pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None

        d2e_from_d2KR  = contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        # the products with dD that are needed below, before the second
        # derivatives take the memory
        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = dD_dot(A * v_grids)
        dD_dot = dDT_dot = None

        d2F, d2A = get_d2F_d2A(pcmobj.surface, pcmobj.surface_discretization_method)
        vK_1_d2K_q  = get_v_dot_d2A_dot_q(d2A, vK_1D, Sq)
        vK_1_d2R_V  = get_v_dot_d2A_dot_q(d2A, vK_1D, v_grids)
        d2A = None
        d2S_sandwich, d2D_sandwich = _second_derivative_sandwiches(pcmobj, dF, d2F, natom, gridslice)
        dF = None
        d2F = None
        vK_1_d2K_q += d2D_sandwich(vK_1, ASq)
        vK_1_d2R_V += d2D_sandwich(vK_1, A * v_grids)
        d2D_sandwich = None
        vK_1_d2K_q += d2S_sandwich(vK_1DA, q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q *= -f_eps_over_2pi
        vK_1_d2K_q += d2S_sandwich(vK_1, q)
        d2S_sandwich = None

        d2e_from_d2KR -= vK_1_d2K_q

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)
        dRdx_dot_V = None

        d2e_from_d2KR -= contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = vK_1D_dot_dAdx
        VK_1_dot_dDdx = vK_1_dot_dDdx
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        # X R for R = -f_eps (1 - D A / (2 pi))
        x = einsum_Adi_ij_Adj_inverseK(vK_1_dot_dKdx, pcmobj)
        dvK_1R = f_epsilon * (x - 1.0/(2.0*PI) * contract('ji,Adj->Adi', D, x) * A) + VK_1_dot_dRdx
        x = None

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dSii = get_dSii(pcmobj.surface, dF)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(4*pi) * (dD*A*S + D*dA*S + D*A*dS + dST*AT*DT + ST*dAT*DT + ST*AT*dDT)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(4*pi) * (d2D*A*S + D*d2A*S + D*A*d2S + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS
        #                           + d2ST*AT*DT + ST*d2AT*DT + ST*AT*d2DT + dST*dAT*DT + dST*dAT*DT + dST*AT*dDT + dST*AT*dDT + ST*dAT*dDT + ST*dAT*dDT)
        f_eps_over_2pi = f_epsilon/(2.0*PI)
        f_eps_over_4pi = f_epsilon/(4.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * contract('ij,Adj->Adi', DA, dSdx_dot_q)
        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq
        dDdx_dot_ASq = None
        dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', AS.T, dDdxT_dot_q)
        dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', S.T, dAdxT_dot_DT_q)
        AT_DT_q = DA.T @ q
        dSdxT_dot_AT_DT_q = get_dS_dot_q(dS, dSii, AT_DT_q, atmlst, gridslice)
        dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q
        dSdxT_dot_AT_DT_q = None

        K_1_dot_dKdx_dot_q = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', AS.T, vK_1_dot_dDdx)
        vK_1D_dot_dAdx = get_dA_dot_q(dA, D.T @ vK_1, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', S.T, vK_1D_dot_dAdx)
        vK_1DA = DA.T @ vK_1
        vK_1DA_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None
        vK_1_dot_dSdxT = get_dS_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', DA, vK_1_dot_dSdxT)
        DA = None
        vK_1_ST_dot_dAdxT = get_dA_dot_q(dA, (S @ vK_1).T, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * contract('ij,Adj->Adi', D, vK_1_ST_dot_dAdxT)
        vK_1_ST_AT = AS @ vK_1
        AS = None
        vK_1_ST_AT_dot_dDdxT = get_dD_dot_q(dD, vK_1_ST_AT, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1_ST_AT_dot_dDdxT
        vK_1_ST_AT_dot_dDdxT = None

        d2e_from_d2KR  = contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2F, d2A = get_d2F_d2A(pcmobj.surface, pcmobj.surface_discretization_method)
        vK_1_d2K_q  = get_v_dot_d2A_dot_q(d2A, (D.T @ vK_1).T, S @ q)
        vK_1_d2K_q += get_v_dot_d2A_dot_q(d2A, (S @ vK_1).T, D.T @ q)
        vK_1_d2R_V  = get_v_dot_d2A_dot_q(d2A, (D.T @ vK_1).T, v_grids)
        d2A = None
        d2Sii = get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2D, d2S = get_d2D_d2S(pcmobj.surface, with_D=True, with_S=True)
        vK_1_d2K_q += get_v_dot_d2D_dot_q(d2D, vK_1, ASq, natom, gridslice)
        vK_1_d2K_q += get_v_dot_d2DT_dot_q(d2D, vK_1_ST_AT, q, natom, gridslice)
        vK_1_d2R_V += get_v_dot_d2D_dot_q(d2D, vK_1, A * v_grids, natom, gridslice)
        d2D = None
        vK_1_d2K_q += get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1DA, q, natom, gridslice)
        vK_1_d2K_q += get_v_dot_d2ST_dot_q(d2S, d2Sii, vK_1, AT_DT_q, natom, gridslice)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->ABdD', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += contract('Adi,BDi->BADd', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q *= -f_eps_over_4pi
        vK_1_d2K_q += get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)

        d2e_from_d2KR -= contract('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= contract('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += contract('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += contract('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        DA = D*A
        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        dvK_1R = -einsum_Adi_ij_Adj_inverseK(vK_1_dot_dKdx, pcmobj) @ R + VK_1_dot_dRdx

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    d2e = d2e_from_d2KR

    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dVdx = get_dvgrids(pcmobj, dm, range(mol.natm), intopt_derivative)
    d2e -= contract('Adi,BDi->BADd', dvK_1R, dVdx)
    d2e -= contract('Adi,BDi->ABdD', dVdx, dK_1Rv)

    d2e *= 0.5
    d2e = d2e.get()
    t1 = log.timer_debug1('solvent hessian d(V * dK-1R/dx * V)/dx contribution', *t1)
    return d2e

def get_dqsym_dx_fix_vgrids(pcmobj, atmlst):
    assert pcmobj._intermediates is not None

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    q            = pcmobj._intermediates['q']
    q_sym        = pcmobj._intermediates['q_sym']
    f_epsilon    = pcmobj._intermediates['f_epsilon']
    if not pcmobj.if_method_in_CPCM_category:
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        S = pcmobj._intermediates['S']

    ngrids = q_sym.shape[0]

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        dF, _ = get_dF_dA(pcmobj.surface, with_dA = False, surface_discretization_method = pcmobj.surface_discretization_method)
        dSii = get_dSii(pcmobj.surface, dF)
        del dF

        available_gpu_memory = get_avail_mem()
        available_gpu_memory = int(available_gpu_memory * 0.9) # Don't use too much gpu memory
        dS_required_memory = 3 * ngrids * ngrids * cupy.dtype(cupy.float64).itemsize

        # dR = 0, dK = dS
        if dS_required_memory <= available_gpu_memory:
            _, dS = get_dD_dS(pcmobj.surface, with_D=False, with_S=True)
            dSdx_dot_q = get_dS_dot_q(dS, dSii, q_sym, atmlst, gridslice)
            del dS
        else:
            dSdx_dot_q = get_dS_dot_q_lowmem(dSii, pcmobj.surface, q_sym, atmlst, gridslice, stream = None)

        dqdx_fix_Vq = einsum_ij_Adj_Adi_inverseK(pcmobj, dSdx_dot_q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        # The products of dS, dD and dD^T with a vector, of the matrices or of
        # the surface data. D A, A S and R are applied as D and S on scaled
        # vectors (three matrices of ngrids^2 less).
        dS_dot, dD_dot, dDT_dot = _first_derivative_products(pcmobj, dSii, atmlst, gridslice)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = dS_dot(q)

        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * contract('ij,Adj->Adi', D, dSdx_dot_q * A)

        Sq = S @ q
        dAdx_dot_Sq = get_dA_dot_q(dA, Sq, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)

        dDdx_dot_ASq = dD_dot(A * Sq)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq

        dqdx_fix_Vq = -einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)

        dDdx_dot_AV = dD_dot(A * v_grids)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dqdx_fix_Vq += einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)

        invKT_V = pcmobj.left_solve_K(v_grids, K_transpose = True)
        dDdxT_dot_invKT_V = dDT_dot(invKT_V)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst)
        dqdx_fix_Vq += f_eps_over_2pi * (contract('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dSdxT_dot_invKT_V = dS_dot(invKT_V)
        dKdxT_dot_invKT_V = dSdxT_dot_invKT_V

        dKdxT_dot_invKT_V -= f_eps_over_2pi * contract('ij,Adj->Adi', S.T, dDdxT_dot_invKT_V * A)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * contract('ij,Adj->Adi', S.T, dAdxT_dot_DT_invKT_V)

        dSdxT_dot_AT_DT_invKT_V = dS_dot(A * DT_invKT_V)
        dS_dot = dD_dot = dDT_dot = None
        dKdxT_dot_invKT_V -= f_eps_over_2pi * dSdxT_dot_AT_DT_invKT_V
        invKT_dKdxT_dot_invKT_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdxT_dot_invKT_V, K_transpose = True)

        # R^T x for R = -f_eps (1 - D A / (2 pi))
        dqdx_fix_Vq += f_epsilon * (
            invKT_dKdxT_dot_invKT_V
            - 1.0/(2.0*PI) * contract('ji,Adj->Adi', D, invKT_dKdxT_dot_invKT_V) * A)

        dqdx_fix_Vq *= -0.5

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dSii = get_dSii(pcmobj.surface, dF)
        dF = None

        dD, dS = get_dD_dS(pcmobj.surface, with_D=True, with_S=True)

        f_eps_over_4pi = f_epsilon/(4.0*PI)
        DA = D*A

        def dK_dot_q(q):
            dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

            dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * contract('ij,Adj->Adi', DA, dSdx_dot_q)

            dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
            dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', D, dAdx_dot_Sq)

            AS = (A * S.T).T # It's just diag(A) @ S
            dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice, ngrids)
            dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq

            dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
            dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', AS.T, dDdxT_dot_q)

            dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst)
            dKdx_dot_q -= f_eps_over_4pi * contract('ij,Adj->Adi', S.T, dAdxT_dot_DT_q)

            dSdxT_dot_AT_DT_q = get_dST_dot_q(dS, dSii, DA.T @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q

            return dKdx_dot_q

        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dKdx_dot_q = dK_dot_q(q)
        dqdx_fix_Vq = -einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + contract('ij,Adj->Adi', D, dAdx_dot_V))
        dqdx_fix_Vq += einsum_ij_Adj_Adi_inverseK(pcmobj, dRdx_dot_V)

        invKT_V = pcmobj.left_solve_K(v_grids, K_transpose = True)
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice, ngrids)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst)
        dqdx_fix_Vq += f_eps_over_2pi * (contract('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dKdx_dot_invKT_V = dK_dot_q(invKT_V)
        invKT_dKdx_dot_invKT_V = einsum_ij_Adj_Adi_inverseK(pcmobj, dKdx_dot_invKT_V, K_transpose = True)

        R = -f_epsilon * (cupy.eye(DA.shape[0]) - 1.0/(2.0*PI)*DA)
        dqdx_fix_Vq += -contract('ij,Adj->Adi', R.T, invKT_dKdx_dot_invKT_V)

        dqdx_fix_Vq *= -0.5

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    return dqdx_fix_Vq

def get_dvgrids(pcmobj, dm, atmlst, intopt_derivative):
    assert pcmobj._intermediates is not None

    mol = pcmobj.mol
    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    atom_coords = atom_coords[atmlst]
    atom_charges = atom_charges[atmlst]
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords.get(), expnt=charge_exp.get()**2)
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    v_ng_ip1 = cupy.array(v_ng_ip1)
    dV_on_charge_dx = contract('dAq,A->Adq', v_ng_ip1, atom_charges)

    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)
    v_ng_ip2 = cupy.array(v_ng_ip2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] += contract('dqA,A->dq', v_ng_ip2[:,g0:g1,:], atom_charges)

    dIdA = int1e_grids_ip1(mol, grid_coords, dm = dm + dm.T, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    dV_on_charge_dx[atmlst,:,:] -= dIdA[atmlst,:,:]

    dIdC = int1e_grids_ip2(mol, grid_coords, intopt = intopt_derivative, dm = dm, charge_exponents = charge_exp**2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] -= dIdC[:,g0:g1]

    return dV_on_charge_dx

def get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, intopt_derivative):
    dV_on_charge_dx = get_dvgrids(pcmobj, dm, atmlst, intopt_derivative)

    f_epsilon = pcmobj._intermediates['f_epsilon']
    if pcmobj.if_method_in_CPCM_category:
        R = -f_epsilon
        R_dVdx = R * dV_on_charge_dx
    else:
        # R = -f_eps (1 - D A / (2 pi)), applied as D on scaled vectors
        A = pcmobj._intermediates['A']
        D = pcmobj._intermediates['D']
        R_dVdx = -f_epsilon * (
            dV_on_charge_dx
            - 1.0/(2.0*PI) * contract('ij,Adj->Adi', D, dV_on_charge_dx * A))

    K_1_R_dVdx = einsum_ij_Adj_Adi_inverseK(pcmobj, R_dVdx)
    K_1T_dVdx = einsum_ij_Adj_Adi_inverseK(pcmobj, dV_on_charge_dx, K_transpose = True)

    if pcmobj.if_method_in_CPCM_category:
        RT_K_1T_dVdx = R * K_1T_dVdx
    else:
        RT_K_1T_dVdx = -f_epsilon * (
            K_1T_dVdx - 1.0/(2.0*PI) * contract('ji,Adj->Adi', D, K_1T_dVdx) * A)

    dqdx_fix_K_R = 0.5 * (K_1_R_dVdx + RT_K_1T_dVdx)

    return dqdx_fix_K_R

def get_dqsym_dx(pcmobj, dm, atmlst, intopt_derivative):
    # The three terms of a Hessian that take these (analytical_hess_nuc,
    # analytical_hess_qv, analytical_grad_vmat) ask with the same density:
    # kept with the charges of that density (PCM._get_vind)
    intermediates = pcmobj._intermediates
    key = tuple(atmlst)
    kept = intermediates.get('dqsym_dx', None)
    if (kept is not None and kept[0] is intermediates.get('dm', None) and kept[1] == key
            and has_intermediates_for(pcmobj, dm)):
        return kept[2]
    dqdx = get_dqsym_dx_fix_vgrids(pcmobj, atmlst) + get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, intopt_derivative)
    if has_intermediates_for(pcmobj, dm):
        intermediates['dqsym_dx'] = (intermediates['dm'], key, dqdx)
    return dqdx

# The integrals of the surface charges with the products of the orbitals and
# the occupied orbitals, (p i|g) [ngrids, nmo nocc], turn the solvent's
# response to the densities of the CPHF equations and the sums over derivative
# charges of analytical_grad_vmat into matrix products (SurfaceMOIntegrals).
# They are kept on the device up to this part of its available memory (a
# small one: the other terms of the response size their batches by what is
# available), on the host up to this part of mol.max_memory, and not made
# beyond that. The two spins of unrestricted orbitals have half of each.
MO_INTEGRALS_DEVICE_FRACTION = .1
MO_INTEGRALS_HOST_FRACTION = .5

class SurfaceMOIntegrals:
    '''(p i|g) = sum_{mu nu} C[mu,p] (mu nu|g) C[nu,i] for the surface charges
    g of pcmobj, all orbitals p and the occupied ones i of one spin.

    With them, for the densities D = C x C_occ^T + C_occ x^T C^T of first
    order orbitals x [nmo,nocc] (no virtual-virtual block),
        sum_{mu nu} D[mu,nu] (mu nu|g) = sum_{p i} w_p (C^T S D S C_occ)[p,i] (p i|g),
    w = 2 for virtual p and 1 for occupied p (potential, mo_density), and the
    matrix of charges between the orbitals is sum_g q[g] (p i|g) (fock).
    '''
    def __init__(self, pcmobj, mo_coeff, mo_occ, t, s):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        mo_coeff = cupy.asarray(mo_coeff)
        self.occidx = cupy.asarray(mo_occ) > 0
        self.nmo = mo_coeff.shape[1]
        self.nocc = int(self.occidx.sum())
        self.ngrids = t.shape[0]
        # [ngrids, nmo nocc], on the device or the host
        self.t = t
        self.sc = s.dot(mo_coeff)
        # points of a product with the integrals on the host
        self.blksize = max(1, int(2e8 / (8 * self.nmo * self.nocc)))

    def matches(self, mo_coeff, mo_occ):
        def same(a, b):
            if a is b:
                return True
            a, b = cupy.asarray(a), cupy.asarray(b)
            return a.shape == b.shape and bool((a == b).all())
        return same(self.mo_coeff, mo_coeff) and same(self.mo_occ, mo_occ)

    def mo_density(self, dms):
        '''x [n, nmo nocc] of potential for the symmetric densities dms
        [n,nao,nao]: w_p (C^T S D S C_occ)[p,i]. None if one of them has a
        virtual-virtual block.'''
        dms = cupy.asarray(dms)
        n = len(dms)
        vir = ~self.occidx
        x = cupy.empty((n, self.nmo, self.nocc))
        size = float(abs(dms).max()) if n else 0.
        for k in range(n):
            half = self.sc.T.dot(dms[k])
            if float(abs(half[vir].dot(self.sc[:,vir])).max()) > 1e-9 * max(size, 1e-30):
                return None
            x[k] = half.dot(self.sc[:,self.occidx])
        x[:,vir] *= 2
        return x.reshape(n, -1)

    def potential(self, x):
        '''[n,ngrids]: sum_{p i} x[n,p,i] (p i|g)'''
        x = cupy.asarray(x).reshape(-1, self.nmo * self.nocc)
        if isinstance(self.t, cupy.ndarray):
            return x.dot(self.t.T)
        v = cupy.empty((len(x), self.ngrids))
        for g0 in range(0, self.ngrids, self.blksize):
            g1 = min(g0 + self.blksize, self.ngrids)
            v[:,g0:g1] = x.dot(cupy.asarray(self.t[g0:g1]).T)
        return v

    def fock(self, q):
        '''[n,nmo,nocc]: sum_g q[n,g] (p i|g)'''
        q = cupy.asarray(q).reshape(-1, self.ngrids)
        if isinstance(self.t, cupy.ndarray):
            out = q.dot(self.t)
        else:
            out = cupy.zeros((len(q), self.nmo * self.nocc))
            for g0 in range(0, self.ngrids, self.blksize):
                g1 = min(g0 + self.blksize, self.ngrids)
                out += q[:,g0:g1].dot(cupy.asarray(self.t[g0:g1]))
        return out.reshape(len(q), self.nmo, self.nocc)

def surface_mo_integrals(pcmobj, mo_coeff, mo_occ, share=1.):
    '''SurfaceMOIntegrals of pcmobj for the orbitals of one spin, or None:
    beyond the angular momenta of the kernels of sgx.int1e_grids, or where
    they fit neither on the device nor on the host
    (MO_INTEGRALS_DEVICE_FRACTION, MO_INTEGRALS_HOST_FRACTION). share: the
    part of those two limits that these integrals may take (.5 for each spin
    of unrestricted orbitals, whose two sets are kept together).'''
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    opt = pcmobj._grid_integrals()
    if opt is None:
        return None
    mol = pcmobj.mol
    grid_coords = pcmobj.surface['grid_coords']
    charge_exp = pcmobj.surface['charge_exp']
    c = cupy.asarray(mo_coeff)
    mocc = c[:,cupy.asarray(mo_occ) > 0]
    ngrids = grid_coords.shape[0]
    nmo, nocc = c.shape[1], mocc.shape[1]
    nbytes = ngrids * nmo * nocc * 8
    if nbytes <= get_avail_mem() * MO_INTEGRALS_DEVICE_FRACTION * share:
        t = cupy.empty((ngrids, nmo, nocc))
    elif nbytes <= mol.max_memory * 1e6 * MO_INTEGRALS_HOST_FRACTION * share:
        t = numpy.empty((ngrids, nmo, nocc))
    else:
        return None
    opt.orbital_contracted(c, mocc, grid_coords, charge_exp**2, out=t)
    s = cupy.asarray(mol.intor('int1e_ovlp'))
    return SurfaceMOIntegrals(pcmobj, mo_coeff, mo_occ, t.reshape(ngrids, nmo * nocc), s)

def _mo_integrals_of(hessobj, mo_coeff, mo_occ):
    '''The SurfaceMOIntegrals of the solvent of a Hessian object for these
    orbitals (a pair for the two spins of unrestricted orbitals), made at the
    first call and kept on the object until its kernel ends; None where they
    are not to be had.'''
    unrestricted = isinstance(hessobj.base, scf.uhf.UHF)
    kept = getattr(hessobj, '_solvent_mo_integrals', None)
    if kept is not None:
        pairs = zip(kept, mo_coeff, mo_occ) if unrestricted else [(kept, mo_coeff, mo_occ)]
        if all(k.matches(c, o) for k, c, o in pairs):
            return kept
        hessobj._solvent_mo_integrals = kept = None
    solvent = hessobj.base.with_solvent
    if unrestricted:
        # the two spins share the limits; the larger set first, so that no
        # integrals are made where the two do not fit
        ints = []
        for k in range(2):
            ints.append(surface_mo_integrals(solvent, mo_coeff[k], mo_occ[k], share=.5))
            if ints[-1] is None:
                break
        ints = None if ints[-1] is None else tuple(ints)
    else:
        ints = surface_mo_integrals(solvent, mo_coeff, mo_occ)
    hessobj._solvent_mo_integrals = ints
    return ints

def _response_from_mo_integrals(solvent, ints, dms):
    '''The solvent's response to the densities dms of the CPHF equations
    ([n,nao,nao], or [2,n,nao,nao] for the two spins) as the matrices between
    the orbitals and the occupied ones [n,nmo,nocc] (a pair for the spins):
    PCM._B_dot_x and its transformation, with the two contractions of the
    integrals as products with SurfaceMOIntegrals. None for densities that
    are not of first order orbitals.'''
    unrestricted = isinstance(ints, tuple)
    if unrestricted:
        v_grids = 0
        for k in range(2):
            x = ints[k].mo_density(dms[k])
            if x is None:
                return None
            v_grids = v_grids - ints[k].potential(x)
    else:
        x = ints.mo_density(dms)
        if x is None:
            return None
        v_grids = -ints.potential(x)
    x = None

    b = solvent.left_multiply_R(v_grids.T)
    q = solvent.left_solve_K(b).T
    vK_1 = solvent.left_solve_K(v_grids.T, K_transpose = True)
    qt = solvent.left_multiply_R(vK_1, R_transpose = True).T
    q_sym = (q + qt)/2.0
    q_sym *= -1
    if unrestricted:
        return ints[0].fock(q_sym), ints[1].fock(q_sym)
    return ints.fock(q_sym)

def analytical_grad_vmat(pcmobj, dm, mo_coeff, mo_occ, atmlst=None, verbose=None, mo_integrals=None):
    '''
    dv_solv / da

    mo_integrals: the SurfaceMOIntegrals of these orbitals, for the matrices
    of the derivatives of the charges
    '''
    if (not pcmobj._intermediates or
        not any(isinstance(x, cupy.ndarray) for x in pcmobj._intermediates.values())):
        pcmobj.build()
    if not has_intermediates_for(pcmobj, dm):
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(pcmobj, verbose)
    t1 = log.init_timer()

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    if atmlst is None:
        atmlst = range(mol.natm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    q_sym        = pcmobj._intermediates['q_sym']

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    intopt_fock = int3c1e.VHFOpt(mol)
    intopt_fock.build(cutoff = 1e-14, aosym = True)
    intopt_derivative = int3c1e.VHFOpt(mol)
    intopt_derivative.build(cutoff = 1e-14, aosym = False)

    dIdx_mo = cupy.empty([len(atmlst), 3, nmo, nocc])

    dIdA = int1e_grids_ip1(mol, grid_coords, charges = q_sym, intopt = intopt_derivative, charge_exponents = charge_exp**2)
    for i_atom in atmlst:
        p0,p1 = aoslice[i_atom, 2:]
        # dIdx[i_atom, :, :, :] = 0
        # dIdx[i_atom, :, p0:p1, :] += dIdA[:, p0:p1, :]
        # dIdx[i_atom, :, :, p0:p1] += dIdA[:, p0:p1, :].transpose(0,2,1)
        dIdA_mo = dIdA[:, p0:p1, :] @ mocc
        dIdA_mo = contract('ip,dpj->dij', mo_coeff[p0:p1, :].T, dIdA_mo)
        dIdB_mo = dIdA[:, p0:p1, :].transpose(0,2,1) @ mocc[p0:p1, :]
        dIdB_mo = contract('ip,dpj->dij', mo_coeff.T, dIdB_mo)
        dIdx_mo[i_atom, :, :, :] = dIdA_mo + dIdB_mo
        del dIdA_mo, dIdB_mo
    del dIdA

    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        if g0 == g1:
            continue
        dIdC = int1e_grids_ip2(mol, grid_coords[g0:g1,:], charges = q_sym[g0:g1],
                               intopt = intopt_derivative, charge_exponents = charge_exp[g0:g1]**2)
        dIdC_mo = dIdC @ mocc
        dIdC_mo = contract('ip,dpj->dij', mo_coeff.T, dIdC_mo)
        dIdx_mo[i_atom, :, :, :] += dIdC_mo
        del dIdC, dIdC_mo

    dV_on_molecule_dx_mo = dIdx_mo

    dqdx = get_dqsym_dx(pcmobj, dm, atmlst, intopt_derivative)
    if mo_integrals is not None:
        dV_on_molecule_dx_mo += mo_integrals.fock(dqdx.reshape(-1, dqdx.shape[-1])).reshape(
            dV_on_molecule_dx_mo.shape)
        atmlst = ()
    for i_atom in atmlst:
        for i_xyz in range(3):
            dIdx_from_dqdx = int1e_grids(mol, grid_coords, charges = dqdx[i_atom, i_xyz, :],
                                         intopt = intopt_fock, charge_exponents = charge_exp**2)
            dV_on_molecule_dx_mo[i_atom, i_xyz, :, :] += mo_coeff.T @ dIdx_from_dqdx @ mocc

    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    return dV_on_molecule_dx_mo

def make_hess_object(base_method):
    '''Create nuclear hessian object with solvent contributions for the given
    solvent-attached method based on its hessian method in vaccum
    '''
    if isinstance(base_method, HessianBase):
        # For backward compatibility. In gpu4pyscf-1.4 and older, the input
        # argument is a hessian object.
        base_method = base_method.base

    # Must be a solvent-attached method
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for Hessian')

    vac_hess = base_method.undo_solvent().Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithSolventHess(vac_hess),
                         (WithSolventHess, vac_hess.__class__), name)

class WithSolventHess:

    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_cpu(self):
        hess_method = self.base.to_cpu().Hessian()
        return utils.to_cpu(self, hess_method)

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        if self.base.with_solvent.frozen_dm0_for_finite_difference_without_response is not None:
            raise NotImplementedError("frozen_dm0_for_finite_difference_without_response not implemented for PCM Hessian")
        if natm_without_ghost(self.mol) != self.mol.natm:
            raise NotImplementedError("Ghost atom not supported for PCM Hessian")

        try:
            with lib.temporary_env(self.base.with_solvent, equilibrium_solvation=True):
                logger.debug(self, 'Compute hessian from solutes')
                self.de_solute = super().kernel(*args, **kwargs)
        finally:
            # the integrals with the orbital products (_mo_integrals_of)
            self._solvent_mo_integrals = None
        logger.debug(self, 'Compute hessian from solvents')
        self.de_solvent = self.base.with_solvent.hess(dm)
        self.de = self.de_solute + self.de_solvent
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        # self.__class__.__bases__ are (WithSolventHess, vac_hess.__class__)
        vac_hess_klass = self.__class__.__bases__[1]
        h1ao = vac_hess_klass.make_h1(self, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        ints = _mo_integrals_of(self, mo_coeff, mo_occ)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1()
            dv = analytical_grad_vmat(self.base.with_solvent, dm, mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose,
                                      mo_integrals=ints)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1()
            dm = dm[0] + dm[1]
            if ints is None:
                ints = (None, None)
            dva = analytical_grad_vmat(solvent, dm, mo_coeff[0], mo_occ[0], atmlst=atmlst, verbose=verbose,
                                       mo_integrals=ints[0])
            dvb = analytical_grad_vmat(solvent, dm, mo_coeff[1], mo_occ[1], atmlst=atmlst, verbose=verbose,
                                       mo_integrals=ints[1])
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dva[i0]
                h1aob[i0] += dvb[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')

    def get_veff_resp_mo(self, mol, dms, mo_coeff, mo_occ, hermi=1):
        # self.__class__.__bases__ are (WithSolventHess, vac_hess.__class__)
        vac_hess_klass = self.__class__.__bases__[1]
        v1vo = vac_hess_klass.get_veff_resp_mo(self, mol, dms, mo_coeff, mo_occ, hermi=hermi)
        if not self.base.with_solvent.equilibrium_solvation:
            return v1vo
        solvent = self.base.with_solvent
        if solvent.frozen_dm0_for_finite_difference_without_response is None:
            # as products with the integrals of the surface charges and the
            # orbital products, where there are such
            ints = _mo_integrals_of(self, mo_coeff, mo_occ)
            v_solvent = None
            if ints is not None:
                v_solvent = _response_from_mo_integrals(solvent, ints, dms)
            if v_solvent is not None:
                if isinstance(ints, tuple):
                    n_dm = dms.shape[1]
                    na = v_solvent[0][0].size
                    v1vo[:,:na] += v_solvent[0].reshape(n_dm,-1)
                    v1vo[:,na:] += v_solvent[1].reshape(n_dm,-1)
                else:
                    v1vo += v_solvent.reshape(dms.shape[0],-1)
                return v1vo
        v_solvent = self.base.with_solvent._B_dot_x(dms)
        if isinstance(self.base, scf.uhf.UHF):
            n_dm = dms.shape[1]
            mocca = mo_coeff[0][:,mo_occ[0]>0]
            moccb = mo_coeff[1][:,mo_occ[1]>0]
            moa, mob = mo_coeff
            nmoa = moa.shape[1]
            nocca = mocca.shape[1]
            v1vo_sol = v_solvent[0] + v_solvent[1]
            v1vo[:,:nmoa*nocca] += _ao2mo(v1vo_sol, mocca, moa).reshape(n_dm,-1)
            v1vo[:,nmoa*nocca:] += _ao2mo(v1vo_sol, moccb, mob).reshape(n_dm,-1)
        elif isinstance(self.base, scf.hf.RHF):
            n_dm = dms.shape[0]
            mocc = mo_coeff[:,mo_occ>0]
            v1vo += _ao2mo(v_solvent, mocc, mo_coeff).reshape(n_dm,-1)
        else:
            raise NotImplementedError('Base object is not supported')
        return v1vo

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
