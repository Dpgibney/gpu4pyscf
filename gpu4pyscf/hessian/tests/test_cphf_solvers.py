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

import pytest
import numpy as np
import cupy as cp
import pyscf
from pyscf import lib
from gpu4pyscf import scf, dft
from gpu4pyscf.scf import cphf, ucphf
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.hessian import uhf as uhf_hess

# The CPHF equations of the Hessians: the residual solver
# (HessianBase.cphf_residual_tol) against the Krylov solver, the two-level
# CPHF (cphf_two_level) against the plain one, and the term that takes the
# Hessian to second order in the residuals

atom = '''
O   0.000   -0.    0.1174
H  -0.757    0.   -0.4696
H   0.757    0.   -0.4696
C   1.4     1.6    0.2
H   2.3     1.2    0.6
H   0.9     2.4    0.7
'''

def _mol(charge, spin):
    return pyscf.M(atom=atom, basis='6-31g*', charge=charge, spin=spin, verbose=0)

def _ks(cls, mol, xc):
    # small grids: the solvers are compared on one model
    mf = cls(mol, xc=xc)
    mf.grids.atom_grid = (40, 110)
    mf.nlcgrids.atom_grid = (20, 50)
    return mf

def _smd(mf):
    mf = mf.SMD()
    mf.with_solvent.solvent = 'water'
    mf.with_solvent.lebedev_order = 11
    return mf

CASES = {
    'rhf': (0, 0, lambda mol: scf.RHF(mol)),
    'uhf': (1, 1, lambda mol: scf.UHF(mol)),
    'rks_pbe': (0, 0, lambda mol: _ks(dft.RKS, mol, 'pbe')),
    'rks_b3lyp': (0, 0, lambda mol: _ks(dft.RKS, mol, 'b3lyp')),
    'rks_wb97mv_smd': (0, 0, lambda mol: _smd(_ks(dft.RKS, mol, 'wb97m-v'))),
    'rks_b3lyp_df': (0, 0, lambda mol: _ks(dft.RKS, mol, 'b3lyp').density_fit()),
    'uks_wb97mv': (1, 1, lambda mol: _ks(dft.UKS, mol, 'wb97m-v')),
}
KS_CASES = [case for case in CASES if 'ks' in case]

_scf = {}
def _converged(case):
    if case not in _scf:
        charge, spin, method = CASES[case]
        mf = method(_mol(charge, spin))
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        _scf[case] = mf
    return _scf[case]

_tight = {}
def _reference(case):
    # the Hessian with the CPHF equations solved to a residual of 1e-9
    if case not in _tight:
        hobj = _converged(case).Hessian()
        hobj.cphf_residual_tol = 1e-9
        hobj.cphf_two_level = False
        _tight[case] = np.asarray(hobj.kernel())
    return _tight[case]

def _exact_responses(hobj):
    '''The list that gets an entry for every response of the model itself
    that the Hessian object builds (not the cheap model's)'''
    calls = []
    resp = hobj.get_veff_resp_mo
    def counted(*args, **kwargs):
        if getattr(hobj, '_cheap_response', None) is None:
            calls.append(1)
        return resp(*args, **kwargs)
    hobj.get_veff_resp_mo = counted
    return calls

@pytest.mark.parametrize('case', CASES)
def test_residual_solver(case):
    # the default tolerance leaves the Hessian within 1e-6 of the converged
    # one (its error goes as the square of the residual)
    ref = _reference(case)
    hobj = _converged(case).Hessian()
    assert hobj.cphf_residual_tol == 1e-4
    hobj.cphf_two_level = False
    h = np.asarray(hobj.kernel())
    assert abs(h - ref).max() < 1e-6

@pytest.mark.parametrize('case', ['rhf', 'rks_b3lyp', 'uks_wb97mv'])
def test_krylov_solver(case):
    # cphf_residual_tol = None: the Krylov solver, here to a tight tolerance
    # (it stops on the size of its vectors, about 1e-6 from the solution)
    ref = _reference(case)
    mf = _converged(case)
    conv_tol_cpscf = mf.conv_tol_cpscf
    mf.conv_tol_cpscf = 1e-10
    try:
        hobj = mf.Hessian()
        hobj.cphf_residual_tol = None
        h = np.asarray(hobj.kernel())
    finally:
        mf.conv_tol_cpscf = conv_tol_cpscf
    assert abs(h - ref).max() < 1e-5

@pytest.mark.parametrize('case', KS_CASES)
def test_two_level_cphf(case):
    # The cheap model's solution and corrections as the start and the trial
    # vectors: the same solution, with the model's own response evaluated for
    # fewer sets of densities (all the parts the cheap model has)
    ref = _reference(case)
    mf = _converged(case)
    counts = {}
    hessians = {}
    for two_level in (False, True):
        hobj = mf.Hessian()
        hobj.cphf_residual_tol = 1e-7
        hobj.cphf_two_level = two_level
        hobj.cphf_two_level_parts = ('xc', 'nlc', 'k')
        calls = _exact_responses(hobj)
        hessians[two_level] = np.asarray(hobj.kernel())
        counts[two_level] = len(calls)
    assert abs(hessians[True] - ref).max() < 1e-7
    assert abs(hessians[False] - ref).max() < 1e-7
    assert counts[True] < counts[False]

@pytest.mark.parametrize('case', ['rks_b3lyp', 'rks_wb97mv_smd', 'uks_wb97mv'])
def test_two_level_default(case):
    # The default: a cheap model that differs in the grids of the VV10 term
    # only; the model's own response is then built once or twice. No cheap
    # model, and the plain solver, without a VV10 term
    ref = _reference(case)
    mf = _converged(case)
    hobj = mf.Hessian()
    assert hobj.cphf_two_level
    assert hobj.cphf_two_level_parts == ('nlc',)
    cheap = hobj.gen_vind_cheap(cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ))
    calls = _exact_responses(hobj)
    h = np.asarray(hobj.kernel())
    assert abs(h - ref).max() < 1e-6
    if 'wb97mv' in case:
        assert cheap is not None
        assert len(calls) <= 2
    else:
        assert cheap is None
        assert len(calls) > 2

@pytest.mark.parametrize('case', ['rks_b3lyp', 'rks_wb97mv_smd', 'uks_wb97mv'])
@pytest.mark.parametrize('parts', [('nlc',), ('xc',), ('k',), ('xc', 'nlc', 'k')])
def test_two_level_parts(case, parts):
    # the parts of the response that the cheap model takes on its own terms
    ref = _reference(case)
    mf = _converged(case)
    hobj = mf.Hessian()
    hobj.cphf_two_level = True
    hobj.cphf_two_level_parts = parts
    cheap = hobj.gen_vind_cheap(cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ))
    has = {'xc': True, 'nlc': 'wb97mv' in case, 'k': True}
    if not any(has[part] for part in parts):
        assert cheap is None
    else:
        assert cheap is not None
    h = np.asarray(hobj.kernel())
    assert abs(h - ref).max() < 1e-6

@pytest.mark.parametrize('case', ['rks_wb97mv_smd', 'uks_wb97mv'])
def test_two_level_correction_step(case):
    # The two kinds of correction steps after the start from the cheap
    # model's solution (TWO_LEVEL_PLAIN_STEP): the preconditioned residuals
    # as trial vectors, which asks nothing more of the cheap model, and the
    # cheap model's solutions for the residuals
    ref = _reference(case)
    mf = _converged(case)
    cheap_responses = {}
    for plain_step in (1e9, 0.):
        hobj = mf.Hessian()
        # below what the cheap model's grids leave: a correction is needed
        hobj.cphf_residual_tol = 1e-7
        resp = hobj.get_veff_resp_mo
        calls = {True: 0, False: 0}
        def counted(*args, _calls=calls, _hobj=hobj, _resp=resp, **kwargs):
            _calls[getattr(_hobj, '_cheap_response', None) is not None] += 1
            return _resp(*args, **kwargs)
        hobj.get_veff_resp_mo = counted
        with lib.temporary_env(rhf_hess, TWO_LEVEL_PLAIN_STEP=plain_step):
            h = np.asarray(hobj.kernel())
        assert abs(h - ref).max() < 1e-7
        # the start and at least one correction
        assert calls[False] >= 2
        cheap_responses[plain_step] = calls[True]
    assert cheap_responses[1e9] < cheap_responses[0.]

@pytest.mark.parametrize('case', ['rks_wb97mv_smd', 'uks_wb97mv'])
def test_cphf_nlcgrids_level(case):
    # The approximation that is off by default: the VV10 response on coarser
    # grids. With the level of the two-level CPHF's cheap model, that model
    # has nothing left to differ in
    ref = _reference(case)
    mf = _converged(case)
    hobj = mf.Hessian()
    assert hobj.cphf_nlcgrids_level is None
    hobj.cphf_nlcgrids_level = hobj.cphf_two_level_nlcgrids_level
    assert hobj.gen_vind_cheap(cp.asarray(mf.mo_coeff), cp.asarray(mf.mo_occ)) is None
    calls = _exact_responses(hobj)
    h = np.asarray(hobj.kernel())
    assert len(calls) > 2
    print(case, 'cphf_nlcgrids_level: max |dH| =', abs(h - ref).max())
    assert abs(h - ref).max() < 1e-4

@pytest.mark.parametrize('case', ['rks_b3lyp', 'uks_wb97mv'])
def test_second_order_term(case):
    # A start from the cheap model's solution leaves the terms that the
    # Hessian takes from the first-order orbitals with an error of first
    # order in the residuals; with what solve_mo1 hands to hess_elec they are
    # right to second order
    ref = _reference(case)
    mf = _converged(case)
    errors = {}
    for with_term in (True, False):
        hobj = mf.Hessian()
        hobj.cphf_residual_tol = 3e-3
        hobj.cphf_two_level = True
        hobj.cphf_two_level_parts = ('xc', 'nlc', 'k')
        if with_term:
            h = np.asarray(hobj.kernel())
        else:
            def none(kept, natm, factor):
                return cp.zeros((natm,natm,3,3))
            with lib.temporary_env(rhf_hess, _second_order_term=none):
                h = np.asarray(hobj.kernel())
        errors[with_term] = abs(h - ref).max()
    assert errors[True] < 3e-5
    assert errors[True] < errors[False] * .2

def test_second_order_term_of_the_plain_solver():
    # nothing for the plain solver on one block of atoms: the residuals are
    # orthogonal to the trial vectors
    mf = _converged('rks_b3lyp')
    hobj = mf.Hessian()
    hobj.cphf_two_level = False
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    h1mo = hobj.make_h1(mo_coeff, mo_occ, None, None, None)
    hobj.solve_mo1(cp.asarray(mf.mo_energy), mo_coeff, mo_occ, h1mo,
                   hobj.gen_vind(mo_coeff, mo_occ))
    assert abs(hobj._cphf_second_order).max() < 1e-12

def _in_atom_blocks(hobj, natm_block):
    '''Makes hobj solve its CPHF equations as on a device with the memory for
    natm_block atoms at a time (solve_mo1 sizes its blocks by the available
    memory). Returns the list that gets the number of perturbations of every
    block.'''
    unrestricted = isinstance(hobj.base, scf.uhf.UHF)
    mod, solver = (uhf_hess, ucphf) if unrestricted else (rhf_hess, cphf)
    nao = hobj.mol.nao
    # the memory in which solve_mo1 counts natm_block atoms
    avail = (natm_block + .5) * 8*nao*nao*3 * (6 / .3 if unrestricted else 3 / .5)
    solve_mo1 = hobj.solve_mo1
    solve_residual = solver.solve_residual
    calls = []

    def counted(*args, **kwargs):
        # the solve of a block with the model's own response (the cheap
        # model's solves of the two-level CPHF go through the same function,
        # without this argument)
        if 'second_order' in kwargs:
            h1 = args[3]
            calls.append(len(h1[0] if unrestricted else h1))
        return solve_residual(*args, **kwargs)

    def in_blocks(*args, **kwargs):
        with lib.temporary_env(mod, get_avail_mem=lambda: avail), \
                lib.temporary_env(solver, solve_residual=counted):
            return solve_mo1(*args, **kwargs)

    hobj.solve_mo1 = in_blocks
    return calls

@pytest.mark.parametrize('case', ['rhf', 'uhf', 'rks_wb97mv_smd', 'rks_b3lyp_df', 'uks_wb97mv'])
def test_atom_blocks(case):
    # The CPHF equations in blocks of atoms, as on a device without the memory
    # for all the perturbations at once: the same Hessian, with the default
    # settings (the two-level CPHF in every block)
    ref = _reference(case)
    mf = _converged(case)
    natm = mf.mol.natm
    for natm_block, blocks in ((natm, [natm*3]), (2, [6]*(natm//2)), (4, [12, (natm-4)*3])):
        hobj = mf.Hessian()
        calls = _in_atom_blocks(hobj, natm_block)
        h = np.asarray(hobj.kernel())
        assert calls == blocks
        print(case, natm_block, 'atoms in a block: max |dH| =', abs(h - ref).max())
        assert abs(h - ref).max() < 1e-6

@pytest.mark.parametrize('case', ['rks_b3lyp', 'uks_wb97mv'])
@pytest.mark.parametrize('two_level', [True, False])
def test_atom_blocks_second_order_term(case, two_level):
    # The term that takes the Hessian to second order in the residuals has a
    # part for every pair of perturbations, also of two blocks, which the
    # plain solver needs too once there are several blocks (the residuals of a
    # block are orthogonal to the trial vectors of that block only). Loose
    # residuals, so that the term shows
    ref = _reference(case)
    mf = _converged(case)
    errors = {}
    for with_term in (True, False):
        hobj = mf.Hessian()
        hobj.cphf_residual_tol = 3e-3
        hobj.cphf_two_level = two_level
        hobj.cphf_two_level_parts = ('xc', 'nlc', 'k')
        calls = _in_atom_blocks(hobj, 2)
        if with_term:
            h = np.asarray(hobj.kernel())
        else:
            def none(kept, natm, factor):
                return cp.zeros((natm,natm,3,3))
            with lib.temporary_env(rhf_hess, _second_order_term=none):
                h = np.asarray(hobj.kernel())
        assert len(calls) == mf.mol.natm // 2
        errors[with_term] = abs(h - ref).max()
    print(case, 'two-level' if two_level else 'plain', 'blocks of 2 atoms: max |dH| with the term',
          errors[True], 'without', errors[False])
    # (measured, with / without: two-level 1.2e-6 / 8.3e-5 and 6.8e-6 / 8.4e-5,
    # plain 1.3e-5 / 1.5e-4 and 1.1e-5 / 5.8e-5)
    assert errors[True] < 3e-5
    assert errors[True] < errors[False] * (.2 if two_level else .5)

def test_cphf_atom_block():
    # all the atoms; multiples of 16; below 16, what fits (upstream: an error)
    f = rhf_hess._cphf_atom_block
    assert [f(n, 57) for n in (1000, 57, 56, 33, 32, 31, 16, 15, 3, 1)] == [57, 57, 48, 32, 32, 16, 16, 15, 3, 1]
    assert [f(n, 3) for n in (40, 3, 2)] == [3, 3, 2]
    with pytest.raises(RuntimeError):
        f(0, 3)

@pytest.mark.parametrize('case', ['rks_b3lyp', 'uks_wb97mv'])
def test_sgx_response(case):
    # The approximation of the response's J and K, off by default: the
    # seminumerical K, with the exact J or RI-J, in every response of the CPHF
    # equations (also the two-level CPHF's cheap model's); the SCF, the
    # derivative integrals and the XC and VV10 terms exact
    ref = _reference(case)
    mf = _converged(case)
    assert not mf.Hessian().sgx_response
    errs = {}
    for auxbasis in (None, 'aug_etb'):
        for dtype in ('double', 'float'):
            hobj = mf.Hessian()
            hobj.sgx_response = True
            hobj.sgx_response_auxbasis = auxbasis
            hobj.sgx_response_dtype = dtype
            errs[auxbasis, dtype] = abs(np.asarray(hobj.kernel()) - ref).max()
    print(case, 'sgx_response: max |dH| against the exact response', errs)
    # (measured: 3.1e-6 and 3.7e-6 with the exact J, 1.4e-4 and 1.2e-4 with RI-J,
    # whose aug_etb basis of 6-31G* is a small one; HCN in def2-TZVP 6e-7 and 2e-6)
    for dtype in ('double', 'float'):
        assert 1e-9 < errs[None, dtype] < 2e-5
        assert errs[None, dtype] < errs['aug_etb', dtype] < 5e-4
    assert abs(errs[None, 'float'] - errs[None, 'double']) < 1e-7

def test_sgx_response_hf():
    hobj = _converged('rhf').Hessian()
    hobj.sgx_response = True
    with pytest.raises(NotImplementedError):
        hobj.kernel()
