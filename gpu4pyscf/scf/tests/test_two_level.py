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
from gpu4pyscf import scf, dft
from gpu4pyscf.scf import two_level
from gpu4pyscf.scf import diis as scf_diis

atom = '''
O   0.000   -0.    0.1174
H  -0.757    0.   -0.4696
H   0.757    0.   -0.4696
C   1.4     1.6    0.2
H   2.3     1.2    0.6
H   0.9     2.4    0.7
'''

def _mol(charge, spin):
    return pyscf.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0)

def _smd(mf):
    mf = mf.SMD()
    mf.with_solvent.solvent = 'water'
    return mf

def _pcm(mf, method):
    mf = mf.PCM()
    mf.with_solvent.method = method
    mf.with_solvent.eps = 78.3553
    return mf

CASES = {
    'rks_pbe': (0, 0, lambda mol: dft.RKS(mol, xc='pbe')),
    'rks_b3lyp': (0, 0, lambda mol: dft.RKS(mol, xc='b3lyp')),
    'rks_r2scan': (0, 0, lambda mol: dft.RKS(mol, xc='r2scan')),
    'rks_wb97x': (0, 0, lambda mol: dft.RKS(mol, xc='wb97x')),
    'rks_wb97mv': (0, 0, lambda mol: dft.RKS(mol, xc='wb97m-v')),
    'uks_b3lyp': (1, 1, lambda mol: dft.UKS(mol, xc='b3lyp')),
    'uks_wb97mv': (1, 1, lambda mol: dft.UKS(mol, xc='wb97m-v')),
    'uks_pbe': (1, 1, lambda mol: dft.UKS(mol, xc='pbe')),
    'rks_wb97mv_smd': (0, 0, lambda mol: _smd(dft.RKS(mol, xc='wb97m-v'))),
    'uks_b3lyp_smd': (1, 1, lambda mol: _smd(dft.UKS(mol, xc='b3lyp'))),
    'rks_b3lyp_cpcm': (0, 0, lambda mol: _pcm(dft.RKS(mol, xc='b3lyp'), 'C-PCM')),
    'rks_b3lyp_iefpcm': (0, 0, lambda mol: _pcm(dft.RKS(mol, xc='b3lyp'), 'IEF-PCM')),
    'rks_b3lyp_df': (0, 0, lambda mol: dft.RKS(mol, xc='b3lyp').density_fit()),
    'rks_b3lyp_level_shift': (0, 0, lambda mol: dft.RKS(mol, xc='b3lyp').set(level_shift=0.2)),
}

@pytest.mark.parametrize('case', CASES)
def test_two_level_energy(case):
    # the two-level solution is the model's own; here with the cheap model
    # that builds the model's J and K (the seminumerical K: SGX_CASES below)
    charge, spin, method = CASES[case]
    mol = _mol(charge, spin)
    e = []
    for on in (False, True):
        mf = method(mol)
        mf.conv_tol = 1e-10
        mf.two_level = on
        mf.two_level_sgx_level = None
        e.append(mf.kernel())
        assert mf.converged
    assert abs(e[1] - e[0]) < 2e-9

def test_two_level_gradient():
    mol = _mol(0, 0)
    g = []
    for on in (False, True):
        mf = _smd(dft.RKS(mol, xc='wb97m-v'))
        mf.conv_tol = 1e-10
        mf.two_level = on
        mf.kernel()
        g.append(mf.Gradients().kernel())
    assert abs(g[1] - g[0]).max() < 1e-6

def test_two_level_max_cycle():
    # max_cycle bounds the cheap model's cycles and the micro cycles together
    mol = _mol(0, 0)
    mf = dft.RKS(mol, xc='b3lyp')
    mf.two_level = True
    mf.max_cycle = 3
    e = mf.kernel()
    assert not mf.converged
    assert mf.cycles == 1
    assert abs(e - mf.energy_tot()) < 1e-8

def test_two_level_without_coarser_grids():
    # nothing to make cheaper: the plain SCF
    mol = _mol(0, 0)
    mf = dft.RKS(mol, xc='pbe')
    mf.grids.level = 1
    assert two_level.cheap_scf(mf) is None
    assert two_level.cheap_scf(scf.RHF(mol)) is None
    mf.two_level = True
    e1 = mf.kernel()
    mf = dft.RKS(mol, xc='pbe')
    mf.grids.level = 1
    assert abs(mf.kernel() - e1) < 1e-9

def test_coarse_grids_keep_coarser_atom_grid():
    mol = _mol(0, 0)
    mf = dft.RKS(mol, xc='pbe')
    mf.grids.atom_grid = {'O': (99, 590), 'C': (99, 590), 'H': (10, 26)}
    g = two_level.coarse_grids(mf.grids, mol, 1)
    assert g.level == 1
    assert g.atom_grid == {'H': (10, 26)}
    assert mf.grids.atom_grid['O'] == (99, 590)

def test_diis_history_with_shifted_fock():
    # add_to_fock gives the history of the Fock matrices F_i + delta at the same densities
    mol = _mol(0, 0)
    mf = dft.RKS(mol, xc='pbe')
    s = mf.get_ovlp()
    x = mf.check_linear_dependency(s)
    nao = s.shape[0]
    rng = np.random.default_rng(7)
    def sym():
        a = cp.asarray(rng.random((nao, nao)))
        return a + a.T
    fs = [sym() for _ in range(4)]
    ds = [sym() for _ in range(4)]
    delta = sym() * .1
    cls = two_level._diis_keeping_densities(scf_diis.SCF_DIIS)
    d1 = cls(mf, None)
    d2 = scf_diis.SCF_DIIS(mf, None)
    d1.Corth = d2.Corth = cp.asarray(x)
    for f, d in zip(fs[:3], ds[:3]):
        d1.update(s, d, f)
        d2.update(s, d, f + delta)
    d1.add_to_fock(s, delta)
    f1 = d1.update(s, ds[3], fs[3] + delta)
    f2 = d2.update(s, ds[3], fs[3] + delta)
    assert abs(f1 - f2).max() < 1e-9

# The cheap model with a seminumerical K (two_level_sgx_level): the model's own
# K is built at the references only
SGX_CASES = ['rks_b3lyp', 'rks_wb97x', 'rks_wb97mv', 'uks_b3lyp', 'uks_wb97mv',
             'rks_wb97mv_smd', 'uks_b3lyp_smd', 'rks_b3lyp_cpcm',
             'rks_b3lyp_level_shift']
_plain = {}

def _plain_energy(case):
    if case not in _plain:
        charge, spin, method = CASES[case]
        mf = method(_mol(charge, spin))
        mf.conv_tol = 1e-10
        _plain[case] = mf.kernel()
        assert mf.converged
    return _plain[case]

@pytest.mark.parametrize('dtype', ['double', 'float'])
@pytest.mark.parametrize('case', SGX_CASES)
def test_two_level_sgx_energy(case, dtype):
    charge, spin, method = CASES[case]
    mf = method(_mol(charge, spin))
    mf.conv_tol = 1e-10
    mf.two_level = True
    mf.two_level_sgx_level = 1
    mf.two_level_sgx_dtype = dtype
    e = mf.kernel()
    assert mf.converged
    assert abs(e - _plain_energy(case)) < 2e-9
    assert mf.cycles <= 3

def test_two_level_sgx_models():
    from gpu4pyscf.sgx.sgx import _COSX
    mol = _mol(0, 0)
    def cheap(mf):
        mf.two_level_sgx_level = 1
        return two_level.cheap_scf(mf)
    # a hybrid with the K of the four-center integrals: also with a fitted J
    assert isinstance(cheap(dft.RKS(mol, xc='b3lyp')), _COSX)
    assert isinstance(cheap(dft.RKS(mol, xc='b3lyp').density_fit(only_dfj=True)), _COSX)
    # no K, or a fitted one
    assert not isinstance(cheap(dft.RKS(mol, xc='pbe')), _COSX)
    assert not isinstance(cheap(dft.RKS(mol, xc='b3lyp').density_fit()), _COSX)
    # the grids of the model are at the cheap model's levels: K alone
    mf = dft.RKS(mol, xc='b3lyp')
    mf.grids.level = 1
    assert isinstance(cheap(mf), _COSX)
    mf.two_level_sgx_level = None
    assert two_level.cheap_scf(mf) is None

def test_two_level_sgx_dfj():
    mol = _mol(0, 0)
    e = []
    for on in (False, True):
        mf = dft.RKS(mol, xc='b3lyp').density_fit(only_dfj=True)
        mf.conv_tol = 1e-10
        mf.two_level = on
        mf.two_level_sgx_level = 1
        e.append(mf.kernel())
        assert mf.converged
    assert abs(e[1] - e[0]) < 2e-9

def test_two_level_sgx_gradient():
    mol = _mol(0, 0)
    g = []
    for on in (False, True):
        mf = _smd(dft.RKS(mol, xc='wb97x'))
        mf.conv_tol = 1e-10
        mf.two_level = on
        mf.two_level_sgx_level = 1
        mf.kernel()
        g.append(mf.Gradients().kernel())
    assert abs(g[1] - g[0]).max() < 1e-6

# A fitted J in the cheap model (two_level_auxbasis): the model's own J is
# built at the references only
@pytest.mark.parametrize('case,sgx_level', [
    ('rks_pbe', None), ('uks_pbe', None), ('rks_b3lyp', None), ('rks_b3lyp', 1),
    ('rks_wb97mv', 1), ('uks_wb97mv', 1), ('rks_wb97mv_smd', 1)])
def test_two_level_fitted_j(case, sgx_level):
    charge, spin, method = CASES[case]
    mf = method(_mol(charge, spin))
    mf.conv_tol = 1e-10
    mf.two_level = True
    mf.two_level_auxbasis = 'def2-universal-jfit'
    mf.two_level_sgx_level = sgx_level
    mf.two_level_sgx_dtype = 'float'
    cheap = two_level.cheap_scf(mf)
    assert isinstance(cheap, two_level._FittedJ)
    e = mf.kernel()
    assert mf.converged
    assert abs(e - _plain_energy(case)) < 2e-9
    assert mf.cycles <= 3

def test_two_level_fitted_j_models():
    mol = _mol(0, 0)
    # the fitted J of the model itself is not replaced
    mf = dft.RKS(mol, xc='pbe').density_fit()
    mf.two_level_auxbasis = 'def2-universal-jfit'
    assert not isinstance(two_level.cheap_scf(mf), two_level._FittedJ)
    # a model on the cheap model's grids: J alone
    mf = dft.RKS(mol, xc='pbe')
    mf.grids.level = 1
    assert two_level.cheap_scf(mf) is None
    mf.two_level_auxbasis = 'def2-universal-jfit'
    assert isinstance(two_level.cheap_scf(mf), two_level._FittedJ)

# the parts a reference builds as a change of the density, and two_level_sgx_ref_level
def test_two_level_reference_energy_form():
    # The energy of V[D1] + V[dD] in the form that takes V[dD] at second order
    from gpu4pyscf.lib.cupy_helper import tag_array
    mol = _mol(0, 0)
    mf = dft.RKS(mol, xc='b3lyp')
    mf.kernel()
    d2 = cp.asarray(mf.make_rdm1())
    rng = np.random.default_rng(3)
    x = cp.asarray(rng.normal(size=d2.shape)) * 1e-4
    d1 = d2 + x + x.T
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    def tr(a, b):
        return float(cp.einsum('ij,ji', a, b))
    k2 = mf.get_k(mol, d2, 1, omega, alpha, hyb)
    dk = mf.get_k(mol, d2 - d1, 1, omega, alpha, hyb)
    j2 = mf.get_j(mol, d2)
    e_true = -.25 * tr(d2, k2)
    zero = cp.zeros_like(d2)
    vx = tag_array(zero, vj=zero, vk=zero, ecoul=0.)

    # K with a k_change that is the exact K[dD] times 1 + eps: off by
    # eps tr(dD K[dD]) / 4, where the plain form is by eps tr(D2 K[dD]) / 4
    eps = 1e-2
    def k_change(ddm, omega, alpha, hyb):
        return mf.get_k(mol, ddm, 1, omega, alpha, hyb) * (1 + eps)
    last = {}
    v = two_level._own_jk(mf, mol, d1, vx, (False, True), last, k_change)
    assert v.e_shift == 0
    v = two_level._own_jk(mf, mol, d2, vx, (False, True), last, k_change)
    e_plain = -.25 * tr(d2, 2 * v.vk)
    second = -.25 * eps * tr(d2 - d1, dk)
    assert abs(e_plain + v.e_shift - e_true - second) < 1e-11
    assert abs(e_plain - e_true) > 100 * abs(second)
    assert abs(v.vj + v.vk).max() == 0 and v.ecoul == 0

    # the model's own K of the change
    last = {}
    for d in (d1, d2):
        v = two_level._own_jk(mf, mol, d, vx, (False, True), last)
    assert abs(2 * v.vk - k2).max() < 1e-10
    assert abs(-.25 * tr(d2, 2 * v.vk) + v.e_shift - e_true) < 1e-11

    # the model's own J: ecoul in the form, and e_shift what takes the trace
    # of a hybrid's J - K/2 to it
    last = {}
    for d in (d1, d2):
        v = two_level._own_jk(mf, mol, d, vx, (True, False), last)
    assert abs(v.vj - j2).max() < 1e-10 and abs(v.vk).max() == 0
    assert abs(v.ecoul - .5 * tr(d2, j2)) < 1e-10
    assert abs(.5 * tr(d2, v.vj) + v.e_shift - v.ecoul) < 1e-10

@pytest.mark.parametrize('case,dtype', [
    ('rks_b3lyp', 'double'), ('rks_b3lyp', 'float'), ('rks_wb97mv_smd', 'double'),
    ('uks_b3lyp', 'double'), ('uks_wb97mv', 'float')])
def test_two_level_reference_k(case, dtype, monkeypatch):
    from gpu4pyscf.scf import jk
    e_plain = _plain_energy(case)
    calls = []
    get_k = jk.get_k
    def counted(*args, **kwargs):
        calls.append(1)
        return get_k(*args, **kwargs)
    monkeypatch.setattr(jk, 'get_k', counted)
    charge, spin, method = CASES[case]
    mf = method(_mol(charge, spin))
    mf.conv_tol = 1e-10
    mf.two_level = True
    mf.two_level_sgx_ref_level = 2
    mf.two_level_sgx_ref_dtype = dtype
    e = mf.kernel()
    assert mf.converged
    # one build of the four-center K, at the first of the references
    assert mf.cycles >= 2 and len(calls) == 1
    assert abs(e - e_plain) < 2e-9
