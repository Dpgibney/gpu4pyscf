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
from gpu4pyscf.scf import hf

# SCF.second_order_energy: the two-electron energy of J and K that are built
# for the changes of the density and summed (hf._e2_of_change)

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

CASES = {
    'rhf': (0, 0, lambda mol: scf.RHF(mol)),
    'uhf': (1, 1, lambda mol: scf.UHF(mol)),
    'rks_pbe': (0, 0, lambda mol: dft.RKS(mol, xc='pbe')),
    'rks_b3lyp': (0, 0, lambda mol: dft.RKS(mol, xc='b3lyp')),
    'rks_wb97x': (0, 0, lambda mol: dft.RKS(mol, xc='wb97x')),
    'uks_pbe': (1, 1, lambda mol: dft.UKS(mol, xc='pbe')),
    'uks_b3lyp': (1, 1, lambda mol: dft.UKS(mol, xc='b3lyp')),
    'rohf': (1, 1, lambda mol: scf.ROHF(mol)),
    'roks_b3lyp': (1, 1, lambda mol: dft.ROKS(mol, xc='b3lyp')),
    'rhf_one_pass': (0, 0, lambda mol: scf.RHF(mol).set(jk_one_pass=True)),
    'uks_b3lyp_one_pass': (1, 1, lambda mol: dft.UKS(mol, xc='b3lyp').set(jk_one_pass=True)),
}

@pytest.mark.parametrize('case', CASES)
def test_scf_energy(case):
    # The two forms on the same Fock matrices: energies apart by what the
    # screening of the builds leaves out, and each the energy of one build of
    # the potential of its final density
    charge, spin, method = CASES[case]
    mol = _mol(charge, spin)
    energies = []
    for on in (True, False):
        mf = method(mol)
        mf.conv_tol = 1e-11
        mf.second_order_energy = on
        e = mf.kernel()
        assert mf.converged
        energies.append(e)
        dm = mf.make_rdm1()
        one = method(mol)
        e_one = one.energy_tot(dm, one.get_hcore(), one.get_veff(mol, dm))
        assert abs(e - e_one) < 2e-10
    assert abs(energies[0] - energies[1]) < 2e-10

@pytest.mark.parametrize('case', ['rhf', 'uhf', 'rks_pbe', 'rks_b3lyp', 'rks_wb97x', 'uks_pbe', 'uks_b3lyp'])
def test_energy_of_a_change(case):
    # get_veff for a change of the density: the tag e2 is continued by
    # hf._e2_of_change, and the energy is that of one build
    charge, spin, method = CASES[case]
    mol = _mol(charge, spin)
    mf = method(mol)
    mf.kernel()
    h1e = mf.get_hcore()
    d2 = cp.asarray(mf.make_rdm1())
    rng = np.random.default_rng(5)
    x = cp.asarray(rng.normal(size=d2.shape[-2:])) * 1e-4
    d1 = d2 + (x + x.T)
    ddm = d2 - d1
    e_one = mf.energy_elec(d2, h1e, mf.get_veff(mol, d2))[0]
    for on in (True, False):
        mf.second_order_energy = on
        v1 = mf.get_veff(mol, d1)
        v2 = mf.get_veff(mol, d2, d1, v1)
        assert abs(mf.energy_elec(d2, h1e, v2)[0] - e_one) < 1e-9
        if not on:
            assert getattr(v2, 'e2', None) is None
            continue
        # the potential of J and K in the two: the array for HF, the tag vj
        # for DFT (J alone without exact exchange)
        if case.endswith('hf'):
            p1, p2 = cp.asarray(v1), cp.asarray(v2)
            assert abs(v2.e2_shift - (v2.e2 - hf._trace(d2, p2) * .5)) < 1e-10
            assert v1.e2_shift == 0
        else:
            p1, p2 = cp.asarray(v1.vj), cp.asarray(v2.vj)
            assert abs(v1.ecoul + v1.exc - (v2.ecoul + v2.exc)
                       - (mf.energy_elec(d1, h1e, v1)[1] - mf.energy_elec(d2, h1e, v2)[1])) < 1e-10
        assert abs(v1.e2 - hf._trace(d1, p1) * .5) < 1e-10
        assert abs(v2.e2 - (v1.e2 + hf._trace(ddm, p1) + hf._trace(ddm, p2 - p1) * .5)) < 1e-10

@pytest.mark.parametrize('case', ['rhf', 'uhf', 'rks_b3lyp', 'uks_b3lyp'])
def test_change_built_with_an_error(case):
    # A K of the change that is off by a factor 1 + eps, standing in for what
    # the screening of such a build leaves out: the summed potential traced
    # with the density is off by eps tr(D K[dD]) c, the form by
    # eps tr(dD K[dD]) c (c = hyb/4 restricted, hyb/2 for the spins)
    charge, spin, method = CASES[case]
    mol = _mol(charge, spin)
    mf = method(mol)
    mf.kernel()
    h1e = mf.get_hcore()
    d2 = cp.asarray(mf.make_rdm1())
    rng = np.random.default_rng(7)
    x = cp.asarray(rng.normal(size=d2.shape[-2:])) * 1e-4
    d1 = d2 + (x + x.T)
    ddm = d2 - d1
    get_k = mf.get_k
    scale = [1.]
    mf.get_k = lambda *args, **kwargs: get_k(*args, **kwargs) * scale[0]
    eps = 1e-2
    c = .25 if d2.ndim == 2 else .5
    if 'b3lyp' in case:
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        c *= hyb
    dk = get_k(mol, ddm)
    first = -c * eps * hf._trace(d2, dk)
    second = -c * eps * hf._trace(ddm, dk)
    # the two are far enough apart for the test to tell the forms
    assert abs(first) > 20 * abs(second)
    e_one = mf.energy_elec(d2, h1e, mf.get_veff(mol, d2))[0]
    for on, expected in ((True, second), (False, first)):
        mf.second_order_energy = on
        v1 = mf.get_veff(mol, d1)
        scale[0] = 1. + eps
        v2 = mf.get_veff(mol, d2, d1, v1)
        scale[0] = 1.
        assert abs(mf.energy_elec(d2, h1e, v2)[0] - e_one - expected) < abs(second) * .05 + 1e-10
