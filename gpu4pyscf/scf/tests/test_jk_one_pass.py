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
import cupy as cp
import pyscf
from gpu4pyscf import scf, dft

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
    'rohf': (1, 1, lambda mol: scf.ROHF(mol)),
    'rks_b3lyp': (0, 0, lambda mol: dft.RKS(mol, xc='b3lyp')),
    'uks_pbe0': (1, 1, lambda mol: dft.UKS(mol, xc='pbe0')),
    'roks_b3lyp': (1, 1, lambda mol: dft.ROKS(mol, xc='b3lyp')),
    # range-separated and pure functionals keep their J/K builders
    'rks_wb97x': (0, 0, lambda mol: dft.RKS(mol, xc='wb97x')),
    'uks_pbe': (1, 1, lambda mol: dft.UKS(mol, xc='pbe')),
}

@pytest.mark.parametrize('case', CASES)
def test_jk_one_pass_energy(case):
    charge, spin, method = CASES[case]
    mol = _mol(charge, spin)
    e = []
    for one_pass in (False, True):
        mf = method(mol)
        mf.conv_tol = 1e-11
        mf.jk_one_pass = one_pass
        e.append(mf.kernel())
        assert mf.converged
    assert abs(e[1] - e[0]) < 1e-9

def test_jk_one_pass_k_scaling():
    # get_veff of global hybrids scales the one-pass K by hyb; get_k applies the same factor in its kernel
    mol = _mol(0, 0)
    mf = dft.RKS(mol, xc='b3lyp').run()
    dm = cp.asarray(mf.make_rdm1())
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    assert omega == 0
    vj, vk = mf.get_jk(mol, dm, 1)
    assert abs(vk * hyb - mf.get_k(mol, dm, 1, omega, alpha, hyb)).max() < 1e-12
    assert abs(vj - mf.get_j(mol, dm, 1)).max() < 1e-10
