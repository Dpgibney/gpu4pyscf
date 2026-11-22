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
    # the two-level solution is the model's own
    charge, spin, method = CASES[case]
    mol = _mol(charge, spin)
    e = []
    for on in (False, True):
        mf = method(mol)
        mf.conv_tol = 1e-10
        mf.two_level = on
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
