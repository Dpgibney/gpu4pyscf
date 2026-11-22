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

'''What the nuclear gradient takes over from the last SCF cycle, which
evaluated the converged density: the surface charges of the solvent and the
VV10 potential.'''

import pytest
import pyscf
from gpu4pyscf import dft

atom = '''
O   0.000   -0.    0.1174
H  -0.757    0.   -0.4696
H   0.757    0.   -0.4696
C   1.4     1.6    0.2
H   2.3     1.2    0.6
H   0.9     2.4    0.7
'''

def _mol(charge=0, spin=0):
    return pyscf.M(atom=atom, basis='def2-svp', charge=charge, spin=spin, verbose=0)

def _solvated(mf, model):
    if model == 'SMD':
        mf = mf.SMD()
        mf.with_solvent.solvent = 'water'
    else:
        mf = mf.PCM()
        mf.with_solvent.method = model
        mf.with_solvent.eps = 78.3553
    return mf

@pytest.mark.parametrize('model', ['C-PCM', 'IEF-PCM', 'SS(V)PE', 'SMD'])
def test_solvent_charges_of_the_scf(model):
    mf = _solvated(dft.RKS(_mol(), xc='b3lyp'), model)
    mf.conv_tol = 1e-10
    mf.kernel()
    pcmobj = mf.with_solvent
    calls = []
    get_vind = pcmobj._get_vind
    def counted(dms):
        calls.append(1)
        return get_vind(dms)
    pcmobj._get_vind = counted
    g1 = mf.Gradients().kernel()
    assert not calls
    # without the kept density the charges are evaluated again
    pcmobj._intermediates.pop('dm')
    g2 = mf.Gradients().kernel()
    assert calls
    assert abs(g1 - g2).max() < 1e-9

def test_solvent_charges_of_the_scf_uks():
    mf = _solvated(dft.UKS(_mol(1, 1), xc='b3lyp'), 'SMD')
    mf.conv_tol = 1e-10
    mf.kernel()
    pcmobj = mf.with_solvent
    calls = []
    get_vind = pcmobj._get_vind
    def counted(dms):
        calls.append(1)
        return get_vind(dms)
    pcmobj._get_vind = counted
    mf.Gradients().kernel()
    assert not calls

def test_surface_matrices_on_demand():
    # K's factors are kept; D and S are built when their matrices are read
    for model in ('IEF-PCM', 'SMD'):
        mf = _solvated(dft.RKS(_mol(), xc='pbe'), model)
        mf.kernel()
        kept = mf.with_solvent._intermediates
        assert 'K_LU' in kept and 'D' not in kept and 'S' not in kept
        n = kept['K_LU'].shape[0]
        assert kept['D'].shape == (n, n) and kept['S'].shape == (n, n)
        assert 'D' in kept and 'S' in kept

@pytest.mark.parametrize('method', [dft.RKS, dft.UKS])
def test_vv10_potential_of_the_scf(method):
    mf = method(_mol(), xc='wb97m-v')
    mf.nlcgrids.level = 1
    mf.conv_tol = 1e-10
    mf.kernel()
    ni = mf._numint
    last = ni._vv10nlc_last
    assert last is not None
    g1 = mf.Gradients().kernel()
    # the pair sums were not evaluated again
    assert ni._vv10nlc_last is last
    ni._vv10nlc_last = None
    g2 = mf.Gradients().kernel()
    assert ni._vv10nlc_last is not None
    assert abs(g1 - g2).max() < 1e-11
