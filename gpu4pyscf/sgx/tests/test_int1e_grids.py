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
from pyscf import df, gto
from gpu4pyscf import dft
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.sgx import int1e_grids

atom = 'O 0 0 0.12; C 0.05 0.1 -1.08; H 0.3 0.94 -1.62; H -0.2 -0.94 -1.66'

def _case(basis, ngrids, seed=11):
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    rng = np.random.default_rng(seed)
    xyz = mol.atom_coords()
    coords = np.vstack([
        xyz,  # at the nuclei
        xyz[rng.integers(0, mol.natm, ngrids//4)] + rng.normal(size=(ngrids//4, 3)) * .05,
        xyz[rng.integers(0, mol.natm, ngrids//2)] + rng.normal(size=(ngrids//2, 3)) * 1.5,
        rng.normal(size=(ngrids, 3)) * 12.,
    ])[:ngrids]
    dm = rng.normal(size=(mol.nao, mol.nao))
    dm = dm + dm.T
    charges = rng.normal(size=ngrids)
    expnt = np.exp(rng.uniform(np.log(.3), np.log(200.), ngrids))
    return mol, coords, dm, charges, expnt

@pytest.mark.parametrize('basis,ngrids', [('def2-tzvp', 300), ('def2-qzvp', 150)])
@pytest.mark.parametrize('gauss', [False, True])
def test_contractions(basis, ngrids, gauss):
    # against libcint: point charges, and normalized Gaussian charges of an
    # exponent per point (one s function each)
    mol, coords, dm, charges, expnt = _case(basis, ngrids)
    if gauss:
        a = df.incore.aux_e2(mol, gto.fakemol_for_charges(coords, expnt=expnt), intor='int3c2e')
        ex = cp.asarray(expnt)
    else:
        a = mol.intor('int1e_grids', grids=coords).transpose(1, 2, 0)
        ex = None
    v_ref = np.einsum('mng,mn->g', a, dm)
    m_ref = a.dot(charges)
    opt = int1e_grids.Int1eGridsOpt(mol)
    for dtype, tol in (('double', 1e-13), ('float', 1e-4)):
        v = opt.density_contracted(cp.asarray(dm), coords, ex, dtype).get()
        m = opt.charge_contracted(cp.asarray(charges), coords, ex, dtype).get()
        assert abs(v - v_ref).max() < tol * abs(v_ref).max()
        assert abs(m - m_ref).max() < tol * abs(m_ref).max()
        assert abs(m - m.T).max() == 0

def test_against_int3c1e():
    # what they stand in for, with several densities and sets of charges
    mol, coords, dm, charges, expnt = _case('def2-tzvp', 200)
    rng = np.random.default_rng(2)
    dms = cp.asarray(np.stack([dm, dm * .3 + np.eye(mol.nao)]))
    qs = cp.asarray(np.stack([charges, rng.normal(size=len(charges))]))
    ex = cp.asarray(expnt)
    old = int3c1e.VHFOpt(mol)
    old.build(1e-14)
    v_old = int3c1e.int1e_grids(mol, cp.asarray(coords), dm=dms, charge_exponents=ex, intopt=old)
    m_old = int3c1e.int1e_grids(mol, cp.asarray(coords), charges=qs, charge_exponents=ex, intopt=old)
    opt = int1e_grids.Int1eGridsOpt(mol)
    v = opt.density_contracted(dms, coords, ex)
    m = opt.charge_contracted(qs, coords, ex)
    assert v.shape == v_old.shape and m.shape == m_old.shape
    assert abs(v - v_old).max() < 1e-10 * abs(v_old).max()
    assert abs(m - m_old).max() < 1e-10 * abs(m_old).max()
    # the functions of the module, and a density that is not symmetric
    d = cp.asarray(rng.normal(size=dm.shape))
    v1 = int1e_grids.density_contracted(mol, coords, d, ex)
    v2 = opt.density_contracted((d + d.T) * .5, coords, ex)
    assert abs(v1 - v2).max() < 1e-11
    m1 = int1e_grids.charge_contracted(mol, coords, qs[0], ex)
    assert abs(m1 - m[0]).max() < 1e-12

@pytest.mark.parametrize('method', ['C-PCM', 'IEF-PCM', 'SMD'])
def test_reaction_field(method):
    # the solvent's potential through these contractions and through int3c1e
    mol = pyscf.M(atom=atom, basis='def2-svp', verbose=0)
    e = []
    for new in (True, False):
        mf = dft.RKS(mol, xc='b3lyp')
        if method == 'SMD':
            mf = mf.SMD()
            mf.with_solvent.solvent = 'water'
        else:
            mf = mf.PCM()
            mf.with_solvent.method = method
            mf.with_solvent.eps = 78.3553
        if not new:
            mf.with_solvent._int1e_opt = False
        mf.conv_tol = 1e-11
        e.append(mf.kernel())
        assert mf.converged
        assert (mf.with_solvent._int1e_opt is not False) == new
    assert abs(e[0] - e[1]) < 1e-10
