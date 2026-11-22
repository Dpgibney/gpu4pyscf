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
from pyscf import scf as cpu_scf
from pyscf import sgx as cpu_sgx
from gpu4pyscf import scf, dft
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.sgx import sgx, sgx_jk, kernels

atom = 'O 0 0 0.12; C 0.05 0.1 -1.08; H 0.3 0.94 -1.62; H -0.2 -0.94 -1.66'

def _points(mol, n, seed=7):
    rng = np.random.default_rng(seed)
    xyz = mol.atom_coords()
    return np.vstack([
        xyz,  # at the nuclei
        xyz[rng.integers(0, mol.natm, n//4)] + rng.normal(size=(n//4, 3)) * .05,
        xyz[rng.integers(0, mol.natm, n//2)] + rng.normal(size=(n//2, 3)) * 1.5,
        rng.normal(size=(n, 3)) * 12.,
    ])[:n]

def _operator_ints(mol, coords, omega, c_full, c_lr):
    a = c_full * mol.intor('int1e_grids', grids=coords)
    if c_lr:
        with mol.with_range_coulomb(omega):
            a += c_lr * mol.intor('int1e_grids', grids=coords)
    return a

@pytest.mark.parametrize('basis,ngrids', [('def2-tzvp', 300), ('def2-qzvp', 160)])
@pytest.mark.parametrize('c_full,c_lr', [(1., 0.), (0., 1.), (.15, .85)])
def test_kernels(basis, ngrids, c_full, c_lr):
    # G[mu,g] = sum_nu (mu nu|g) F[nu,g] through f (g) functions, for 1/r,
    # erf(omega r)/r and their mix, against the integrals of libcint
    mol = pyscf.M(atom=atom, basis=basis, verbose=0)
    omega = .3
    coords = _points(mol, ngrids)
    rng = np.random.default_rng(3)
    f = rng.normal(size=(mol.nao, ngrids))
    ref = np.einsum('gmn,ng->mg', _operator_ints(mol, coords, omega, c_full, c_lr), f)
    smol = SortedGTO.from_mol(mol, decontract=True, diffuse_cutoff=0.3)
    classes = sgx_jk.pair_data(smol, omega, c_full, c_lr)
    for dtype, tol in (('double', 1e-11), ('float', 2e-5)):
        dev = sgx_jk._DevicePairs(classes, omega, c_full, c_lr, dtype)
        fc = cp.ascontiguousarray(smol.apply_C_dot(cp.asarray(f)), dtype=dev.real)
        gc = sgx_jk.contract(dev, fc, cp.asarray(coords.T))
        g = smol.apply_CT_dot(cp.asarray(gc, dtype=np.float64)).get()
        assert abs(g - ref).max() < tol * abs(ref).max()

def test_boys_table():
    from scipy.special import gammainc, gamma
    t = np.linspace(.05, 60, 400)
    f = kernels.boys_exact(12, t)
    for m in range(13):
        ref = gammainc(m + .5, t) * gamma(m + .5) / (2 * t**(m + .5))
        assert abs(f[m] / ref - 1).max() < 1e-12

@pytest.fixture(scope='module')
def model():
    mol = pyscf.M(atom=atom, basis='def2-tzvp', verbose=0)
    mf = dft.RKS(mol, xc='pbe0')
    mf.conv_tol = 1e-10
    mf.kernel()
    return mol, mf, cp.asarray(mf.make_rdm1())

@pytest.mark.parametrize('omega,lr,sr', [(None, None, None), (.3, 1., .15)])
def test_k_dense_reference(model, omega, lr, sr):
    # the driver against its formula with the integrals of libcint
    mol, mf, dm = model
    obj = sgx.SGX(mol)
    obj.build()
    coords = obj._coords.T.get()
    weights = obj._weights.get()
    proj = obj._proj.get()
    ao = mol.eval_gto('GTOval', coords)
    c_full, c_lr = (1., 0.) if omega is None else (sr, lr - sr)
    a = _operator_ints(mol, coords, omega, c_full, c_lr)
    fg = ao.dot(proj.dot(dm.get())) * weights[:,None]
    ref = proj.T.dot(ao.T.dot(np.einsum('gmn,gn->gm', a, fg)))
    ref = (ref + ref.T) * .5
    vk = obj.get_k(dm, 1, omega, lr, sr).get()
    assert abs(vk - ref).max() < 1e-11
    # single precision
    obj.kernel_dtype = 'float'
    assert abs(obj.get_k(dm, 1, omega, lr, sr).get() - ref).max() < 2e-5

def test_k_pyscf_sgx(model):
    # the grid, its overlap fit and K are those of PySCF's sgx
    mol, mf, dm = model
    cmf = cpu_sgx.sgx_fit(cpu_scf.RHF(mol), pjs=True)
    cmf.with_df.grids_level_i = cmf.with_df.grids_level_f = 1
    cmf.with_df.use_opt_grids = True
    cmf.with_df.build()
    ref = cmf.with_df.get_jk(dm.get(), with_j=False)[1]
    obj = sgx.SGX(mol)
    assert abs(obj.get_k(dm).get() - ref).max() < 1e-9

def test_k_against_exact(model):
    mol, mf, dm = model
    vk = mf.get_k(mol, dm)
    ek = -.25 * float(cp.einsum('ij,ji', dm, vk))
    errs = []
    for level in (0, 1, 2):
        obj = sgx.SGX(mol)
        obj.grids_level = level
        errs.append(abs(-.25 * float(cp.einsum('ij,ji', dm, obj.get_k(dm))) - ek))
    assert errs[0] < 2e-4 and errs[1] < 5e-5 and errs[2] < 1e-5

def test_k_batches_and_lists(model):
    # the grid in batches (they follow the memory that is free), and the
    # pairs of a block from a list
    mol, mf, dm = model
    obj = sgx.SGX(mol)
    vk = obj.get_k(dm)
    n = obj.size // obj.blksize // 3 * obj.blksize
    obj._batches = lambda: [(0, n), (n, 2*n), (2*n, obj.size)]
    assert abs(obj.get_k(dm) - vk).max() < 1e-11
    smol = obj._sorted_mol
    dev = obj._device_pairs(0., 1., 0.)
    rng = np.random.default_rng(2)
    ngrids = 3 * obj.blksize - 5
    fc = cp.asarray(rng.normal(size=(int(smol.p_ao_loc[-1]), ngrids)))
    coords = obj._coords[:,:ngrids]
    ref = sgx_jk.contract(dev, fc, coords, blksize=obj.blksize)
    # every pair on the first and the last block, none on the second
    lists = []
    for c in dev.classes:
        npairs = c[2]
        loc = cp.asarray([0, npairs, npairs, 2*npairs], dtype=np.int32)
        lists.append((loc, cp.tile(cp.arange(npairs, dtype=np.int32), 2)))
    g = sgx_jk.contract(dev, fc, coords, lists=lists, blksize=obj.blksize)
    b = obj.blksize
    assert abs(g[:,:b] - ref[:,:b]).max() < 1e-12
    assert abs(g[:,2*b:] - ref[:,2*b:]).max() < 1e-12
    assert abs(g[:,b:2*b]).max() == 0

def test_k_not_hermitian(model):
    # a density matrix that is not symmetric: K[mu,nu] = (mu lambda|sigma nu) D[lambda,sigma],
    # the grid on the electron of mu and lambda
    mol, mf, dm = model
    rng = np.random.default_rng(5)
    d = rng.normal(size=dm.shape)
    obj = sgx.SGX(mol)
    obj.build()
    coords = obj._coords.T.get()
    weights = obj._weights.get()
    proj = obj._proj.get()
    ao = mol.eval_gto('GTOval', coords)
    a = mol.intor('int1e_grids', grids=coords)
    fg = ao.dot(proj.dot(d)) * weights[:,None]
    ref = proj.T.dot(ao.T.dot(np.einsum('gmn,gn->gm', a, fg)))
    vk = obj.get_k(cp.asarray(d), hermi=0).get()
    assert abs(vk - ref).max() < 1e-10
    # and against the exact one
    exact = mf.get_k(mol, cp.asarray(d), hermi=0).get()
    assert abs(vk - exact).max() < 2e-2 * abs(exact).max()

@pytest.mark.parametrize('xc', ['b3lyp', 'wb97x', 'hf'])
def test_cosx_scf(xc):
    mol = pyscf.M(atom=atom, basis='def2-svp', verbose=0)
    if xc == 'hf':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=xc)
    e0 = mf.kernel()
    mf = mf.COSX()
    e1 = mf.kernel()
    assert mf.converged
    assert 1e-9 < abs(e1 - e0) < 1e-4
    # another grid: built at the next K
    mf.with_sgx.grids_level = 3
    assert abs(mf.kernel() - e0) < 3e-6

def test_cosx_uks():
    mol = pyscf.M(atom=atom, basis='def2-svp', charge=1, spin=1, verbose=0)
    mf = dft.UKS(mol, xc='pbe0')
    e0 = mf.kernel()
    mf = mf.COSX()
    e1 = mf.kernel()
    assert mf.converged
    assert 1e-9 < abs(e1 - e0) < 1e-4

def test_cosx_solvent_and_dfj():
    mol = pyscf.M(atom=atom, basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='b3lyp').density_fit(only_dfj=True).PCM()
    mf.with_solvent.eps = 78.3553
    e0 = mf.kernel()
    mf = mf.COSX()
    e1 = mf.kernel()
    assert mf.converged
    assert 1e-9 < abs(e1 - e0) < 1e-4
