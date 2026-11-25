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

import unittest
import numpy
from pyscf import gto, scf, lib, dft
from pyscf import grad, hessian
from pyscf.hessian import uks as uks_cpu
from gpu4pyscf.hessian import uks as uks_gpu
try:
    from gpu4pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 6
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'sto3g'
    mol.spin = 1
    mol.charge = 1
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _check_vxc(method, xc='LDA'):
    mf = dft.uks.UKS(mol, xc=xc)
    mf.conv_tol = 1e-14
    mf.grids.level = 1
    mf.kernel()
    dm = mf.make_rdm1()
    hobj = mf.Hessian()
    fn = getattr(uks_cpu, method)
    vxc_cpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    mf = mf.to_gpu()
    mf.kernel()
    hobj = mf.Hessian()
    fn = getattr(uks_gpu, method)
    vxc_gpu = fn(hobj, mf.mo_coeff, mf.mo_occ, 12000)

    # outputs of _get_vxc_deriv2 and _get_vxc_deriv1 in GPU4PYSCF are different from PySCF for memory efficiency
    if method == '_get_vxc_deriv2':
        vxc0 = numpy.einsum('nxyij,ij->nxyi', vxc_cpu[0], dm[0])
        vxc1 = numpy.einsum('nxyij,ij->nxyi', vxc_cpu[1], dm[1])
        vxc_cpu = (vxc0, vxc1)

    if method == '_get_vxc_deriv1':
        mo_occ = mf.mo_occ.get()
        mo_coeff = mf.mo_coeff.get()
        mocca = mo_coeff[0][:,mo_occ[0]>0]
        moccb = mo_coeff[1][:,mo_occ[1]>0]
        vxc0 = numpy.einsum('nxij,jq->nxiq', vxc_cpu[0], mocca)
        vxc0 = numpy.einsum('nxiq,ip->nxpq', vxc0, mo_coeff[0])
        vxc1 = numpy.einsum('nxij,jq->nxiq', vxc_cpu[1], moccb)
        vxc1 = numpy.einsum('nxiq,ip->nxpq', vxc1, mo_coeff[1])
        vxc_cpu = (vxc0, vxc1)

    #print(f'testing {method} for {xc}')
    #print(numpy.linalg.norm(vxc_cpu[0] - vxc_gpu[0].get()))
    #print(numpy.linalg.norm(vxc_cpu[1] - vxc_gpu[1].get()))
    assert numpy.linalg.norm(vxc_cpu[0] - vxc_gpu[0].get()) < 1e-6
    assert numpy.linalg.norm(vxc_cpu[1] - vxc_gpu[1].get()) < 1e-6

def _vs_cpu(mf, tol=1e-7):
    mf.conv_tol_cpscf = 1e-8
    ref = mf.Hessian().kernel()
    hessobj = mf.Hessian().to_gpu()
    hessobj.base.cphf_grids = hessobj.base.grids
    e2_gpu = hessobj.kernel()
    assert abs(ref - e2_gpu).max() < tol

class KnownValues(unittest.TestCase):
    def test_vxc_diag(self):
        _check_vxc('_get_vxc_diag', xc='LDA')
        _check_vxc('_get_vxc_diag', xc='PBE')
        _check_vxc('_get_vxc_diag', xc='TPSS')

    def test_vxc_deriv1(self):
        _check_vxc('_get_vxc_deriv1', xc='LDA')
        _check_vxc('_get_vxc_deriv1', xc='PBE')
        _check_vxc('_get_vxc_deriv1', xc='TPSS')

    def test_vxc_deriv2(self):
        _check_vxc('_get_vxc_deriv2', xc='LDA')
        _check_vxc('_get_vxc_deriv2', xc='PBE')
        _check_vxc('_get_vxc_deriv2', xc='TPSS')

    def test_hessian_lda(self, disp=None):
        print('-----testing LDA Hessian----')
        mf = mol.UKS(xc='LDA').run()
        _vs_cpu(mf)

    def test_hessian_gga(self):
        print('-----testing PBE Hessian----')
        mf = mol.UKS(xc='PBE').run()
        _vs_cpu(mf, tol=1e-6)

    def test_hessian_hybrid(self):
        print('-----testing B3LYP Hessian----')
        mf = mol.UKS(xc='b3lyp').run()
        _vs_cpu(mf, tol=2e-6)

    def test_hessian_mgga(self):
        print('-----testing M06 Hessian----')
        mf = mol.UKS(xc='m06').run()
        _vs_cpu(mf)

    def test_hessian_rsh(self):
        print('-----testing wb97 Hessian----')
        mf = mol.UKS(xc='wb97').run()
        _vs_cpu(mf)
    
    def test_hessian_hf(self):
        print('-----testing hf Hessian----')
        mf = mol.UKS(xc='hf').run()
        _vs_cpu(mf)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_hessian_pbe_D3(self):
        print('-----testing PBE Hessian with D3BJ----')
        mf = mol.UKS(xc='PBE')
        mf.conv_tol = 1e-14
        mf.disp = 'd3bj'
        mf.run()
        _vs_cpu(mf, tol=1e-6)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_hessian_pbe_D4(self):
        print('-----testing PBE Hessian with D4----')
        mf = mol.UKS(xc='PBE')
        mf.conv_tol = 1e-14
        mf.disp = 'd4'
        mf.run()
        _vs_cpu(mf, tol=1e-6)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_consistency_b3lyp_d3_hess(self):
        print('----- testing B3LYP-D3 consistency ------')
        mf1 = mol.UKS()
        mf1.conv_tol = 1e-14
        mf1.xc = 'b3lyp'
        mf1.disp = 'd3bj'
        mf1.kernel()
        hess1 = mf1.Hessian().kernel()

        mf2 = mol.UKS()
        mf2.conv_tol = 1e-14
        mf2.xc = 'b3lyp-d3bj'
        mf2.kernel()
        hess2 = mf2.Hessian().kernel()
        assert abs(hess1 - hess2).max() < 1e-8

    def test_nr_uks_fxc_mo(self):
        # the XC response of the CPHF in the MO basis from the first-order orbitals of both spins, against
        # the AO matrices of NumInt.nr_uks_fxc: LDA, GGA, meta-GGA for a doublet (one occupied orbital less
        # for beta), with the grid blocks in the MO form, in the AO form (see hessian.rks._fxc_mo_add_block)
        # and as the count of GEMMs decides
        import cupy
        from gpu4pyscf.hessian import uhf as uhf_gpu
        from gpu4pyscf.hessian import rks as rks_gpu
        from gpu4pyscf.hessian.rhf import _ao2mo
        from gpu4pyscf.lib.cupy_helper import contract, tag_array
        mol1 = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0.1 0.757 0.587', basis='def2-svp',
                     charge=1, spin=1, verbose=0)
        nset = 5
        for xc in ('lda,vwn', 'pbe', 'r2scan'):
            mf = dft.uks.UKS(mol1, xc=xc).to_gpu()
            mf.grids.level = 1
            mf.cphf_grids = mf.grids
            mf.kernel()
            hobj = mf.Hessian()
            mo_coeff, mo_occ = cupy.asarray(mf.mo_coeff), cupy.asarray(mf.mo_occ)
            nmo = mo_coeff.shape[-1]
            nocc = [int((mo_occ[s] > 0).sum()) for s in range(2)]
            self.assertEqual(nocc, [5, 4])
            rng = numpy.random.default_rng(3)
            mo1 = cupy.asarray(rng.random((nset, nmo * sum(nocc))) - .5) * .1

            # the densities as hessian.uhf.gen_vind tags them
            orbo = uhf_gpu._stack_uhf_occ_oribtals(mo_coeff[0][:,:nocc[0]], mo_coeff[1][:,:nocc[1]])
            x = uhf_gpu._stack_uhf_occ_oribtals(
                mo1[:,:nmo*nocc[0]].reshape(nset, nmo, nocc[0]),
                mo1[:,nmo*nocc[0]:].reshape(nset, nmo, nocc[1]))
            mo1_ao = contract('snai,spa->snpi', x, mo_coeff)
            dm1 = contract('snpi,sqi->snpq', mo1_ao, orbo)
            dm1 = dm1 + dm1.transpose(0, 1, 3, 2)
            dm1 = tag_array(dm1, mo1=mo1_ao, occ_coeff=orbo, symmetrize=1)
            ni = mf._numint
            rho0, vxc, fxc = ni.cache_xc_kernel(mol1, mf.grids, mf.xc, mo_coeff, mo_occ, 1)
            v1 = ni.nr_uks_fxc(mol1, mf.grids, mf.xc, None, dm1, 0, 1, rho0, vxc, fxc)
            ref = [_ao2mo(v1[s], mo_coeff[s][:,:nocc[s]], mo_coeff[s]) for s in range(2)]
            scale = float(max(abs(ref[0]).max(), abs(ref[1]).max()))
            self.assertTrue(scale > 1e-3)
            for gemms in (None, (0, 1), (1, 0)):   # as counted; every block in the MO form; in the AO form
                table = rks_gpu._FXC_MO_GEMMS if gemms is None else dict.fromkeys(rks_gpu._FXC_MO_GEMMS, gemms)
                with lib.temporary_env(rks_gpu, _FXC_MO_GEMMS=table):
                    va, vb = uks_gpu.nr_uks_fxc_mo(ni, mol1, mf.grids, mf.xc, None, dm1, mo_coeff, mo_occ,
                                                   0, 1, rho0, vxc, fxc)
                self.assertEqual(va.shape, (nset, nmo, nocc[0]))
                self.assertEqual(vb.shape, (nset, nmo, nocc[1]))
                self.assertLess(float(abs(va - ref[0]).max()), 1e-11 * scale)
                self.assertLess(float(abs(vb - ref[1]).max()), 1e-11 * scale)

            # the same through the response function of the CPHF equations
            res = []
            for mo_form in (True, False):
                with lib.temporary_env(uks_gpu, NR_UKS_FXC_MO=mo_form):
                    res.append(uhf_gpu.gen_vind(hobj, mo_coeff, mo_occ)(mo1))
            self.assertLess(float(abs(res[0] - res[1]).max()), 1e-11 * float(abs(res[1]).max()))

if __name__ == "__main__":
    print("Full Tests for UKS Hessian")
    unittest.main()
