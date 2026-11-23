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

import pyscf
import numpy as np
import unittest
import pytest
from gpu4pyscf.dft import RKS, UKS
from gpu4pyscf.scf import HF, UHF
from pyscf.hessian import thermo

def setUpModule():
    global mol_close, mol_open

    mol_close = pyscf.M(
        atom = '''
            O     -1.168500    0.182500    0.000000
            O      1.114600    0.210300    0.000000
            C      0.053800   -0.392700    0.000000
            H     -0.328661   -1.494191   -0.538879
            H     -1.582685    0.639818    1.199294
        ''',
        basis = '6-31g',
        charge = 0,
        spin = 0,
        output='/dev/null',
        verbose = 0,
    )

    mol_open = pyscf.M(
        atom = '''
            O     -1.168500    0.182500    0.000000
            O      1.114600    0.210300    0.000000
            C      0.053800   -0.392700    0.000000
            H     -0.328661   -1.494191   -0.538879
        ''',
        basis = '6-31g',
        charge = 0,
        spin = 1,
        output='/dev/null',
        verbose = 0,
    )

def tearDownModule():
    global mol_close, mol_open
    mol_close.stdout.close()
    mol_open.stdout.close()
    del mol_close, mol_open

class KnownValues(unittest.TestCase):
    # All reference results from the same calculation with mf.level_shift = 0,
    # the CPHF equations solved to a residual of 1e-9 (cphf_residual_tol). The
    # Krylov solver at conv_tol_cpscf, which the references were made with
    # before, is 1.7 (RKS) and 2.5 cm^-1 (UKS) from these, 0.001 (RHF) and
    # 0.06 cm^-1 (UHF)

    def test_level_shift_hessian_rks(self):
        mf = RKS(mol_close, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_close, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -189.52569283262818
        ref_frequency = np.array([-293.6647  ,  653.627964,  879.084602, 1150.084798,
            1208.695842, 1340.444446, 1349.276006, 1761.905008,
            1792.488249])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-7
        assert np.max(np.abs(test_frequency - ref_frequency)) < 5e-2

    def test_level_shift_hessian_uks(self):
        mf = UKS(mol_open, xc = 'wB97X')
        mf.grids.atom_grid = (99,590)
        mf.nlcgrids.atom_grid = (50,194)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0
        mf.max_cycle = 200

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_open, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -188.92925230031926
        ref_frequency = np.array([ 290.17636 ,  877.677068, 1081.500649, 1277.54542 ,
            1693.164015, 1759.986779])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-7
        assert np.max(np.abs(test_frequency - ref_frequency)) < 5e-2

    def test_level_shift_hessian_rhf(self):
        mf = HF(mol_close)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_close, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -188.53825152772055
        ref_frequency = np.array([-286.232322,  703.646173,  988.934769, 1234.535995,
            1246.903379, 1391.310072, 1443.064264, 1774.960853,
            1818.527194])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1e-2

    def test_level_shift_hessian_uhf(self):
        mf = UHF(mol_open)
        mf.conv_tol = 1e-12
        mf = mf.density_fit(auxbasis = "def2-universal-JKFIT")

        mf.level_shift = 1.0
        mf.max_cycle = 200

        test_energy = mf.kernel()
        assert mf.converged
        hobj = mf.Hessian()
        hobj.auxbasis_response = 2
        hessian = hobj.kernel()
        results = thermo.harmonic_analysis(mol_open, hessian, imaginary_freq = False)
        test_frequency = results['freq_wavenumber']

        ref_energy = -188.00032587095123
        ref_frequency = np.array([ 624.635893, 1030.204167, 1183.823855, 1362.457241,
            1693.987845, 1797.99353 ])
        assert np.max(np.abs(test_energy - ref_energy)) < 1e-10
        assert np.max(np.abs(test_frequency - ref_frequency)) < 1e-1

if __name__ == "__main__":
    print("Tests for HF and KS hessian with level shift")
    unittest.main()
