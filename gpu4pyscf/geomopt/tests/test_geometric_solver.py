# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
from unittest import mock
import numpy
import cupy
import pyscf
from pyscf.hessian import thermo
from gpu4pyscf import scf
from gpu4pyscf.hessian import rhf as rhf_hess
try:
    import geometric  # noqa: F401
    from gpu4pyscf.geomopt import geometric_solver
except ImportError:
    geometric_solver = None


@unittest.skipIf(geometric_solver is None, 'requires geomeTRIC')
class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # a guess of the saddle point of HCN <-> HNC
        cls.mol = pyscf.M(atom='C 0 0 0; N 1.19 0 0; H 0.486 1.042 0', basis='6-31g',
                          output='/dev/null')

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_transition_state_from_hessian_array(self):
        # the saddle-point search from a Hessian given as an array: [natm,natm,3,3] as the
        # Hessian objects return it, the same on the device, [3*natm,3*natm] on the host.
        # The solver computes none of its own
        mol = self.mol
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        h0 = mf.Hessian().kernel()
        natm = mol.natm
        self.assertEqual(h0.shape, (natm, natm, 3, 3))
        h0_host = cupy.asnumpy(h0).transpose(0, 2, 1, 3).reshape(3*natm, 3*natm)
        geoms = []
        for hessian in (h0, cupy.asarray(h0), h0_host):
            with mock.patch.object(rhf_hess.Hessian, 'kernel', side_effect=AssertionError):
                conv, mol_ts = geometric_solver.kernel(mf, transition=True, hessian=hessian)
            self.assertTrue(conv)
            geoms.append(mol_ts.atom_coords())
        self.assertAlmostEqual(abs(geoms[0] - geoms[1]).max(), 0, 6)
        self.assertAlmostEqual(abs(geoms[0] - geoms[2]).max(), 0, 6)
        mf_ts = scf.RHF(mol_ts).run(conv_tol=1e-12)
        self.assertAlmostEqual(float(abs(mf_ts.Gradients().kernel()).max()), 0, 3)
        freq = thermo.harmonic_analysis(mol_ts, cupy.asnumpy(mf_ts.Hessian().kernel()),
                                        imaginary_freq=False)['freq_wavenumber']
        self.assertEqual(int((freq < 0).sum()), 1)

    def test_other_hessian_options(self):
        # anything but an array goes to PySCF's solver as it is
        mf = scf.RHF(self.mol)
        with mock.patch.object(geometric_solver._pyscf_solver, 'kernel', return_value=(True, None)) as m:
            geometric_solver.kernel(mf, transition=True, hessian=True, maxsteps=7)
            geometric_solver.optimize(mf)
        self.assertIs(m.call_args_list[0].kwargs['hessian'], True)
        self.assertEqual(m.call_args_list[0].args[5], 7)
        self.assertNotIn('hessian', m.call_args_list[1].kwargs)
        self.assertRaises(ValueError, geometric_solver.kernel, mf, hessian=numpy.zeros((3, 3)))


if __name__ == '__main__':
    print('Full Tests for the geomeTRIC interface')
    unittest.main()
