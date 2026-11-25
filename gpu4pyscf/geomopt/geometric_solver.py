#!/usr/bin/env python
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

'''
geomeTRIC through PySCF's interface (pyscf.geomopt.geometric_solver), with
a start Hessian given as an array.

PySCF's solver takes hessian=True (the Hessian of method.Hessian().kernel()
with its defaults) or geomeTRIC's own options; a Hessian of the starting
geometry made by the caller, with other options (grid_response) or at
another level, could not be given: PySCF 2.14 computes the method's own
Hessian into the file of hessian='file:path'.

    >>> h0 = mf.Hessian().set(grid_response=True).kernel()
    >>> conv, mol_ts = geometric_solver.kernel(mf, transition=True, hessian=h0)
'''

import os
import tempfile
import uuid
import numpy
import cupy
from pyscf import lib
from pyscf.geomopt import geometric_solver as _pyscf_solver

ASSERT_CONV = _pyscf_solver.ASSERT_CONV
INCLUDE_GHOST = _pyscf_solver.INCLUDE_GHOST


def _hessian_to_file(hessian, natm, tmpdir):
    '''The Hessian (Eh/Bohr^2), [natm,natm,3,3] as the Hessian objects return
    it or [3*natm,3*natm], on the host or the device, written as geomeTRIC
    reads it: the option 'file:path' '''
    h = cupy.asnumpy(hessian) if isinstance(hessian, cupy.ndarray) else numpy.asarray(hessian)
    if h.shape == (natm, natm, 3, 3):
        h = h.transpose(0, 2, 1, 3)
    elif h.shape != (3*natm, 3*natm):
        raise ValueError(f'A Hessian of shape {h.shape} for {natm} atoms')
    hessian_file = os.path.join(tmpdir, str(uuid.uuid4()))
    numpy.savetxt(hessian_file, h.reshape(3*natm, 3*natm))
    return f'file:{hessian_file}'


def kernel(method, assert_convergence=ASSERT_CONV, include_ghost=INCLUDE_GHOST,
           constraints=None, callback=None, maxsteps=100, **kwargs):
    '''pyscf.geomopt.geometric_solver.kernel, with hessian also an array:
    the Hessian of the starting geometry, [natm,natm,3,3] or
    [3*natm,3*natm] in Eh/Bohr^2 (numpy or cupy). Returns (converged, mol).
    '''
    hessian = kwargs.get('hessian')
    if not isinstance(hessian, (numpy.ndarray, cupy.ndarray)):
        return _pyscf_solver.kernel(method, assert_convergence, include_ghost,
                                    constraints, callback, maxsteps, **kwargs)
    natm = method.mol.natm
    with tempfile.TemporaryDirectory(dir=lib.param.TMPDIR) as tmpdir:
        kwargs['hessian'] = _hessian_to_file(hessian, natm, tmpdir)
        # The file is the Hessian: PySCF's _make_hessian would compute the
        # method's own into it
        with lib.temporary_env(_pyscf_solver, _make_hessian=lambda g_scanner, option, tmp: option):
            return _pyscf_solver.kernel(method, assert_convergence, include_ghost,
                                        constraints, callback, maxsteps, **kwargs)


def optimize(method, assert_convergence=ASSERT_CONV, include_ghost=INCLUDE_GHOST,
             constraints=None, callback=None, maxsteps=100, **kwargs):
    '''kernel, returning the optimized molecule only (as PySCF's optimize)'''
    return kernel(method, assert_convergence, include_ghost, constraints,
                  callback, maxsteps, **kwargs)[1]
