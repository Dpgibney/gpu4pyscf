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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by: Xiaojie Wu <wxj6000@gmail.com>

'''
Restricted coupled pertubed Hartree-Fock solver
'''


import numpy
import scipy.linalg
import cupy
from pyscf import lib
from gpu4pyscf.lib.cupy_helper import krylov
from gpu4pyscf.lib import logger

def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-7, hermi=False, verbose=logger.WARN,
          level_shift=0, mo10=None):
    '''
    Args:
        fvind : function
            Given density matrix, compute (ij|kl)D_{lk}*2 - (ij|kl)D_{jk}
    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
        mo10 : ndarray
            Initial guess for the first order orbitals.
    '''

    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose, 
                          level_shift=level_shift, mo10=mo10)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose, 
                            level_shift=level_shift, mo10=mo10)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN,
               level_shift=0, mo10=None):
    '''For field independent basis. First order overlap matrix is zero'''
    log = logger.new_logger(verbose=verbose)
    t0 = log.init_timer()

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a[:,None] + level_shift - e_i)
    mo1base = h1 * -e_ai
    nvir, nocc = e_ai.shape

    def vind_vo(mo1):
        v = fvind(mo1.reshape(-1,nvir,nocc)).reshape(-1,nvir,nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v *= e_ai
        return v.reshape(-1,nvir*nocc)
    
    x0 = mo10.reshape(-1,nvir*nocc) if mo10 is not None else None
    lindep = max(tol**2, numpy.finfo(mo1base.real.dtype).eps)
    mo1 = krylov(vind_vo, mo1base.reshape(-1,nvir*nocc), x0=x0,
                 tol=tol, max_cycle=max_cycle, lindep=lindep,
                 hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)
    return mo1.reshape(h1.shape), None

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                max_cycle=50, tol=1e-9, hermi=False,
                verbose=logger.WARN, level_shift=0, mo10=None):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]
    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
        mo10 : ndarray
            Initial guess for the first order orbitals.
    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = log.init_timer()

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = 1 / (e_a[:,None] + level_shift - e_i)
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir

    s1 = s1.reshape(-1,nmo,nocc)
    hs = mo1base = h1.reshape(-1,nmo,nocc) - s1*e_i

    mo1base = hs.copy()
    mo1base[:,viridx] *= -e_ai
    mo1base[:,occidx] = -s1[:,occidx] * .5

    def vind_vo(mo1):
        mo1 = mo1.reshape(-1,nmo, nocc)
        v = fvind(mo1).reshape(-1,nmo, nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v[:,viridx,:] *= e_ai
        v[:,occidx,:] = 0
        return v.reshape(-1,nmo*nocc)
    
    x0 = mo10.reshape(-1,nmo*nocc) if mo10 is not None else None
    lindep = max(tol**2, numpy.finfo(mo1base.real.dtype).eps)
    mo1 = krylov(vind_vo, mo1base.reshape(-1,nmo*nocc), x0=x0,
                 tol=tol, max_cycle=max_cycle, lindep=lindep,
                 hermi=hermi, verbose=log)
    mo1 = mo1.reshape(mo1base.shape)
    mo1[:,occidx] = mo1base[:,occidx]
    log.timer('krylov solver in CPHF', *t0)

    hs += fvind(mo1).reshape(mo1base.shape)
    mo1[:,viridx] = hs[:,viridx] / (e_i - e_a[:,None])

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo - s1*lib.direct_sum('i+j->ij',e_i,e_i)
    mo_e1 = hs[:,occidx,:]
    mo_e1 += mo1[:,occidx] * (e_i[:,None] - e_i)

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        return mo1.reshape(h1.shape), mo_e1.reshape(nocc,nocc)

def _orthonormal_rows(t, vs, drop):
    '''Orthonormal rows spanning the part of the rows of t orthogonal to the
    rows of the blocks vs, without the directions whose norm is below drop,
    and the size of each row in t (the diagonal of the pivoted QR, which runs
    on the host: a few vectors). t is overwritten.'''
    for _ in range(2):
        for v in vs:
            t -= t.dot(v.T).dot(v)
    q, r = scipy.linalg.qr(t.T.get(), mode='economic', pivoting=True)[:2]
    sizes = abs(r.diagonal())
    keep = sizes > drop
    q = cupy.asarray(numpy.ascontiguousarray(q[:,keep].T))
    # the rows are combinations of t's, by large coefficients for near-dependent t
    for v in vs:
        q -= q.dot(v.T).dot(v)
    return q, sizes[keep]

def _combine_blocks(c, blocks):
    '''sum_k c[k,r] vectors[k] of the vectors in the list of blocks [nk,...]'''
    out = 0
    p0 = 0
    for b in blocks:
        p1 = p0 + len(b)
        out = out + cupy.asarray(c[p0:p1].T).dot(b)
        p0 = p1
    return out

def solve_residual(fvind, mo_energy, mo_occ, h1, s1, max_cycle=50, tol=1e-4,
                   verbose=logger.WARN, level_shift=0, scale_trials=False,
                   x0=None, trials=None, second_order=None):
    '''The equations of solve_withs1 for real orbitals and perturbations, solved
    until the true residual of every perturbation is below tol.

    The virtual block of the solution is x0 + V c, x0 the uncoupled solution
    and V one space of trial vectors for all perturbations. With the operator
    M = (e_a - e_i) + fvind on the virtual block, which is symmetric, the
    coefficients solve V^T M V c = V^T r0 for r0 the residual of x0, in the
    plain dot product, and the virtual block then gets the last update of
    solve_withs1, from the response of the solution (a combination of the
    responses already computed: no response is built after the solve). The
    error of a Hessian built from this solution goes about as the square of
    the residuals, so a loose tol is enough for it; quantities linear in the
    solution need a tighter one.

    A cycle adds the preconditioned residuals of the perturbations that are
    not converged, made orthonormal against V and among themselves. Unlike
    krylov, which stops on the sizes of Krylov vectors that shrink by orders
    of magnitude, the stop and the choice of vectors look at the residuals.

    This is cphf.solve_residual of the CPU tree this work comes from.

    Kwargs:
        tol : float
            Upper limit of the norm of the preconditioned residual,
            r / (e_a - e_i), of each perturbation
        max_cycle : int
            Max. number of cycles of trial vectors
        level_shift : float
            Added to (e_a - e_i) in the preconditioner
        scale_trials : bool
            Give fvind each trial vector at its size in the residuals (not
            below 1e-6), and divide the response by that size. fvind is
            linear, so only what the density screening of its integrals can
            skip changes: the later cycles cost less. Only for a fvind whose
            absolute thresholds are far below 1e-6, such as direct SCF's
        x0 : array [nset,nvir,nocc]
            The virtual block the solution starts from, in place of the
            uncoupled solution (the solution of a cheaper model, for example)
        trials : function
            For the residuals r [n,nvir*nocc] of the perturbations that are
            not converged, the vectors [m,nvir*nocc] whose span is added to
            V, in place of r / (e_a - e_i): approximate solutions d of
            ((e_a - e_i) + fvind) d = r from a cheaper response. fvind is
            then evaluated for the start and for a few such sets only, and
            the solution is still fvind's to tol.
        second_order : list
            Gets (d, r), [nset,nvir*nocc] each: the virtual block of the
            solution before its last update minus the uncoupled solution,
            and the residual. The bilinear form of two perturbations A and B
            that a Hessian takes from the returned orbitals differs from its
            value at second order in the residuals by the dot products
            d_A r_B: they vanish where d is in the span of V (x0 = None and
            A, B solved together), and are the first-order error otherwise
            (hessian.rhf.solve_mo1 collects them).

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    assert not (cupy.iscomplexobj(h1) or cupy.iscomplexobj(s1))
    log = logger.new_logger(verbose=verbose)
    t0 = log.init_timer()

    mo_energy = cupy.asarray(mo_energy)
    mo_occ = cupy.asarray(mo_occ)
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_i = mo_energy[occidx]
    e_a = mo_energy[viridx]
    e_ai = e_a[:,None] - e_i
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir
    ndim = nvir * nocc
    diag = e_ai.ravel()
    precond = diag + level_shift

    s1 = cupy.asarray(s1).reshape(-1,nmo,nocc)
    hs = cupy.asarray(h1).reshape(-1,nmo,nocc) - s1*e_i
    nset = hs.shape[0]

    mo1 = cupy.empty_like(hs)
    if x0 is None:
        mo1[:,viridx] = -hs[:,viridx] / e_ai
    else:
        mo1[:,viridx] = cupy.asarray(x0).reshape(nset,nvir,nocc)
    mo1[:,occidx] = -s1[:,occidx] * .5
    v1 = fvind(mo1).reshape(nset,nmo,nocc)
    # -(e_ai x0 + fvind(x0) + hs) on the virtual block
    if x0 is None:
        r0 = -v1[:,viridx].reshape(nset,ndim)
    else:
        r0 = -(v1[:,viridx] + hs[:,viridx] + mo1[:,viridx] * e_ai).reshape(nset,ndim)

    vs = []  # the trial vectors V, a block per cycle
    mvs = []  # M V
    voo = []  # the occupied rows of fvind(V), for the first order energies
    vtmv = numpy.zeros((0,0))
    vtr0 = numpy.zeros((0,nset))
    c = vtr0
    r = r0
    drop = max(tol*1e-2, 1e-13)
    for cycle in range(max_cycle+1):
        rnorm = cupy.linalg.norm(r / precond, axis=1).get()
        active = rnorm > tol
        log.info('CPHF cycle %d: max residual %g, %d of %d perturbations left',
                 cycle, rnorm.max(initial=0), active.sum(), nset)
        if not active.any():
            break
        if cycle == max_cycle:
            raise RuntimeError('CPHF solver failed to converge.')

        t = r[cupy.asarray(numpy.where(active)[0])]
        if trials is None:
            t = t / precond
        else:
            t = cupy.array(trials(t), dtype=cupy.float64, order='C').reshape(-1,ndim)
        x1, sizes = _orthonormal_rows(t, vs, drop)
        t = None
        nq = len(x1)
        if nq == 0:
            log.warn('CPHF stopped with no new trial vector; residual %g '
                     'above tol %g', rnorm.max(), tol)
            break
        scale = 1.
        if scale_trials:
            scale = cupy.asarray(numpy.maximum(sizes, 1e-6)).reshape(nq,1,1)
        trial = cupy.zeros((nq,nmo,nocc))
        trial[:,viridx] = x1.reshape(nq,nvir,nocc) * scale
        w = fvind(trial).reshape(nq,nmo,nocc) / scale
        trial = None
        mx1 = w[:,viridx].reshape(nq,ndim) + x1 * diag
        voo.append(w[:,occidx].reshape(nq,nocc*nocc))
        w = None
        vs.append(x1)
        mvs.append(mx1)

        nd = vtmv.shape[0]
        vtmv1 = numpy.empty((nd+nq,nd+nq))
        vtmv1[:nd,:nd] = vtmv
        p0 = 0
        for v, mv in zip(vs, mvs):
            p1 = p0 + len(v)
            vtmv1[p0:p1,nd:] = v.dot(mx1.T).get()
            vtmv1[nd:,p0:p1] = x1.dot(mv.T).get()
            p0 = p1
        vtmv = vtmv1
        vtr0 = numpy.vstack((vtr0, x1.dot(r0.T).get()))
        c = numpy.linalg.solve(vtmv, vtr0)
        r = r0 - _combine_blocks(c, mvs)

    nvec = vtmv.shape[0]
    log.info('CPHF: %d cycles, %d trial vectors, max residual %.3g',
             cycle, nvec, float(cupy.linalg.norm(r / precond, axis=1).max()) if nset else 0.)
    if vs:
        xv = _combine_blocks(c, vs)
        mo1[:,viridx] += xv.reshape(nset,nvir,nocc)
        # fvind(mo1), from the responses of x0 and of the trial vectors
        v1[:,viridx] += (_combine_blocks(c, mvs) - xv * diag).reshape(nset,nvir,nocc)
        v1[:,occidx] += _combine_blocks(c, voo).reshape(nset,nocc,nocc)
    log.timer('CPHF solver', *t0)
    if second_order is not None:
        second_order.append(((mo1[:,viridx] + hs[:,viridx] / e_ai).reshape(nset,ndim), r))

    hs += v1
    mo1[:,viridx] = hs[:,viridx] / (e_i - e_a[:,None])
    mo_e1 = hs[:,occidx,:]
    mo_e1 += mo1[:,occidx] * (e_i[:,None] - e_i)
    return mo1, mo_e1

