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

'''
Seminumerical exchange (COSX, PySCF's sgx): one electron coordinate of the
exchange integrals on a grid,

    K[mu,nu] = sum_g X[mu,g] G[nu,g],   G[nu,g] = sum_sigma A[nu,sigma](g) F[sigma,g],
    F[sigma,g] = w_g sum_lambda D[lambda,sigma] X[lambda,g],

with X the AO values on the grid, w its weights and A[nu,sigma](g) the
potential of the AO pair at grid point g. The products with X are matrix
products; G comes from the kernels of gpu4pyscf.sgx.kernels, which hold no
integrals.
'''

import math
import numpy as np
import cupy as cp
from scipy.special import gammaln
from pyscf.gto.mole import ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF, ATOM_OF
from gpu4pyscf.gto.mole import _scale_sp_ctr_coeff
from gpu4pyscf.sgx import kernels
from gpu4pyscf.sgx.kernels import NREC

# grid points of a kernel's block (one thread each)
BLKSIZE = 128
# elements of the temporary arrays of pair_data, per chunk of shells
_PAIR_CHUNK = 2000000
# work of a kernel launch, in the units of _class_cost (about a nanosecond in
# double precision on a GTX 1650 Super: half a second a launch there)
_LAUNCH_WORK = 5e8

def _class_cost(L):
    '''Relative cost of a primitive pair on a grid point for the shell pairs
    of total angular momentum L (measured: 0.28, 0.5, 1.1, 2.3, 5.7, 9, 17 ns
    for L = 0..6)'''
    return .25 * (1 + .9 * L**2.3)


class _PairClass:
    '''The shell pairs of angular momenta (li, lj), li >= lj, of a sorted mol'''
    def __init__(self, li, lj):
        self.li = li
        self.lj = lj
        self.npairs = 0
        self.ish = self.jsh = None  # shells of the pair in the sorted mol
        self.pairs = None  # [npairs,4]: first AO of i, of j, its primitive pairs k0:k1
        self.ab = None  # [npairs,3]: A - B
        self.q = None  # [npairs]: bound of |(a|b)_g| over the functions and g
        self.prims = None  # [nprims,NREC]


def pair_data(mol, omega=None, c_full=1., c_lr=0., pair_tol=1e-13, prim_tol=1e-15):
    '''The data of the kernels for every class of shell pairs of the sorted
    mol (gpu4pyscf.gto.mole.SortedMole), in numpy arrays, for the operator
    c_full / r + c_lr erf(omega r) / r.

    A primitive pair is left out where the bound of its integrals is below
    prim_tol, a shell pair where the sum of these bounds is below pair_tol.
    The bound of a primitive pair holds at every point: the potential, at its
    center, of |c_a c_b| K r_A^la r_B^lb exp(-p r_P^2), with r_A <= r_P + |PA|.
    '''
    bas = mol._bas
    env = _scale_sp_ctr_coeff(mol)
    ls = bas[:,ANG_OF]
    nbas = len(bas)
    coords = mol.atom_coords()[bas[:,ATOM_OF]]
    ao_loc = np.asarray(mol.ao_loc_nr(cart=True))
    counts = np.asarray(mol.l_ctr_counts)
    offsets = np.append(0, np.cumsum(counts))
    assert offsets[-1] == nbas
    if not omega or c_lr == 0:
        omega = 0.
        c_lr = 0.

    def shells(g):
        s0, s1 = offsets[g], offsets[g+1]
        l, nprim = (int(x) for x in mol.uniq_l_ctr[g])
        assert np.all(ls[s0:s1] == l) and np.all(bas[s0:s1,NPRIM_OF] == nprim)
        prim = np.arange(nprim)
        exps = env[bas[s0:s1,PTR_EXP,None] + prim]
        coef = env[bas[s0:s1,PTR_COEFF,None] + prim]
        return s0, s1, l, nprim, exps, coef

    parts = {}
    ngroups = len(counts)
    for gi in range(ngroups):
        i0, i1, li, npi, exps_i, coef_i = shells(gi)
        for gj in range(gi+1):
            j0, j1, lj, npj, exps_j, coef_j = shells(gj)
            assert li >= lj
            nj = j1 - j0
            step = max(1, _PAIR_CHUNK // (nj * npi * npj))
            for c0 in range(i0, i1, step):
                c1 = min(c0 + step, i1)
                ish = np.arange(c0, c1)
                jsh = np.arange(j0, j1)
                a = exps_i[c0-i0:c1-i0,None,:,None]
                b = exps_j[None,:,None,:]
                ca = coef_i[c0-i0:c1-i0,None,:,None]
                cb = coef_j[None,:,None,:]
                ra = coords[ish][:,None,:]
                rb = coords[jsh][None,:,:]
                ab = ra - rb
                ab2 = (ab**2).sum(axis=2)[:,:,None,None]
                p = a + b
                cc = ca * cb * np.exp(-a * b / p * ab2) * (2*np.pi / p)
                if gi == gj:
                    # i > j once; i == j with half its weight, added from both sides
                    same = ish[:,None] == jsh[None,:]
                    cc = np.where(same[:,:,None,None], .5 * cc, cc)
                    keep_pair = ish[:,None] >= jsh[None,:]
                else:
                    keep_pair = np.ones((len(ish), nj), dtype=bool)
                # bound of the integrals of each primitive pair
                dab = np.sqrt(ab2)
                pa = b / p * dab
                pb = a / p * dab
                q = np.zeros_like(cc)
                for s in range(li+1):
                    for t in range(lj+1):
                        n = s + t
                        q += (math.comb(li, s) * math.comb(lj, t)
                              * pa**(li-s) * pb**(lj-t)
                              * np.exp(gammaln(n/2 + 1)) / p**(n/2))
                q *= abs(cc)
                if gi == gj:
                    q = np.where(same[:,:,None,None], 2 * q, q)
                keep = (q > prim_tol) & keep_pair[:,:,None,None]
                qpair = np.where(keep, q, 0).sum(axis=(2,3))
                keep &= (qpair > pair_tol)[:,:,None,None]
                nprim = keep.sum(axis=(2,3))
                sel = nprim > 0
                if not sel.any():
                    continue
                pi, pj = np.nonzero(sel)
                # the primitive pairs kept, pair after pair
                idx = np.nonzero(keep)
                pp = np.broadcast_to(p, keep.shape)[idx]
                aa = np.broadcast_to(a, keep.shape)[idx]
                bb = np.broadcast_to(b, keep.shape)[idx]
                rai = coords[ish][idx[0]]
                rbj = coords[jsh][idx[1]]
                center = (aa[:,None] * rai + bb[:,None] * rbj) / pp[:,None]
                rec = np.zeros((len(pp), NREC))
                rec[:,0] = pp
                rec[:,1:4] = center
                rec[:,4] = cc[idx] * c_full
                rec[:,5:8] = center - rai
                rec[:,8] = .5 / pp
                if omega:
                    theta = omega**2 / (omega**2 + pp)
                    rec[:,9] = theta
                    rec[:,10] = cc[idx] * c_lr * np.sqrt(theta)
                part = parts.setdefault((li, lj), [])
                part.append((ish[pi], jsh[pj], nprim[sel], (ra - rb)[sel], qpair[sel], rec))

    classes = {}
    for (li, lj), part in parts.items():
        pc = _PairClass(li, lj)
        pc.ish = np.hstack([x[0] for x in part]).astype(np.int32)
        pc.jsh = np.hstack([x[1] for x in part]).astype(np.int32)
        nprim = np.hstack([x[2] for x in part])
        pc.ab = np.vstack([x[3] for x in part])
        pc.q = np.hstack([x[4] for x in part])
        pc.prims = np.vstack([x[5] for x in part])
        pc.npairs = len(pc.ish)
        k1 = np.cumsum(nprim)
        pairs = np.empty((pc.npairs, 4), dtype=np.int32)
        pairs[:,0] = ao_loc[pc.ish]
        pairs[:,1] = ao_loc[pc.jsh]
        pairs[:,2] = k1 - nprim
        pairs[:,3] = k1
        pc.pairs = pairs
        classes[li, lj] = pc
    return classes


class _DevicePairs:
    '''pair_data on the device, in the precision of the kernels'''
    def __init__(self, classes, omega, c_full, c_lr, dtype='double'):
        real = np.float64 if dtype == 'double' else np.float32
        self.dtype = dtype
        self.real = real
        if c_lr == 0 or not omega:
            self.mode = 'full'
        elif c_full == 0:
            self.mode = 'lr'
        else:
            self.mode = 'mixed'
        self.classes = []
        for (li, lj), pc in sorted(classes.items()):
            self.classes.append((
                li, lj, pc.npairs,
                cp.asarray(pc.pairs),
                cp.asarray(pc.ab.astype(real)),
                cp.asarray(pc.prims.astype(real)),
                cp.asarray(pc.ish), cp.asarray(pc.jsh),
                cp.asarray(pc.q.astype(np.float32))))


def contract(dev, F, coords, G=None, lists=None, blksize=BLKSIZE):
    '''G[a,g] += sum_b (a|b)_g F[b,g] for the Cartesian functions of the
    sorted mol and the grid points of coords.

    Args:
        dev : _DevicePairs
        F : [nao_cart,ngrids] in the precision of dev
        coords : [3,ngrids]
        lists : for each class of dev, None for all its pairs on every block
            of blksize points, or (loc, pairs), int32 arrays on the device:
            block b takes the pairs pairs[loc[b]:loc[b+1]] of the class. (No
            caller screens yet. A bound of the exchange energy a pair carries
            on a block, the product of the norms of F of its two shells there,
            drops what the Fock matrix needs: all pairs with a d shell for a
            minimal-basis guess density. A bound of the matrix elements is
            the safe form.)
    '''
    real = dev.real
    assert F.dtype == real and F.flags.c_contiguous
    nao, ngrids = F.shape
    coords = cp.ascontiguousarray(coords, dtype=real)
    assert coords.shape == (3, ngrids)
    if G is None:
        G = cp.zeros_like(F)
    nblk = (ngrids + blksize - 1) // blksize
    dummy = cp.zeros(1, dtype=np.int32)
    for n, (li, lj, npairs, pairs, ab, prims, ish, jsh, q) in enumerate(dev.classes):
        kern = kernels.get_kernel(li, lj, dev.mode, dev.dtype)
        boys = kernels.boys_table(li + lj, dev.dtype)
        if lists is None or lists[n] is None:
            loc, sub, listed = dummy, dummy, 0
            ntasks = npairs * nblk
        else:
            loc, sub = lists[n]
            listed = 1
            ntasks = len(sub)
            if ntasks == 0:
                continue
        # blocks per launch, by the work of a block: a launch should end
        # well within the time the driver of a display allows a kernel
        work = ntasks / nblk * blksize * len(prims) / npairs * _class_cost(li + lj)
        step = max(1, int(_LAUNCH_WORK / work))
        for b0 in range(0, nblk, step):
            kern((min(step, nblk - b0),), (blksize,),
                 (G, F, coords, pairs, ab, prims, loc, sub, boys,
                  np.int32(listed), np.int32(npairs), np.int32(ngrids), np.int32(ngrids),
                  np.int32(b0)))
    return G
