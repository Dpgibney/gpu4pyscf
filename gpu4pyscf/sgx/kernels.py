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
The CUDA kernels of the seminumerical exchange, generated per class of shell
pairs and compiled when first used (cupy.RawKernel).

For the grid points g of a block and the shell pairs (i,j) of its list, a
kernel adds

    G[a,g] += sum_b (a|b)_g F[b,g],    G[b,g] += sum_a (a|b)_g F[a,g]

for the Cartesian functions a of shell i and b of shell j, with
(a|b)_g = int a(r) b(r) O(|r - r_g|) dr and the operator
O(r) = c_full / r + c_lr erf(omega r) / r.

One thread takes one grid point and goes through the pairs: the threads of a
warp are on the same pair, read its data with one load, and read and write
their own columns of F and G, which are adjacent.

The integrals are Obara-Saika's: the Boys functions of the primitive pair at
the point, the vertical recurrence to [e|0]^(0) for the Cartesian powers e of
degree li..li+lj at the center of shell i, summed over the primitive pairs.
The horizontal recurrence is not carried out: with x_B = x_A + AB,

    (a|b) = sum_{k <= b} w(b,k) [a+k|0],   w(b,k) = prod_d binom(b_d,k_d) AB_d^(b_d-k_d),

so that  sum_b (a|b) F[b] = sum_k T[k] [a+k|0]  with  T[k] = sum_b w(b,k) F[b],
and      sum_a (a|b) F[a] = sum_k w(b,k) U[k]   with  U[k] = sum_a F[a] [a+k|0].

Both operators take the same recurrence: the base values of erf(omega r)/r
are those of 1/r at theta T, theta = omega^2 / (omega^2 + p), scaled by
theta^(m + 1/2), so a range-separated hybrid costs a second set of Boys
functions and nothing else.
'''

import math
import numpy as np
import cupy as cp

__all__ = ['get_kernel', 'boys_table', 'cart_tuples', 'NREC', 'LMAX']

# highest angular momentum of a shell
LMAX = 4
# doubles of a primitive pair's record: p, P (3), coefficient of 1/r, P - A (3),
# 1/(2p), theta, coefficient of erf(omega r)/r times sqrt(theta), unused
NREC = 12

# The Boys functions F_m(T) for T < boys_tmax(L) come from a table at the
# multiples of BOYS_STEP: a Taylor series for the highest order, downward
# recursion for the others. Beyond, the asymptotic form.
BOYS_STEP = .125
BOYS_ORDER = {'double': 6, 'float': 3}

def boys_tmax(L):
    '''Where F_m(T) = (2m-1)!! / 2^(m+1) sqrt(pi / T^(2m+1)) holds for the
    integrals of total angular momentum L: its error, exp(-T) (2T)^(m-1)
    relative to the leading term, is below 1e-12 there'''
    return 30 + 4 * L

def boys_exact(mmax, t):
    '''F_m(t) for m = 0..mmax'''
    t = np.asarray(t, dtype=np.longdouble)
    f = np.empty((mmax+1,) + t.shape, dtype=np.longdouble)
    term = np.ones_like(t) / (2*mmax+1)
    s = term.copy()
    k = 0
    while True:
        k += 1
        term = term * (2*t) / (2*mmax + 2*k + 1)
        s += term
        if k > 2*t.max() + 8 and np.all(term <= 1e-21 * s):
            break
    e = np.exp(-t)
    f[mmax] = e * s
    for m in range(mmax, 0, -1):
        f[m-1] = (2*t*f[m] + e) / (2*m-1)
    return f

_boys_tables = {}
def boys_table(L, dtype='double'):
    '''The rows [exp(-T_i), F_L(T_i), F_{L+1}(T_i)/1!, ..., F_{L+K}(T_i)/K!]
    of the kernels of total angular momentum L, on the device'''
    key = (L, dtype, cp.cuda.Device().id)
    if key not in _boys_tables:
        K = BOYS_ORDER[dtype]
        n = int(round(boys_tmax(L) / BOYS_STEP)) + 2
        t = np.arange(n) * np.longdouble(BOYS_STEP)
        f = boys_exact(L+K, t)
        tab = np.empty((n, K+2), dtype=np.longdouble)
        tab[:,0] = np.exp(-t)
        for k in range(K+1):
            tab[:,1+k] = f[L+k] / math.factorial(k)
        _boys_tables[key] = cp.asarray(tab.astype(np.float64 if dtype == 'double' else np.float32))
    return _boys_tables[key]

def cart_tuples(l):
    '''Powers of the Cartesian functions of a shell, in the order of the AOs'''
    return [(lx, ly, l-lx-ly) for lx in range(l, -1, -1) for ly in range(l-lx, -1, -1)]

def _s(e):
    return '%d%d%d' % tuple(e)

def _minus(e, d, n=1):
    e = list(e)
    e[d] -= n
    return tuple(e)

def _plus(a, b):
    return tuple(x + y for x, y in zip(a, b))

def _fnum(x, dtype):
    r = repr(float(x))
    return r + 'f' if dtype == 'float' else r


def _gen_boys(L, mode, dtype):
    '''Lines that set f0..fL to the base values [0|0]^(m) of the primitive
    pair, from T, the pointer q to its record and the table boys'''
    K = BOYS_ORDER[dtype]
    nrow = K + 2
    tmax = boys_tmax(L)
    c = lambda x: _fnum(x, dtype)
    rsq = 'rsqrt' if dtype == 'double' else 'rsqrtf'

    def one(pre, tvar, coef, scale):
        # pre: prefix of the outputs; scale: None or the name of theta, the
        # factor per order
        out = ['REAL %s;' % ', '.join('%s%d' % (pre, m) for m in range(L+1))]
        out.append('if (%s < %s) {' % (tvar, c(tmax)))
        out.append('    const int it = (int)(%s * %s + %s);' % (tvar, c(1/BOYS_STEP), c(.5)))
        out.append('    const REAL x = (REAL)it * %s - %s;' % (c(BOYS_STEP), tvar))
        out.append('    const REAL* __restrict__ row = boys + it * %d;' % nrow)
        out.append('    REAL s = row[%d];' % (K+1))
        for k in range(K, 0, -1):
            out.append('    s = s * x + row[%d];' % k)
        if L == 0:
            out.append('    %s0 = s * %s;' % (pre, coef))
        else:
            # exp(-T) = exp(-T_i) exp(x)
            out.append('    REAL ex = %s;' % c(1/math.factorial(K)))
            for k in range(K-1, -1, -1):
                out.append('    ex = ex * x + %s;' % c(1/math.factorial(k)))
            out.append('    const REAL e = row[0] * ex * %s;' % coef)
            out.append('    const REAL t2 = %s + %s;' % (tvar, tvar))
            out.append('    %s%d = s * %s;' % (pre, L, coef))
            for m in range(L, 0, -1):
                out.append('    %s%d = (t2 * %s%d + e) * %s;' % (pre, m-1, pre, m, c(1/(2*m-1))))
            if scale is not None:
                out.append('    REAL sc = %s;' % scale)
                for m in range(1, L+1):
                    out.append('    %s%d *= sc;' % (pre, m))
                    if m < L:
                        out.append('    sc *= %s;' % scale)
        out.append('} else {')
        out.append('    const REAL r = %s(%s);' % (rsq, tvar))
        out.append('    %s0 = %s * %s * r;' % (pre, c(math.sqrt(math.pi)/2), coef))
        if L > 0:
            if scale is None:
                out.append('    const REAL hr = %s * r * r;' % c(.5))
            else:
                out.append('    const REAL hr = %s * r * r * %s;' % (c(.5), scale))
            for m in range(1, L+1):
                if m == 1:
                    out.append('    %s1 = %s0 * hr;' % (pre, pre))
                else:
                    out.append('    %s%d = %s%d * (%s * hr);' % (pre, m, pre, m-1, c(2*m-1)))
        out.append('}')
        return out

    lines = []
    if mode == 'full':
        lines += one('f', 'T', 'q[4]', None)
    elif mode == 'lr':
        lines.append('const REAL theta = q[9];')
        lines.append('const REAL Tl = theta * T;')
        lines += one('f', 'Tl', 'q[10]', 'theta')
    elif mode == 'mixed':
        lines += one('f', 'T', 'q[4]', None)
        lines.append('const REAL theta = q[9];')
        lines.append('const REAL Tl = theta * T;')
        lines += one('h', 'Tl', 'q[10]', 'theta')
        for m in range(L+1):
            lines.append('f%d += h%d;' % (m, m))
    else:
        raise ValueError(mode)
    return lines


def _gen_vrr(L, lmin):
    '''Lines of the vertical recurrence, from f0..fL to v{e}_0 for the powers
    e of degree lmin..L, which are added to the accumulators E{e}'''
    need = {}
    order = []
    diffs = []

    def request(e, m):
        if sum(e) == 0 or (e, m) in need:
            return
        # raise a component from 0 to 1 where there is one: no lower term then
        d = next((i for i in (2, 1, 0) if e[i] == 1), None)
        if d is None:
            d = next(i for i in (2, 1, 0) if e[i] > 0)
        e1 = _minus(e, d)
        request(e1, m)
        request(e1, m+1)
        lower = e[d] >= 2
        if lower:
            e2 = _minus(e1, d)
            request(e2, m)
            request(e2, m+1)
        need[e, m] = (d, lower)
        order.append((e, m))

    targets = [e for n in range(lmin, L+1) for e in cart_tuples(n)]
    for e in targets:
        request(e, 0)

    def v(e, m):
        if sum(e) == 0:
            return 'f%d' % m
        return 'v%s_%d' % (_s(e), m)

    lines = []
    for n in range(1, L):
        if n == 1:
            lines.append('const REAL c1 = q[8];')
        else:
            lines.append('const REAL c%d = %d * q[8];' % (n, n))
    done_diff = set()
    for e, m in order:
        d, lower = need[e, m]
        ax = 'xyz'[d]
        e1 = _minus(e, d)
        expr = 'pa%s * %s + cp%s * %s' % (ax, v(e1, m), ax, v(e1, m+1))
        if lower:
            e2 = _minus(e1, d)
            if (e2, m) not in done_diff:
                done_diff.add((e2, m))
                lines.append('const REAL d%s_%d = %s - %s;' % (_s(e2), m, v(e2, m), v(e2, m+1)))
            expr += ' + c%d * d%s_%d' % (e[d]-1, _s(e2), m)
        lines.append('const REAL %s = %s;' % (v(e, m), expr))
    for e in targets:
        lines.append('E%s += %s;' % (_s(e), v(e, 0)))
    return lines, targets, len(order)


def _gen_contract(li, lj):
    '''Lines before the primitive loop (T[k] from fj and AB) and after it
    (gi, U[k], gj)'''
    A = cart_tuples(li)
    B = cart_tuples(lj)
    ks = [k for n in range(lj+1) for k in cart_tuples(n)]
    pre = []
    post = []

    # monomials of AB of degree 1..lj
    mono = {(0, 0, 0): None}
    if lj > 0:
        pre.append('const REAL abx = pair_ab[3*ip], aby = pair_ab[3*ip+1], abz = pair_ab[3*ip+2];')
    for n in range(1, lj+1):
        for e in cart_tuples(n):
            d = next(i for i in (0, 1, 2) if e[i] > 0)
            lower = mono[_minus(e, d)]
            name = 'm%s' % _s(e)
            if lower is None:
                pre.append('const REAL %s = ab%s;' % (name, 'xyz'[d]))
            else:
                pre.append('const REAL %s = %s * ab%s;' % (name, lower, 'xyz'[d]))
            mono[e] = name

    weights = {}
    def w(b, k):
        '''The factor of fj[b] in T[k] and of U[k] in gj[b]: None for 1'''
        if (b, k) not in weights:
            diff = _minus(_minus(_minus(b, 0, k[0]), 1, k[1]), 2, k[2])
            coef = math.comb(b[0], k[0]) * math.comb(b[1], k[1]) * math.comb(b[2], k[2])
            m = mono[diff]
            if m is None:
                assert coef == 1
                weights[b, k] = None
            elif coef == 1:
                weights[b, k] = m
            else:
                name = 'w%s_%s' % (_s(b), _s(k))
                pre.append('const REAL %s = %d * %s;' % (name, coef, m))
                weights[b, k] = name
        return weights[b, k]

    def leq(k, b):
        return all(x <= y for x, y in zip(k, b))

    # T[k] = sum_b w(b,k) fj[b]
    for k in ks:
        terms = []
        for ib, b in enumerate(B):
            if leq(k, b):
                wt = w(b, k)
                terms.append('fj%d' % ib if wt is None else '%s * fj%d' % (wt, ib))
        pre.append('const REAL T%s = %s;' % (_s(k), ' + '.join(terms)))

    # gi[a] = sum_k T[k] E[a+k]
    for ia, a in enumerate(A):
        terms = ['T%s * E%s' % (_s(k), _s(_plus(a, k))) for k in ks]
        post.append('const REAL gi%d = %s;' % (ia, ' + '.join(terms)))
    # U[k] = sum_a fi[a] E[a+k]
    for k in ks:
        terms = ['fi%d * E%s' % (ia, _s(_plus(a, k))) for ia, a in enumerate(A)]
        post.append('const REAL U%s = %s;' % (_s(k), ' + '.join(terms)))
    # gj[b] = sum_{k <= b} w(b,k) U[k]
    for ib, b in enumerate(B):
        terms = []
        for k in ks:
            if leq(k, b):
                wt = w(b, k)
                terms.append('U%s' % _s(k) if wt is None else '%s * U%s' % (wt, _s(k)))
        post.append('const REAL gj%d = %s;' % (ib, ' + '.join(terms)))
    return pre, post


def kernel_name(li, lj, mode, dtype):
    return 'sgx_k_%d%d_%s_%s' % (li, lj, mode, dtype)

def kernel_source(li, lj, mode='full', dtype='double'):
    '''The CUDA source of the kernel of the shell pairs with angular momenta
    li >= lj.

    mode: 'full' for c_full/r, 'lr' for c_lr erf(omega r)/r, 'mixed' for both.
    '''
    assert li >= lj
    L = li + lj
    ni = (li+1)*(li+2)//2
    nj = (lj+1)*(lj+2)//2
    boys = _gen_boys(L, mode, dtype)
    vrr, targets, nvrr = _gen_vrr(L, li)
    pre, post = _gen_contract(li, lj)

    def ind(lines, n):
        return '\n'.join(' ' * n + x for x in lines)

    load = []
    for a in range(ni):
        load.append('const REAL fi%d = F[(size_t)(ai+%d) * ld + g];' % (a, a))
    for b in range(nj):
        load.append('const REAL fj%d = F[(size_t)(aj+%d) * ld + g];' % (b, b))
    zero = ['REAL %s;' % ', '.join('E%s = 0' % _s(e) for e in targets)]
    geom = []
    if L > 0:
        geom.append('const REAL pax = q[5], pay = q[6], paz = q[7];')
    store = []
    for a in range(ni):
        store.append('G[(size_t)(ai+%d) * ld + g] += gi%d;' % (a, a))
    for b in range(nj):
        store.append('G[(size_t)(aj+%d) * ld + g] += gj%d;' % (b, b))

    return '''
extern "C" __global__
void %(name)s(REAL* __restrict__ G, const REAL* __restrict__ F, const REAL* __restrict__ grid,
        const int* __restrict__ pairs, const REAL* __restrict__ pair_ab,
        const REAL* __restrict__ prims,
        const int* __restrict__ blk_loc, const int* __restrict__ blk_pairs,
        const REAL* __restrict__ boys, const int listed, const int npairs,
        const int ngrids, const int ld, const int blk0)
{
    const int blk = blk0 + blockIdx.x;
    const int g = blk * blockDim.x + threadIdx.x;
    if (g >= ngrids) {
        return;
    }
    const REAL gx = grid[g], gy = grid[(size_t)ld + g], gz = grid[(size_t)2*ld + g];
    int n0 = 0;
    int n1 = npairs;
    if (listed) {
        n0 = blk_loc[blk];
        n1 = blk_loc[blk+1];
    }
    for (int n = n0; n < n1; ++n) {
        const int ip = listed ? blk_pairs[n] : n;
        const int ai = pairs[4*ip], aj = pairs[4*ip+1];
        const int k0 = pairs[4*ip+2], k1 = pairs[4*ip+3];
%(load)s
%(pre)s
%(zero)s
        for (int k = k0; k < k1; ++k) {
            const REAL* __restrict__ q = prims + %(nrec)d * (size_t)k;
            const REAL cpx = gx - q[1], cpy = gy - q[2], cpz = gz - q[3];
            const REAL T = q[0] * (cpx*cpx + cpy*cpy + cpz*cpz);
%(boys)s
%(geom)s
%(vrr)s
        }
%(post)s
%(store)s
    }
}
''' % dict(name=kernel_name(li, lj, mode, dtype), nrec=NREC,
           load=ind(load, 8), pre=ind(pre, 8), zero=ind(zero, 8),
           boys=ind(boys, 12), geom=ind(geom, 12), vrr=ind(vrr, 12),
           post=ind(post, 8), store=ind(store, 8))


_kernels = {}
def get_kernel(li, lj, mode='full', dtype='double'):
    '''The compiled kernel (cupy.RawKernel) of a class of shell pairs'''
    key = (li, lj, mode, dtype)
    if key not in _kernels:
        if max(li, lj) > LMAX:
            raise NotImplementedError('angular momentum %d' % max(li, lj))
        code = '#define REAL %s\n' % dtype + kernel_source(li, lj, mode, dtype)
        _kernels[key] = cp.RawKernel(code, kernel_name(li, lj, mode, dtype),
                                     options=('-std=c++17',))
    return _kernels[key]
