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
Two-level SCF: the SCF solution of a DFT model from a few evaluations of its
XC and VV10 terms on its own grids.

The Fock matrix of a model and that of a cheap one of the same Hamiltonian
differ by a matrix that changes little with the density. With the model's
potential V[D_ref] of a reference density, the SCF iterates on

    F[D] = h + V[D_ref] + (V_cheap[D] - V_cheap[D_ref])

whose builds are the cheap model's; its error is the difference of the two
models applied to D - D_ref, second order. The reference is the converged
density of the cheap model first, then that of each such iteration; every
reference costs one build of the model, which also gives the energy and the
orbital gradient of the model there, the tests of its convergence. The
solution is the model's own, to its convergence thresholds.

The cheap model here is a copy on coarser XC and VV10 grids (levels
two_level_grids_level and two_level_nlcgrids_level). Its J and K are the
model's own, so the J and K of a reference are those the cheap model has
built for that density, and a reference costs the XC and VV10 terms on the
model's grids alone. Every iteration still builds J and K, so the references
are the cheaper step here: the first is taken once the cheap model is
converged to two_level_conv_tol only (1e-5: later, the correction sets the
iterations back by one; earlier, the second reference may not pass), and the
DIIS history goes on across the references (the constant
V[D_ref] - V_cheap[D_ref] is added to its Fock matrices, with their error
vectors following), so that the iterations are those of the plain SCF.

Caffeine, wB97M-V with XC grid (99,590) and VV10 grid (50,194) in SMD water,
on a GTX 1650 Super: def2-SVP 127 s -> 54 s, the energy of the plain SCF to
1e-11 Eh, with its 14 cycles and 2 references.

    >>> mf = dft.RKS(mol, xc='wb97m-v')
    >>> mf.two_level = True
    >>> mf.kernel()

This follows pyscf.scf.two_level of the CPU tree this work comes from, whose
cheap model also fits J and evaluates K seminumerically.
'''

import numpy
import cupy
from pyscf.data.elements import charge as _charge
from pyscf.dft import gen_grid as gen_grid_cpu
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, tag_array, pack_tril

# The micro cycles stop when the predicted energy changes by less than this
# fraction of conv_tol and the orbital gradient is below this fraction of
# conv_tol_grad. (Every micro cycle builds J and K here: 1 and .7 stop them
# where the plain SCF stops, .1 and .3 cost caffeine two cycles more.)
MICRO_ETOL = 1.
MICRO_GTOL = .7
# The model is also converged at a reference density where 2 |g|^2 / gap, a
# bound of the energy left in its orbital gradient g at second order (twice
# that of orbital-energy denominators no smaller than the gap), is below
# conv_tol. False: only by the energy predicted for that density.
SECOND_ORDER_TEST = True


def _energy_left(norm_gorb, mo_energy, mo_occ):
    '''2 |g|^2 / gap for the orbital gradient norm of get_grad (twice the Fock
    matrix elements for restricted orbitals: the bound is four times theirs),
    numpy.inf where the occupations leave no gap (fractional occupations)'''
    mo_energy = cupy.asnumpy(mo_energy)
    mo_occ = cupy.asnumpy(mo_occ)
    if mo_energy.ndim == 1:
        mo_energy = mo_energy[None]
        mo_occ = mo_occ[None]
    gap = numpy.inf
    for e, occ in zip(mo_energy, mo_occ):
        filled = occ > 0
        empty = occ < occ.max()
        if filled.any() and empty.any():
            gap = min(gap, e[empty].min() - e[filled].max())
    if not 1e-3 < gap < numpy.inf:
        return numpy.inf
    return 2 * norm_gorb**2 / gap


class _KeepsDensities:
    '''Mixin for scf.diis.CDIIS: the density of each Fock matrix of the
    history is kept next to it, so that a constant matrix can be added to all
    of them with their error vectors (SDF - FDS, linear in F) following.'''

    def update(self, s, d, f, *args, **kwargs):
        # the slot that push_err_vec and push_vec are about to fill
        slot = self._head if self._head < self.space else 0
        self._store('d%d' % slot, cupy.asarray(d).copy())
        return super().update(s, d, f, *args, **kwargs)

    def add_to_fock(self, s, delta):
        '''The history of the Fock matrices F_i + delta at the same densities'''
        slots = sorted(set(self._bookkeep))
        if not slots:
            return
        if self.Corth.ndim == 3:
            nao, nmo = self.Corth.shape[-2:]
        else:
            nao, nmo = self.Corth.shape
        delta = cupy.asarray(delta)
        d_tril = pack_tril(delta.reshape(-1,nao,nao)).ravel()
        for i in slots:
            d = cupy.asarray(self._buffer['d%d' % i])
            de = self._sdf_err_vec(s, d, delta)
            de = pack_tril(de.reshape(-1,nmo,nmo)).ravel()
            self._store('e%d' % i, cupy.asarray(self._buffer['e%d' % i]) + de)
            self._store('x%d' % i, cupy.asarray(self._buffer['x%d' % i]) + d_tril)
        errs = [cupy.asarray(self._buffer['e%d' % i]) for i in slots]
        for a, i in enumerate(slots):
            for j, ej in zip(slots[:a+1], errs):
                tmp = cupy.asnumpy(errs[a].conj().dot(ej))
                self._H[i+1,j+1] = tmp
                self._H[j+1,i+1] = tmp.conjugate()


def _diis_keeping_densities(diis_cls):
    return type(diis_cls.__name__ + 'KeepsDensities', (_KeepsDensities, diis_cls), {})


def coarse_grids(grids, mol, level):
    '''A copy of grids (not built) at most as fine as the default grids of the
    given level. A user's atom_grid, which takes precedence over the level, is
    kept only for the elements it gives fewer points than that level.
    '''
    g = grids.copy()
    g.reset(mol)
    g.level = min(grids.level, level)
    atom_grid = grids.atom_grid
    if atom_grid:
        symbs = {mol.atom_symbol(ia) for ia in range(mol.natm)}
        if isinstance(atom_grid, (list, tuple)):
            atom_grid = {symb: atom_grid for symb in symbs}
        default = atom_grid.get('default', None)
        kept = {}
        for symb in symbs:
            conf = atom_grid.get(symb, default)
            if conf is None:
                continue
            chg = _charge(symb)
            n = gen_grid_cpu._default_rad(chg, g.level) * gen_grid_cpu._default_ang(chg, g.level)
            if conf[0] * conf[1] < n:
                kept[symb] = conf
        g.atom_grid = kept
    return g


def _same_grids(a, b):
    return a.level == b.level and a.atom_grid == b.atom_grid


def cheap_scf(mf):
    '''The cheap model of mf for the two-level SCF: a copy with the grids of
    levels two_level_grids_level and two_level_nlcgrids_level where mf's are
    finer. It shares the integral screening data, the density fitting and the
    solvent object of mf: its J, K and solvent terms are mf's own. None if mf
    has no grids, or none finer than those levels, or is not a plain restricted
    or unrestricted Kohn-Sham model (also with density fitting or a solvent):
    the low-memory, DFT+U, constrained and smearing variants have potentials or
    iterations of their own, which the kernel below has not been tried with.
    '''
    from gpu4pyscf.dft import rks, uks, rks_lowmem, rkspu, ukspu, ucdft
    from gpu4pyscf.scf import rohf, smearing
    if not isinstance(mf, (rks.RKS, uks.UKS)):
        return None
    if isinstance(mf, (rohf.ROHF, rks_lowmem.RKS, rkspu.RKSpU, ukspu.UKSpU,
                       ucdft.CDFTBaseMixin, smearing._SmearingSCF)):
        return None
    mol = mf.mol
    cheap = mf.copy()
    if not isinstance(cheap.diis, bool):
        # a DIIS object of mf: not to be filled by the cheap model
        cheap.diis = True
    cheap.diis_file = None
    cheap.chkfile = None
    cheap.callback = None
    cheap.scf_summary = {}
    cheap.two_level = False
    cheaper = False
    for key, level in (('grids', mf.two_level_grids_level),
                       ('nlcgrids', mf.two_level_nlcgrids_level)):
        grids = getattr(mf, key, None)
        if grids is None:
            continue
        g = coarse_grids(grids, mol, level)
        setattr(cheap, key, g)
        if not _same_grids(g, grids) and (key == 'grids' or mf.do_nlc()):
            cheaper = True
    if not cheaper:
        return None
    return cheap


def _shares_jk(mf, cheap):
    '''Whether the two models build the same J and K: then those of a
    potential of the cheap model serve the model at the same density'''
    if type(cheap) is not type(mf) or getattr(cheap, 'xc', None) != getattr(mf, 'xc', None):
        return False
    if getattr(cheap, 'with_df', None) is not getattr(mf, 'with_df', None):
        return False
    return (cheap._opt_gpu is mf._opt_gpu and cheap._opt_jengine is mf._opt_jengine
            and cheap.direct_scf and mf.direct_scf
            and cheap.direct_scf_tol == mf.direct_scf_tol)


def _veff_at_reference(mf, cheap, mol, dm, vx):
    '''The model's potential of the reference density dm, where the cheap
    model's potential is vx. Where the two share J and K, get_veff's
    incremental build with no change of the density takes them from vx and
    evaluates the XC terms on the model's grids only.'''
    if _shares_jk(mf, cheap) and getattr(vx, 'vj', None) is not None:
        return mf.get_veff(mol, dm, dm_last=dm, vhf_last=vx)
    return mf.get_veff(mol, dm)


def _tagged(v, like):
    '''The array v with the solvent terms of the potential like, which
    get_fock and energy_elec of a solvated model read from the potential'''
    tags = {key: getattr(like, key) for key in ('e_solvent', 'v_solvent')
            if getattr(like, key, None) is not None}
    if tags:
        return tag_array(v, **tags)
    return v


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None, dump_chk=True, dm0=None,
           callback=None, conv_check=True, **kwargs):
    '''The SCF of mf by the two-level scheme, with the arguments and the
    return values of hf._kernel: (converged, e_tot, mo_energy, mo_coeff,
    mo_occ). mf.cycles counts the references, the evaluations of mf's own
    potential.

    Converged when, at a reference density, the orbital gradient of mf is below
    conv_tol_grad and either the energy of mf is within conv_tol of the one
    predicted for that density from the previous reference (its energy there,
    the change of the cheap model's energy, and the first-order term of the
    difference of the two potentials), or the energy left in that gradient at
    second order is below conv_tol (_energy_left). As in hf._kernel, the
    orbitals returned are those the last density was built from, and e_tot is
    the energy of mf at that density.
    '''
    from gpu4pyscf.scf import hf
    cheap = None
    if mf.max_cycle > 0:
        cheap = mf.two_level_scf
        if cheap is None:
            cheap = cheap_scf(mf)
        if cheap is None:
            logger.info(mf, 'two-level SCF: no cheaper model of this one, the plain SCF instead')
    if cheap is None:
        return hf._kernel(mf, conv_tol, conv_tol_grad, dump_chk, dm0, callback,
                          conv_check, **kwargs)

    conv_tol = mf.conv_tol
    mol = mf.mol
    log = logger.new_logger(mf)
    cput0 = cput1 = log.init_timer()
    if conv_tol_grad is None:
        conv_tol_grad = conv_tol**.5
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    # The cheap model's SCF, to two_level_conv_tol. Its last cycle leaves the
    # potential of its final density, whose J and K (and solvent terms) the
    # first reference takes over, and its DIIS history, which the micro cycles
    # continue.
    last = {}

    def keep(envs):
        last['dm'] = envs['dm']
        last['vhf'] = envs['vhf']
        last['mo_coeff'] = envs['mo_coeff']
        last['diis'] = envs.get('mf_diis')
    cheap.callback = keep
    cheap.conv_tol = max(conv_tol, mf.two_level_conv_tol)
    cheap.conv_tol_grad = None
    if cheap.diis is True and isinstance(cheap.DIIS, type):
        cheap.DIIS = _diis_keeping_densities(cheap.DIIS)
    # its "converged SCF energy" line would read as the result of mf
    cheap._finalize = lambda: None
    try:
        cheap.kernel(dm0=dm0)
    finally:
        cheap.callback = None
    if not cheap.converged:
        log.warn('two-level SCF: the cheap model did not converge')
    log.info('two-level SCF: the cheap model  E= %.15g', cheap.e_tot)
    mo_energy, mo_coeff, mo_occ = cheap.mo_energy, cheap.mo_coeff, cheap.mo_occ
    mf_diis = last.get('diis')
    if not isinstance(mf_diis, _KeepsDensities):
        mf_diis = None
    if last.get('mo_coeff') is mo_coeff:
        dm, vx = last['dm'], last['vhf']
    else:
        # a cycle after the last callback (the one that removes a level shift)
        dm = cheap.make_rdm1(mo_coeff, mo_occ)
        vx = cheap.get_veff(mol, dm)
        dm = asarray(dm)
        mf_diis = None
    last = None
    ncycle = cheap.cycles
    cput1 = log.timer('two-level SCF: the cheap model, %d cycles' % cheap.cycles, *cput1)

    h1e = cupy.asarray(mf.get_hcore(mol))
    s1e = cupy.asarray(mf.get_ovlp(mol))
    x_orth = mf.check_linear_dependency(s1e, log)
    dump_chk = dump_chk and mf.chkfile is not None
    if dump_chk:
        from pyscf.scf import chkfile
        chkfile.save_mol(mol, mf.chkfile)

    def gnorm(fock):
        return float(cupy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock)))

    def with_orbitals(dm):
        # numint evaluates the density from the orbitals where it has them
        return tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

    scf_conv = False
    e_pred = None
    e_tot = 0
    ncheap = 0
    dv = 0
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        # a reference: the potential, energy and orbital gradient of mf itself
        vhf = vref = _veff_at_reference(mf, cheap, mol, with_orbitals(dm), vx)
        e_tot = mf.energy_tot(dm, h1e, vref)
        fock = mf.get_fock(h1e, s1e, vref, dm)  # = h1e + vref, no DIIS
        norm_gorb = gnorm(fock)
        fock = None
        mf.cycles = cycle + 1
        e_left = _energy_left(norm_gorb, mo_energy, mo_occ) if SECOND_ORDER_TEST else numpy.inf
        if e_pred is None:
            log.info('reference %d  E= %.15g  |g|= %4.3g  (2 |g|^2/gap= %4.3g)',
                     cycle+1, e_tot, norm_gorb, e_left)
        else:
            log.info('reference %d  E= %.15g  E - predicted= %4.3g  |g|= %4.3g  '
                     '(2 |g|^2/gap= %4.3g, %d cheap builds)', cycle+1, e_tot, e_tot-e_pred,
                     norm_gorb, e_left, ncheap)
        if norm_gorb < conv_tol_grad and (
                e_left < conv_tol or (e_pred is not None and abs(e_tot-e_pred) < conv_tol)):
            scf_conv = True
        if dump_chk:
            mf.dump_chk({'mol': mol, 'mo_energy': mo_energy, 'mo_occ': mo_occ,
                         'mo_coeff': mo_coeff, 'e_tot': e_tot})
        if callable(callback):
            callback(locals())
        cput1 = log.timer('reference %d' % (cycle+1), *cput1)
        if scf_conv:
            break
        if ncycle >= mf.max_cycle:
            # max_cycle counts the cycles of the cheap model's SCF and the
            # micro cycles together, as it counts the cycles of the plain SCF
            log.warn("SCF failed to converge")
            break

        # micro cycles on h + V[D_ref] + (V_cheap[D] - V_cheap[D_ref])
        ex_ref = cheap.energy_tot(dm, h1e, vx)
        dm_ref = dm
        dv_last, dv = dv, asarray(vref) - asarray(vx)
        vref = None
        if isinstance(mf_diis, _KeepsDensities):
            # the history of the Fock matrices with this reference's dv
            mf_diis.add_to_fock(s1e, dv - dv_last)
        elif mf.diis:
            mf_diis = mf.DIIS(mf, None)
            mf_diis.space = mf.diis_space
            mf_diis.rollback = mf.diis_space_rollback
            mf_diis.Corth = cupy.asarray(x_orth)
        else:
            mf_diis = None
        dv_last = None
        e_micro = e_tot
        vhf = _tagged(asarray(vx) + dv, vx)
        for micro in range(mf.two_level_micro_cycles):
            if ncycle >= mf.max_cycle:
                break
            ncycle += 1
            f = mf.get_fock(h1e, s1e, vhf, dm, ncycle, mf_diis)
            mo_energy, mo_coeff = mf.eig(f, s1e, x=x_orth)
            f = None
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            dm_last = dm
            dm = mf.make_rdm1(mo_coeff, mo_occ)
            vx = cheap.get_veff(mol, dm, dm_last, vx)
            dm = asarray(dm)  # Remove the attached attributes
            ncheap += 1
            vhf = _tagged(asarray(vx) + dv, vx)
            # the energy of mf predicted for dm: that of the reference, the change of the cheap
            # model's, and the first order of the difference of the two models
            e_last = e_micro
            e_micro = (e_tot + cheap.energy_tot(dm, h1e, vx) - ex_ref
                       + float((dv * (dm - dm_ref).swapaxes(-1, -2)).sum().real))
            g = gnorm(mf.get_fock(h1e, s1e, vhf, dm))
            log.debug('  micro %d  E(predicted)= %.15g  delta_E= %4.3g  |g|= %4.3g',
                      micro+1, e_micro, e_micro-e_last, g)
            if abs(e_micro-e_last) < conv_tol * MICRO_ETOL and g < conv_tol_grad * MICRO_GTOL:
                break
        e_pred = e_micro
        vhf = None
        cput1 = log.timer('micro cycles of reference %d' % (cycle+1), *cput1)
    else:
        log.warn("SCF failed to converge")

    if scf_conv and mf.level_shift:
        # the orbitals of the model's own Fock matrix, without the level shift
        fock = mf.get_fock(h1e, s1e, vref, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        if dump_chk:
            mf.dump_chk({'mol': mol, 'mo_energy': mo_energy, 'mo_occ': mo_occ,
                         'mo_coeff': mo_coeff, 'e_tot': e_tot})
    log.info('two-level SCF: %d evaluations of the model, %d builds of the cheap one after its SCF '
             'of %d cycles', mf.cycles, ncheap, cheap.cycles)
    log.timer('two-level scf_cycle', *cput0)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ
