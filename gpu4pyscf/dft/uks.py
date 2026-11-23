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

import cupy
from pyscf.dft import uks as uks_cpu
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf, uhf, j_engine
from gpu4pyscf.lib.cupy_helper import tag_array, asarray
from gpu4pyscf.lib import utils


def get_veff(ks, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
    '''Coulomb + XC functional for UKS.  See pyscf/dft/rks.py
    :func:`get_veff` fore more details.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    # The density that vhf_last was built from: its J and K are in vhf_last,
    # only the XC terms are evaluated (see rks.get_veff)
    unchanged = dm_last is dm and getattr(vhf_last, 'vj', None) is not None
    dm_in = dm
    if isinstance(dm, cupy.ndarray) and dm.ndim == 2:
        dm = cupy.repeat(dm[None]*.5, 2, axis=0)
    else:
        dm = asarray(dm)
    assert dm.ndim == 3
    log = logger.new_logger(ks)
    t0 = log.init_timer()
    if ks.grids.coords is None:
        rks.initialize_grids(ks, mol, dm[0]+dm[1])

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_uks(mol, ks.grids, ks.xc, dm.view(cupy.ndarray), max_memory=max_memory)
        log.debug('nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            # The VV10 density from the orbitals where the density carries
            # them, as RKS and the nuclear gradient evaluate it (the gradient
            # then finds the potential of this density kept: numint._vv10nlc)
            dm_nlc = dm
            mo_coeff = getattr(dm_in, 'mo_coeff', None)
            if (isinstance(mo_coeff, cupy.ndarray) and mo_coeff.ndim == 3
                    and getattr(dm_in, 'ndim', 0) == 3):
                dm_nlc = dm_in
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm_nlc)
            exc += enlc
            vxc += vnlc
            log.debug('nelec with nlc grids = %s', n)
    t1 = log.timer('vxc', *t0)

    dm_orig = dm
    vj_last = getattr(vhf_last, 'vj', None)
    if vj_last is not None:
        dm_last = asarray(dm_last)
        dm = dm - dm_last
    else:
        dm_last = None
    hybrid = ni.libxc.is_hybrid_xc(ks.xc)
    if hybrid:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
    vk = None
    if unchanged:
        vj = cupy.zeros_like(dm[0])
        if hybrid:
            vk = cupy.zeros_like(dm)
    elif hybrid and ks.jk_one_pass and omega == 0:
        # J and K in one Rys pass, see hf.get_veff. jk.get_jk has no such pass for range-separated K.
        vj, vk = ks.get_jk(mol, dm, hermi)
        vj = vj[0] + vj[1]
        vk *= hyb
    else:
        vj = ks.get_j(mol, dm[0]+dm[1], hermi)
    vhf = vj
    ecoul = uhf._trace_ecoul(vj, dm, dm_last, vhf_last)
    # The energy of J and K built for the change dm of the density at second
    # order in it (hf._e2_of_change, see rks.get_veff)
    second_order = ks.second_order_energy and ecoul is not None
    e2 = None
    vk_last = None
    if vj_last is not None:
        vk_last = getattr(vhf_last, 'vk', None)
        if second_order and (vk_last is not None or not hybrid):
            if hybrid:
                j_last = asarray(vj_last[0]) + asarray(vk_last[0])
            else:
                j_last = asarray(vj_last)
            ecoul = hf._e2_of_change(dm, vj, j_last, vhf_last.ecoul)
            j_last = None
    cput2 = log.timer_debug1('vj', *t1)

    if hybrid:
        if vk is None:
            vk = ks.get_k(mol, dm, hermi, omega, alpha, hyb)
        vhf = vj - vk
        if vj_last is not None:
            if second_order and getattr(vhf_last, 'e2', None) is not None:
                e2 = hf._e2_of_change(dm, vhf, asarray(vj_last), vhf_last.e2)
            vhf += asarray(vhf_last.vj)
            # the exchange part of vhf, kept apart (see rks.get_veff)
            if vk_last is None:
                vk = None
            else:
                vk += asarray(vk_last)
        vxc += vhf
        if e2 is None:
            e2_trace = float(cupy.einsum('nij,nji->', dm_orig, vhf).real.get()) * .5
            exc += e2_trace
            if second_order and vj_last is None:
                e2 = e2_trace
        else:
            exc += e2
        if ecoul is not None:
            exc -= ecoul
        log.timer_debug1('vk', *cput2)
    else:
        if vj_last is not None:
            vhf += asarray(vhf_last.vj)
        vxc += vhf
        if second_order:
            e2 = ecoul
    t0 = log.timer('veff', *t0)
    vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vhf, vk=vk, e2=e2)
    return vxc


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if not (isinstance(dm, cupy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    return rks.energy_elec(ks, dm, h1e, vhf)


class UKS(rks.KohnShamDFT, uhf.UHF):
    def __init__(self, mol, xc='LDA,VWN'):
        uhf.UHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)

    get_veff = get_veff
    get_vasp = uks_cpu.get_vsap
    energy_elec = energy_elec
    energy_tot = hf.RHF.energy_tot
    init_guess_by_vsap = uks_cpu.UKS.init_guess_by_vsap

    def to_hf(self):
        '''Convert to UHF object.'''
        from gpu4pyscf import scf
        return self._transfer_attrs_(scf.UHF(self.mol))

    def Gradients(self):
        from gpu4pyscf.grad import uks as uks_grad
        return uks_grad.Gradients(self)

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        mf = uks_cpu.UKS(self.mol, xc=self.xc)
        utils.to_cpu(self, mf)
        return mf
