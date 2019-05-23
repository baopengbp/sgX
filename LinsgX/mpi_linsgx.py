#!/usr/bin/env python
# LinsgX
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
linear scaling semi-grid eXchange with direct DF-J and differencial density matrix

To lower the scaling of exchange matrix construction for large system, one 
coordinate is analitical and the other is grid. The traditional two electron 
integrals turn to analytical one electron integrals and numerical integration 
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39 and
Neese et. al. Chem. Phys. 2009, 356, 98)

Select a batch of grids in an atom, then use the AO value to screen u and
use u and overlap matrix to screen v.

Minimizing numerical errors using overlap fitting correction(only effective for energy 
of L0 grids. No effect for energy of L1 grids and convergent density of L0 grids).(see 
Lzsak, R. et. al. J. Chem. Phys. 2013, 139, 094111)
Grid screening for max(u)max(Fg). 
Two SCF steps: coarse grid then fine grid. There are several parameters can be changed:
# threshold for u and v
gthrdu = 1e-10
gthrdvs = 1e-10
gthrdvd = 1e-10
# initial and final grids level
grdlvl_i = 0
grdlvl_f = 1
# norm_ddm threshold for grids change
thrd_nddm = 0.2
# set block size to adapt memory 
sblk = 100

Set mf.direct_scf = False because no traditional 2e integrals

'MKL_NUM_THREADS=7 OMP_NUM_THREADS=7 mpirun -np 4 python mpi_linsgx.py'

or openMP: 'MKL_NUM_THREADS=28 OMP_NUM_THREADS=28 python mpi_linsgx.py'
'''

import time
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.scf import hf
from pyscf.df import addons
from pyscf.scf import uhf
from pyscf.scf import jk
from pyscf.scf import _vhf
from pyscf import df
from pyscf import dft
from pyscf.df.incore import aux_e2

from moleintor_sgx import getints3c_scr

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

def loop(self):
# direct  blocksize
    mol = self.mol
    auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
    if auxmol is None:
        auxmol = make_auxmol(mol, auxbasis)
    int3c='int3c2e'
    int3c = mol._add_suffix(int3c)
    int3c = gto.moleintor.ascint3(int3c)
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]
    naoaux = ao_loc[-1] - nao
        # TODO: Libcint-3.14 and newer version support to compute int3c2e without
    # the opt for the 3rd index.
    #if '3c2e' in int3c:
    #    cintopt = gto.moleintor.make_cintopt(atm, mol._bas, env, int3c)
    #else:
    #    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)

    comp = 1
    aosym = 's2ij'
    # banlence the auxbasis by splitting shells
    segsize = (naoaux+mpi.pool.size-1) // mpi.pool.size
    global paux
    aa = 0
    while naoaux-aa > segsize-1:
        aa = 0
        j = 0
        b1 = []
        paux = []
        for i in range(auxmol.nbas+1):
            if ao_loc[mol.nbas+i]-nao-aa > segsize and j < mpi.pool.size-1:
                paux.append(ao_loc[mol.nbas+i-1]-nao-aa)
                aa = ao_loc[mol.nbas+i-1]-nao
                b1.append(i-1)
                j += 1
        if naoaux-aa <= segsize:
            b1.append(auxmol.nbas)
            paux.append(naoaux-aa)
            # average the last two to two+losted
            if len(b1) != mpi.pool.size: 
                nb1 = len(b1)
                nbl2 = b1[nb1-1] - b1[nb1-3]
                vb0 = b1[nb1-3]
                b1 = b1[:nb1-2]
                paux = paux[:nb1-2]
                segs = nbl2 // (mpi.pool.size - nb1 +2)
                for i in range(mpi.pool.size - nb1 +1):
                    vb1 = b1[nb1-3] + segs * (i+1)
                    b1.append(vb1)
                    paux.append(ao_loc[mol.nbas+vb1] - ao_loc[mol.nbas+vb0])
                    vb0 = vb1
                vb1 = b1[mpi.pool.size-2] + nbl2 - (mpi.pool.size - nb1 +1)*segs
                b1.append(vb1)
                paux.append(ao_loc[mol.nbas+vb1] - ao_loc[mol.nbas+b1[mpi.pool.size-2]])
        segsize += 1
    stop = b1[rank]    
    if rank ==0:
        start = 0
    else:
       start = b1[rank-1]
    if start == stop: 
        print('rank=',rank, '!!!!!- no shell in this rank, need less threads -!!!!!')
        print('aux basis',paux)
        print('aux shell',b1)
### use 1/10 of the block to adapt memory
    BLKSIZE = min(80, (stop-start)//10+1)
    #if rank==0: print('blksize for rij=', 10)  
    for p0, p1 in lib.prange(start,stop, BLKSIZE):
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+p0, mol.nbas+p1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                  aosym, ao_loc, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1

# pre for get_k
# Use default mesh grids and weights
def get_gridss(mol,lvl, sblk):
    grids = dft.gen_grid.Grids(mol)
    grids.level = lvl
    if rank == 0:
        grids.build()
        print('ngrids is',grids.coords.shape)
        grids.coords = numpy.array_split(grids.coords, mpi.pool.size)
        grids.weights = numpy.array_split(grids.weights, mpi.pool.size)
    grids.coords = mpi.scatter(grids.coords)
    grids.weights = mpi.scatter(grids.weights)

    coords0 = grids.coords
    ngrids0 = coords0.shape[0]
    weights = grids.weights
    ao_v = dft.numint.eval_ao(mol, coords0)
    wao_v0 = ao_v  
    return wao_v0, ngrids0, coords0, weights  

'''
# need modify bcast_tagged_array(arr) in mpi4pyscf/tools/mpi.py for very big array to:

def bcast_tagged_array_occdf(arr):
#   'Broadcast big nparray or tagged array.'
    if comm.bcast(not isinstance(arr, numpy.ndarray)):
        return comm.bcast(arr)

    new_arr = bcast(arr)

    if comm.bcast(isinstance(arr, lib.NPArrayWithTag)):
        new_arr = lib.tag_array(new_arr)
        if rank == 0:
            kv = []
            for k, v in arr.__dict__.items():
                kv.append((k, v))
            comm.bcast(kv)
        else:
            kv = comm.bcast(None)
            new_arr.__dict__.update(kv)

        for k, v in kv:
            if v is 'NPARRAY_TO_BCAST':
                new_arr.k = bcast(v)

    if rank != 0:
        arr = new_arr
    return arr
'''

@mpi.parallel_call(skip_args=[1])
#@numba.autojit
#@profile
def get_jk(mol_or_mf, dm, hermi, dmcur):
#, *args, **kwargs
    '''MPI version of scf.hf.get_jk function'''
    #vj = get_j(mol_or_mf, dm, hermi)
    #vk = get_k(mol_or_mf, dm, hermi)
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(isinstance(dm, str) and dm == 'SKIPPED_ARG')):
        #dm = mpi.bcast_tagged_array_occdf(dm)
        dm = mpi.bcast_tagged_array(dm)

    mf.unpack_(comm.bcast(mf.pack()))

# initial and final grids level
    grdlvl_i = 0
    grdlvl_f = 1
# norm_ddm threshold for grids change
    thrd_nddm = 0.01
# set block size to adapt memory 
    sblk = 100
# interspace betweeen v shell
    intsp = 1
# threshold for u and v
    gthrdu = 1e-7
    gthrdvs = 1e-4
    gthrdvd = 1e-4

    global cond, wao_vx, ngridsx, coordsx, weightsx

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

# DF-J and sgX set
    mf.with_df = mf
    mol = mf.mol
    global int2c, ovlp, ao_loc, rao_loc, ov_scr
    # need set mf.initsgx = None in scf.SCF __init_
    if mf.initsgx is None:
        mf.initsgx = 0
        cond = 0
# set auxbasis in input file, need self.auxbasis = None in __init__ of hf.py
#        mf.auxbasis = 'weigend' 
        auxbasis = mf.auxbasis
        auxbasis = comm.bcast(auxbasis)
        mf.auxbasis = comm.bcast(mf.auxbasis)
        auxmol = df.addons.make_auxmol(mol, auxbasis)
# (P|Q)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        if rank == 0: print('auxmol.basis',auxmol.basis,'number of aux basis',int2c.shape[0])
# for sgX
        # ao_loc and rao_loc
        intbn = mol._add_suffix('int3c2e')
        intbn = gto.moleintor.ascint3(intbn)
        ao_loc = gto.moleintor.make_loc(mol._bas, intbn)
        rao_loc = numpy.zeros((nao),dtype=int)
        ovlp = mol.intor_symmetric('int1e_ovlp')
        for i in range(mol.nbas):
            for jj in range(ao_loc[i],ao_loc[i+1]):
                rao_loc[jj] = i 
        # ovlp screening
        ov_scr = numpy.zeros((mol.nbas,mol.nbas),dtype=int)
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                if mol.bas_atom(i) == mol.bas_atom(j):
                    ov_scr[i,j] = 1
                else:
                    movlp = numpy.amax(numpy.absolute(ovlp[ao_loc[i]:ao_loc[i+1],ao_loc[j]:ao_loc[j+1]]))  
                    if movlp > gthrdvs:
                        ov_scr[i,j] = 1
        if rank == 0: print('thrd_nddm',thrd_nddm, 'sblk',sblk, 'intsp',intsp, 'gthrdu',gthrdu)
        if rank == 0: print('gthrdvs',gthrdvs, 'gthrdvd',gthrdvd)
# coase and fine grids change
    grdchg = 0
    norm_ddm = 0
    for k in range(nset):
        norm_ddm += numpy.linalg.norm(dms[k])
    if norm_ddm < thrd_nddm and cond == 2 :
        cond = 1
    if cond == 0:
        wao_vx, ngridsx, coordsx, weightsx  = get_gridss(mol,grdlvl_i, sblk)
        if rank == 0: print('grids level at first is', grdlvl_i) 
        cond = 2
    elif cond == 1:
        wao_vx, ngridsx, coordsx, weightsx = get_gridss(mol,grdlvl_f, sblk)
        if rank == 0: print('grids level change to', grdlvl_f)
        dms = numpy.asarray(dmcur)
        dms = dms.reshape(-1,nao,nao)
        grdchg = 1
        cond = 3

    if rank==0: Jtime=time.time()
# DF-J
    dmtril = []
    for k in range(nset):
        dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
        i = numpy.arange(nao)
        dmtril[k][i*(i+1)//2+i] *= .5
    rho = []
    b0 = 0
    for eri1 in loop(mf.with_df):
        naux, nao_pair = eri1.shape
        #if rank==0: print('slice-naux',naux,'rank',rank)
        b1 = b0 + naux
        assert(nao_pair == nao*(nao+1)//2)
        for k in range(nset):
            if b0 == 0: rho.append(numpy.empty(paux[rank]))
            rho[k][b0:b1] = numpy.dot(eri1, dmtril[k])
        b0 = b1
    orho = []
    rec = []
    for k in range(nset):
        orho.append(mpi.gather(rho[k]))
        if rank == 0:
            ivj0 = scipy.linalg.solve(int2c, orho[k])
        else:
            ivj0 = None
        rec.append(numpy.empty(paux[rank]))
        comm.Scatterv([ivj0,paux],rec[k],root=0)
    b0 = 0
    for eri1 in loop(mf.with_df):
        naux, nao_pair = eri1.shape
        b1 = b0 + naux
        assert(nao_pair == nao*(nao+1)//2)
        for k in range(nset):
            vj[k] += numpy.dot(rec[k][b0:b1].T, eri1)
        b0 = b1
    for k in range(nset):
        vj[k] = comm.reduce(vj[k])
    if rank==0: print "Took this long for J: ", time.time()-Jtime

    if rank==0: Jtime=time.time()
# sgX
    wao_v = wao_vx
    coords = coordsx
    weights = weightsx
    for k in range(nset):
        '''# Plus density screening
        ov_scr = numpy.zeros((mol.nbas,mol.nbas),dtype=int)
        for i in range(mol.nbas):
            for j in range(mol.nbas):
                movlp = numpy.amax(numpy.absolute(dms[k][ao_loc[i]:ao_loc[i+1],ao_loc[j]:ao_loc[j+1]]))    
                if movlp > gthrdvd:
                    ov_scr[i,j] = 1'''
# xfg screening
        ngrids = coords.shape[0]
        blksize = min(ngrids, sblk)
        gscr = []
        for i0, i1 in lib.prange(0, ngrids, blksize):
            # screening u by value of grids 
            umaxg = numpy.amax(numpy.absolute(wao_v[i0:i1]), axis=0)
            usi = numpy.argwhere(umaxg > gthrdu).reshape(-1)
            if len(usi) != 0:
                # screening v by ovlp then triangle matrix bn
                uovl = ovlp[usi, :]
                vmaxu = numpy.amax(numpy.absolute(uovl), axis=0)
                osi = numpy.argwhere(vmaxu > gthrdvs).reshape(-1) 
                udms = dms[k][usi, :]
                # screening v by dm and ovlp then triangle matrix bn
                dmaxg = numpy.amax(numpy.absolute(udms), axis=0)
                dsi = numpy.argwhere(dmaxg > gthrdvd).reshape(-1) 
                vsi = numpy.intersect1d(dsi, osi)
                if len(vsi) != 0:
                    vsh = numpy.unique(rao_loc[vsi]) 
                    vshi = []
                    for i in range(vsh.shape[0]):
                        ista = ao_loc[vsh[i]]
                        iend = ao_loc[vsh[i]+1]
                        vshi.append(numpy.arange(ista, iend))
                    vshi = numpy.asarray(numpy.hstack(vshi))
                    dmsi = dms[k][usi]
                    fg = weights[i0:i1,None] * numpy.dot(wao_v[i0:i1,usi],dmsi[:,vshi])
                    gmaxfg = numpy.amax(numpy.absolute(fg), axis=1)
                    gmaxwao_v = numpy.amax(numpy.absolute(wao_v[i0:i1,usi]), axis=1)
                    gmaxtt = gmaxfg * gmaxwao_v
                    gscr0 = numpy.argwhere(gmaxtt > gthrdu).reshape(-1)
                    if gscr0.shape[0] > 0: 
                        gscr.append(gscr0 + i0)
        hgscr = numpy.hstack(gscr).reshape(-1)
        coords = mpi.gather(coords[hgscr])
        wao_v = mpi.gather(wao_v[hgscr])
        weights = mpi.gather(weights[hgscr])
        if rank == 0:
            print('screened grids', coords.shape[0])
            coords = numpy.array_split(coords, mpi.pool.size)
            wao_v = numpy.array_split(wao_v, mpi.pool.size)
            weights = numpy.array_split(weights, mpi.pool.size)
        coords = mpi.scatter(coords)
        wao_v = mpi.scatter(wao_v)
        weights = mpi.scatter(weights)

# Kuv = Sum(Xug Avt Dkt Xkg)
        ngrids = coords.shape[0]
        for i0, i1 in lib.prange(0, ngrids, blksize):
            # screening u by value of grids 
            umaxg = numpy.amax(numpy.absolute(wao_v[i0:i1]), axis=0)
            usi = numpy.argwhere(umaxg > gthrdu).reshape(-1)
            if len(usi) != 0:
                # screening v by ovlp then triangle matrix bn
                uovl = ovlp[usi, :]
                vmaxu = numpy.amax(numpy.absolute(uovl), axis=0)
                osi = numpy.argwhere(vmaxu > gthrdvs).reshape(-1) 
                udms = dms[k][usi, :]
                # screening v by dm and ovlp then triangle matrix bn
                dmaxg = numpy.amax(numpy.absolute(udms), axis=0)
                dsi = numpy.argwhere(dmaxg > gthrdvd).reshape(-1) 
                vsi = numpy.intersect1d(dsi, osi)
                if len(vsi) != 0:
                    vsh = numpy.unique(rao_loc[vsi])
                    nvsh = vsh.shape[0]
                    vov0 = ov_scr[vsh]
                    vov = vov0[:,vsh]  
                    
                    vshi = []
                    xvsh = vsh
                    ivx = [0]
                    vx = 0
                    for i in range(vsh.shape[0]):
                        ista = ao_loc[vsh[i]]
                        iend = ao_loc[vsh[i]+1]
                        vshi.append(numpy.arange(ista, iend))
                        vx += iend - ista
                        ivx.append(vx)
                    vshi = numpy.asarray(numpy.hstack(vshi))
                    nvshi = vshi.shape[0]
                    #print('ee',nvshi)
                    ivx = numpy.asarray(ivx)

                    vshbeg = vsh[0]
                    vshfin = vsh[-1]+1
                    dmsi = dms[k][usi]
                    fg = weights[i0:i1,None] * numpy.dot(wao_v[i0:i1,usi],dmsi[:,vshi])

                    fakemol = gto.fakemol_for_charges(coords[i0:i1])
                    #pmol = gto.mole.conc_mol(mol, fakemol)
                    intor = mol._add_suffix('int3c2e')
                    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                                      fakemol._atm, fakemol._bas, fakemol._env)
                    shls_slice = (vshbeg, vshfin, vshbeg, vshfin, mol.nbas, mol.nbas+fakemol.nbas)
                    comp=1
                    #aosym='s1'
                    aosym='s2ij'
                    if aosym == 's2ij': 
                        gv = getints3c_scr(intor, atm, bas, env, shls_slice, comp,
                                                           xvsh, nvshi, ivx, vov, fg, aosym)
                    else:
                        gv = getints3c_scr(intor, atm, bas, env, shls_slice, comp,
                                                           xvsh, nvshi, ivx, vov, fg, aosym)
                    vk0 = numpy.zeros((nao,nao))
                    vksp = lib.einsum('gu,gv->uv', wao_v[i0:i1,usi], gv)
                    vk1 = vk0[usi]
                    vk1[:,vshi] = vksp
                    vk0[usi] = vk1
                    vk[k] += vk0
        wao_vw = weights[:,None] * wao_v  
        sn = lib.einsum('gu,gv->uv', wao_v, wao_vw)
        vk[k] = comm.reduce(vk[k])
        sn = comm.reduce(sn)
        # SSn^-1 for grids to analitic
        if rank == 0:
            snsgk = scipy.linalg.solve(sn, vk[k])
            vk[k] = numpy.matmul(ovlp, snsgk)
            if hermi == 1:
                vk[k] = (vk[k] + vk[k].T)*.5

    if rank == 0:
        print "Took this long for K: ", time.time()-Jtime
        vj = lib.unpack_tril(numpy.asarray(vj), 1).reshape(dm_shape)
        vk = numpy.asarray(vk).reshape(dm_shape)

    return vj, vk, grdchg

@mpi.register_class
class SCF(hf.SCF):

    @lib.with_doc(hf.SCF.get_veff.__doc__)
    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        grdchg = 0
        vj, vk, grdchg = get_jk(self, ddm, hermi=hermi, dmcur=dm)
        if grdchg == 0 :
            return numpy.asarray(vhf_last) + vj - vk * .5
        else:
            return vj - vk * .5
#    @lib.with_doc(hf.SCF.get_veff.__doc__)
    def get_veffuhf(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.asarray((dm*.5,dm*.5))
        ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
        grdchg = 0
        vj, vk, grdchg = get_jk(self, ddm, hermi=hermi, dmcur=dm)
        if grdchg == 0 :
            vhf = vj[0] + vj[1] - vk
            vhf += numpy.asarray(vhf_last)
        else:
            vhf = vj[0] + vj[1] - vk
        return vhf

    def pack(self):
        return {'verbose': self.verbose,
                'direct_scf_tol': self.direct_scf_tol}
    def unpack_(self, mf_dic):
        self.__dict__.update(mf_dic)
        return self

class RHF(SCF):
    pass

@mpi.register_class
class UHF(uhf.UHF,SCF):
    get_veff = SCF.get_veffuhf

if __name__ == '__main__':
    import time
    import numpy
    from pyscf import gto, scf, lib
    from mpi4pyscf import scf as mpi_scf
    from pyscf import dft
    from pyscf.lib import logger
    from mpi4pyscf.lib import logger
    lib.logger.TIMER_LEVEL = 0
    import mpi_linsgx as sgx
        
    mol = gto.M(atom='''
       H       -0.89830571    0.63485971    0.00000000
       C        0.00000000    0.00000000    0.00000000
       H        0.00000000   -0.63485971    0.89830571
       H        0.00000000   -0.63485971   -0.89830571
       C        1.25762799    0.88880359    0.00000000
       H        1.25762799    1.52366330    0.89830571
       H        1.25762799    1.52366330   -0.89830571
       C        2.51525599    0.00000000    0.00000000
       H        2.51525599   -0.63485971    0.89830571
       H        2.51525599   -0.63485971   -0.89830571
       C        3.77288398    0.88880359    0.00000000
       H        3.77288398    1.52366330    0.89830571
       H        3.77288398    1.52366330   -0.89830571
       C        5.03051198    0.00000000    0.00000000
       H        5.03051198   -0.63485971    0.89830571
       H        5.03051198   -0.63485971   -0.89830571
       C        6.28813997    0.88880359    0.00000000
       H        6.28813997    1.52366330    0.89830571
       H        6.28813997    1.52366330   -0.89830571
       C        7.54576797    0.00000000    0.00000000
       H        7.54576797   -0.63485971    0.89830571
       H        7.54576797   -0.63485971   -0.89830571
       C        8.80339596    0.88880359    0.00000000
       H        8.80339596    1.52366330    0.89830571
       H        8.80339596    1.52366330   -0.89830571
       C       10.06102396    0.00000000    0.00000000
       H       10.06102396   -0.63485971    0.89830571
       H       10.06102396   -0.63485971   -0.89830571
       C       11.31865195    0.88880359    0.00000000
       H       11.31865195    1.52366330    0.89830571
       H       11.31865195    1.52366330   -0.89830571
       H       12.21695767    0.25394388    0.00000000
''', basis='ccpvdz')

    print('basis=',mol.basis,'nao',mol.nao)

    Jtime=time.time()
    mf = mpi_scf.RHF(mol)
    #mf = scf.RHF(mol)
    mf.verbose = 4
    mf.max_cycle = 50
    #mf.kernel()
    print "Took this long for intg: ", time.time()-Jtime

    Jtime=time.time()
    mf = sgx.RHF(mol)
    mf.auxbasis = 'weigend'
    mf.direct_scf = False
    mf.verbose = 4
    mf.conv_tol = 2e-5
    mf.max_cycle = 50
    mf.kernel()
    print "Took this long for Rs: ", time.time()-Jtime










