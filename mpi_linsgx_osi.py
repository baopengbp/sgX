#!/usr/bin/env python
# LinsgX
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
linear semi-grid eXchange with direct DF-J and differencial density matrix

To lower the scaling of exchange matrix construction for large system, one 
coordinate is analitical and the other is grid. The traditional two electron 
integrals turn to analytical one electron integrals and numerical integration 
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39 and
Neese et. al. Chem. Phys. 2009, 356, 98)

Select a batch of grids in an atom, then use the AO value to screen u and
use u and overlap matrix to screen v.

Minimizing numerical errors using overlap fitting correction.(see 
Lzsak, R. et. al. J. Chem. Phys. 2011, 135, 144105)
Grid screening for weighted AO value and DktXkg. 
Two SCF steps: coarse grid then fine grid. There are several parameters can be changed:
# threshold for u and v
gthrdu = 1e-10
# initial and final grids level
grdlvl_i = 0
grdlvl_f = 1
# norm_ddm threshold for grids change
thrd_nddm = 0.2
# set block size to adapt memory 
sblk = 200
# interspace betweeen v shell
intsp = 1

Set mf.direct_scf = False because no traditional 2e integrals

'MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 mpirun -np 28 python mpi_linsgx_osi.py'

Because there is no omp parralel for concatenate, hstack and vstack in NumPy, omp version is slow

TODO: generate sblk grids in a very small district, grids screening by XF and XAF, grids load banlence.
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

### use 1/10 of the block to adapt memory
    BLKSIZE = min(80, (stop-start)//10+1)  
    for p0, p1 in lib.prange(start,stop, BLKSIZE):
        shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+p0, mol.nbas+p1)
        buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                  aosym, ao_loc, cintopt)
        eri1 = numpy.asarray(buf.T, order='C')
        yield eri1

# pre for get_k
# Use default mesh grids and weights
def get_gridss(mol,lvl, sblk):
    mfg = dft.RKS(mol)
    mfg.grids.level = lvl
    grids = mfg.grids
    crdnum = numpy.asarray([])
    if rank == 0:
        grids.build()
        print('ngrids is',grids.coords.shape)
        crdnum = numpy.array_split(numpy.asarray(range(grids.coords.shape[0])), mpi.pool.size)
        grids.coords = numpy.array_split(grids.coords, mpi.pool.size)
        grids.weights = numpy.array_split(grids.weights, mpi.pool.size)
    crdnum = mpi.scatter(crdnum)
    grids.coords = mpi.scatter(grids.coords)
    grids.weights = mpi.scatter(grids.weights)

    coords0 = mfg.grids.coords
    ngrids0 = coords0.shape[0]
    weights = mfg.grids.weights
    ao_v = dft.numint.eval_ao(mol, coords0)
    # wao=w**0.5 * ao
    xx = numpy.sqrt(abs(weights)).reshape(-1,1)    
    wao_v0 = xx*ao_v   
    # split grids by atom then by sblk
    mfa = dft.gen_grid.Grids(mol)
    atom_grids_tab = mfa.gen_atomic_grids(mol, level = lvl) 
    aaa=0
    gridatm = [0]   
    for ia in range(mol.natm):
        coordsa, vol = atom_grids_tab[mol.atom_symbol(ia)]
        aaa += coordsa.shape[0]
        gridatm.append(aaa) 

    gridatm = numpy.intersect1d(crdnum, numpy.asarray(gridatm))        
    gridatm -= numpy.asarray([crdnum[0]]*gridatm.shape[0])
    gridatm = numpy.unique(numpy.append(gridatm,[0,crdnum.shape[0]]))        
    hsblk = sblk // 2
    for ii in range(gridatm.shape[0]-1):
        i = gridatm[ii]
        while i+sblk < gridatm[ii+1]-hsblk:
            i += sblk
            gridatm = numpy.append(gridatm, i)
    gridatm = numpy.unique(gridatm) 
    return wao_v0, ngrids0, coords0, gridatm  

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

#@profile
@mpi.parallel_call(skip_args=[1])
def get_jk(mol_or_mf, dm, hermi, dmcur, *args, **kwargs):
    '''MPI version of scf.hf.get_jk function'''
    #vj = get_j(mol_or_mf, dm, hermi)
    #vk = get_k(mol_or_mf, dm, hermi)
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(isinstance(dm, str) and dm == 'SKIPPED_ARG')):
        dm = mpi.bcast_tagged_array_occdf(dm)

    mf.unpack_(comm.bcast(mf.pack()))

# initial and final grids level
    grdlvl_i = 0
    grdlvl_f = 1
# norm_ddm threshold for grids change
    thrd_nddm = 0.2
# set block size to adapt memory 
    sblk = 200
# interspace betweeen v shell
    intsp = 1
# threshold for u 
    gthrdu = 1e-10

    global cond, wao_vx, ngridsx, coordsx, gridatm

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
    global int2c, ovlp, ao_loc, rao_loc
# use mf.opt to calc int2c once, cond, dm0, and rao, ao_loc, ovlp for sgX
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
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
        #print('dsssa',mol.nbas, ao_loc.shape,ao_loc[0],ao_loc[-1],ao_loc[1],ao_loc[2],ao_loc[3],ao_loc[115])
        rao_loc = numpy.zeros((nao),dtype=int)
        for i in range(mol.nbas):
            for j in range(ao_loc[i],ao_loc[i+1]):
                rao_loc[j] = i 
        ovlp = mol.intor_symmetric('int1e_ovlp')

        if rank == 0: print('thrd_nddm',thrd_nddm, 'sblk',sblk, 'intsp',intsp, 'gthrdu',gthrdu)

# coase and fine grids change
    grdchg = 0
    norm_ddm = 0
    for k in range(nset):
        norm_ddm += numpy.linalg.norm(dms[k])
    if norm_ddm < thrd_nddm and cond == 2 :
        cond = 1
    if cond == 0:
        wao_vx, ngridsx, coordsx, gridatm  = get_gridss(mol,grdlvl_i, sblk)
        if rank == 0: print('grids level at first is', grdlvl_i) 
        cond = 2
    elif cond == 1:
        wao_vx, ngridsx, coordsx, gridatm = get_gridss(mol,grdlvl_f, sblk)
        if rank == 0: print('grids level change to', grdlvl_f)
        dms = numpy.asarray(dmcur)
        dms = dms.reshape(-1,nao,nao)
        grdchg = 1
        cond = 3

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

# sgX
    wao_v = wao_vx
    coords = coordsx
    for k in range(nset):
# Kuv = Sum(Xug Avt Dkt Xkg)
        ngrids = coords.shape[0]
        for ii in range(gridatm.shape[0]-1):
            i0 = gridatm[ii]
            i1 = gridatm[ii+1]
            # screening u by value of grids 
            umaxg = numpy.amax(numpy.absolute(wao_v[i0:i1]), axis=0)
            usi = numpy.argwhere(umaxg > gthrdu).reshape(-1)
            # screening v by ovlp then triangle matrix bn
            uovl = ovlp[usi, :]
            vmaxu = numpy.amax(numpy.absolute(uovl), axis=0)
            vsi = numpy.argwhere(vmaxu > gthrdu).reshape(-1) 
            if len(vsi) != 0:
                vsh = numpy.unique(rao_loc[vsi])          
                #vshbeg = vsh[0]
                vshfin = vsh[-1]+1
                # use gap between continurous v to save time
                vsh1 = vsh
                vsh1= numpy.delete(vsh1, 0)
                vsh1 = numpy.append(vsh1, [vshfin])
                vshd = numpy.argwhere(vsh1-vsh > intsp)
                vshd = numpy.append(vshd, vsh.shape[0]-1)
                nvshd = vshd.shape[0]                 
                #vbeg = ao_loc[vshbeg]
                vfin = ao_loc[vshfin]
                fakemol = gto.fakemol_for_charges(coords[i0:i1])
                pmol = gto.mole.conc_mol(mol, fakemol)
                bn = []
                dmsk = [] 
                bntp = [[0 for col in range(nvshd)] for row in range(nvshd)]        
                for i in range(nvshd):
                    if i==0:
                        ii0 = vsh[0]
                        ii1 = vsh[vshd[0]]+1
                    else:
                        ii0 = vsh[vshd[i-1]+1]
                        ii1 = vsh[vshd[i]]+1
                    dmsk.append(dms[k][:,ao_loc[ii0]:ao_loc[ii1]])
                    bnh = []
                    for j in range(0, i):                       
                        bnh.append(bntp[j][i].swapaxes(0,1))
                    for j in range(i, nvshd):
                        if j==0:
                            jj0 = vsh[0]
                            jj1 = vsh[vshd[0]]+1
                        else:
                            jj0 = vsh[vshd[j-1]+1]
                            jj1 = vsh[vshd[j]]+1
                        shls_slice = (ii0, ii1, jj0, jj1, mol.nbas, mol.nbas+fakemol.nbas)
                        bntp[i][j] = pmol.intor(intor='int3c2e', comp=1, aosym='s1', shls_slice=shls_slice)
                        bnh.append(bntp[i][j])
                    bnrow = numpy.concatenate(bnh, axis=1)
                    bn.append(bnrow)                 
                bn = numpy.concatenate(bn, axis=0)
                abn = numpy.absolute(bn)
                #if cond==3: print(rank,'wet',numpy.amax(abn), numpy.median(abn))
                dmsk = numpy.asarray(numpy.hstack(dmsk))
                fg = numpy.dot(wao_v[i0:i1,usi],dmsk[usi])      
                gv = lib.einsum('vtg,gt->gv', bn, fg) 
                vk0 = numpy.zeros((nao,nao))
                vksp = lib.einsum('gu,gv->uv', wao_v[i0:i1,usi], gv)
                blen = 0
                for i in range(nvshd):
                    if i==0:
                        ii0 = vsh[0]
                        ii1 = vsh[vshd[0]]+1
                    else:
                        ii0 = vsh[vshd[i-1]+1]
                        ii1 = vsh[vshd[i]]+1
                    baa = ao_loc[ii1]-ao_loc[ii0]
                    vk0[usi,ao_loc[ii0]:ao_loc[ii1]] = vksp[:,blen:(blen+baa)]
                    blen += baa
                vk[k] += vk0
            else:
                vk0 = numpy.zeros((nao,nao))
                vk[k] += vk0 
        sn = lib.einsum('gu,gv->uv', wao_v, wao_v)
        vk[k] = comm.reduce(vk[k])
        sn = comm.reduce(sn)
        # SSn^-1 for grids to analitic
        if rank == 0:
            snsgk = scipy.linalg.solve(sn, vk[k])
            vk[k] = numpy.matmul(ovlp, snsgk)

    if rank == 0:
        vj = lib.unpack_tril(numpy.asarray(vj), 1).reshape(dm_shape)
        vk = numpy.asarray(vk).reshape(dm_shape)

    #if cond==3: cond=4

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
    import mpi_linsgx_osi as sgx
        
    mol = gto.M(atom='''
  H        0.00000        0.00000        0.00000
  C        1.10000        0.00000        0.00000
  C        1.61406        1.45167        0.00000
  C        3.15406        1.45167        0.00000
  C        3.66813        2.90334        0.00000
  C        5.20813        2.90334        0.00000
  C        5.72219        4.35500       -0.00000
  C        7.26219        4.35500        0.00000
  C        7.77625        5.80667       -0.00000
  C        9.31625        5.80667        0.00000
  C        9.83031        7.25834       -0.00000
  C       11.37031        7.25834        0.00000
  C       11.88438        8.71001       -0.00000
  C       13.42438        8.71001        0.00000
  C       13.93844       10.16168       -0.00000
  C       15.47844       10.16168        0.00000
  C       15.99250       11.61334       -0.00000
  C       17.53250       11.61334        0.00000
  C       18.04656       13.06501       -0.00000
  C       19.58656       13.06501        0.00000
  C       20.10063       14.51668       -0.00000
  H        1.46615       -0.51919       -0.89799
  H        1.46615       -0.51919        0.89799
  H        1.24792        1.97086       -0.89799
  H        1.24792        1.97086        0.89799
  H        3.52125        0.93322        0.89799
  H        3.52125        0.93322       -0.89799
  H        3.30198        3.42253       -0.89799
  H        3.30198        3.42253        0.89799
  H        5.57531        2.38488        0.89799
  H        5.57531        2.38488       -0.89799
  H        5.35604        4.87419       -0.89799
  H        5.35604        4.87419        0.89799
  H        7.62938        3.83655        0.89799
  H        7.62938        3.83655       -0.89799
  H        7.41010        6.32586       -0.89799
  H        7.41010        6.32586        0.89799
  H        9.68344        5.28822        0.89799
  H        9.68344        5.28822       -0.89799
  H        9.46417        7.77753       -0.89799
  H        9.46417        7.77753        0.89799
  H       11.73750        6.73989        0.89799
  H       11.73750        6.73989       -0.89799
  H       11.51823        9.22920       -0.89799
  H       11.51823        9.22920        0.89799
  H       13.79156        8.19155        0.89799
  H       13.79156        8.19155       -0.89799
  H       13.57229       10.68086       -0.89799
  H       13.57229       10.68086        0.89799
  H       15.84563        9.64322        0.89799
  H       15.84563        9.64322       -0.89799
  H       15.62636       12.13253       -0.89799
  H       15.62636       12.13253        0.89799
  H       17.89969       11.09489        0.89799
  H       17.89969       11.09489       -0.89799
  H       17.68042       13.58420       -0.89799
  H       17.68042       13.58420        0.89799
  H       19.95375       12.54656        0.89799
  H       19.95375       12.54656       -0.89799
  H       21.20063       14.51668        0.00000
  H       19.73448       15.03587       -0.89799
  H       19.73448       15.03587        0.89799
''', basis='ccpvdz')

    print('basis=',mol.basis,'nao',mol.nao)

    Jtime=time.time()
    #mf = mpi_scf.RHF(mol)
    mf = scf.RHF(mol)
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










