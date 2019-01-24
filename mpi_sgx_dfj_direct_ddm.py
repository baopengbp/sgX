#!/usr/bin/env python
# sgX
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
semi-grid eXchange with direct DF-J and differencial density matrix

To lower the scaling of exchange matrix construction for large system, one 
coordinate is analitical and the other is grid. The traditional two electron 
integrals turn to analytical one electron integrals and numerical integration 
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39 and
Neese et. al. Chem. Phys. 2009, 356, 98)

Minimizing numerical errors using overlap fitting correction.(see 
Lzsak, R. et. al. J. Chem. Phys. 2013, 139, 094111)
Grid screening for weighted AO value and DktXkg. 
Two SCF steps: coarse grid then fine grid. There are 5 parameters can be changed:
# threshold for Xg and Fg screening
gthrd = 1e-10
# initial and final grids level
grdlvl_i = 0
grdlvl_f = 1
# norm_ddm threshold for grids change
thrd_nddm = 0.03
# set block size to adapt memory 
sblk = 200

Set mf.direct_scf = False because no traditional 2e integrals

'MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 mpirun -np 7 python mpi_sgx_dfj_direct_ddm.py'
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
def get_gridss(mol,lvl):
    mfg = dft.RKS(mol)
    mfg.grids.level = lvl
    grids = mfg.grids
    if rank == 0:
        grids.build()
        print('ngrids is',grids.coords.shape)
        grids.coords = numpy.array_split(grids.coords, mpi.pool.size)
        grids.weights = numpy.array_split(grids.weights, mpi.pool.size)
    grids.coords = mpi.scatter(grids.coords)
    grids.weights = mpi.scatter(grids.weights)

    coords0 = mfg.grids.coords
    ngrids0 = coords0.shape[0]
    weights = mfg.grids.weights
    ao_v = dft.numint.eval_ao(mol, coords0)
    # wao=w**0.5 * ao
    xx = numpy.sqrt(abs(weights)).reshape(-1,1)    
    wao_v0 = xx*ao_v   
    
#    Ktime=time.time()
    # threshold for Xg and Fg
    gthrd = 1e-10
    if rank == 0: print('threshold for grids screening', gthrd)
    fg0 = numpy.amax(numpy.absolute(wao_v0), axis=1)
    sngds = numpy.argwhere(fg0 < gthrd)
    wao_vx = numpy.delete(wao_v0, sngds, 0)
    coordsx = numpy.delete(coords0, sngds, 0)
#    print ("Took this long for Xg screening: ", time.time()-Ktime)
    ngridsx = coordsx.shape[0]
    return wao_vx, ngridsx, coordsx, gthrd

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
    thrd_nddm = 0.03
# set block size to adapt memory 
    sblk = 200

    global cond, wao_vx, ngridsx, coordsx, gthrd

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

# DF-J set
    mf.with_df = mf
    mol = mf.mol
    global int2c
# use mf.opt to calc int2c once, cond, dm0
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
        if rank == 0: print('auxmol.basis',auxmol.basis)

# coase and fine grids change
    grdchg = 0
    norm_ddm = 0
    for k in range(nset):
        norm_ddm += numpy.linalg.norm(dms[k])
    if norm_ddm < thrd_nddm and cond == 2 :
        cond = 1
    if cond == 0:
        wao_vx, ngridsx, coordsx, gthrd  = get_gridss(mol,grdlvl_i)
        if rank == 0: print('grids level at first is', grdlvl_i) 
        cond = 2
    elif cond == 1:
        wao_vx, ngridsx, coordsx, gthrd = get_gridss(mol,grdlvl_f)
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
        # if rank==0: print('slice-naux',naux,'rank',rank)
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
    for k in range(nset):
# screening from Fg
        fg = numpy.dot(wao_vx, dms[k])
        fg0 = numpy.amax(numpy.absolute(fg), axis=1)
        sngds = numpy.argwhere(fg0 < gthrd)
        if sngds.shape[0] < ngridsx: 
            wao_v = numpy.delete(wao_vx, sngds, 0)
            fg = numpy.delete(fg, sngds, 0)
            coords = numpy.delete(coordsx, sngds, 0)
        else:
            wao_v = wao_vx
            coords = coordsx
        # rescatter data
        coords = mpi.gather(coords)
        wao_v = mpi.gather(wao_v)
        fg = mpi.gather(fg)
        if rank == 0:
            coords = numpy.array_split(coords, mpi.pool.size)
            wao_v = numpy.array_split(wao_v, mpi.pool.size)
            fg = numpy.array_split(fg, mpi.pool.size)
        coords = mpi.scatter(coords)
        wao_v = mpi.scatter(wao_v)
        fg = mpi.scatter(fg)
        
# Kuv = Sum(Xug Avt Dkt Xkg)
        ngrids = coords.shape[0]
        blksize = min(ngrids, sblk)
        for i0, i1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1])
            bn = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s1', out=None)
            gbn = bn.swapaxes(0,2)
            gv = lib.einsum('gvt,gt->gv', gbn, fg[i0:i1]) 
            vk[k] += lib.einsum('gu,gv->uv', wao_v[i0:i1], gv)
        sn = lib.einsum('gu,gv->uv', wao_v, wao_v)
        vk[k] = comm.reduce(vk[k])
        sn = comm.reduce(sn)
        # SSn^-1 for grids to analitic
        if rank == 0:
            snsgk = scipy.linalg.solve(sn, vk[k])
            ovlp = mol.intor_symmetric('int1e_ovlp')
            vk[k] = numpy.matmul(ovlp, snsgk)

    if rank == 0:
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
    import mpi_sgx_dfj_direct_ddm as sgx
        
    mol = gto.M(atom='''C     0.      0.      0.    
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.    
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
            charge=-1,
            spin=1,
            basis='ccpvdz')

    print('basis=',mol.basis,'nao',mol.nao)

    Jtime=time.time()
    mf = mpi_scf.RHF(mol)
#    mf = scf.UHF(mol)
    mf.verbose = 4
    mf.max_cycle = 50
#    mf.kernel()
    print "Took this long for intg: ", time.time()-Jtime

    Jtime=time.time()
    mf = sgx.UHF(mol)
    mf.auxbasis = 'weigend'
    mf.direct_scf = False
    mf.verbose = 4
    mf.conv_tol = 2e-5
    mf.max_cycle = 50
    mf.kernel()
    print "Took this long for Rs: ", time.time()-Jtime










