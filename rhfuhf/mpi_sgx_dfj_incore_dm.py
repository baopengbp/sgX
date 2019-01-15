#!/usr/bin/env python
# sgX
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
semi-grid eXchange with DF-J incore and without differencial density matrix

To lower the scaling of exchange matrix construction for large system, one 
coordinate is analitical and the other is grid. The traditional two electron 
integrals turn to analytical one electron integrals and numerical integration 
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39)

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

'MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 mpirun -np 7 python mpi_sgx_dfj_incore_dm.py'
'''

from pyscf.scf import hf
from pyscf.scf import jk
from pyscf.scf import _vhf
import sys
import copy
import time
import ctypes
from functools import reduce
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
import numpy
import scipy.linalg
from pyscf import gto, scf, dft, lib
from pyscf.df import addons
from pyscf import df
from pyscf.scf import uhf
import pyscf

import time
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.scf import hf
from pyscf.scf import jk
from pyscf.scf import _vhf
from pyscf import df
from pyscf import dft

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

def loop(self, blksize=None):
# direct  blocksize
    mol = self.mol
    auxmol = self.auxmol = addons.make_auxmol(self.mol, self.auxbasis)
#    if auxmol is None:
#        auxmol = make_auxmol(mol, auxbasis)
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

#    if blksize is None:
#        max_memory = (self.max_memory - lib.current_memory()[0]) * .8
#        blksize = min(int(max_memory*1e6/48/nao/(nao+1)), 80)
#        print('blksize1',blksize,int(max_memory*1e6/48/nao/(nao+1)),max_memory,self.max_memory)
    comp = 1
    aosym = 's2ij'

#    for i in range(rank*local_num, (rank + 1)*local_num):
    segsize = (naoaux+mpi.pool.size-1) // mpi.pool.size
    global pauxz,paux
    aa = 0
    while naoaux-aa > segsize-1:
        aa = 0
        j = 0
        b1 = []
        pauxz = []
        paux = []
        for i in range(auxmol.nbas+1):
            if ao_loc[mol.nbas+i]-nao-aa > segsize and j < mpi.pool.size-1:
                paux.append(ao_loc[mol.nbas+i-1]-nao-aa)
                aa = ao_loc[mol.nbas+i-1]-nao
                pauxz.append(aa)
                b1.append(i-1)
                j += 1
        if naoaux-aa <= segsize:
            b1.append(auxmol.nbas)
            paux.append(naoaux-aa)            
            pauxz.append(ao_loc[-1])
        segsize += 1
    stop = b1[rank]    
    if rank ==0:
        start = 0
    else:
       start = b1[rank-1]

    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas+start, mol.nbas+stop)
    buf = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                  aosym, ao_loc, cintopt)
    global eri1
    eri1 = numpy.asarray(buf.T, order='C')
#    print(rank,'bbb',eri1.shape)
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
    sngds = []
    ss = 0
    for i in range(ngrids0):
        if numpy.amax(numpy.absolute(wao_v0[i,:])) < gthrd:  
            sngds.append(i)              
            ss += 1
    wao_vx = numpy.delete(wao_v0, sngds, 0)
    coordsx = numpy.delete(coords0, sngds, 0)
#    print ("Took this long for Xg screening: ", time.time()-Ktime)
    ngridsx = coordsx.shape[0]
    return wao_vx, ngridsx, coordsx, gthrd

def batch_nuc(mol, grid_coords, out=None):
    fakemol = gto.fakemol_for_charges(grid_coords)
    j3c = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s1', out=out)
    return j3c

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
def get_jk(mol_or_mf, dm, hermi=1):
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

    global cond, wao_vx, ngridsx, coordsx, gthrd, dm0

    dms = numpy.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = [0] * nset
    vk = [0] * nset

# DF-J 
    mf.with_df = mf
    mol = mf.mol
    global int2c
# use mf.opt to calc int2c once, cond, dm0
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
        cond = 0
        dm0 = numpy.zeros((nset,nao,nao))
# set auxbasis in input file, need self.auxbasis = None in __init__ of hf.py
#        mf.auxbasis = 'weigend' 
        auxbasis = mf.auxbasis
        auxbasis = comm.bcast(auxbasis)
        mf.auxbasis = comm.bcast(mf.auxbasis)
#        if rank == 0: print('auxbasis',auxbasis)
        auxmol = df.addons.make_auxmol(mol, auxbasis)
# (P|Q)
        int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
        if rank == 0: print('auxmol.basis',auxmol.basis)
    naux0 = int2c.shape[0]
    if rank == 0: print('naux0',naux0)
    with_j=True

# coase and fine grids change
    norm_ddm = 0
    for k in range(nset):
        norm_ddm += numpy.linalg.norm(dms[k]-dm0[k])
    dm0 = dms
    if norm_ddm < thrd_nddm and cond == 2 :
        cond = 1
    if cond == 0:
        wao_vx, ngridsx, coordsx, gthrd  = get_gridss(mol,grdlvl_i)
        if rank == 0: print('grids level at first is', grdlvl_i) 
        cond = 2
    elif cond == 1 :
        wao_vx, ngridsx, coordsx, gthrd = get_gridss(mol,grdlvl_f)
        if rank == 0: print('grids level change to', grdlvl_f)
        cond = 3

    if 0==0:
        dmtril = []
        for k in range(nset):
            if with_j:
                dmtril.append(lib.pack_tril(dms[k]+dms[k].T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5
        rho = []
        for eri1 in loop(mf.with_df):
            naux, nao_pair = eri1.shape
#            print('slice-naux',naux,'rank',rank,auxmol.basis)
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    rho.append(numpy.dot(eri1, dmtril[k]))
#        global paux
#        paux = []
#        paux = comm.gather(naux)
#        comm.bcast(paux)
        orho = []
        rec = []
        for k in range(nset):
            orho.append(mpi.gather(rho[k]))
            if rank == 0:
                ivj0 = scipy.linalg.solve(int2c, orho[k])
            else:
                ivj0 = None
            rec.append(numpy.empty(naux))
            comm.Scatterv([ivj0,paux],rec[k],root=0)
#        for eri1 in loop(mf.with_df):
        if 1==1:
            naux, nao_pair = eri1.shape
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    vj[k] += numpy.dot(rec[k].T, eri1)
#                    vj[k] += numpy.tensordot(rec[k], eri1, axes=([0],[0]))
        for k in range(nset):
            vj[k] = comm.reduce(vj[k])

    for k in range(nset):
# screening from Fg
        fg = numpy.dot(wao_vx, dms[k])
        sngds = []
        ss = 0
        for i in range(ngridsx):
            if numpy.amax(numpy.absolute(fg[i,:])) < gthrd:  
                sngds.append(i)              
                ss += 1
        if ss < ngridsx: 
            wao_v = numpy.delete(wao_vx, sngds, 0)
            fg = numpy.delete(fg, sngds, 0)
            coords = numpy.delete(coordsx, sngds, 0)
        else:
            wao_v = wao_vx
            coords = coordsx

# Kuv = Sum(Xug Avt Dkt Xkg)
        ngrids = coords.shape[0]
        blksize = min(ngrids, sblk)
        for i0, i1 in lib.prange(0, ngrids, blksize):
            bn=batch_nuc(mol, coords[i0:i1])
            gbn = bn.swapaxes(0,2)
#            jg = numpy.dot(gbn.reshape(-1,nao*nao), dms[k].reshape(-1))
#            xj = lib.einsum('gv,g->gv', wao_v[i0:i1], jg)
#            vj[k] += lib.einsum('gu,gv->uv', wao_v[i0:i1], xj)
            gv = lib.einsum('gvt,gt->gv', gbn, fg[i0:i1]) 
            vk[k] += lib.einsum('gu,gv->uv', wao_v[i0:i1], gv)
        sn = lib.einsum('gu,gv->uv', wao_v, wao_v)
#    comm.Barrier()
#        vj[k] = comm.reduce(vj[k])
        vk[k] = comm.reduce(vk[k])
        sn = comm.reduce(sn)
        # SSn^-1 for grids to analitic
        if rank == 0:
#        sn = lib.einsum('gu,gv->uv', wao_v, wao_v)
            snsgk = scipy.linalg.solve(sn, vk[k])
            ovlp = mol.intor_symmetric('int1e_ovlp')
            vk[k] = numpy.matmul(ovlp, snsgk)
#            snsgj = scipy.linalg.solve(sn, vj[k])
#            vj[k] = numpy.matmul(ovlp, snsgj)
    if rank == 0:
        vj = lib.unpack_tril(numpy.asarray(vj), 1).reshape(dm_shape)
        vk = numpy.asarray(vk).reshape(dm_shape)
    return vj, vk

@mpi.register_class
class SCF(hf.SCF):

    @lib.with_doc(hf.SCF.get_jk.__doc__)
    def get_jk(self, mol, dm, hermi=1):
        assert(mol is self.mol)
        return get_jk(self, dm, hermi)

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
    get_jk = SCF.get_jk

if __name__ == '__main__':
    import time
    import numpy
    from pyscf import gto, scf, lib
    from mpi4pyscf import scf as mpi_scf
    from pyscf import dft
    from pyscf.lib import logger
    from mpi4pyscf.lib import logger
    lib.logger.TIMER_LEVEL = 0
    import mpi_sgx_dfj_incore_dm as sgx
        
    mol = gto.M(atom='''C     0.      0.      0.    
                      C     0.8917  0.8917  0.8917
                      C     1.7834  1.7834  0.    
                      C     2.6751  2.6751  0.8917
                      C     1.7834  0.      1.7834
                      C     2.6751  0.8917  2.6751
                      C     0.      1.7834  1.7834
                      C     0.8917  2.6751  2.6751''',
#            charge=-1,
#            spin=1,
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
    mf = sgx.RHF(mol)
    mf.auxbasis = 'weigend'
    mf.direct_scf = False
    mf.verbose = 4
    mf.conv_tol = 2e-5
    mf.max_cycle = 50
    mf.kernel()
    print "Took this long for Rs: ", time.time()-Jtime










