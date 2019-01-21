#!/usr/bin/env python
# sgX
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
very begin semi-grid Coulomb and eXchange without differencial density matrix
To lower the scaling of coulomb and exchange matrix construction for large system, one 
coordinate is analitical and the other is grid. The traditional two electron 
integrals turn to analytical one electron integrals and numerical integration 
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39)

'MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 mpirun -np 16 python test.py'
In test.py, need:
import mpi_sgx_verybegin as sgx
mf = sgx.RHF(mol)
'''

import time
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.scf import hf
from pyscf.scf import jk
from pyscf.scf import _vhf

from pyscf import dft


from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

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
        dm = mpi.bcast_tagged_array(dm)

    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()

    mol = mf.mol    
    mfg = dft.RKS(mol)
    mfg.grids.level =0

    mfg.grids.build()
    coords = mfg.grids.coords
    ngrids = coords.shape[0]
    if rank == 0:
        print('grids level', mfg.grids.level, ngrids)
    weights = mfg.grids.weights
    ao_v = dft.numint.eval_ao(mol, coords)
#    print(ao_v.shape)
    # wao=w**0.5 * ao
    xx = numpy.sqrt(abs(weights)).reshape(-1,1)    
    wao_v = xx*ao_v

#    for i in range(rank*local_num, (rank + 1)*local_num):
    segsize = (ngrids+mpi.pool.size-1) // mpi.pool.size
    start = rank * segsize
    stop = min(ngrids, start+segsize)

    nao = dm.shape[0]
    sgk = numpy.zeros((nao,nao))
    sgj = numpy.zeros((nao,nao))
#    for i in range(ngrids):
    for i in range(start, stop):
        with mol.with_rinv_origin(coords[i]):
            z = mol.intor('int1e_rinv') 
        #sgk += lib.einsum('g,gu,vtg,kt,gk->uv', weights[i0:i1], ao_v[i0:i1], z, dm, ao_v[i0:i1])
        t = wao_v[i].dot(dm)
        v = t.dot(z)
        sgk += numpy.outer(wao_v[i], v)
        uv = numpy.outer(wao_v[i], wao_v[i])
        jg = lib.einsum('kt,kt', dm, z)
        sgj += jg * uv
    comm.Barrier()
    sgj = comm.reduce(sgj)
    sgk = comm.reduce(sgk)
    return sgj, sgk

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
