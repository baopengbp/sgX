#!/usr/bin/env python
# sgX
# Author: Peng Bao <baopeng@iccas.ac.cn>

'''
semi-grid eXchange with DF-J incore and without differencial density matrix

To lower the scaling of exchange matrix construction for large system, one 
coordinate is analitical and the other is grid. The traditional two electron 
integrals turn to analytical one electron integrals and numerical integration 
based on grid.(see Friesner, R. A. Chem. Phys. Lett. 1985, 116, 39 and
Neese et. al. Chem. Phys. 2009, 356, 98)

Minimizing numerical errors using overlap fitting correction.(see 
Lzsak, R. et. al. J. Chem. Phys. 2013, 139, 094111)
Grid screening for weighted AO value and DktXkg. 
Two SCF steps: coarse grid then fine grid. There are 5 parameters can be changed:
# threshold for Xg and Fg
gthrd = 1e-10
# initial and final grids level
grdlvl_i = 0
grdlvl_f = 1
# norm_ddm threshold for grids change
thrd_nddm = 0.03
# set block size to adapt memory 
sblk = 200

Set mf.direct_scf = False because no traditional 2e integrals

'MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 python omp_dfj_incore_dm.py'
'''

import numpy
import scipy.linalg
from pyscf import gto, scf, lib
lib.logger.TIMER_LEVEL = 0
from pyscf import df
from pyscf import dft
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools as pbctools
import time

cell=pbcgto.Cell()

#Molecule
cell.a=[[0., 3.37013733, 3.37013733],
       [3.37013733, 0., 3.37013733],
       [3.37013733, 3.37013733, 0.]]
cell.atom="""
C 0 0 0
C 1.68506866 1.68506866 1.68506866
"""
cell.build()
cell.verbose=5
scell = pbctools.super_cell(cell, [1,1,4])
mol = gto.Mole()
mol.atom = scell._atom
mol.unit = 'Bohr'
mol.basis = 'ccpvdz'
#mol.verbose = 4
mol.build()
#mol.verbose = 6
#, output = '1ewcn.out'
print('basis=',mol.basis,'nao',mol.nao)
# HF reference with analytical two-electron integrals
mf = scf.RHF(mol)
mf.max_memory = 120000
mf.max_cycle = 200
mf.conv_tol = 2e-5
mf.verbose = 4
#mf.kernel()
print('reference HF total energy =', mf.e_tot)

mf = scf.RHF(mol)
# Use density fitting to compute Coulomb matrix
# Build auxmol to hold auxiliary fitting basis
auxbasis = 'ccpvdz-jk-fit'
auxmol = df.addons.make_auxmol(mol, auxbasis)
# (P|Q)
int2c = auxmol.intor('int2c2e', aosym='s1', comp=1)
naux = int2c.shape[0]
# (ij|P)
int3c = df.incore.aux_e2(mol, auxmol, 'int3c2e', aosym='s2ij', comp=1)
nao = mol.nao

# pre for get_k
# Use default mesh grids and weights
def get_gridss(mol,lvl):
    mfg = dft.RKS(mol)
    mfg.grids.level = lvl
    mfg.grids.build()
    coords0 = mfg.grids.coords
    ngrids0 = coords0.shape[0]
    weights = mfg.grids.weights
    ao_v = dft.numint.eval_ao(mol, coords0)
    # wao=w**0.5 * ao
    xx = numpy.sqrt(abs(weights)).reshape(-1,1)    
    wao_v0 = xx*ao_v   
    
    Ktime=time.time()
    # threshold for Xg and Fg
    gthrd = 1e-10
    print('threshold for grids screening', gthrd)
    sngds = []
    ss = 0
    for i in range(ngrids0):
        if numpy.amax(numpy.absolute(wao_v0[i,:])) < gthrd:  
            sngds.append(i)              
            ss += 1
    wao_vx = numpy.delete(wao_v0, sngds, 0)
    coordsx = numpy.delete(coords0, sngds, 0)
    print ("Took this long for Xg screening: ", time.time()-Ktime)
    ngridsx = coordsx.shape[0]
    return wao_vx, ngridsx, coordsx, gthrd

cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
def batch_nuc(mol, grid_coords, out=None):
    fakemol = gto.fakemol_for_charges(grid_coords)
    j3c = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s1', out=out)
    return j3c

# initial and final grids level
grdlvl_i = 0
grdlvl_f = 1
# norm_ddm threshold for grids change
thrd_nddm = 0.03
# set block size to adapt memory 
sblk = 200

cond = 0 
dm0 = numpy.zeros((nao,nao))
# Redefine get_jk function. This function is used by HF/DFT object, to compute
# the effective potential.
#@profile
def get_jk(mol, dm, *args, **kwargs):
# coase and fine grids change
    global cond, wao_vx, ngridsx, coordsx, gthrd, dm0
    norm_ddm = numpy.linalg.norm(dm-dm0)
    dm0 = dm
    if norm_ddm < thrd_nddm and cond == 2 :
        cond = 1
    if cond == 0:
        wao_vx, ngridsx, coordsx, gthrd = get_gridss(mol,grdlvl_i)
        print('grids level at first is', grdlvl_i) 
        cond = 2
    elif cond == 1 :
        wao_vx, ngridsx, coordsx, gthrd = get_gridss(mol,grdlvl_f)
        print('grids level change to', grdlvl_f)
        cond = 3

# RI-J
#    rho = lib.einsum('ijP,ji->P', int3c, dm)
#    vj_P = scipy.linalg.solve(int2c, rho)
#    vj = lib.einsum('ijP,P->ij', int3c, vj_P)
    dmtril = lib.pack_tril(dm+dm.T)
    i = numpy.arange(nao)
    dmtril[i*(i+1)//2+i] *= .5
    rho = numpy.dot(dmtril, int3c)
    vj_P = scipy.linalg.solve(int2c, rho)
    blksize = min(naux, sblk)
    vj = 0
    for i0, i1 in lib.prange(0, naux, blksize):
        vj += numpy.dot(int3c[:,i0:i1], vj_P[i0:i1])
    vj = lib.unpack_tril(vj, 1).reshape(nao,nao)

# screening from Fg
    Ktime=time.time()
    fg = numpy.dot(wao_vx, dm)
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
    ngrids = coords.shape[0]

# Kuv = Sum(Xug Avt Dkt Xkg)
    sgk = numpy.zeros((nao,nao))
    blksize = min(ngrids, int(min(sblk, mf.max_memory*1e6/8/nao**2)))
    for i0, i1 in lib.prange(0, ngrids, blksize):
        bn=batch_nuc(mol, coords[i0:i1])
        gbn = bn.swapaxes(0,2)
        gv = lib.einsum('gvt,gt->gv', gbn, fg[i0:i1]) 
        sgk += lib.einsum('gu,gv->uv', wao_v[i0:i1], gv)
    # SSn^-1 for grids to analitic
    sn = lib.einsum('gu,gv->uv', wao_v, wao_v)
    snsgk = scipy.linalg.solve(sn, sgk)
    ovlp = mol.intor_symmetric('int1e_ovlp')
    vk = numpy.matmul(ovlp, snsgk)
    return vj, vk

# Overwrite the default get_jk to apply the new J/K builder
mf.direct_scf = False
mf.get_jk = get_jk
mf.max_cycle = 50
mf.conv_tol = 2e-5
mf.verbose = 4
mf.kernel()
print('Approximate HF total energy =', mf.e_tot)







