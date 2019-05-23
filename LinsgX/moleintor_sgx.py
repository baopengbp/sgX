#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
#
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Modified for LinsgX from gto/moleintor.py
#

'''
A low level interface to libcint library. It's recommended to use the
Mole.intor method to drive the integral evaluation funcitons.
'''

import warnings
import ctypes
import numpy
from pyscf import lib

libcgto = lib.load_library('libcgto')

ANG_OF     = 1
NPRIM_OF   = 2
NCTR_OF    = 3
KAPPA_OF   = 4
PTR_EXP    = 5
PTR_COEFF  = 6
BAS_SLOTS  = 8

#@profile
def getints3c_scr(intor_name, atm, bas, env, shls_slice=None, comp=1, xvsh=None,
                  nvshi=None, ivx=None, vov=None, fg=None, aosym='s1', 
                  ao_loc=None, cintopt=None, out=None):
    atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    # scr 
    vsh = numpy.asarray(xvsh, dtype=numpy.int32, order='C')
    nvsh = xvsh.shape[0]
    ivx = numpy.asarray(ivx, dtype=numpy.int32, order='C')
    vov = numpy.asarray(vov.reshape(-1), dtype=numpy.int32, order='C')
    fg = numpy.asarray(fg.reshape(-1), dtype=numpy.double, order='C')

    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas)
        if 'ssc' in intor_name or 'spinor' in intor_name:
            bas = numpy.asarray(numpy.vstack((bas,bas)), dtype=numpy.int32)
            shls_slice = (0, nbas, 0, nbas, nbas, nbas*2)
            nbas = bas.shape[0]
    else:
        assert(shls_slice[1] <= nbas and
               shls_slice[3] <= nbas and
               shls_slice[5] <= nbas)

    i0, i1, j0, j1, k0, k1 = shls_slice[:6]
    if ao_loc is None:
        ao_loc = make_loc(bas, intor_name)
        if 'ssc' in intor_name:
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'cart')
        elif 'spinor' in intor_name:
            # The auxbasis for electron-2 is in real spherical representation
            ao_loc[k0:] = ao_loc[k0] + make_loc(bas[k0:], 'sph')

    naok = ao_loc[k1] - ao_loc[k0]

    if aosym in ('s1',):
        naoi = ao_loc[i1] - ao_loc[i0]
        naoj = ao_loc[j1] - ao_loc[j0]
        #shape = (naoi, naoj, naok, comp)
        shape = (nvshi, nvshi, naok, comp)

    else:
        aosym = 's2ij'
        #nij = ao_loc[i1]*(ao_loc[i1]+1)//2 - ao_loc[i0]*(ao_loc[i0]+1)//2
        #nij = (ao_loc[i1]-ao_loc[i0])*(ao_loc[i1]-ao_loc[i0]+1)//2
        nij = nvshi*(nvshi+1)//2
        shape = (nij, naok, comp)

    if 'spinor' in intor_name:
        mat = numpy.ndarray(shape, numpy.complex, out, order='F')
        drv = libcgto.GTOr3c_drv
        fill = getattr(libcgto, 'GTOr3c_fill_'+aosym)
    else:
        mat = numpy.zeros(shape, numpy.double, order='F')
        gv = numpy.zeros((naok,nvshi), dtype=numpy.double, order='C')
        drv = libcgto.GTOnr3c_scr_drv
        fill = getattr(libcgto, 'GTOnr3c_fill_scr_'+aosym)

    if mat.size > 0:
        # Generating opt for all indices leads to large overhead and poor OMP
        # speedup for solvent model and COSX functions. In these methods,
        # the third index of the three center integrals corresponds to a
        # large number of grids. Initializing the opt for the third index is
        # not necessary.
        if cintopt is None:
            if '3c2e' in intor_name:
                # TODO: Libcint-3.14 and newer version support to compute
                # int3c2e without the opt for the 3rd index.
                #cintopt = make_cintopt(atm, bas[:max(i1, j1)], env, intor_name)
                cintopt = lib.c_null_ptr()
            else:
                cintopt = make_cintopt(atm, bas, env, intor_name)
        #print('rtyery')
        drv(getattr(libcgto, intor_name), fill,
            mat.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(comp),
            (ctypes.c_int*6)(*(shls_slice[:6])),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
            env.ctypes.data_as(ctypes.c_void_p),
            vsh.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nvsh),
            ctypes.c_int(nvshi),
            ivx.ctypes.data_as(ctypes.c_void_p),
            vov.ctypes.data_as(ctypes.c_void_p),
            fg.ctypes.data_as(ctypes.c_void_p),
            gv.ctypes.data_as(ctypes.c_void_p))

    return gv

def make_loc(bas, key):
    if 'cart' in key:
        l = bas[:,ANG_OF]
        dims = (l+1)*(l+2)//2 * bas[:,NCTR_OF]
    elif 'sph' in key:
        dims = (bas[:,ANG_OF]*2+1) * bas[:,NCTR_OF]
    else:  # spinor
        l = bas[:,ANG_OF]
        k = bas[:,KAPPA_OF]
        dims = (l*4+2) * bas[:,NCTR_OF]
        dims[k<0] = (l[k<0] * 2 + 2) * bas[k<0,NCTR_OF]
        dims[k>0] = (l[k>0] * 2    ) * bas[k>0,NCTR_OF]

    ao_loc = numpy.empty(len(dims)+1, dtype=numpy.int32)
    ao_loc[0] = 0
    dims.cumsum(dtype=numpy.int32, out=ao_loc[1:])
    return ao_loc




