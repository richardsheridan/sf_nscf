# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:22:37 2015

@author: Richard Sheridan

This module is for small functions that haven't changed in a while and are
cluttering the flow of another module.

Also for functions or classes that have import-time logic associated with them
(2nd half)

Nothing here should have in-package dependencies or the tests will have circular
import problems

"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import exp, log
from scipy.special import gammaln
from collections import OrderedDict

# faster version of numpy.convolve for ndarray only
# This is okay to use as long as LAMBDA_ARRAY is symmetric,
# otherwise a slice LAMBDA_ARRAY[::-1] is necessary
from numpy.core.multiarray import correlate

LAMBDA_1 = np.float64(1.0)/6.0 #always assume cubic lattice (1/6) for now
LAMBDA_0 = 1.0-2.0*LAMBDA_1
LAMBDA_ARRAY = np.array([LAMBDA_1,LAMBDA_0,LAMBDA_1])
MINLAT = 25
MINBULK = 5

def lattice_parameters(a, m, l, p_l):
    """

        l is the real polymer's bond length, m is the real segment mass,
        and a is the ratio between molecular weight and radius of gyration at
        theta conditions. The lattice persistence, p_l, is:

        p_l = 1/6 * (1+1/Z)/(1-1/Z)

        with coordination number Z = 6 for a cubic lattice, p_l = .233.

        >>> a, m, l, p_l = .003, 104, 0.025, .2333333 # Ang, amu, ang, --
        >>> lattice_parameters(a, m, l, p_l)
        (1.6045716577959512, 667.5018096431157)
    """

    if not 0.001 < a < 1:
        raise ValueError('polymer size parameter a out of spec at:', a)
    if not 30 < m < 300:
        raise ValueError('monomer weight m out of spec at:', m)
    if not 0 < l < 100:
        raise ValueError('monomer length l out of spec at:', l)
    if not 0 < p_l < 1:
        raise ValueError('lattice persistence p_l out of spec at:', l)

    l_lat = a**2 * m / l / p_l
    m_lat = (a * m / l)**2 / p_l
    return l_lat, m_lat

class NotImplementedAttribute:
    """http://stackoverflow.com/a/32536493/4504950"""

    def __get__(self, obj, type):
        raise NotImplementedError("This attribute must be set")

_sz_dict = OrderedDict()
def schultz_zimm(pdi,nn,cache=_sz_dict):
    """ Calculate Shultz-Zimm distribution from PDI and number average DP

    Shultz-Zimm is a "realistic" distribution for linear polymers. Numerical
    problems arise when the distribution gets too uniform, so if we find them,
    default to an exact uniform calculation.
    """
    args = pdi,nn
    if args in cache:
        cache.move_to_end(args)
        return cache[args]

    uniform = False

    if pdi == 1.0:
        uniform = True
    elif pdi < 1.0:
        raise ValueError('Invalid PDI',pdi)
    else:
        x = 1.0/(pdi-1.0)
        # Calculate the distribution in chunks so we don't waste CPU time
        chunk = 256
        p_ni_list = []
        pdi_underflow = False

        for i in range(max(1,int((100*nn)/chunk))):
            ni = np.arange(chunk*i+1,chunk*(i+1)+1,dtype=np.float64)
            r = ni/nn
            xr = x*r

            p_ni = exp(log(x/ni) - gammaln(x+1) + xr*(log(xr)/r-1))

            pdi_underflow = (p_ni>=1.0).any() # catch "too small PDI"
            if pdi_underflow: break # and break out to uniform calculation

            # Stop calculating when species account for less than 1ppm
            keep = (r < 1.0) | (p_ni >= 1e-6)
            if keep.all():
                p_ni_list.append(p_ni)
            else:
                p_ni_list.append(p_ni[keep])
                break
        else: # Belongs to the for loop. Executes if no break statement runs.
            raise RuntimeError('SZdist overflow')

    if uniform or pdi_underflow:
        # NOTE: rounding here allows nn to be a double in the rest of the logic
        p_ni = np.zeros(int(round(nn)))
        p_ni[-1] = 1.0
    else:
        p_ni = np.concatenate(p_ni_list)
        p_ni /= p_ni.sum()
    cache[args]=p_ni

    if len(cache)>9000:
        cache.popitem(last=False)

    return p_ni


def sumkd(array, axis=None):
    return np.sum(array, axis=axis, keepdims=True)

def meankd(array, axis=None):
    return np.mean(array, axis=axis, keepdims=True)


class Propagator:

    def __init__(self, g_z, segments):
        self.g_z = g_z
        self.shape = g_z.size, segments

    def ta(self):
        # terminally attached beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = 0.0
        g_zs[0, 0] = self.g_z[0]
        _calc_g_zs_uniform(self.g_z, g_zs)
        return g_zs

    def free(self):
        # free beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = self.g_z
        _calc_g_zs_uniform(self.g_z, g_zs)
        return g_zs

    def ngts_u(self, c):
        # free ends of uniform chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c * self.g_z
        _calc_g_zs_uniform(self.g_z, g_zs)
        return g_zs

    def ngts(self, c_i):
        # free ends of disperse chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c_i[-1] * self.g_z
        _calc_g_zs(self.g_z, c_i, g_zs)
        return g_zs

    def _new(self):
        return np.empty(self.shape, order='F')


""" BEGIN IMPORT-TIME SHENANIGANS """

# compatability for systems lacking compiler capability
CEX = JIT = False
try:
    from numba import njit
    JIT = True
except ImportError:
    try:
        from calc_g_zs_cex import _calc_g_zs_cex, _calc_g_zs_uniform_cex
    except ImportError:
        from warnings import warn
        warn('Compiled inner loop unavailable, using slow-as-molasses version!')
    else:
        CEX = True


if JIT:
    def fastsum(g_zs, axis=0):
        output = np.zeros(g_zs.shape[1])
        _fastsum(g_zs, output)
        return output

    @njit('void(f8[:,:],f8[:])')
    def _fastsum(g_zs, output):
        layers, segments = g_zs.shape
        for s in range(segments):
            for z in range(layers):
                output[s] += g_zs[z, s]

    def compose(g_zs, g_zs_ngts, g_z):
        output = np.zeros_like(g_z)
        _compose(g_zs, g_zs_ngts, output)
        output /= g_z
        return output

    @njit('void(f8[:,:],f8[:,:],f8[:])')
    def _compose(g_zs, g_zs_ngts, output):
        layers, segments = g_zs.shape
        for s in range(segments):
            for z in range(layers):
                if g_zs[z, s] and g_zs_ngts[z, segments-s-1]: # Prevent NaNs
                    output[z] += g_zs[z, s] * g_zs_ngts[z, segments-s-1]

else:
    fastsum = np.sum

    def compose(g_zs, g_zs_ngts, g_z):
        prod = g_zs * np.fliplr(g_zs_ngts)
        prod[np.isnan(prod)] = 0
        return np.sum(prod, axis=1) / g_z


# _calc_g_zs and _calc_g_zs_uniform need a special 3 way branch because of the
# optional c-extension
if JIT:
    @njit('void(f8[:],f8[:],f8[:,:])')
    def _calc_g_zs(g_z,c_i,g_zs):
        layers, segments = g_zs.shape
        for r in range(1, segments):
            c = c_i[segments-r-1]
            g_zs[0, r] = (g_zs[0, r-1] * LAMBDA_0
                          + g_zs[1, r-1] * LAMBDA_1
                          + c) * g_z[0]
            for z in range(1, layers-1):
                g_zs[z, r] = (g_zs[z-1, r-1] * LAMBDA_1
                              + g_zs[z, r-1] * LAMBDA_0
                              + g_zs[z+1, r-1] * LAMBDA_1
                              + c) * g_z[z]
            g_zs[layers-1, r] = (g_zs[layers-1, r-1] * LAMBDA_0
                                 + g_zs[layers-2, r-1] * LAMBDA_1
                                 + c) * g_z[layers-1]

    @njit('void(f8[:],f8[:,:])')
    def _calc_g_zs_uniform(g_z, g_zs):
        layers, segments = g_zs.shape
        for r in range(1, segments):
            g_zs[0, r] = (g_zs[0, r-1] * LAMBDA_0
                          + g_zs[1, r-1] * LAMBDA_1
                          ) * g_z[0]
            for z in range(1, layers-1):
                g_zs[z, r] = (g_zs[z-1, r-1] * LAMBDA_1
                              + g_zs[z, r-1] * LAMBDA_0
                              + g_zs[z+1, r-1] * LAMBDA_1
                              ) * g_z[z]
            g_zs[layers-1, r] = (g_zs[layers-1, r-1] * LAMBDA_0
                                 + g_zs[layers-2, r-1] * LAMBDA_1
                                 ) * g_z[layers-1]

elif CEX:
    def _calc_g_zs(g_z, c_i, g_zs):
        return _calc_g_zs_cex(g_z, c_i, g_zs, LAMBDA_0, LAMBDA_1)

    def _calc_g_zs_uniform(g_z, g_zs):
        return _calc_g_zs_uniform_cex(g_z, g_zs, LAMBDA_0, LAMBDA_1)

else:
    def _calc_g_zs(g_z, c_i, g_zs):
        pg_zs = g_zs[:, 0]
        segment_iterator = enumerate(c_i[::-1])
        next(segment_iterator)
        for r, c in segment_iterator:
            g_zs[: ,r] = pg_zs = (correlate(pg_zs, LAMBDA_ARRAY, 1) + c) * g_z

    def _calc_g_zs_uniform(g_z, g_zs):
        segments = g_zs.shape[1]
        pg_zs = g_zs[:, 0]
        for r in range(1, segments):
            g_zs[:, r] = pg_zs = correlate(pg_zs, LAMBDA_ARRAY, 1) * g_z