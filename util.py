# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:22:37 2015

@author: Richard Sheridan

This module is for small functions that haven't changed in a while and are
cluttering the flow of another module.

Also for functions or classes that have import-time logic associated with them

Nothing here should have in-package dependencies or the tests will have circular
import problems

"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy import exp, log
from scipy.special import gammaln
from collections import OrderedDict

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
        raise ValueError('Invalid PDI')
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
        p_ni = np.zeros(round(nn))
        p_ni[-1] = 1.0
    else:
        p_ni = np.concatenate(p_ni_list)
        p_ni /= p_ni.sum()
    cache[args]=p_ni

    if len(cache)>9000:
        cache.popitem(last=False)

    return p_ni