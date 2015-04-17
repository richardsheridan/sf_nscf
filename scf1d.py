# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 13:40:43 2014

@author: Richard Sheridan

Numerical Self-consistent Field (SCF) End-Tethered Polymer
Profile\ [#Cosgrove]_\ [#deVos]_\ [#Sheridan]_


.. [#Cosgrove] Cosgrove, T., Heath, T., Van Lent, B., Leermakers, F. A. M.,
    & Scheutjens, J. M. H. M. (1987). Configuration of terminally attached
    chains at the solid/solvent interface: self-consistent field theory and
    a Monte Carlo model. Macromolecules, 20(7), 1692–1696.
    doi:10.1021/ma00173a041

.. [#deVos] De Vos, W. M., & Leermakers, F. A. M. (2009). Modeling the
    structure of a polydisperse polymer brush. Polymer, 50(1), 305–316.
    doi:10.1016/j.polymer.2008.10.025

.. [#Sheridan] Sheridan, R. J., Beers, K. L., et. al (2014). Direct observation
    of "surface theta" conditions. [in prep]
"""

import numpy as np
from time import time
from collections import OrderedDict
from numpy import exp, log, sqrt, hstack, fabs
from scipy.special import gammaln

# compatability for systems lacking compiler capability
PYONLY = JIT = False
try:
    from numba import njit
    JIT = True
except ImportError:
    try:
        from calc_g_zs_cex import _calc_g_zs, _calc_g_zs_uniform
    except ImportError:
        from warnings import warn
        warn('Compiled inner loop unavailable, using slow as molasses version!')
        PYONLY = True

# Micro-optimizations

# faster version of numpy.convolve for ndarray only
# This is okay to use as long as LAMBDA_ARRAY is symmetric,
# otherwise a slice LAMBDA_ARRAY[::-1] is necessary
from numpy.core.multiarray import correlate

if not JIT:
    fastsum = np.sum

# Precalculate some global constants
LAMBDA_1 = np.float64(1.0)/6.0 #always assume cubic lattice (1/6) for now
LAMBDA_0 = 1.0-2.0*LAMBDA_1
LAMBDA_ARRAY = np.array([LAMBDA_1,LAMBDA_0,LAMBDA_1])
MINLAT = 25
MINBULK = 5

def SCFprofile(z, chi=None, chi_s=None, h_dry=None, l_lat=1, mn=None,
               m_lat=1, phi_b=0, pdi=1, disp=False):
    """
    Generate volume fraction profile for Refl1D based on real parameters.

    The field theory is a lattice-based one, so we need to move between lattice
    and real space. This is done using the parameters l_lat and m_lat, the
    lattice size and the mass of a lattice segment, respectivley. We use h_dry
    (dry thickness) as a convenient measure of surface coverage, along with mn
    (number average molecular weight) as the real inputs.

    Make sure your inputs for h_dry/l_lat and mn/m_lat match dimensions!
    Angstroms and daltons are good choices.

    This function is suitable for use as a VolumeProfile, as well as the
    default EndTetheredPolymer class.
    """

    # calculate lattice space parameters
    theta = h_dry/l_lat
    segments = mn/m_lat
    sigma = theta/segments

    # solve the self consistent field equations using the cache
    if disp: print("\n=====Begin calculations=====\n")
    phi_lat = SCFcache(chi,chi_s,pdi,sigma,phi_b,segments,disp)
    if disp: print("\n============================\n")

    # Chop edge effects out
    for x,layer in enumerate(reversed(phi_lat)):
        if abs(layer - phi_b) < 1e-6:
            break
    phi_lat = phi_lat[:-(x+1)]

    # re-dimensionalize the solution
    layers = len(phi_lat)
    z_end = l_lat*layers
    z_lat = np.linspace(0.0,z_end,num=layers)
    phi = np.interp(z,z_lat,phi_lat,right=phi_b)

    return phi

_SCFcache_dict = OrderedDict()
def SCFcache(chi,chi_s,pdi,sigma,phi_b,segments,disp=False,cache=_SCFcache_dict):
    """Return a memoized SCF result by walking from a previous solution.

    Using an OrderedDict because I want to prune keys FIFO
    """
    # prime the cache with a known easy solutions
    if not cache:
        cache[(0,0,0,.1,.1,.1)] = SCFsolve(sigma=.1,phi_b=.1,segments=50,disp=disp)
        cache[(0,0,0,0,.1,.1)] = SCFsolve(sigma=0,phi_b=.1,segments=50,disp=disp)
        cache[(0,0,0,.1,0,.1)] = SCFsolve(sigma=.1,phi_b=0,segments=50,disp=disp)

    if disp: starttime = time()

    # Try to keep the parameters between 0 and 1. Factors are arbitrary.
    scaled_parameters = (chi,chi_s*3,pdi-1,sigma,phi_b,segments/500)

    # longshot, but return a cached result if we hit it
    if scaled_parameters in cache:
        if disp: print('SCFcache hit at:', scaled_parameters)
        cache.move_to_end(scaled_parameters)
        return cache[scaled_parameters]

    # Find the closest parameters in the cache: O(len(cache))

    # Numpy setup
    cached_parameters = tuple(dict.__iter__(cache))
    cp_array = np.array(cached_parameters)
    p_array = np.array(scaled_parameters)

    # Calculate distances to all cached parameters
    deltas = p_array - cp_array # Parameter space displacement vectors
    closest_index = np.sum(deltas*deltas,axis=1).argmin()

    # Organize closest point data for later use
    closest_cp = cached_parameters[closest_index]
    closest_cp_array = cp_array[closest_index]
    closest_delta = deltas[closest_index]

    cache.move_to_end(closest_cp)
    phi = cache[closest_cp]

    if disp:
        print("Walking from nearest:", closest_cp_array)
        print("to:", p_array)

    """
    We must walk from the previously cached point to the desired region.
    This is goes from step=0 (cached) and step=1 (finish), where the step=0
    is implicit above. We try the full step first, so that this function only
    calls SCFsolve one time during normal cache misses.

    The solver may not converge if the step size is too big. In that case,
    we retry with half the step size. This should find the edge of the basin
    of attraction for the solution eventually. On successful steps we increase
    stepsize slightly to accelerate after getting stuck.

    It might seem to make sense to bin parameters into a coarser grid, so we
    would be more likely to have cache hits and use them, but this rarely
    happened in practice.
    """

    step = 1.0 # Fractional distance between cached and requested
    dstep = 1.0 # Step size increment
    flag = True

    while flag:
        # end on 1.0 exactly every time
        if step >= 1.0:
            step = 1.0
            flag = False

        # conditional math because, "why risk floating point error"
        if flag:
            p_tup = tuple(closest_cp_array + step*closest_delta)
        else:
            p_tup = scaled_parameters

        if disp:
            print('Parameter step is', step)
            print('current parameters:', p_tup)

        try:
            phi = SCFsolve(p_tup[0], p_tup[1]/3, p_tup[2]+1, p_tup[3], p_tup[4],
                           p_tup[5]*500, disp=disp, phi0=phi)
        except (NoConvergence, ValueError):
            if disp: print('Step failed')
            flag = True # Reset this so we don't quit if step=1.0 fails
            dstep *= .5
            step -= dstep
            if dstep < 1e-5:
                raise RuntimeError('Cache walk appears to be stuck')
        else: # Belongs to try, executes if no exception is raised
            cache[p_tup] = phi
            dstep *= 1.05
            step += dstep

    if disp: print('SCFcache execution time:', round(time()-starttime,3), "s")

    # keep the cache from consuming all things
    while len(cache)>1000:
        cache.popitem(last=False)

    return phi


def SCFsolve(chi=0,chi_s=0,pdi=1,sigma=None,phi_b=0,segments=None,
             disp=False,phi0=None,maxiter=30):
    """Solve SCF equations using an initial guess and lattice parameters

    This function finds a solution for the equations where the lattice size
    is sufficiently large.

    The Newton-Krylov solver really makes this one. Krylov+gmres was faster
    than the other scipy.optimize alternatives by quite a lot.
    """

    from scipy.optimize import root

    if sigma >= 1:
        raise ValueError('Chains that short cannot be squeezed that high')

    if disp: starttime = time()

    p_i = SZdist(pdi,segments)

    if phi0 is None:
        # TODO: Better initial guess for chi>.6
        phi0 = default_guess(segments,sigma)
        if disp: print('No guess passed, using default phi0: layers =',len(phi0))
    else:
        phi0 = fabs(phi0)
        phi0[phi0>.99999] = .99999
        if disp: print("Initial guess passed: layers =", len(phi0))

    # resizing loop variables
    jac_solve_method = 'gmres'
    lattice_too_small = True

    # We tolerate up to 10 ppm deviation from bulk phi
    # when counting layers_near_phi_b
    tol = 1e-5

    # callback to detect an undersized lattice early
    callback = None # TODO: no good strategy yet ;_;

    while lattice_too_small:
        if disp: print("Solving SCF equations")

        try:
            result = root(SCFeqns,
                          phi0,
                          args=(chi,chi_s,sigma,segments,p_i,phi_b),
                          method='Krylov',
                          callback=callback,
                          options={'disp':bool(disp),'maxiter':maxiter,
                                  'jac_options':{'method':jac_solve_method}})
        except ShortCircuitRoot as e:
            # dumping out to resize since we've exceeded resize tol by 4x
            phi = fabs(e.x)
            if disp: print(e.value)
        except RuntimeError as e:
            if str(e) == 'gmres is not re-entrant':
                # Threads are racing to use gmres. Lose the race and use
                # something slower but thread-safe.
                jac_solve_method = 'lgmres'
                continue
            else:
                raise
        else: # Belongs to try, executes if no exception is raised
            if disp:
                print('Solver exit code:',result.status,result.message)

            if result.success:
                # carry on to resize logic.
                phi = fabs(result.x)
            elif result.status == 2:
                raise NoConvergence
            else:
                raise AssertionError # was: assert result.status in {1,2}

        if disp:
            print('lattice size:', len(phi))

        layers_near_phi_b = np.sum(fabs(phi - phi_b) < tol)

        lattice_too_small = layers_near_phi_b < MINBULK
        if lattice_too_small:
            # if there aren't enough layers_near_phi_b, grow the lattice 20%
            newlayers = max(1,round(len(phi0)*0.2))
            if disp: print('Growing undersized lattice by', newlayers)
            phi0 = hstack((phi[:-5],np.linspace(phi[-6],phi[-5],num=newlayers),
                           phi[-5:]))

    if disp:
        print("SCFsolve execution time:", round(time()-starttime,3), "s")

    return phi


_SZdist_dict = OrderedDict()
def SZdist(pdi,nn,cache=_SZdist_dict):
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
        p_ni = hstack(p_ni_list)

    cache[args]=p_ni

    if len(cache)>9000:
        cache.popitem(last=False)

    return p_ni

def default_guess(segments=100,sigma=.5,phi_b=.1,chi=0,chi_s=0):
    """ Produce an initial guess for phi via analytical approximants.

    For now, a line using numbers from scaling theory
    """
    ss=sqrt(sigma)
    default_layers = round(max(MINLAT,segments*ss))
    default_phi0 = np.linspace(ss,phi_b,num=default_layers)
    return default_phi0

class ShortCircuitRoot(Exception):
    """ Special error to stop root() before a solution is found.

    """
    def __init__(self, value,x):
         self.value = value
         self.x = x
    def __str__(self):
         return repr(self.value)

#class NoConvergence(Exception):
#    """ Special exception to stop SCFsolve if root() is not converging
#
#    """
#    pass
from scipy.optimize.nonlin import NoConvergence

def short_circuit_callback(x,tol,phi_b=0):
    """ Special callback to stop root() before solution is found.

    This kills root if the tolerances are exceeded by 4 times the tolerances
    of the lattice resizing loop. This seems to work well empirically to
    restart the solver when necessary without cutting out of otherwise
    reasonable solver trajectories.
    """
    if abs(x[-1]) > 4*tol:
        raise ShortCircuitRoot('Stopping, lattice too small!',x)

def SCFeqns(phi_z, chi, chi_s, sigma, navgsegments, p_i,
            phi_b = 0, dump_u = False):
    """ System of SCF equation for terminally attached polymers.

        Formatted for input to a nonlinear minimizer or solver.
    """

    # let the solver go negative if it wants
    phi_z = fabs(phi_z)

    layers = phi_z.size
    cutoff = p_i.size
    uniform = cutoff == round(navgsegments) # if uniform, we can take shortcuts!

    # calculate new g_z (Boltzmann weighting factors)
    g_z = calc_g_z(phi_z, chi, chi_s, phi_b)
    u = -log(g_z)

    if dump_u:
        return u

    # normalize g_z for numerical stability
    uavg = np.sum(u)/layers
    g_z_norm = g_z*exp(uavg)

    # calculate weighting factors, normalization constants, and density fields

    # for terminally attached chains
    if sigma:
        g_zs_ta_norm = calc_g_zs(g_z_norm,-1,layers,cutoff)
        if uniform:
            c_i_ta_norm = sigma/np.sum(g_zs_ta_norm[:,-1])
        else:
            c_i_ta_norm = sigma*p_i/fastsum(g_zs_ta_norm,axis=0)
        g_zs_ta_ngts_norm = calc_g_zs(g_z_norm,c_i_ta_norm,layers,cutoff)
        phi_z_ta = calc_phi_z(g_zs_ta_norm,
                              g_zs_ta_ngts_norm,
                              g_z_norm,
                              layers,
                              cutoff)
    else:
        phi_z_ta = 0

    # for free chains
    if phi_b:
        g_zs_free_norm = calc_g_zs(g_z_norm,0,layers,cutoff)
        if uniform:
            r_i = cutoff
            c_i_free = phi_b/r_i
        else:
            r_i = np.arange(1,cutoff+1)
            c_i_free = phi_b*p_i/r_i
        normalizer = exp(-uavg*r_i)
        c_i_free_norm = c_i_free*normalizer
        g_zs_free_ngts_norm = calc_g_zs(g_z_norm,c_i_free_norm,layers,cutoff)
        phi_z_free = calc_phi_z(g_zs_free_norm,
                                g_zs_free_ngts_norm,
                                g_z_norm,
                                layers,
                                cutoff)
    else:
        phi_z_free = 0

    phi_z_new = phi_z_ta + phi_z_free

    eps_z = phi_z - phi_z_new

    return eps_z


def calc_g_z(phi_z, chi, chi_s, phi_b=0):
    layers = phi_z.size
    delta = np.zeros(layers)
    delta[0] = 1.0
    phi_z_avg = calc_phi_z_avg(phi_z)

    # calculate new g_z (Boltzmann weighting factors)
    g_z = (1.0 - phi_z)/(1.0 - phi_b)*exp(2*chi*(phi_z_avg-phi_b) + delta*chi_s)

    return g_z

def calc_phi_z_avg(phi_z):
    return correlate(phi_z,LAMBDA_ARRAY,1)

def calc_phi_z(g_zs,g_zs_ngts,g_z,layers,segments):
    prod = g_zs*np.fliplr(g_zs_ngts)
    prod[np.isnan(prod)]=0
#    prod=np.nan_to_num(prod)
    return np.sum(prod,axis=1)/g_z

def calc_g_zs(g_z,c_i,layers,segments):
    # initialize
    g_zs=np.empty((layers,segments),order='F')

    # choose special case
    if np.size(c_i) == 1: # floats need np.size() rather than ndarray.size
        if c_i > 0:
            # uniform chains
            g_zs[:,0] = c_i*g_z
        elif c_i == 0:
            # free beginnings
            g_zs[:,0] = g_z
        elif c_i == -1:
            # terminally attached beginings
            g_zs[:,0] = 0.0
            g_zs[0,0] = g_z[0]
        else:
            raise ValueError('Unsupported special case')
        _calc_g_zs_uniform(g_z,g_zs,LAMBDA_0,LAMBDA_1,layers,segments)
    else:
        # free ends
        g_zs[:,0] = c_i[-1]*g_z
        _calc_g_zs(g_z,c_i,g_zs,LAMBDA_0,LAMBDA_1,layers,segments)

    return g_zs

if PYONLY:
    def _calc_g_zs(g_z,c_i,g_zs,LAMBDA_0,LAMBDA_1,layers,segments):
        pg_zs = g_zs[:,0]
        segment_iterator = enumerate(c_i[::-1])
        next(segment_iterator)
        for r,c in segment_iterator:
            g_zs[:,r] = pg_zs = (correlate(pg_zs,LAMBDA_ARRAY,1) + c) * g_z

    def _calc_g_zs_uniform(g_z,g_zs,LAMBDA_0,LAMBDA_1,layers,segments):
        pg_zs = g_zs[:,0]
        for r in range(1,segments):
            g_zs[:,r] = pg_zs = correlate(pg_zs,LAMBDA_ARRAY,1) * g_z

if JIT:
    def fastsum(g_zs,axis=0):
        layers, segments = g_zs.shape
        output = np.zeros(segments)
        _fastsum(g_zs,output,layers,segments)
        return output

    @njit('void(f8[:,:],f8[:],i4,i4)')
    def _fastsum(g_zs,output,layers,segments):
        for s in range(segments):
            for z in range(layers):
                output[s]+=g_zs[z,s]

    def calc_phi_z(g_zs,g_zs_ngts,g_z,layers,segments):
        output = np.zeros((layers))
        _calc_phi_z(g_zs,g_zs_ngts,output,layers,segments)
        output /= g_z
        return output

    @njit('void(f8[:,:],f8[:,:],f8[:],i4,i4)')
    def _calc_phi_z(g_zs,g_zs_ngts,output,layers,segments):
        for s in range(segments):
            for z in range(layers):
                if g_zs[z,s] and g_zs_ngts[z,segments-s-1]: # Prevent NaNs
                    output[z]+=g_zs[z,s]*g_zs_ngts[z,segments-s-1]

    @njit('void(f8[:],f8[:],f8[:,:],f8,f8,i4,i4)')
    def _calc_g_zs(g_z,c_i,g_zs,LAMBDA_0,LAMBDA_1,layers,segments):
        for r in range(1,segments):
            c = c_i[segments-r-1]
            g_zs[0,r] = (g_zs[0,r-1]*LAMBDA_0
                         + g_zs[1,r-1]*LAMBDA_1
                         + c) * g_z[0]
            for z in range(1,(layers-1)):
                g_zs[z,r]=(g_zs[z-1,r-1]*LAMBDA_1
                           + g_zs[z,r-1]*LAMBDA_0
                           + g_zs[z+1,r-1]*LAMBDA_1
                           + c) * g_z[z]
            g_zs[layers-1,r]=(g_zs[layers-1,r-1]*LAMBDA_0
                              + g_zs[layers-2,r-1]*LAMBDA_1
                              + c) * g_z[layers-1]

    @njit('void(f8[:],f8[:,:],f8,f8,i4,i4)')
    def _calc_g_zs_uniform(g_z,g_zs,LAMBDA_0,LAMBDA_1,layers,segments):
        for r in range(1,segments):
            g_zs[0,r] = (g_zs[0,r-1]*LAMBDA_0
                         + g_zs[1,r-1]*LAMBDA_1
                         ) * g_z[0]
            for z in range(1,(layers-1)):
                g_zs[z,r]=(g_zs[z-1,r-1]*LAMBDA_1
                           + g_zs[z,r-1]*LAMBDA_0
                           + g_zs[z+1,r-1]*LAMBDA_1
                           ) * g_z[z]
            g_zs[layers-1,r]=(g_zs[layers-1,r-1]*LAMBDA_0
                              + g_zs[layers-2,r-1]*LAMBDA_1
                              ) * g_z[layers-1]
