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

from __future__ import division, print_function, unicode_literals

import numpy as np
from time import time
from collections import OrderedDict
from numpy import exp, log, sqrt, hstack, fabs
from scipy.special import gammaln
from calc_g_zs_cex import _calc_g_zs, _calc_g_zs_uniform

# This is okay to use as long as LAMBDA_ARRAY is symmetric,
# otherwise a slice LAMBDA_ARRAY[::-1] is necessary
from numpy.core.multiarray import correlate as raw_convolve

from numpy.core import add
addred = add.reduce

LAMBDA_1 = np.float64(1.0)/6.0 #always assume cubic lattice (1/6) for now
LAMBDA_0 = 1.0-2.0*LAMBDA_1
LAMBDA_ARRAY = np.array([LAMBDA_1,LAMBDA_0,LAMBDA_1])
MINLAT = 25


def SCFprofile(z, chi=None, chi_s=None, h_dry=None, l_lat=1, mn=None, 
               m_lat=1, pdi=1, disp=False):
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
    phi_lat = SCFcache(chi,chi_s,pdi,sigma,segments,disp)
    if disp: print("\n============================\n")
    
    # re-dimensionalize the solution
    layers = len(phi_lat)
    z_end = l_lat*layers
    z_lat = np.linspace(0.0,z_end,num=layers)
    phi = np.interp(z,z_lat,phi_lat,right=0.0)

    return phi
    
_SCFcache_dict = OrderedDict()
def SCFcache(chi,chi_s,pdi,sigma,segments,disp=False,cache=_SCFcache_dict):
    """Return a memoized SCF result by walking from a previous solution.
    
    Using an OrderedDict because I want to prune keys FIFO
    """
    # prime the cache with a known easy solution
    if not cache: 
        cache[(0,0,0,.1,.2)] = SCFsolve(sigma=.1,segments=100,disp=disp)
        
    if disp: starttime = time()

    # Try to keep the parameters between 0 and 1. Factors are arbitrary.
    scaled_parameters = (chi,chi_s*3,pdi-1,sigma,segments/500)
    
    # longshot, but return a cached result if we hit it
    if scaled_parameters in cache:
        if disp: print('SCFcache hit at:', scaled_parameters)
        phi = cache.pop(scaled_parameters) # pop and assign to shift the key
        cache[scaled_parameters] = phi     # to the end as "recently used"
        return phi
    
    # Find the closest parameters in the cache: O(len(cache))
    
    # Numpy setup
    cached_parameters = list(dict.__iter__(cache))
    cp_array = np.array(cached_parameters)
    p_array = np.array(scaled_parameters)
    
    # Calculate distances to all cached parameters
    deltas = p_array - cp_array # Parameter space displacement vectors
    closest_index = addred(deltas*deltas,axis=1).argmin()
    
    # Organize closest point data for later use
    closest_cp = cached_parameters[closest_index]
    closest_cp_array = cp_array[closest_index]
    closest_delta = deltas[closest_index]
    
    phi = cache.pop(closest_cp) # pop and assign to shift the key
    cache[closest_cp] = phi     # to the end as "recently used"
    
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
            phi = SCFsolve(p_tup[0], p_tup[1]/3, p_tup[2]+1, p_tup[3], 
                           p_tup[4]*500, disp, phi)
            cache[p_tup] = phi
            dstep *= 1.05
            step += dstep
        except NoConvergence:
            flag = True # Reset this so we don't quit if step=1.0 fails
            dstep *= .5
            step -= dstep
    
    if disp: print('SCFcache execution time:', round(time()-starttime,3), "s")
    
    # keep the cache from consuming all things
    while len(cache)>1000:
        cache.popitem(last=False)
        
    return phi


def SCFsolve(chi=0,chi_s=0,pdi=1,sigma=None,segments=None,
             disp=False,phi0=None,maxiter=15):
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
        layers, phi0 = default_guess(segments,sigma)
        if disp: print('No guess passed, using default phi0: layers =',layers)
    else:
        phi0 = fabs(phi0)
        phi0[phi0>.99999] = .99999
        layers = len(phi0)
        if disp: print("Initial guess passed: layers =", layers)
    
    # Loop resizing variables
    
    # We tolerate up to 2ppm of our polymer in the last layer,
    theta = sigma*segments
    tol = 2e-6*theta
    # otherwise we grow it by 20%.
    ratio = .2
    
    # callback to detect an undersized lattice early
    def callback(x,fx): 
        short_circuit_callback(x,tol)
    
    # other loop variables        
    jac_solve_method = 'gmres'
    
    while True:
        if disp: print("Solving SCF equations")
        
        try:
            result = root(
                SCFeqns,phi0,args=(chi,chi_s,sigma,segments,p_i),
                method='Krylov',callback=callback,
                options={'disp':bool(disp),'maxiter':maxiter,
                         'jac_options':{'method':jac_solve_method}})
        except ShortCircuitRoot as e:
            # dumping out to resize since we've exceeded resize tol by 4x
            phi = fabs(e.x)
            if disp: print(e.value)
        # except ValueError as e:
            # if str(e) == 'array must not contain infs or NaNs':
                # # TODO: Handle this error better. Caused by double overflows.
                # raise #RuntimeError("solver couldn't converge")
            # else:
                # raise
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
            
            assert result.status in (1,2)
            if result.status == 1:
                # success! carry on to resize logic.
                phi = fabs(result.x)
            else:
                raise NoConvergence
                
        if disp: print('phi(M)/sum(phi) =', phi[-1] / theta * 1e6, '(ppm)')
        
        if phi[-1] > tol:
            # if the last layer is beyond tolerance, grow the lattice
            newlayers = max(1,round(len(phi0)*ratio))
            if disp: print('Growing undersized lattice by', newlayers)
            phi0 = hstack((phi,np.linspace(phi[-1],0,num=newlayers)))
        else:
            # otherwise, we are done for real
            break
    
    # chop off extra layers
    chop = addred(phi>tol)+1
    phi = phi[:max(MINLAT,chop)]
    
    if disp:
        print('After chopping: phi(M)/sum(phi) =',
              phi[-1] / theta * 1e6, '(ppm)')
        print("lattice size:", len(phi))
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
        p_ni = cache.pop(args)
        cache[args] = p_ni
        return p_ni
        
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
        p_ni = np.zeros((1,round(nn)))
        p_ni[0,-1] = 1.0
    else:
        p_ni = hstack(p_ni_list).reshape(1,-1)
    
    cache[args]=p_ni
    
    if len(cache)>9000:
        cache.popitem(last=False)
    
    return p_ni

def default_guess(segments=100,sigma=.5,chi=0,chi_s=0):
    """ Produce an initial guess for phi via analytical approximants.
    
    For now, a line using numbers from scaling theory
    """
    ss=sqrt(sigma)
    default_layers = round(max(MINLAT,segments*ss))
    default_phi0 = np.linspace(ss,0,num=default_layers)
    return default_layers, default_phi0
    
class ShortCircuitRoot(Exception):
    """ Special error to stop root() before a solution is found.
    
    """
    def __init__(self, value,x):
         self.value = value
         self.x = x
    def __str__(self):
         return repr(self.value)
         
class NoConvergence(Exception):
    """ Special exception to stop SCFsolve if root() is not converging
    
    """
    pass
         
def short_circuit_callback(x,tol):
    """ Special callback to stop root() before solution is found.
    
    This kills root if the tolerances are exceeded by 4 times the tolerances
    of the lattice resizing loop. This seems to work well empirically to 
    restart the solver when necessary without cutting out of otherwise 
    reasonable solver trajectories.
    """
    if abs(x[-1]) > 4*tol:
        raise ShortCircuitRoot('Stopping, lattice too small!',x)

def SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i):
    """ System of SCF equation for terminally attached polymers.
    
        Formatted for input to a nonlinear minimizer or solver.
    """
    
    # let the solver go negative if it wants
    phi_z = fabs(phi_z)
    
    # attempts to try fields with values greater than one are penalized
    toomuch = phi_z>.99999
    if toomuch.any():
        penalty = np.where(toomuch,1e5*(phi_z-.99999),0)
        phi_z[toomuch] = .99999
    else:
        penalty = 0.0
    
    layers = phi_z.size
    cutoff = p_i.size
    
    # calculate all needed quantities for new g_z
    delta = np.zeros(layers)
    delta[0] = 1.0
    phi_z_avg = calc_phi_z_avg(phi_z)
    
    # calculate new g_z (Boltzmann weighting factors)
    g_z = (1.0 - phi_z)*exp(2*chi*phi_z_avg + delta*chi_s)
    
    # normalize g_z for numerical stability
    u = -log(g_z)
    uavg = addred(u)/layers
    g_z_norm = g_z*exp(uavg)
    
    # calculate weighting factors for terminally attached chains
    g_zs_ta_norm = calc_g_zs(g_z_norm,0,layers,cutoff)
    
    # calculate normalization constants from 1/(single chain partition fn)
    if cutoff == round(navgsegments): # if uniform,
        c_i_norm = sigma/addred(g_zs_ta_norm[:,-1]) # take a shortcut!
    else:
        c_i_norm = sigma*p_i/addred(g_zs_ta_norm,axis=0)
    
    # calculate weighting factors for free chains
    g_zs_free_ngts_norm = calc_g_zs(g_z_norm,c_i_norm,layers,cutoff)
    
    # calculate new polymer density field
    phi_z_new = calc_phi_z(g_zs_ta_norm,g_zs_free_ngts_norm,g_z_norm)
    
    # Handle float overflows only if they show themselves
    if np.isnan(phi_z_new).any():
        maxfloat=_getmax(g_zs_ta_norm.dtype.type)
        g_zs_ta_norm[np.isinf(g_zs_ta_norm)]=maxfloat
        g_zs_free_ngts_norm[np.isinf(g_zs_free_ngts_norm)]=maxfloat
        phi_z_new = calc_phi_z(g_zs_ta_norm,g_zs_free_ngts_norm,g_z_norm)
        
    eps_z = phi_z - phi_z_new
    return eps_z + penalty*np.sign(eps_z)

def _getmax(t, seen_t={}):
    try:
        return seen_t[t]
    except KeyError:
        from numpy.core import getlimits
        fmax = getlimits.finfo(t).max
        seen_t[t]=fmax
        return fmax

def calc_phi_z_avg(phi_z):
    return raw_convolve(phi_z,LAMBDA_ARRAY,1)

def calc_phi_z(g_ta,g_free,g_z):
    return addred(g_ta*np.fliplr(g_free),axis=1)/g_z

def calc_g_zs(g_z,c_i,layers,segments):
    # initialize
    g_zs=np.empty((layers,segments),dtype=np.float64,order='F')
    
    # choose special case
    if np.size(c_i) == 1:
        if c_i:
            # uniform chains
            g_zs[:,0] = c_i*g_z
        else:
            # terminally attached ends
            g_zs[:,0] = 0.0
            g_zs[0,0] = g_z[0]
        _calc_g_zs_uniform(g_z,g_zs,LAMBDA_0,LAMBDA_1,layers,segments)

    else:
        # free ends
        g_zs[:,0] = c_i[0,-1]*g_z
        _calc_g_zs(g_z,c_i,g_zs,LAMBDA_0,LAMBDA_1,layers,segments)
    
    # Older versions of inner loops
    
#    if np.size(c_i)==1:
#        c_i = np.zeros((1,round(segments)))
#        g_zs[:,0] = 0.0
#        g_zs[0,0] = g_z[0]
#    else:
#        # free chains
#        g_zs[:,0] = c_i[0,segments-1]*g_z
    
    # FASTEST: call some custom C code identical to "SLOW" loop
#    _calc_g_zs_pointers(g_z,c_i,g_zs,LAMBDA_0,LAMBDA_1,layers,segments)
    
    # FASTER: use the convolve function to partially vectorize  
#    pg_zs=g_zs[:,0]    
#    for r in range(1,segments):
#        pg_zs=g_z*(c_i[0,segments-r-1]+raw_convolve(pg_zs,LAMBDA_ARRAY,1))
#        g_zs[:,r]=pg_zs
    
    # SLOW: loop outright, pulling some slicing out of the innermost loop  
#    for r in range(1,segments):
#        c=c_i[0,segments-r-1]
#        g_zs[0,r]=(pg_zs[0]*LAMBDA_0+pg_zs[1]*LAMBDA_1+c)*g_z[0]
#        for z in range(1,(layers-1)):
#            g_zs[z,r]=(pg_zs[z-1]*LAMBDA_1
#                       + pg_zs[z]*LAMBDA_0
#                       + pg_zs[z+1]*LAMBDA_1
#                       + c) * g_z[z]
#        g_zs[-1,r]=(pg_zs[-1]*LAMBDA_0+pg_zs[-2]*LAMBDA_1+c)*g_z[-1]
#        pg_zs=g_zs[:,r]
               
    return g_zs
