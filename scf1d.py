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

from util import schultz_zimm
import numpy as np
from time import time
from collections import OrderedDict
from numpy import exp, log, sqrt, fabs
from scipy.optimize.nonlin import newton_krylov, NoConvergence

# compatability for systems lacking compiler capability
PYONLY = JIT = False
try:
    from numba import njit
    JIT = True
except ImportError:
    try:
        from calc_g_zs_cex import _calc_g_zs_cex, _calc_g_zs_uniform_cex
    except ImportError:
        from warnings import warn
        warn('Compiled inner loop unavailable, using slow as molasses version!')
        PYONLY = True

# Micro-optimizations

# faster version of numpy.convolve for ndarray only
# This is okay to use as long as LAMBDA_ARRAY is symmetric,
# otherwise a slice LAMBDA_ARRAY[::-1] is necessary
from numpy.core.multiarray import correlate

def sumkd(array, axis=None):
    return np.sum(array, axis=axis, keepdims=True)

def meankd(array, axis=None):
    return np.mean(array, axis=axis, keepdims=True)

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

def SCFsqueeze(chi,chi_s,pdi,sigma,phi_b,segments,layers,disp=False):
    """ Return a self consistent field within a specified number of layers.


    """

    phi = SCFcache(chi,chi_s,pdi,sigma,phi_b,segments,disp)
    squeezing = layers - len(phi) < 0

    p_i = schultz_zimm(pdi,segments)
    jac_solve_method = 'gmres'
    def curried(phi):
        return SCFeqns(phi,chi,chi_s,sigma,segments,p_i,phi_b)

    while layers - len(phi):
        if squeezing:
            phi = np.delete(phi, -1)
        else:
            phi = np.append(phi, phi[-1])
        phi = fabs(newton_krylov(curried,
                                 phi,
                                 verbose=bool(disp),
                                 maxiter=30,
                                 method=jac_solve_method,
                                 ))

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
        except (NoConvergence, ValueError) as e:
            if isinstance(e, ValueError):
                if str(e) != "array must not contain infs or NaNs":
                    raise
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


def build_chi_cvl(x_av, x_ws, x_vw, x_aw):
    """ Build an interaction matrix based on a reduced number of parameters
        and some rules.

        Based on the system from Cohen-Stuart, de Vos, and Leermakers (2006)
    """
#    x_av=1 # goal: >1
#    x_ws=-0.6 # goal: -1.5
#    x_vw = 2.5 # goal: 3.5
    x_as=x_ws-1
    x_av=-x_ws
    x_sv=0
    return np.array((
                    (0,x_ws,x_as,x_sv),
                    (x_ws,0,x_aw,x_vw),
                    (x_as,x_aw,0,x_av),
                    (x_sv,x_vw,x_av,0),
                    ))


def SCFsolve_multi(chi_jk, sigma_j, phi_b_j, n_avg_j, pdi_j=None,
                   phi_jz_solid=None, disp=False, u_jz0=None, maxiter=30):
    """ Solve SCF equations involving multiple material types using an initial
        guess and lattice parameters

        This function finds a solution for the equations where the lattice
        size is sufficiently large.
    """
    if np.any(sigma_j >= 1):
        raise ValueError('Chains that short cannot be squeezed that high')
    if np.sum(phi_b_j) != 1:
        raise ValueError('Bulk volume fractions should add up to 1')

    if disp: starttime = time()

    if pdi_j is None:
        p_ji = None
    else:
        p_ji = [schultz_zimm(pdi, n_avg) if n_avg > 1 else None
                for n_avg, pdi in zip(n_avg_j, pdi_j)]

    if u_jz0 is None:
        u_jz0 = np.zeros((np.sum(n_avg_j != 0),
                          max(MINLAT, n_avg_j.max()*sqrt(sigma_j.max())),
                          ))
        if disp:
            print('No guess passed, using default u_jz0: '
                  'types = {}, layers = {}'.format(*u_jz0.shape))
    else:
        if disp:
            print('Initial guess passed: '
                  'types = {}, layers = {}'.format(*u_jz0.shape))

    # resizing loop variables
    jac_solve_method = 'gmres'
    lattice_too_small = True

    # We tolerate up to 1 ppm deviation from bulk phi
    # when counting layers_near_phi_b
    tol = 1e-6

    def curried(u_jz):
        return SCFeqns_multi(u_jz, chi_jk, sigma_j, phi_b_j, n_avg_j, p_ji,
                             phi_jz_solid, dump_phi=False)

    while lattice_too_small:
        if disp: print("Solving SCF equations")

        try:
            u_jz = newton_krylov(curried,
                                 u_jz0,
                                 verbose=bool(disp),
                                 maxiter=maxiter,
                                 method=jac_solve_method,
                                 )
        except RuntimeError as e:
            if str(e) == 'gmres is not re-entrant':
                # Threads are racing to use gmres. Lose the race and use
                # something slower but thread-safe.
                jac_solve_method = 'lgmres'
                continue
            else:
                raise

        if disp:
            print('lattice size:', u_jz.shape[1])

        u_deviation = fabs(u_jz).sum(axis=0)
        layers_near_zero = u_deviation < tol
        nbulk = np.sum(layers_near_zero)
        lattice_too_small = nbulk < MINBULK

        if lattice_too_small:
            # if there aren't enough layers_near_zero, grow the lattice 20%
            newlayers = max(1, round((u_jz0.shape[1])*0.2))
            if disp: print('Growing undersized lattice by', newlayers)

            # TODO: comment on inscrutable indexing and stacking
            if nbulk:
                i = np.diff(layers_near_zero).nonzero()[0].max()
            else:
                i = u_deviation.argmin()
            u_jz0 = np.hstack((u_jz[:,:i-1],
                               [np.linspace(u_z[i-1], u_z[i], num=newlayers)
                                for u_z in u_jz], # XXX: vectorize?
                               u_jz[:,i:]))

    if nbulk > 2*MINBULK:
        chop_end = np.diff(layers_near_zero).nonzero()[0].max()
        chop_start = chop_end - MINBULK
        i = np.arange(u_jz.shape[1])
        u_jz = u_jz[:,(i <= chop_start) | (i > chop_end)]

    if disp:
        print("SCFsolve execution time:", round(time()-starttime,3), "s")

    return u_jz

def SCFsolve(chi=0,chi_s=0,pdi=1,sigma=None,phi_b=0,segments=None,
             disp=False,phi0=None,maxiter=30):
    """Solve SCF equations using an initial guess and lattice parameters

    This function finds a solution for the equations where the lattice size
    is sufficiently large.

    The Newton-Krylov solver really makes this one. With gmres, it was faster
    than the other solvers by quite a lot.
    """

    if sigma >= 1:
        raise ValueError('Chains that short cannot be squeezed that high')

    if disp: starttime = time()

    p_i = schultz_zimm(pdi,segments)

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

    # We tolerate up to 1 ppm deviation from bulk phi
    # when counting layers_near_phi_b
    tol = 1e-6

    def curried(phi):
        return SCFeqns(phi,chi,chi_s,sigma,segments,p_i,phi_b)

    while lattice_too_small:
        if disp: print("Solving SCF equations")

        try:
            phi = fabs(newton_krylov(curried,
                                     phi0,
                                     verbose=bool(disp),
                                     maxiter=maxiter,
                                     method=jac_solve_method,
                                     ))
        except RuntimeError as e:
            if str(e) == 'gmres is not re-entrant':
                # Threads are racing to use gmres. Lose the race and use
                # something slower but thread-safe.
                jac_solve_method = 'lgmres'
                continue
            else:
                raise

        if disp:
            print('lattice size:', len(phi))

        phi_deviation = fabs(phi - phi_b)
        layers_near_phi_b = phi_deviation < tol
        nbulk = np.sum(layers_near_phi_b)
        lattice_too_small = nbulk < MINBULK

        if lattice_too_small:
            # if there aren't enough layers_near_phi_b, grow the lattice 20%
            newlayers = max(1, round(len(phi0)*0.2))
            if disp: print('Growing undersized lattice by', newlayers)
            if nbulk:
                i = np.diff(layers_near_phi_b).nonzero()[0].max()
            else:
                i = phi_deviation.argmin()
            phi0 = np.insert(phi,i,np.linspace(phi[i-1], phi[i], num=newlayers))

    if nbulk > 2*MINBULK:
        chop_end = np.diff(layers_near_phi_b).nonzero()[0].max()
        chop_start = chop_end - MINBULK
        i = np.arange(len(phi))
        phi = phi[(i <= chop_start) | (i > chop_end)]

    if disp:
        print("SCFsolve execution time:", round(time()-starttime,3), "s")

    return phi




def default_guess(segments=100,sigma=.5,phi_b=.1,chi=0,chi_s=0):
    """ Produce an initial guess for phi via analytical approximants.

    For now, a line using numbers from scaling theory
    """
    ss=sqrt(sigma)
    default_layers = round(max(MINLAT,segments*ss))
    default_phi0 = np.linspace(ss,phi_b,num=default_layers)
    return default_phi0

def SCFeqns_multi(u_jz, chi_jk, sigma_j, phi_b_j, n_avg_j, p_ji=None,
                  phi_jz_solid=None, dump_phi=False):
    """ System of SCF equation for mixed homopolymers, solvents and solids.
        Formatted for input to a nonlinear minimizer or solver.

        The j dimension is an index for the type of material, e.g.
        j = 0 -> surface
        j = 1 -> solvent
        j = 2 -> polymer
        j = 3 -> air

        define solids concentration with separate density field,
        so pass in a u_jz with size (J-1,Z)

        All inputs should be ndarrays. For speed, we don't check or convert.
        p_ji and phi_jz_solid can be lists or dicts of arrays
        instead of 2D arrays if sparsity is a concern.

        plenty of inputs don't converge with raw newton_krylov
        TODO: create scfcache equivalent

        Does this agree with SCFeqns?
        TODO: plot differences at a variety of solutions
    """

    solids = n_avg_j < 1
    monomers = n_avg_j == 1
    polymers = n_avg_j > 1

    phi_jz = np.empty_like(u_jz)
    for j in np.nonzero(solids)[0]:
        phi_jz = np.insert(phi_jz,
                           j,
                           0 if phi_jz_solid is None else phi_jz_solid[j],
                           axis=0)
        u_jz = np.insert(u_jz, j, 0, axis=0)

    for j in np.nonzero(monomers)[0]:
        phi_jz[j] = phi_b_j[j] / exp(u_jz[j])
        phi_jz[j,0] += sigma_j[j]

    u_jz_avg = meankd(u_jz, axis=1)
    g_jz_norm = exp(u_jz_avg - u_jz)
    for j in np.nonzero(polymers)[0]:
        # TODO: introduce thread-based parallelism
        phi_jz[j] = calc_phi_z(g_jz_norm[j],
                               n_avg_j[j],
                               sigma_j[j],
                               phi_b_j[j],
                               u_jz_avg[j],
                               None if p_ji is None else p_ji[j],
                               )
    if dump_phi:
        return phi_jz[~solids]

    u_int_jz = np.dot(chi_jk, phi_jz_avg(phi_jz) - phi_b_j[:, None])

    # TODO: see if subtracting a reference potential helps
    u_prime_jz = (u_jz - u_int_jz)[~solids]

    # HINT: Don't try to square these errors, it prevents solution
    eps_jz = (u_prime_jz - meankd(u_prime_jz, axis=0)) + (1 - sumkd(phi_jz, axis=0))

#    import matplotlib.pyplot as plt

#    plt.subplot(311)
#    plt.cla()
#    plt.plot(u_jz.T)
#    plt.legend((0,1,2,3))

#    plt.subplot(312)
#    plt.cla()
#    plt.plot(phi_jz.T)
#
#    plt.subplot(313)
#    plt.cla()
#    plt.plot(eps_jz.T)

#    plt.draw()
#    plt.show(block=0)

    return eps_jz

def SCFeqns(phi_z, chi, chi_s, sigma, n_avg, p_i, phi_b=0):
    """ System of SCF equation for terminally attached polymers.

        Formatted for input to a nonlinear minimizer or solver.

        The sign convention here on u is "backwards" and always has been.
        It saves a few sign flips, and looks more like Cosgrove's.
    """

    # let the solver go negative if it wants
    phi_z = fabs(phi_z)

    # calculate new g_z (Boltzmann weighting factors)
    u_prime = log((1.0 - phi_z)/(1.0 - phi_b))
    u_int = 2*chi*(correlate(phi_z, LAMBDA_ARRAY, 1)-phi_b)
    u_int[0] += chi_s
    u_z = u_prime + u_int
    g_z = exp(u_z)

    # normalize g_z for numerical stability
    u_z_avg = np.mean(u_z)
    g_z_norm = g_z/exp(u_z_avg)

    phi_z_new = calc_phi_z(g_z_norm, n_avg, sigma, phi_b, u_z_avg, p_i)

    eps_z = phi_z - phi_z_new

    return eps_z

def calc_phi_z(g_z, n_avg, sigma, phi_b, u_z_avg=0, p_i=None):

    if p_i is None:
        segments = n_avg
        uniform = 1
    else:
        segments = p_i.size
        uniform = segments == round(n_avg)

    g_zs = Propagator(g_z, segments)

    # for terminally attached chains
    if sigma:
        g_zs_ta = g_zs.ta()

        if uniform:
            c_i_ta = sigma/np.sum(g_zs_ta[:, -1])
            g_zs_ta_ngts = g_zs.ngts_u(c_i_ta)
        else:
            c_i_ta = sigma*p_i/fastsum(g_zs_ta, axis=0)
            g_zs_ta_ngts = g_zs.ngts(c_i_ta)

        phi_z_ta = compose(g_zs_ta, g_zs_ta_ngts, g_z)
    else:
        phi_z_ta = 0

    # for free chains
    if phi_b:
        g_zs_free = g_zs.free()

        if uniform:
            r_i = segments
            c_free = phi_b/r_i
            normalizer = exp(u_z_avg*r_i)
            c_free = c_free*normalizer
            g_zs_free_ngts = g_zs.ngts_u(c_free)
        else:
            r_i = np.arange(1, segments+1)
            c_i_free = phi_b*p_i/r_i
            normalizer = exp(u_z_avg*r_i)
            c_i_free = c_i_free*normalizer
            g_zs_free_ngts = g_zs.ngts(c_i_free)

        phi_z_free = compose(g_zs_free, g_zs_free_ngts, g_z)
    else:
        phi_z_free = 0

    return phi_z_ta + phi_z_free

def phi_jz_avg(phi_jz):
    """ Convolve the transition matrix with the density field.

        For now, treat ends with special cases, add j[0] to phi[0], j[J] to phi[Z]
        later, pass in phi_b_below_j, phi_b_above_j and add to each correlation
    """
    types, layers = phi_jz.shape
    avg = np.empty_like(phi_jz)

    for j, phi_z in enumerate(phi_jz):
        avg[j] = correlate(phi_z, LAMBDA_ARRAY, 1)

    avg[0,0] += LAMBDA_1
#    avg[types-1,layers-1] += LAMBDA_1

    return avg


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

class Propagator():

    def __init__(self, g_z, segments):
        self.g_z = g_z
        self.shape = g_z.size, segments

    def ta(self):
        # terminally attached beginings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = 0.0
        g_zs[0, 0] = self.g_z[0]
        self._calc_g_zs_uniform(self.g_z, g_zs)
        return g_zs

    def free(self):
        # free beginnings
        # forward propagator

        g_zs = self._new()
        g_zs[:, 0] = self.g_z
        self._calc_g_zs_uniform(self.g_z, g_zs)
        return g_zs

    def ngts_u(self, c):
        # free ends of uniform chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c * self.g_z
        self._calc_g_zs_uniform(self.g_z, g_zs)
        return g_zs

    def ngts(self, c_i):
        # free ends of disperse chains
        # reverse propagator

        g_zs = self._new()
        g_zs[:, 0] = c_i[-1] * self.g_z
        self._calc_g_zs(self.g_z, c_i, g_zs)
        return g_zs

    def _new(self):
        return np.empty(self.shape, order='F')

    if JIT:
        @staticmethod
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

        @staticmethod
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

    elif PYONLY:
        @staticmethod
        def _calc_g_zs(g_z, c_i, g_zs):
            pg_zs = g_zs[:, 0]
            segment_iterator = enumerate(c_i[::-1])
            next(segment_iterator)
            for r, c in segment_iterator:
                g_zs[: ,r] = pg_zs = (correlate(pg_zs, LAMBDA_ARRAY, 1) + c) * g_z

        @staticmethod
        def _calc_g_zs_uniform(g_z, g_zs):
            segments = g_zs.shape[1]
            pg_zs = g_zs[:, 0]
            for r in range(1, segments):
                g_zs[:, r] = pg_zs = correlate(pg_zs, LAMBDA_ARRAY, 1) * g_z

    else:
        @staticmethod
        def _calc_g_zs(g_z, c_i, g_zs):
            return _calc_g_zs_cex(g_z, c_i, g_zs, LAMBDA_0, LAMBDA_1)

        @staticmethod
        def _calc_g_zs_uniform(g_z, g_zs):
            return _calc_g_zs_uniform_cex(g_z, g_zs, LAMBDA_0, LAMBDA_1)

if 0:
    u_jz0 = np.zeros((3,100))

    x_av=1 # goal: >1
    x_ws=-0.6 # goal: -1.5
    x_vw = 2.5 # goal: 3.5
    x_as=x_ws-1
    x_av=-x_ws
    x_sv=0
    x_aw=0
    chi_jk = np.array((
                (0,x_ws,x_as,x_sv),
                (x_ws,0,x_aw,x_vw),
                (x_as,x_aw,0,x_av),
                (x_sv,x_vw,x_av,0),
                ))
    chi_jk = (1-np.eye(4))*0.10

    sigma_j = np.array((0,0,.01,0))
    phi_b_j = np.array((0,0.1,0,.9))
    n_avg_j = np.array((0,1,175,1))

    p_ji = None
    def curried(phi, dump=0):
        return SCFeqns_multi(phi,chi_jk, sigma_j, phi_b_j, n_avg_j,
                             dump_phi=dump)

    start = time()
    for x in range(10):
        ans=SCFsolve_multi(chi_jk, sigma_j, phi_b_j, n_avg_j)
#    ans=newton_krylov(curried,u_jz0,verbose=1, maxiter=None, method='gmres')
    print(time()-start)
    print(repr(ans))
#    import matplotlib.pyplot as plt
#    phi = SCFeqns_multi(ans,chi_jk, sigma_j, phi_b_j, n_avg_j,dump_phi=1)
#    plt.plot(phi.T,'x-')
