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

from util import (schultz_zimm, sumkd, meankd, MINLAT, MINBULK, NotImplementedAttribute,
                LAMBDA_1, LAMBDA_ARRAY, correlate, _calc_g_zs,
                _calc_g_zs_uniform, compose, fastsum)

import numpy as np
from time import time
from collections import OrderedDict
from numpy import exp, log, fabs
from scipy.optimize.nonlin import newton_krylov, NoConvergence


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
    bs = BasicSystem()
    parameters = (chi,chi_s,pdi,sigma,phi_b,segments)
    u = bs.walk(parameters,disp)
    phi_lat = bs.field_equations(parameters)(u, 1).squeeze()
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
    bs = BasicSystem()
    parameters = (chi,chi_s,pdi,sigma,phi_b,segments)
    u = bs.walk(parameters,disp).squeeze()
    squeezing = layers - len(u) < 0

    jac_solve_method = 'gmres'
    while layers - len(u):
        if squeezing:
            u = np.delete(u, -1)
        else:
            u = np.append(u, u[-1])
        u = newton_krylov(bs.field_equations(parameters),
                                 u[None,:],
                                 verbose=bool(disp),
                                 maxiter=30,
                                 method=jac_solve_method,
                                 ).squeeze()


    phi = bs.field_equations(parameters)(u,1).squeeze()

    return phi


class BaseSystem(object):
    """ base class for physical system encapulators

        should produce wrapped field equations from a parameters tuple
        should cache every valid field equation solution across all instances
        should have methods for managing the cache without exposing it
        manages scaling of parameters for the cache so the user never
        sees scaled parameters
    """

    _cache = NotImplementedAttribute # equivalent of NotImplementedError
    _cache_limit = 100

    _scale = NotImplementedAttribute
    _offset = NotImplementedAttribute

    def __init__(self, disp=False):
        if not self._cache:
            self._prime_cache(disp)

    def _prime_cache(self):
        raise NotImplementedError

    def field_equations(self, parameters):
        raise NotImplementedError

    def in_cache(self, key):
        return key in self._cache

    def from_cache(self, key):
        self._cache.move_to_end(key)
        return self._cache[key]

    def add_to_cache(self, key, value):
        self._cache[key] = value
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

    def cached_parameters(self):
        return tuple(dict.__iter__(self._cache))

    def scale_parameters(self, param):
        return tuple(s*(p-o) for p,s,o in zip(param, self._scale, self._offset))

    def unscale_parameters(self, param):
        return tuple(p/s+o for p,s,o in zip(param, self._scale, self._offset))

    def walk(self, unscaled_parameters, disp=False):
        """Return a memoized SCF result by walking from a previous solution.

        """

        if disp: starttime = time()

        scaled_parameters = self.scale_parameters(unscaled_parameters)

        # longshot, but return a cached result if we hit it
        if self.in_cache(scaled_parameters):
            if disp: print('cache hit at:', unscaled_parameters)
            return self.from_cache(scaled_parameters)

        # Find the closest parameters in the cache: O(len(cache))

        # Numpy setup
        cached_parameters = self.cached_parameters()
        cp_array = np.array(cached_parameters)
        p_array = np.array(scaled_parameters)

        # Calculate distances to all cached parameters
        deltas = p_array - cp_array # Parameter space displacement vectors
        closest_index = np.sum(deltas*deltas,axis=1).argmin()

        # Organize closest point data for later use
        closest_cp = cached_parameters[closest_index]
        closest_cp_array = cp_array[closest_index]
        closest_delta = deltas[closest_index]

        u = self.from_cache(closest_cp)

        if disp:
            print("Walking from nearest:", self.unscale_parameters(closest_cp))
            print("to:", unscaled_parameters)

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

            up_tup = self.unscale_parameters(p_tup)

            if disp:
                print('Parameter step is', step)
                print('current parameters:', up_tup)

            fe = self.field_equations(up_tup)

            try:
                u = SCFsolve(fe, u, disp)
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
                self.add_to_cache(p_tup, u)
                dstep *= 1.05
                step += dstep

        if disp: print('walk execution time:', round(time()-starttime,3), "s")

        return u


class BasicSystem(BaseSystem):
    """ Physcal system representing homopolymer solvated by a single solvent

    """

    _scale = (1,3,1,1,1,1/500)
    _offset = (0,0,1,0,0,0)
    _cache = OrderedDict()

    def _prime_cache(self, disp=False):
        x0 = np.random.normal(0,.001,(1,MINLAT))

        p = (0,0,0,.1,.1,.1)
        up = self.unscale_parameters(p)
        fe = self.field_equations(up)
        self._cache[p] = SCFsolve(fe,x0,disp)
        p = (0,0,0,0,.1,.1)
        up = self.unscale_parameters(p)
        fe = self.field_equations(up)
        self._cache[p] = SCFsolve(fe,x0,disp)
        p = (0,0,0,.1,0,.1)
        up = self.unscale_parameters(p)
        fe = self.field_equations(up)
        self._cache[p] = SCFsolve(fe,x0,disp)

    def field_equations(self, parameters):
        """ Accept parameters and return the corresponding field equations
            (a function that requires one argument, a 2D ndarray).
        """

        chi, chi_s, pdi, sigma, phi_b, n_avg = parameters

        if sigma >= 1:
            raise ValueError('Chains that short cannot be squeezed that high!')

        if phi_b >= 1:
            raise ValueError('Bulk chain concentration impossibly high!')

        p_i = schultz_zimm(pdi,n_avg)

        # TODO: any utility in using functools.partial?
        def wrapped_field_equations(u_jz, dump_phi=False):
            return SCFeqns(u_jz.squeeze(), chi, chi_s, sigma, n_avg, p_i, phi_b,
                           dump_phi)[None,:]

        return wrapped_field_equations


class VaporSwollenSystem(BaseSystem):
    """ Physcal system representing homopolymer solvated by a single vapor
        according to Cohen-Stuart, de Vos, and Leermakers (2006)

    """

    _scale = (1,)*7
    _offset = (0,)*7
    _cache = OrderedDict()

    def _prime_cache(self, disp=False):
        x0 = np.random.normal(0,.001,(3,MINLAT))

        p = (1, -.6, 2.5, .01, .1, 75, 1)
        fe = self.field_equations(p)
        self._cache[p] = SCFsolve(fe,x0,disp)

    def field_equations(self, parameters):
        """ Accept parameters and return the corresponding field equations
            (a function that requires one argument, a 2D ndarray).

            x_av = air-vapor chi.  goal: >1
            x_ws = water-surface chi.  goal: -1.5
            x_vw = vapor-water chi.  goal: 3.5

            parameters = x_av, x_ws, x_vw, sigma_a, phi_b_w, n_avg, pdi
        """

        x_av, x_ws, x_vw, sigma_a, phi_b_w, n_avg, pdi = parameters

        # Build an interaction matrix

        x_as = x_ws-1
        x_av = -x_ws
        x_sv = 0
        x_aw = 0
        chi_jk = np.array((
                           (0.00, x_ws, x_as, x_sv),
                           (x_ws, 0.00, x_aw, x_vw),
                           (x_as, x_aw, 0.00, x_av),
                           (x_sv, x_vw, x_av, 0.00),
                           ))

        sigma_j = np.array((0.00, 0.00, sigma_a, 0.00))
        if np.sum(sigma_j) >= 1:
            raise ValueError('Surface overloaded with grafted species')

        phi_b_j = np.array((0.00, phi_b_w, 0.00, 1-phi_b_w))
        if np.sum(phi_b_j) != 1:
            raise ValueError('Bulk volume fractions should add up to 1')

        if n_avg < 1:
            raise ValueError('Invalid number average molecular weight')
        n_avg_j = np.array((0.00, 1.0, n_avg, 1.0))

        # XXX: this will be useful someday, but for now it's excess work
    #    pdi_j = (None, None, pdi, None)
    #    if pdi_j is None:
    #        p_ji = None
    #    else:
    #        p_ji = [schultz_zimm(pdi, n_avg) if n_avg > 1 else None
    #                for n_avg, pdi in zip(n_avg_j, pdi_j)]
        if pdi is None:
            p_ji = None
        else:
            p_ji = (None, None, schultz_zimm(pdi, n_avg), None)


        def wrapped_field_equations(u_jz, dump_phi=False):
            return SCFeqns_multi(u_jz, chi_jk, sigma_j, phi_b_j, n_avg_j, p_ji,
                                 None, dump_phi)

        return wrapped_field_equations


def SCFsolve(field_equations, u_jz_guess, disp=False, maxiter=30):
    """Solve SCF equations using an initial guess and lattice parameters

    This function finds a solution for the equations where the lattice size
    is sufficiently large.

    The Newton-Krylov solver really makes this one. With gmres, it was faster
    than the other solvers by quite a lot.
    """

    if disp: starttime = time()

    # resizing loop variables
    jac_solve_method = 'gmres'
    lattice_too_small = True

    # We tolerate up to 1 ppm deviation from bulk
    # when counting layers_near_bulk
    tol = 1e-6

    while lattice_too_small:
        if disp: print("Solving SCF equations")

        try:
            u_jz = newton_krylov(field_equations,
                                 u_jz_guess,
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

        field_deviation = fabs(u_jz).sum(axis=0)
        layers_near_bulk = field_deviation < tol
        nbulk = layers_near_bulk.sum()
        lattice_too_small = nbulk < MINBULK

        if lattice_too_small:
            # if there aren't enough layers_near_zero, grow the lattice 20%
            newlayers = max(1, round((u_jz_guess.shape[1])*0.2))
            if disp: print('Growing undersized lattice by', newlayers)

            # find the layer closest to eqm with the bulk
            i = field_deviation.argmin()

            # make newlayers there via linear interpolation for each species
            interpolation = [np.linspace(field_z[i-1],
                                         field_z[i],
                                         num=newlayers) for field_z in u_jz]

            # then sandwich the interpolated layers between the originals
            # interpolation includes the first and last points, so shift i
            u_jz_guess = np.hstack((u_jz[:,:i-1],
                                    interpolation,
                                    u_jz[:,i+1:]))

            # TODO: vectorize this interpolation and stacking?

    if nbulk > 2*MINBULK:
        chop_end = np.diff(layers_near_bulk).nonzero()[0].max()
        chop_start = chop_end - MINBULK
        i = np.arange(u_jz.shape[1])
        u_jz = u_jz[:,(i <= chop_start) | (i > chop_end)]

    if disp:
        print("SCFsolve execution time:", round(time()-starttime,3), "s")
        print('lattice size:', u_jz.shape[1])

    return u_jz


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

    return eps_jz


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


def SCFeqns(u_z, chi, chi_s, sigma, n_avg, p_i, phi_b=0, dump_phi=False):
    """ System of SCF equation for terminally attached polymers.

        Formatted for input to a nonlinear minimizer or solver.

        The sign convention here on u is "backwards" and always has been.
        It saves a few sign flips, and looks more like Cosgrove's.
    """

    g_z = exp(u_z)

    # normalize g_z for numerical stability
    u_z_avg = np.mean(u_z)
    g_z_norm = g_z/exp(u_z_avg)

    phi_z = calc_phi_z(g_z_norm, n_avg, sigma, phi_b, u_z_avg, p_i)

    if dump_phi:
        return phi_z

    # penalize attempts that overfill the lattice
    toomuch = phi_z>.99999
    penalty_flag = toomuch.any()
    if penalty_flag:
        penalty = np.where(toomuch, phi_z-.99999, 0)
        phi_z[toomuch] = .99999

    # calculate new potentials
    u_prime = log((1.0 - phi_z)/(1.0 - phi_b))
    u_int = 2*chi*(correlate(phi_z, LAMBDA_ARRAY, 1)-phi_b)
    u_int[0] += chi_s
    u_z_new = u_prime + u_int

    eps_z = u_z - u_z_new
    if penalty_flag:
        np.copysign(penalty, eps_z, penalty)
        eps_z += penalty

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


if 0:# __name__ == '__main__':
#    bs = BasicSystem()
#    param = 0,0,1,.1,0,160.52
#    param = bs.scale_parameters(param)
#    for _ in range(1):
#        ans = SCFwalk(param, bs, 1)
#
#    import matplotlib.pyplot as plt
#    phi = bs.field_equations(param)(ans, dump_phi=1)
#    plt.plot(phi.T, 'x-')

    vs = VaporSwollenSystem()
    param = (.1, -.15, .35, .01, .1, 75, 1)
    phi = vs.walk(param,1)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(phi.T)