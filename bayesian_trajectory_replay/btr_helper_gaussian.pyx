# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import math
from libc.math cimport sqrt, exp, erf, fabs
cimport numpy as np
cimport cython

# types and helper functions

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef double _1sqrt2pi = 1. / sqrt(2. * math.pi)
cdef double _sqrt2 = sqrt(2.)

@cython.cdivision(True) # turn off division-by-zero checking
cdef double _norm(double x, double u, double s):
	cdef double factor = _1sqrt2pi / s
	cdef double dxus = (x - u) / s
	return factor * exp(- (dxus * dxus) / 2.)

@cython.cdivision(True) # turn off division-by-zero checking
cdef double _convolve(double x_t, double x_tp1, double sigma, double mu):
	cdef double dxt = x_tp1 - x_t
	if fabs(dxt) < 1e-150:
		return _norm(x_t, mu, sigma) / 2.
	cdef double x_tmmu = x_t - mu
	cdef double x_tp1mmu = x_tp1 - mu
	cdef double sqrt2s2 = _sqrt2 * sigma
	return (erf( x_tp1mmu / sqrt2s2) - erf( x_tmmu / sqrt2s2)) / (dxt * 4.)


# main functions

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.cdivision(True) # turn off division-by-zero checking
def tau_update_obs(np.ndarray[DTYPE_t, ndim=1] p_tau not None, np.ndarray[DTYPE_t, ndim=1] Z not None, np.ndarray[DTYPE_t, ndim=2] sensor_norm not None, np.ndarray[DTYPE_t, ndim=1] sigmas not None, np.ndarray[DTYPE_t, ndim=1] sensor_range not None, np.ndarray[DTYPE_t, ndim=2] weights not None, double I_factor, double alpha, double additive_eps, int traj_idx, int temporal_res):
	
	assert p_tau.dtype == DTYPE and Z.dtype == DTYPE and sensor_norm.dtype == DTYPE and weights.dtype == DTYPE and sensor_range.dtype == DTYPE
	assert Z.shape[0] == sensor_norm.shape[1] == sigmas.shape[0] == weights.shape[1] == sensor_range.shape[0]
	assert sensor_norm.shape[0]*temporal_res == p_tau.shape[0] == weights.shape[0]
	assert all(sigmas > 0)
	assert all(sensor_range > 0)
	assert temporal_res > 0
	
	cdef np.ndarray[DTYPE_t, ndim=1] uniform = alpha/sensor_range
	cdef double beta = 1. - alpha
	cdef int l = p_tau.shape[0]
	cdef int l_data = sensor_norm.shape[0]
	cdef int c = Z.shape[0]
	cdef int i, j, i_data
	cdef double m
	cdef double d_j
	cdef double factor
	cdef double w
	cdef double v
	
	for i in range(l):
		m = I_factor
		# the length of sensor_norm is 1/temporal_res of the latent space, hence this division
		i_data = i / temporal_res
		for j in range(c):
			d_j = 0.
			factor = 1.
			if i_data != 0:
				d_j += _convolve(sensor_norm[i_data-1,j], sensor_norm[i_data,j], sigmas[j], Z[j])
			else:
				factor = 2.
			if i_data != l_data-1:
				d_j += _convolve(sensor_norm[i_data,j], sensor_norm[i_data+1,j], sigmas[j], Z[j])
			else:
				factor = 2.
			d_j *= factor
			#print traj_idx, i, j, d_j, Z[j], sensor_norm[i_data,j]
			w = weights[i, j]
			m *= (beta * (d_j + additive_eps) + uniform[j]) ** w
		p_tau[i] *= m

