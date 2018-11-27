# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import math
from libc.math cimport sqrt, exp, erf, fabs, atan
cimport numpy as np
cimport cython

# types and helper functions

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef double _pi = math.pi
cdef double _1pi = 1. / math.pi

@cython.cdivision(True) # turn off division-by-zero checking
cdef double _cauchy(double x, double x0, double one_over_gamma):
	cdef double factor = (x-x0) * one_over_gamma
	return (_1pi * one_over_gamma) / (1 + factor*factor)

@cython.cdivision(True) # turn off division-by-zero checking
cdef double _convolve(double x_t, double x_tp1, double one_over_gamma, double z):
	cdef double dxt = x_tp1 - x_t
	if fabs(dxt) < 1e-150:
		return _cauchy(z, x_t, one_over_gamma)
	cdef double factor = 1. / (_pi * dxt)
	return factor * (atan((x_tp1-z)*one_over_gamma) - atan((x_t-z)*one_over_gamma))

# main functions

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.cdivision(True) # turn off division-by-zero checking
def tau_update_obs(np.ndarray[DTYPE_t, ndim=1] p_tau not None, np.ndarray[DTYPE_t, ndim=1] Z not None, np.ndarray[DTYPE_t, ndim=2] sensor_norm not None, np.ndarray[DTYPE_t, ndim=1] gammas not None, np.ndarray[DTYPE_t, ndim=1] sensor_range not None, np.ndarray[DTYPE_t, ndim=2] weights not None, double I_factor, double alpha, double additive_eps, int traj_idx, int temporal_res):
	
	assert p_tau.dtype == DTYPE and Z.dtype == DTYPE and sensor_norm.dtype == DTYPE and weights.dtype == DTYPE and sensor_range.dtype == DTYPE
	assert Z.shape[0] == sensor_norm.shape[1] == gammas.shape[0] == weights.shape[1] == sensor_range.shape[0]
	assert sensor_norm.shape[0]*temporal_res == p_tau.shape[0] == weights.shape[0]
	assert all(gammas > 0)
	assert all(sensor_range > 0)
	assert temporal_res > 0
	
	cdef np.ndarray[DTYPE_t, ndim=1] one_over_gammas = 1./gammas
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
	
	#print traj_idx, ' tau_update_obs', I_factor, additive_eps, gammas
	
	for i in range(l):
		m = I_factor
		# the length of sensor_norm is 1/temporal_res of the latent space, hence this division
		i_data = i / temporal_res
		for j in range(c):
			d_j = 0.
			factor = 1.
			if i_data != 0:
				d_j += _convolve(sensor_norm[i_data-1,j], sensor_norm[i_data,j], one_over_gammas[j], Z[j])
			else:
				factor = 2.
			if i_data != l_data-1:
				d_j += _convolve(sensor_norm[i_data,j], sensor_norm[i_data+1,j], one_over_gammas[j], Z[j])
			else:
				factor = 2.
			d_j *= factor
			w = weights[i, j]
			m *= (beta * (d_j + additive_eps) + uniform[j]) ** w
		#if I_factor != 0 and m/I_factor > 1e-3:
		#	print traj_idx, i, m/I_factor, sensor_norm[i_data], weights[i]
		p_tau[i] *= m


