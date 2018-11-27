# -*- coding: utf-8 -*-
# cython: profile=False
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
cimport numpy as np
cimport cython

# types

DTYPE = np.double
ctypedef np.double_t DTYPE_t

# main functions

@cython.boundscheck(False) # turn off bounds-checking for entire function
def tau_update_temporal(np.ndarray[DTYPE_t, ndim=1] p_tau not None, np.ndarray[DTYPE_t, ndim=1] p_update not None):
	
	cdef int l_p_tau = len(p_tau)
	cdef int l_p_update = len(p_update)
	cdef int safe_l = l_p_tau - l_p_update
	
	assert safe_l >= 0
	
	cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros([len(p_tau)], dtype=DTYPE)
	cdef DTYPE_t value
	cdef int i, j, idx
	
	# first safe range
	for i in range(safe_l):
		for j in range(l_p_update):
			out[i+j] += p_tau[i] * p_update[j]
	
	# then end of range, slower but accumulate overflow in last element
	for i in range(safe_l, l_p_tau):
		for j in range(l_p_update):
			idx = min(i+j, l_p_tau-1)
			out[idx] += p_tau[i] * p_update[j]
	
	# return data
	return out