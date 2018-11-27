# -*- coding: utf-8 -*-
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

import numpy as np
import scipy as sp
import os
from math import exp
from scipy.signal import convolve
from scipy.stats.distributions import norm
from scipy.stats import lognorm
import sys
from btr_helper_temporal import tau_update_temporal

def load_obs_function(name = 'Gaussian'):
	""" Load one of two observation models, either 'Gaussian' (default) or 'Cauchy' """
	global tau_update_obs
	if name == 'Gaussian':
		from btr_helper_gaussian import tau_update_obs
	elif name == 'Cauchy':
		from btr_helper_cauchy import tau_update_obs
	else:
		raise RuntimeError('Unknown observation function ' + name)

class DegeneratedLatentSpace(RuntimeError):
	def __init__(self, arg):
		super(DegeneratedLatentSpace, self).__init__(arg)


class BayesianTrajectoryReplay(object):
	""" A controller using all recorded trajectories and doing bayesian inference """
	
	def __init__(self, data_logs, training_idx, theta_tau, theta_I, alpha, temporal_res, obs_add_eps, weights = None, debug_dump = False):
		""" Construct using training_idx of data_logs.run, mutate data_logs (dilate using temporal_res) """
		
		self.temporal_res = temporal_res
		
		# check value ranges
		if theta_tau <= 0:
			raise RuntimeError('Invalid theta_tau value: ' + str(theta_tau))
		if theta_I < 0 or theta_I >1:
			raise RuntimeError('Invalid theta_I value: ' + str(theta_I))
		if alpha < 0 or alpha > 1:
			raise RuntimeError('Invalid alpha value: ' + str(alpha))
		if obs_add_eps < 0:
			raise RuntimeError('Invalid obs_add_eps value: ' + str(obs_add_eps))
		#print('obs_add_eps: ' + str(obs_add_eps))
		
		# Build latent space
		
		# find limit for temporal transition model window
		def p_d_tau_f(x):
			return lognorm.pdf(x, theta_tau)
		modePos = exp(-theta_tau*theta_tau)
		modeValue = p_d_tau_f(modePos)
		p_d_tau = []
		for i in range(1, temporal_res+1):
			p_d_tau.append(p_d_tau_f(float(i)/temporal_res))
		i = temporal_res + 1
		while p_d_tau[-1] > modeValue * 0.02:
			p_d_tau.append(p_d_tau_f(float(i)/temporal_res))
			i += 1
		
		# simplified p(τ_t|τ_{t-1},I_{t-1}) using convolution
		# normal (no subsampling)
		#self.p_d_tau = np.array([0, 0, theta_tau, 1.-2.*theta_tau, theta_tau])
		# subsampling log-normal
		#self.p_d_tau = np.array(([0.]*(len(p_d_tau)+1)) + p_d_tau)
		self.p_d_tau = np.array([0.] + p_d_tau)
		self.p_d_tau /= self.p_d_tau.sum()
		#print('Temporal transition model:' + str(self.p_d_tau))
		
		# Π
		self.data = []
		self.data_descr = data_logs.descr
		self.norm = data_logs.norm
		if weights:
			self.weights = np.kron(weights, np.ones((temporal_res,1)))
		else:
			self.weights = []
			for i in training_idx:
				row_count, col_count = data_logs.runs[i].sensor_norm.shape
				self.weights.append(np.ones((row_count*temporal_res, col_count)))
		
		# create and initialize latent space
		self.traj_count = len(training_idx)
		idx_count_f = float(self.traj_count)
		# p(I) marginals
		self.p_I = np.empty((self.traj_count),dtype=float)
		self.p_I[:] = 1./idx_count_f
		# p(I_t,τ_t)
		self.p_tau_I = []
		max_len = 0
		for idx in training_idx:
			run = data_logs.runs[idx]
			self.data.append(run)
			a = np.zeros( (run.sample_count * temporal_res) )
			max_len = max(max_len, len(a))
			a[0] = 1./idx_count_f
			self.p_tau_I.append(a)
		# p(I'|I) transition matrix
		norm_p_I_eps = theta_I
		main_I_factor = (1. - norm_p_I_eps)
		other_I_factor = norm_p_I_eps / (idx_count_f - 1.)
		self.p_I_trans = np.identity(self.traj_count, dtype=float)
		self.p_I_trans *= main_I_factor - other_I_factor
		self.p_I_trans += other_I_factor
		self.alpha = alpha
		self.obs_add_eps = obs_add_eps
		
		# reference prior for observation
		self.priorZmin = data_logs.priorZmin
		self.invPriorZrange = data_logs.invPriorZrange
		self.priorZ = data_logs.priorZ
		
		# debug
		self.debug_img = np.zeros((self.traj_count, max_len))
		self.debug_dump = debug_dump
		
	def _normalize_latent_space(self):
		# sums unnormalized probabilities
		tot = 0.
		for i, p_tau in enumerate(self.p_tau_I):
			# sanity check; only required for debug sessions:
			#assert np.min(p_tau) >= 0.
			sum_p_tau = np.sum(p_tau)
			tot += sum_p_tau
			self.p_I[i] = sum_p_tau
		
		# sanity check; keep all the time
		assert np.isfinite(tot)
		if tot < 1e-150:
			raise DegeneratedLatentSpace('tot < 1e-150 : ' + str(tot))
		if tot > 1e150:
			raise DegeneratedLatentSpace('tot > 1e150 : ' + str(tot))
		
		# renormalize space
		for p_tau in self.p_tau_I:
			p_tau /= tot
			# sanity check; keep all the time
			max_p_tau = np.max(p_tau)
			assert np.isfinite(max_p_tau)
			if max_p_tau > 1e150:
				raise DegeneratedLatentSpace('max(p_tau) > 1e150 : ' + str(max_p_tau))
		
		# post-normalization sanity check; keep all the time
		assert all(np.isfinite(self.p_I))
		min_corr_f = np.min(self.p_I)
		if min_corr_f < 1e-150:
			raise DegeneratedLatentSpace('min(abs(p_I)) < 1e-150 : ' + str(min_corr_f))
		max_corr_f = np.max(self.p_I)
		if max_corr_f > 1e150:
			raise DegeneratedLatentSpace('max(abs(p_I)) > 1e150 : ' + str(max_corr_f))
		
		return tot
		
	def _step_latent_space(self):
		for p_tau in self.p_tau_I:
			p_tau[:] = tau_update_temporal(p_tau, self.p_d_tau)
	
	def _get_I_corr_factor(self):
		corr_f = np.dot(self.p_I_trans, self.p_I) / self.p_I
		assert all(np.isfinite(corr_f))
		assert np.min(corr_f) > 1e-150
		return corr_f
	
	def update_latent_image(self):
		for i, p_tau in enumerate(self.p_tau_I):
			#self.debug_img[i,0:100] = p_tau[0:100]
			self.debug_img[i,0:len(p_tau)] = p_tau
	
	def get_command(self):
		""" Get command """
		assert self.data
		command = np.zeros((self.data[0].motor_raw.shape[1],))
		for p_tau, run in zip(self.p_tau_I, self.data):
			reshaped_p_tau = p_tau.reshape((run.motor_raw.shape[0], self.temporal_res))
			summed_p_tau = reshaped_p_tau.sum(axis=1)
			command += np.dot(summed_p_tau, run.motor_raw)
		return command
	
	def apply_temporal_transition(self):
		self._step_latent_space()
		# the following code displays information about taus and Is that contribute significantly to the command
		#for i, p_tau in enumerate(self.p_tau_I):
			#for j, v in enumerate(p_tau):
				#if v > 1e-3:
					#print i, j, v, self.weights[i][j], self.data[i].motor_raw[j/self.temporal_res], self.data[i].sensor_norm[j/self.temporal_res]
		
	
	def apply_observation(self, Z):
		""" Apply normalized observation, return whether the application was a success (led to non-denerate latent space) """
		
		# make sure Z is within bounds
		Z = np.minimum(Z, self.norm.normalize_sensor_values(self.data_descr.sensor_max))
		Z = np.maximum(Z, self.norm.normalize_sensor_values(self.data_descr.sensor_min))
		
		# Note that we assume that we executed get_command() prior to this
		# call, that called _step_latent_space()
		tot = sum(self.p_I)
		if self.debug_dump:
			print 'prob tot and p_I before obs: ', tot, self.p_I
			print 'obs: ', Z, ' '
		norm_res = self.norm.normalized_sensor_resolution()
		if self.debug_dump:
			print 'sd:  ', norm_res
		# apply observation
		corr_f = self._get_I_corr_factor()
		if self.debug_dump:
			print 'corr_f: ', corr_f
		sensor_range = self.norm.normalized_sensor_range()
		for p_tau, run, w, i in zip(self.p_tau_I, self.data, self.weights, range(len(self.p_tau_I))):
			#print i, p_tau
			tau_update_obs(p_tau, Z, run.sensor_norm, norm_res, sensor_range, w, corr_f[i], self.alpha, self.obs_add_eps, i, self.temporal_res)
			assert all(np.isfinite(p_tau))
		# normalization
		tot = self._normalize_latent_space()
		if self.debug_dump:
			print 'prob tot after obs: ', tot


# load the Gaussian observation function
load_obs_function()
