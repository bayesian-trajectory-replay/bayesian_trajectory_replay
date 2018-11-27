#!/usr/bin/env python

# -*- coding: utf-8 -*-
# kate: replace-tabs off; indent-width 4; indent-mode normal
# vim: ts=4:sw=4:noexpandtab

from bayesian_trajectory_replay.data_log import *
from bayesian_trajectory_replay.bayesian_trajectory_replay import *


test_sensor_data = [
	[1, 1],
	[1, 1],
	[1, 1],
	[1, 1],
	[1, 1],
	[1, 0],
	[0, 0],
	[0, 0]
]


def main():
	# load data and setup parameters
	data_logs = DataLogs('config.yaml', normalization='none')
	theta_tau = 0.3
	theta_I = 1e-8
	alpha = 1e-2
	temporal_res = 3
	obs_add_eps = 1e-100
	
	# create replay
	replay = BayesianTrajectoryReplay(data_logs, range(len(data_logs.runs)), theta_tau, theta_I, alpha, temporal_res, obs_add_eps)
	
	for sensor_data in test_sensor_data:
		# consider observation 
		print 'Sensor data:', sensor_data
		normalized_sensor_data = data_logs.norm.normalize_sensor_values(sensor_data)
		replay.apply_observation(normalized_sensor_data)
		
		# check if we have finished the sequence
		p_of_last_ts = sum(p_tau[-1] for p_tau in replay.p_tau_I)
		if p_of_last_ts > 0.9:
			print 'Finished with p = ', p_of_last_ts
			return
		
		# compute command
		U = replay.get_command()
		print 'Cmd:', U
		
		# apply temporal transition
		replay.apply_temporal_transition()
	
	# this should never happen
	p_of_last_ts = sum(p_tau[-1] for p_tau in replay.p_tau_I)
	print 'Oups, algorithm did not finish before the robot died (p =', p_of_last_ts, ')'


if __name__ == '__main__':
	main()
