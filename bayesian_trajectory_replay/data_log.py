import csv
import numpy as np
import sys
import yaml
import os
import collections

class DataLog(object):
	"""Data from a single run, pre-filtered by the constructor """
	
	def __init__(self, dirname, filename, descr, data_preprocessors):
		""" Construct by loading data """
		
		# Load data
		raw_data = np.genfromtxt(open(os.path.join(dirname, filename)))
		
		# process data
		if data_preprocessors:
			try:
				for data_preprocessor in data_preprocessors:
					raw_data = data_preprocessor(raw_data)
			except TypeError:
				raw_data = data_preprocessors(raw_data)
		
		# build container and views
		nr = raw_data.shape[0]
		sensor_count = descr.sensor_count
		motor_count = descr.motor_count
		sensor_and_motor_raw = np.empty((nr, sensor_count+motor_count), dtype=float)
		sensor_raw = sensor_and_motor_raw[:,0:sensor_count]
		motor_raw = sensor_and_motor_raw[:,sensor_count:sensor_count+motor_count]
		if 'time' in descr.by_name:
			time = np.empty((nr, 1), dtype=float)
		else:
			time = np.linspace(0,nr-1,nr)
		if 'segmentation_ground_truth' in descr.by_name:
			self.segmentation_ground_truth = np.empty((nr, 1), dtype=int)
		
		# copy data using views
		sensor_id = 0
		motor_id = 0
		for col in range(raw_data.shape[1]):
			if col in descr.by_orig_number:
				col_type = descr.by_orig_number[col]
				raw_value = raw_data[:,col]
				if col_type == 'sensor':
					assert sensor_id < sensor_count
					sensor_raw[:,sensor_id] = raw_value
					sensor_id += 1
				elif col_type == 'motor':
					assert motor_id < motor_count
					motor_raw[:,motor_id] = raw_value
					motor_id += 1
				elif col_type == 'time':
					time = raw_data[:,col].astype(float)
				elif col_type == 'segmentation_ground_truth':
					self.segmentation_ground_truth = raw_data[:,col]
				elif col_type == 'unused':
					pass
				else:
					print 'Warning, unknown column type ' + col_type + ', ignoring'
		
		# check integrity
		if sensor_id != descr.sensor_count:
			raise RuntimeError('Data file ' + filename + ' has not enough sensor data, wanted ' + str(descr.sensor_count) + ' channels, found ' + str(sensor_id))
		if motor_id != descr.motor_count:
			raise RuntimeError('Data file ' + filename + ' has not enough motor data, wanted ' + str(descr.motor_count) + ' channels, found ' + str(motor_id))
		
		# store useful variables inside self
		self.sample_count = nr
		self.sensor_and_motor_raw = sensor_and_motor_raw
		self.sensor_raw = sensor_raw
		self.motor_raw = motor_raw
		self.time = time
		self.filename = filename
		self.descr = descr
	
	def normalize(self, norm):
		""" Normalize data using norm """
		
		sc = norm.sensor_count_norm
		mc = norm.motor_count_norm
		self.sensor_and_motor_norm = np.empty((self.sample_count, sc+mc), dtype=float)
		self.sensor_norm = self.sensor_and_motor_norm[:,0:sc]
		self.motor_norm = self.sensor_and_motor_norm[:,sc:sc+mc]
		self.sensor_norm[:] = norm.normalize_sensor_values(self.sensor_raw)
		self.motor_norm[:] = norm.normalize_motor_values(self.motor_raw)
	
	def dilateTemporally(self, factor):
		""" Temporally dilate data of a given factor """
		
		def scale(a):
			return np.kron(a, np.ones((factor,1)))
		
		# raw data
		self.sample_count *= factor
		self.sensor_and_motor_raw = scale(self.sensor_and_motor_raw)
		sc = self.descr.sensor_count
		mc = self.descr.motor_count
		self.sensor_raw = self.sensor_and_motor_raw[:,0:sc]
		self.motor_raw = self.sensor_and_motor_raw[:,sc:sc+mc]
		self.time = scale(self.time)
		if hasattr(self, 'segmentation_ground_truth'):
			self.segmentation_ground_truth = scale(self.segmentation_ground_truth)
		# normalized data
		self.sensor_and_motor_norm = scale(self.sensor_and_motor_norm)
		sc = self.sensor_norm.shape[1]
		mc = self.motor_norm.shape[1]
		self.sensor_norm = self.sensor_and_motor_norm[:,0:sc]
		self.motor_norm = self.sensor_and_motor_norm[:,sc:sc+mc]
		
		
class ChannelDescription(object):
	""" A description of a channel """
	
	def __init__(self, name, min_value, max_value, value_range, resolution, step_value):
		""" Construct from parameters """
		self.name = name
		self.min_value = min_value
		self.max_value = max_value
		self.value_range = value_range
		self.resolution = resolution
		self.step_value = step_value
	
	def __repr__(self):
		return 'ChannelDescription(\'' + self.name + '\', ' + str(self.min_value) + ', ' + str(self.max_value) + ', ' + str(self.value_range) + ', ' + str(self.resolution) + ', ' + str(self.step_value) + ')'
	
	def __str__(self):
		return '{ name: \'' + self.name + '\', min_value: ' + str(self.min_value) + ', max_value: ' + str(self.max_value) + ', value_range: ' + str(self.value_range) + ', resolution: ' + str(self.resolution) + ', step_value: ' + str(self.step_value) + '}'

class DataLogsDescription(object):
	""" A description of the data, extract from a YAML file """
	
	def __init__(self, description):
		""" Construct from a loaded YAML document """
		self.name = description['name']
		# Check consistency of YAML description and create a new description 
		# with bi-directional lookup
		self.by_orig_number = {}
		self.by_name = {}
		self.sensor = [] # an array of sensor descriptions
		self.motor = [] # an array of motor descriptions
		for i,channel_descr in enumerate(description['channels']):
			if isinstance(channel_descr, str):
				if channel_descr in self.by_name:
					raise Exception('Channel ' + channel_name + ' defined twice')
				self.by_name[channel_descr] = i
				self.by_orig_number[i] = channel_descr
			elif channel_descr is not None:
				assert len(channel_descr) >= 5
				channel_name = channel_descr[0]
				channel_type = channel_descr[1]
				min_value = channel_descr[2]
				max_value = channel_descr[3]
				value_range = max_value - min_value
				if value_range == 0:
					print 'Channel ' + channel_name + ' has 0 range, ignoring'
					continue
				
				resolution = channel_descr[4]
				step_value = float(resolution) / float(value_range)
				if channel_type == 'sensor':
					self.sensor.append(ChannelDescription(channel_name, min_value, max_value, value_range, resolution, step_value))
				elif channel_type == 'motor':
					self.motor.append(ChannelDescription(channel_name, min_value, max_value, value_range, resolution, step_value))
				else:
					raise Exception('Channel ' + channel_name + ', unknown type ' + channel_type)
				if channel_name in self.by_name:
					raise Exception('Channel ' + channel_name + ' defined twice')
				self.by_name[channel_name] = i
				self.by_orig_number[i] = channel_type
		self.sensor_count = len(self.sensor)
		self.motor_count = len(self.motor)
		self.sensor_min = np.array([s.min_value for s in self.sensor], dtype=float)
		self.sensor_max = np.array([s.max_value for s in self.sensor], dtype=float)
		self.sensor_range = self.sensor_max - self.sensor_min
		self.sensor_resolution = np.array([s.resolution for s in self.sensor], dtype=float)
		self.motor_min = np.array([m.min_value for m in self.motor], dtype=float)
		self.motor_max = np.array([m.max_value for m in self.motor], dtype=float)
		self.motor_range = self.motor_max - self.motor_min
		self.motor_resolution = np.array([m.resolution for m in self.motor], dtype=float)
		# add description for unified array
		self.sensor_and_motor = []
		for sensor in self.sensor:
			self.sensor_and_motor.append(sensor)
		for motor in self.motor:
			self.sensor_and_motor.append(motor)
		self.sensor_and_motor_min = np.array([c.min_value for c in self.sensor_and_motor], dtype=float)
		self.sensor_and_motor_max = np.array([c.max_value for c in self.sensor_and_motor], dtype=float)
		self.sensor_and_motor_range = self.sensor_and_motor_max - self.sensor_and_motor_min
		# weight groups
		if 'weight_groups' in description:
			self.weight_groups = description['weight_groups']
		else:
			self.weight_groups = None
	
	def __str__(self):
		return 'name=' + str(self.name) + \
			', by_orig_number=' + str(self.by_orig_number) + \
			', by_name=' + str(self.by_name) + \
			', sensor=' + str(self.sensor) + \
			', motor=' + str(self.motor)

class NormalizationIdentity(object):
	def __init__(self, descr):
		self.descr = descr
		self.sensor_count_norm = descr.sensor_count
		self.motor_count_norm = descr.motor_count
	def normalized_sensor_range(self):
		return self.descr.sensor_range
	def normalized_sensor_resolution(self):
		return self.descr.sensor_resolution
	def normalize_sensor_values(self, raw_values):
		return np.array(raw_values, dtype=float)
	def raw_sensor_values(self, normalized_values):
		return normalized_values
	def normalized_motor_range(self):
		return self.descr.motor_range
	def normalized_motor_resolution(self):
		return self.descr.motor_resolution
	def normalize_motor_values(self, raw_values):
		return np.array(raw_values, dtype=float)
	def raw_motor_values(self, normalized_values):
		return normalized_values


class NormalizationDescription(object):
	def __init__(self, descr):
		self.descr = descr
		self.sensor_count_norm = descr.sensor_count
		self.motor_count_norm = descr.motor_count
	
	def normalized_sensor_range(self):
		return 2.
	
	def normalized_sensor_resolution(self):
		""" Normalize sensor resolution """
		return (self.descr.sensor_resolution * 2.) / self.descr.sensor_range
	
	def normalize_sensor_values(self, raw_values):
		""" Normalize raw sensor values """
		return (raw_values - self.descr.sensor_min) * 2. / self.descr.sensor_range - 1.
	
	def raw_sensor_values(self, normalized_values):
		""" Transform normalized sensor values into raw values """
		return np.int_(np.round((normalized_values + 1) * self.descr.sensor_range * 0.5 + self.descr.sensor_min))
	
	def normalized_motor_range(self):
		return 2.
	
	def normalized_motor_resolution(self):
		""" Normalize motor resolution """
		return (self.descr.motor_resolution * 2.) / self.descr.motor_range
	
	def normalize_motor_values(self, raw_values):
		""" Normalize raw motor values """
		return (raw_values - self.descr.motor_min) * 2. / self.descr.motor_range - 1.
	
	def raw_motor_values(self, normalized_values):
		""" Transform normalized motor values into raw values """
		return np.int_(np.round((normalized_values + 1) * self.descr.motor_range * 0.5 + self.descr.motor_min))


class DataLogs(object):
	""" Data from a complete experiment, encompassing several runs """
	
	def __init__(self, description_filename, data_preprocessors=None, normalization='descr', **kwargs):
		""" Construct by loading data using a YAML description file """
		
		# load YAML file
		stream = file(description_filename, 'r')
		description = yaml.load(stream)
		self.descr = DataLogsDescription(description)
		
		# load all runs
		dirname = os.path.dirname(description_filename)
		self.runs = []
		for filename in description['files']:
			self.runs.append(DataLog(dirname, filename, self.descr, data_preprocessors))
		
		# bound check, clamp anyway
		for run_id, run in enumerate(self.runs):
			for i in range(self.descr.sensor_count):
				min_value = run.sensor_raw[:,i].min()
				if min_value < self.descr.sensor_min[i]:
					min_arg = run.sensor_raw[:,i].argmin()
					print ('Warning: at run ' + str(run_id) + ' sensor data value ' + str(min_value) + ' smaller than given minimum ' + str(self.descr.sensor_min[i]) + ' on channel ' + str(i) + ' at pos ' + str(min_arg) + ': clamping')
				max_value = run.sensor_raw[:,i].max()
				if max_value > self.descr.sensor_max[i]:
					max_arg = run.sensor_raw[:,i].argmax()
					print ('Warning: at run ' + str(run_id) + ' sensor data value ' + str(max_value) + ' larger than given maximum ' + str(self.descr.sensor_max[i]) + ' on channel ' + str(i) + ' at pos ' + str(max_arg) + ': clamping')
			run.sensor_raw = np.maximum(run.sensor_raw, self.descr.sensor_min)
			run.sensor_raw = np.minimum(run.sensor_raw, self.descr.sensor_max)
			for i in range(self.descr.motor_count):
				min_value = run.motor_raw[:,i].min()
				if min_value < self.descr.motor_min[i]:
					min_arg = run.motor_raw[:,i].argmin()
					print ('Warning: at run ' + str(run_id) + ' motor data value ' + str(min_value) + ' smaller than given minimum ' + str(self.descr.motor_min[i]) + ' on channel ' + str(i) + ' at pos ' + str(min_arg) + ': clamping')
				max_value = run.motor_raw[:,i].max()
				if max_value > self.descr.motor_max[i]:
					max_arg = run.motor_raw[:,i].argmax()
					print ('Warning: at run ' + str(run_id) + ' motor data value ' + str(max_value) + ' larger than given maximum ' + str(self.descr.motor_max[i]) + ' on channel ' + str(i) + ' at pos ' + str(max_arg) + ': clamping')
			run.motor_raw = np.maximum(run.motor_raw, self.descr.motor_min)
			run.motor_raw = np.minimum(run.motor_raw, self.descr.motor_max)
		
		# normalize using method
		if normalization == 'none':
			self.norm = NormalizationIdentity(self.descr)
		elif normalization == 'descr':
			self.norm = NormalizationDescription(self.descr)
		else:
			raise RuntimeError('Unknown normalization method ' + method)
		
		# normalize data
		for run in self.runs:
			run.normalize(self.norm)
		
		# compute prior on sensor data, do Laplace smoothing with rule of succession
		priorHistSize = 16
		sensor_range = self.norm.normalized_sensor_range()
		self.priorZmin = self.norm.normalize_sensor_values(self.descr.sensor_min)
		self.invPriorZrange = float(priorHistSize) / (sensor_range * (1 + np.finfo(float).eps*100))
		self.priorZ = np.ones((priorHistSize, self.norm.sensor_count_norm), dtype=float)
		for run in self.runs:
			for row in run.sensor_norm:
				bins_i = ((row - self.priorZmin) * self.invPriorZrange).astype(int)
				for i in range(len(row)):
					self.priorZ[bins_i[i], i] += 1
		self.priorZ = (self.priorZ * priorHistSize) / (np.sum(self.priorZ, axis=0) * sensor_range)
		# print self.priorZ
		# dump histogram
		#for i in range(self.priorZ.shape[1]):
			#print i
			#for j in range(self.priorZ.shape[0]):
				#v = self.priorZ[j,i]
				#print '|'+('*'*int(v*100*sensor_range[i]/priorHistSize))
			#print '\n'
			
	def dilateTemporally(self, factor = 2):
		""" Temporally dilate data of a given factor """
		
		for run in self.runs:
			run.dilateTemporally(factor)
