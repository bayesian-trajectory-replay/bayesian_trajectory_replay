import numpy as np

def DuplicateFirstLine(n):
	# we want one or more lines
	n = int(n)
	assert n >= 1
	
	def _duplicate(data):
		rows, cols = data.shape
		new_data = np.empty((rows+n-1, cols))
		for i in range(n):
			new_data[i] = data[0]
		new_data[n:,] = data[1:,]
		return new_data
	
	return _duplicate