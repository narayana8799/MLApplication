import numpy as np
from math import ceil

def onehotencode(y):

	labels = np.unique(y).size
	encoded_y = np.zeros((y.size, labels), dtype=np.int8)
	for i in range(y.size):
		encoded_y[i, y[i]] = 1

	return encoded_y

def MSE(ytrue, ypred, lambd):

	m = ytrue.size
	error = np.square(ypred - ytrue)
	error = np.sum(error) / (2*m)
	return error

def create_mini_batches(mbs, x, y):

	if mbs == -1:
		return [x], [y]

	else:
		mbx, mby = [], []
		total_mbs = ceil(x.shape[0] / mbs)
		start, end = 0, mbs
		for i in range(total_mbs):

			if i==ceil(x.shape[0] / mbs)-1:

				mbx.append(x[start:, :])
				mby.append(y[start:, :])				
			else:

				mbx.append(x[start:end, :])
				mby.append(y[start:end, :])

			start += mbs
			end += mbs

		return mbx, mby

def initialize_weights(features):

	W = np.random.random((1, features))
	B = np.random.random(1)
	return W, B

def standard_scaler(x):

	mean = np.mean(x, axis=0)
	std = np.std(x, axis=0)
	scaled_x = (x - mean) / std

	return scaled_x