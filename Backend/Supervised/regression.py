import numpy as np
import utils
import optimizers
import matplotlib.pyplot as plt

class LinearRegression:

	def __init__(self, lr=0.001, lambd=0, iters=500, optimizer=None, mini_batch_size=128):
		
		self.lr = lr
		self.lambd = lambd
		self.iters = iters
		self.mbs = mini_batch_size
		self.opt = optimizer

		self.x, self.y, self.mbx, self.mby = None, None, None, None
		self.W, self.B = None, None
		self.costs = None

	def fit(self, x, y):
		
		self.x = utils.standard_scaler(x).T
		self.y = y.reshape((1, -1))
		self.W, self.B = utils.initialize_weights(self.x.shape[0])
		self.mbx, self.mby = utils.create_mini_batches(self.mbs, self.x, self.y)

		gd = optimizers.GradientDescent(self.mbx, self.mby, self.W, self.B, self.lr, self.iters, self.lambd, self.opt, cost_kind='MSE')
		self.W, self.B, self.costs = gd.run()

		
		plt.plot(self.x[0, :].ravel(), (self.W@self.x+self.B).ravel())
		plt.show()
