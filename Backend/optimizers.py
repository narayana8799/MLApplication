import numpy as np
import utils

class GradientDescent:

	def __init__(self, mbx, mby, W, B, lr, iters, lambd, optimizer, cost_kind):

		self.lr = lr
		self.iters = iters
		self.lambd = lambd
		self.opt = optimizer
		self.cost_kind = cost_kind
		self.costs = []
		self.x, self.y = mbx, mby
		self.W, self.B = W, B
		self.m = None
		self.dw, self.db = None, None

	def run(self):
		
		for i in range(1, self.iters+1):
			for x, y in zip(self.x, self.y):

				A = np.matmul(self.W, x) + self.B
				self.m = x.shape[1]
			
				if self.opt == 'adam':
					pass

				else:
					self.dw = np.matmul((A - y), x.T)
					self.db = np.sum(A - y)
					self.update_parameters()
					if self.cost_kind == 'MSE':
						cst = utils.MSE(y, A, self.lambd)
						self.costs.append(cst)

		return self.W, self.B, self.costs


	def update_parameters(self):
		self.W = self.W*(1-(self.lambd*self.lr) / self.m) - self.lr*self.dw
		self.B -= self.lr*self.db
