import theano
import theano.tensor as T
import numpy as np
import time
from collections import OrderedDict

def epoch(batch_size_to_use, n_train, training_function):
	i=0
	costs = []
	while i + batch_size_to_use <= n_train:
		costs.append(training_function(i, batch_size_to_use))
		i += batch_size_to_use

	return costs



class AETrainer(object):
	def __init__(self, model, x, shared_input, x_mask=None, shared_mask=None, batch_size=64, momentum=0.0):
		self.i, self.bs = T.iscalars('i', 'bs')

		self.model = model
		self.x = x
		self.x_mask = x_mask
		self.shared_input = shared_input
		self.shared_mask = shared_mask
		self.batch_size = batch_size
		self.mask=mask
		self.momentum = momentum

		if momentum:
			self.momentums = {param : theano.shared(np.zeros(param.get_value().shape)) 
									for param in self.model.parameters}

		self.steps = 0
		self.costs = [(0, 999999)]


	def get_training_function(self):
		given = {self.x : self.shared_input[self.i:self.i+self.bs]}

		if self.model.mask:
			given[self.x_mask] = self.shared_mask[self.i:self.i+self.bs]

		if self.momentum:
			l_r = self.model.learning_rate
			self.updates = OrderedDict()
			for param, grad in zip(self.model.parameters, self.model.gradients):
				update = self.momentum * self.momentums[param] - l_r * grad
				self.updates[self.momentums[param]] = update
				self.updates[param] = param + update

			return theano.function([self.i, self.bs], self.model.cost, 
							updates=self.updates, givens=given)
		else:
			return theano.function([self.i, self.bs], self.model.cost, 
							updates=self.model.updates, givens=given)


	def run_epochs(self, min_epochs=50, min_improvement=1.001, 
					lr_decay=0.0, decay_modulo=25):
		start = time.time()
		training_function = self.get_training_function()

		# train for at least this many epochs
		epoch_stop = min_epochs
		n_train = len(self.shared_input.get_value())
		since_last_decay = 0

		while self.steps < epoch_stop:
			self.steps += 1
			costs = epoch(self.batch_size, n_train, training_function)
			print "=== epoch {} ===".format(self.steps)
			print "costs: {}".format([line[()] for line in costs])
			print "avg: {}".format(np.mean(costs))
			
			# keep training as long as we are improving enough
			if (np.mean(costs) * min_improvement) < self.costs[-1][1]:
				epoch_stop += 1
				since_last_decay += 1
			elif lr_decay and since_last_decay - decay_modulo > 0:
				self.model.learning_rate *= (1 - lr_decay)
				since_last_decay = 0
				print "min improvement not seen; decreasing learning rate to {}".format(
																self.model.learning_rate)
				print "epochs left: {}".format(epoch_stop)
				# recompile training function with new learning rate
				training_function = self.get_training_function()

			self.costs.append((self.steps, np.mean(costs)))


		elapsed = (time.time() - start)
		print "ELAPSED TIME: {}".format(elapsed)

