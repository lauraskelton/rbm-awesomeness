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
		# import pdb;pdb.set_trace()
	return costs



class AETrainer(object):
	# takes a list of all of which will be updated with respect to the provided cost function
	def __init__(self, layers, cost, x, shared_input, x_mask=None, shared_mask=None, batch_size=64, 
								learning_rate=0.05, momentum=0.9, weight_decay=0.0002):
		self.i, self.bs = T.iscalars('i', 'bs')

		if type(layers) == list:
			self.layers = layers
		else:
			self.layers = [layers]
		self.cost = cost
		self.x = x
		self.x_mask = x_mask
		self.shared_input = shared_input
		self.shared_mask = shared_mask
		self.batch_size = batch_size
		self.momentum = momentum
		self.weight_decay = weight_decay
		self.learning_rate = learning_rate

		self.params = [param for layer in self.layers for param in layer.parameters]
		self.gradients = T.grad(self.cost, self.params)
		self.lr = T.dscalar('lr')

		if momentum:
			self.momentums = {param : theano.shared(np.zeros(param.get_value().shape)) 
									for param in self.params}

		self.steps = 0
		self.costs = [(0, 999999)]


	def get_training_function(self):
		given = {self.x : self.shared_input[self.i:self.i+self.bs],
				self.lr : self.learning_rate}

		if self.x_mask:
			given[self.x_mask] = self.shared_mask[self.i:self.i+self.bs]

		self.updates = OrderedDict()
		for param, grad in zip(self.params, self.gradients):
			if self.momentum:
				update = self.momentum * self.momentums[param] - self.lr * grad
				self.updates[self.momentums[param]] = update
			else:
				update = - self.learning_rate * grad

			self.updates[param] = param * (1-self.weight_decay) + update


		return theano.function([self.i, self.bs], self.cost, 
						updates=self.updates, givens=given)



	def run_epochs(self, min_epochs=50, min_improvement=1.001, 
					lr_decay=0.1, decay_modulo=25, custom_training=None):
		start = time.time()

		if custom_training is not None:
			training_function = custom_training
			print "Training using custom training function..."
		else:
			training_function = self.get_training_function()

		# train for at least this many epochs
		epoch_stop = min_epochs
		n_train = len(self.shared_input.get_value())
		since_last_decay = 0
		best_cost = float('inf')

		while self.steps < epoch_stop:
			self.steps += 1
			costs = epoch(self.batch_size, n_train, training_function)

			print "=== epoch {} ===".format(self.steps)
			# print "costs: {}".format([line[()] for line in costs])
			print "avg cost: {:.5f} (previous best: {:.5f}; ratio: {:.5f})".format(
								np.mean(costs), best_cost, best_cost/np.mean(costs))
			
			# keep training as long as we are improving enough
			if (np.mean(costs) * min_improvement) < best_cost:
				epoch_stop += 1
				since_last_decay += 1
			elif lr_decay and since_last_decay - decay_modulo > 0:
				self.learning_rate *= (1 - lr_decay)
				since_last_decay = 0
				print "min improvement not seen; decreasing learning rate to {}".format(
																self.learning_rate)
				print "epochs left: {}".format(epoch_stop)

			if np.mean(costs) < best_cost:
				best_cost = np.mean(costs)

			self.costs.append((self.steps, np.mean(costs)))


		elapsed = (time.time() - start)
		print "ELAPSED TIME: {}".format(elapsed)

