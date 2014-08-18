import theano
import theano.tensor as T
import numpy as np
import time

def epoch(batch_size_to_use, n_train, theano_function):
	i=0
	costs = []
	while i + batch_size_to_use <= n_train:
		costs.append(theano_function(i, batch_size_to_use))
		i += batch_size_to_use

	return costs



class AETrainer(object):
	def __init__(self, model, inputs, mask=None, batch_size=64):
		self.i, self.bs = T.iscalars('i', 'bs')
		self.model = model
		self.inputs = inputs
		self.batch_size = batch_size
		self.mask=mask

		self.steps = 0
		self.costs = [(0, 999999)]

	def get_training_function(self):
		given = {self.model.inputs : self.inputs[self.i:self.i+self.bs]}

		if self.model.mask:
			given[self.model.mask] = self.mask[self.i:self.i+self.bs]

		return theano.function([self.i, self.bs], self.model.cost, 
						updates=self.model.updates, givens=given)


	def run_epochs(self, min_epochs=50, min_improvement=1.001, 
					lr_decay=0.0, decay_modulo=25):
		start = time.time()
		training_function = self.get_training_function()

		# train for at least this many epochs
		epoch_stop = min_epochs
		n_train = len(self.inputs.get_value())

		while self.steps < epoch_stop:
			self.steps += 1
			costs = epoch(self.batch_size, n_train, training_function)
			print "=== epoch {} ===".format(self.steps)
			print "costs: {}".format([line[()] for line in costs])
			print "avg: {}".format(np.mean(costs))
			
			# keep training as long as we are improving enough
			if (np.mean(costs) * min_improvement) < self.costs[-1][1]:
				epoch_stop += 1
			elif lr_decay and self.steps % decay_modulo == 0:
				self.model.learning_rate *= (1 - lr_decay)
				print "min improvement not seen; decreasing learning rate to {}".format(network.learning_rate)
				print "epochs left: {}".format(epoch_stop)
				# recompile training function with new learning rate
				training_function = self.get_training_function()

			self.costs.append((self.steps, np.mean(costs)))


		elapsed = (time.time() - start)
		print "ELAPSED TIME: {}".format(elapsed)