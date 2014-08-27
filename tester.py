import theano
import theano.tensor as T
import numpy as np

class AETester(object):
	def __init__(self, layers, testfunc, x, shared_input, x_mask=None, shared_mask=None):
		self.layers = layers
		self.testfunc = testfunc
		self.params = [param for layer in self.layers for param in layer.parameters]
		self.errors = [theano.shared(np.zeros(param.get_value().shape), "{}_errors".format(param.name)) for param in self.params]
		