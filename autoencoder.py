import theano
import theano.tensor as T
import numpy as np

class CFAutoencoder(object):
    def __init__(self, input_tensor, mask_tensor, n_in, n_hidden, learning_rate, pct_blackout=0.2, 
                    W=None, b_in=None, b_out=None):
        if W == None:
            # initialization of weights as suggested in theano tutorials
            W = np.asarray(np.random.uniform(
                                        low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                        high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                        size=(n_in, n_hidden)), 
                                        dtype=theano.config.floatX)

        self.W = theano.shared(W, 'W')

        if b_in == None:
            self.b_in = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'b_in')
        else:
            self.b_in = theano.shared(b_in, 'b_in')

        if b_out == None:
            self.b_out = theano.shared(np.zeros(n_in, dtype=theano.config.floatX), 'b_out')
        else:
            self.b_out = theano.shared(b_out, 'b_out')

        matrixType = T.TensorType(theano.config.floatX, (False,)*2)


        self.n_in = n_in
        self.n_hidden = n_hidden
        self.inputs = input_tensor
        self.mask = mask_tensor
        self.x = matrixType('x')
        self.x_mask = matrixType('empty')

        self.pct_blackout = pct_blackout
        self.noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-(self.pct_blackout), 
                            dtype=theano.config.floatX)
        self.noisy = self.noise * self.x

        self.active_hidden = T.nnet.sigmoid(T.dot(self.noisy, self.W) + self.b_in)
        self.output = T.nnet.sigmoid(T.dot(self.active_hidden, self.W.T) + self.b_out)

        self.entropy = -T.sum(self.x * T.log(self.output) + 
                                (1 - self.x) * T.log(1 - self.output), axis=1)

        # multiply by zeros where all 5 inputs were zero

        self.cost = T.mean(T.dot(self.entropy, self.x_mask))

        self.parameters = [self.W, self.b_in, self.b_out]
        self.gradients = T.grad(self.cost, self.parameters)

        self.learning_rate = learning_rate

        self.updates = []
        for param, grad in zip(self.parameters, self.gradients):
            self.updates.append((param, param - self.learning_rate * grad))

        i, batch_size = T.iscalars('i', 'batch_size')
        self.train_step = theano.function([i, batch_size], self.cost, 
                                            updates=self.updates, 
                                            givens={self.x:        self.inputs[i:i+batch_size],
                                                    self.x_mask:   self.mask[i:i+batch_size]})


    def get_activation_function(self):
        self.clean_hidden = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b_in)
        self.clean_output = T.nnet.sigmoid(T.dot(self.clean_hidden, self.W.T) + self.b_out)
        self.clean_activation_function = theano.function([self.x], self.clean_output)

        return self.clean_activation_function

    def get_testing_function(self, test_data, test_mask, pct_blackout=0.5):
        self.test_noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-pct_blackout, 
                            dtype=theano.config.floatX)
        self.test_noisy = self.test_noise * self.x
        self.test_active_hidden = T.nnet.sigmoid(T.dot(self.test_noisy, self.W) + self.b_in)
        self.test_output = T.nnet.sigmoid(T.dot(self.test_active_hidden, self.W.T) + self.b_out)

        # root mean squared error of unknowns only
        self.only_originally_unknown = T.dot(1-self.test_noise, T.dot(self.x_mask, self.test_output))
        self.test_error = T.pow(T.mean(T.pow(T.dot(self.x_mask, self.test_output) - self.x, 2)), 0.5)

        self.testing_function = theano.function([i, batch_size], self.test_error, 
                                                givens={self.x:        test_data[i:i+batch_size],
                                                        self.x_mask:   test_mask[i:i+batch_size]})

        return self.testing_function


    def save(self, f):
        params = {thing.name : thing.get_value() for thing in nn.parameters}
        params['n_in'] = self.n_in
        params['n_hidden'] = self.n_hidden
        params['learning_rate'] = self.learning_rate
        params['pct_blackout'] = self.pct_blackout
        np.savez_compressed(f, **params)

    def load(f):
        stuff = np.load(f)
        return stuff

