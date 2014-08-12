import theano
import theano.tensor as T
import numpy as np

class CFAutoencoder(object):
    def __init__(self, n_in, n_hidden, learning_rate, prior_self=None, input_tensor=None, mask_tensor=None, pct_blackout=0.2, 
                    W=None, b_in=None, b_out=None):
        if W == None:
            # initialization of weights as suggested in theano tutorials

            # initialize random starting weights in an intelligent way
            W = np.asarray(np.random.uniform(
                                        low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                        high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                        size=(n_in, n_hidden)), 
                                        dtype=theano.config.floatX)

        self.W = theano.shared(W, 'W')

        if b_in == None:
            # initialize input biases as zeros
            self.b_in = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'b_in')
        else:
            self.b_in = theano.shared(b_in, 'b_in')

        if b_out == None:
            # initialize output biases as zeros
            self.b_out = theano.shared(np.zeros(n_in, dtype=theano.config.floatX), 'b_out')
        else:
            self.b_out = theano.shared(b_out, 'b_out')

        matrixType = T.TensorType(theano.config.floatX, (False,)*2)


        self.n_in = n_in
        self.n_hidden = n_hidden
        self.inputs = input_tensor
        self.prior_self = prior_self

        # we send the input tensor to x when we actually call the function chain we're setting up here.
        self.x = matrixType('x')
        self.x_mask = matrixType('empty')

        self.pct_blackout = pct_blackout

        # this is a mask of which values we are blacking out to corrupt the input data
        self.noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-(self.pct_blackout), 
                            dtype=theano.config.floatX)

        # noisy is the input data after it has been corrupted (after some values were masked out by the noise mask)
        # x is a placeholder for the input tensor.
        self.noisy = self.noise * self.x

        # we are running the calculation of all of the activations (weights ⋅ inputs) + biases of this node
        # sigmoid is scaling the resulting activation smoothly from 0 to 1 for our sigmoid neuron
        # which gives us how activated (from 0 to 1) this node is given this input vector
        self.active_hidden = T.nnet.sigmoid(T.dot(self.noisy, self.W) + self.b_in)

        # the clever autoencoder part!
        # we transpose the exact same weights matrix so that we can feed the hidden node activations
        # back through the very same weights the inputs were using
        # in order to get our predicted outputs. This is great because we have way fewer dimensions we're working in to optimize,
        # and it's reasonable to assume that the ideal weights should be the same between inputs and hidden nodes
        # moving either forwards or backwards.
        # So here, we're taking the activation (0 to 1) of the hidden node with the transposed weights (so we can map nodes back to the input)
        # which is (activations ⋅ transposed weights), and adding the output biases (which are separate from the input biases, I think
        # due to the fact that we scaled the original activations with the sigmoid, so we need to offset them with the biases now
        # then we are scaling the output activations to be smoothly between 0 and 1 with the sigmoid function,
        # which should give us our rating predictions that are between 0.2 and 1.0
        # (is this true? should we rescale the sigmoid in some way to map to the ratings better?)
        # or does the stochastic gradient descent take care of remapping things to the original values of 0.2 to 1.0 well enough on its own?
        # my mental image is worried that the sigmoid is sending too many things to 0.5, which is biasing the predicted ratings to be lower (3 rating maps to 0.6 in input vector)
        self.output = T.nnet.sigmoid(T.dot(self.active_hidden, self.W.T) + self.b_out)

        # entropy is our cost function. it represents how much information was lost.
        # this is applying the entropy cost function to each value of output relative to each value of the uncorrupted original input matrix
        self.entropy = -T.sum(self.x * T.log(self.output) + 
                                (1 - self.x) * T.log(1 - self.output), axis=1)

        if mask_tensor == None:
            # all values are meaningful, so fill the mask with 1s
            self.cost = T.mean(self.entropy)
        else:
            # we're taking (entropy ⋅ mask) to ignore the cost where the input data was unknown/meaningless
            self.cost = T.mean(T.dot(self.entropy, self.x_mask))
        

        # now we save the current version of the weights matrix, the input biases, and the output biases to represent this state of the neural network
        self.parameters = [self.W, self.b_in, self.b_out]
        # calculate the gradient (direction of greatest change of the cost vector) so we can step towards a lower cost in the next iteration
        self.gradients = T.grad(self.cost, self.parameters)

        # we only want to take a tiny step in the direction of lower cost so we can slowly follow the curve down to the point of lowest cost (local min)
        self.learning_rate = learning_rate

        # make a vector of the new values of each parameter by descending slightly along the gradient (opposite direction of gradient, to move towards lower cost)
        self.updates = []
        # zip is confusing... tuples?
        for param, grad in zip(self.parameters, self.gradients):
            self.updates.append((param, param - self.learning_rate * grad))

        # i and batch size are general Theano placeholders for the real i and batch size we will set later
        i, batch_size = T.iscalars('i', 'batch_size')

        if self.prior_self == None:
            # here we are setting the actual values of all of those theano scalar and matrix placeholders
            self.train_step = theano.function([i, batch_size], self.cost, 
                                            updates=self.updates, 
                                            givens={self.x:        self.inputs[i:i+batch_size],
                                                    self.x_mask:   self.mask[i:i+batch_size]})

        else:
            # here we are setting the actual values of all of those theano scalar and matrix placeholders
            self.train_step2 = theano.function([i, batch_size], self.cost2, 
                                            updates=self.updates2, 
                                            givens={self.x2:         get_activation_function(self.prior_self)[i:i+batch_size]


    def get_activation_function(self):
        # the hidden node activations with uncorrupted (original) input data
        self.clean_hidden = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b_in)
        # the output activactions with uncorrupted (original) input data fed through the network
        self.clean_output = T.nnet.sigmoid(T.dot(self.clean_hidden, self.W.T) + self.b_out)
        # the theano function we call with our uncorrupted input matrix
        self.clean_activation_function = theano.function([self.x], self.clean_output)
        # closure/block function we return from this function (matrix -> matrix)
        return self.clean_activation_function


    def get_testing_function(self, test_data, test_mask, pct_blackout=0.5):
        i, batch_size = T.iscalars('i', 'batch_size')
        self.test_noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-pct_blackout, 
                            dtype=theano.config.floatX)
        self.test_noisy = self.test_noise * self.x
        self.test_active_hidden = T.nnet.sigmoid(T.dot(self.test_noisy, self.W) + self.b_in)
        self.test_output = T.nnet.sigmoid(T.dot(self.test_active_hidden, self.W.T) + self.b_out)

        # root mean squared error of unknowns only

        # taking the original input vector's mask of which beers had no input information (no rating)
        # mask out any output predicted ratings where there was no rating of the original beer
        # so we aren't affecting the error factor in dimensions where we don't have any meaningful information in the original input data
        # flattenedOutputVector = dot product ( (mask vector of which items we sent through the network to test, so we only test accuracy of non-inputted answers) with dot product ( inputMask with full output vector ) )
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

