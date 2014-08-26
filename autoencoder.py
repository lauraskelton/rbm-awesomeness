import theano
import theano.tensor as T
import numpy as np

matrixType = T.TensorType(theano.config.floatX, (False,)*2)

class CFAutoencoder(object):
    def __init__(self, n_in, n_hidden, inputs, mask=None, learning_rate=0.05, 
                pct_noise=0.5, W=None, b_in=None, b_out=None, original_input=None, weight_decay=0.0,
                activation=T.nnet.sigmoid):
        if W is None:
            # initialization of weights as suggested in theano tutorials

            # initialize random starting weights in an intelligent way
            W = np.asarray(np.random.uniform(
                                        low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                        high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                        size=(n_in, n_hidden)), 
                                        dtype=theano.config.floatX)

        self.W = theano.shared(W, 'W')

        if b_in is None:
            # initialize input biases as zeros
            self.b_in = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'b_in')
        else:
            self.b_in = theano.shared(b_in, 'b_in')

        if b_out is None:
            # initialize output biases as zeros
            self.b_out = theano.shared(np.zeros(n_in, dtype=theano.config.floatX), 'b_out')
        else:
            self.b_out = theano.shared(b_out, 'b_out')


        self.n_in = n_in
        self.n_hidden = n_hidden

        # we only want to take a tiny step in the direction of lower cost so we can slowly follow 
        #   the curve down to the point of lowest cost (local min)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pct_noise = pct_noise

        self.inputs = inputs
        self.mask = mask
        self.original_input = original_input
        self.activation = activation

        self.set_noise(self.pct_noise)
        self.set_cost_and_updates(self.mask)

    def set_noise(self, pct_noise):
        self.pct_noise = pct_noise

        if self.pct_noise != 0:
            # this is a mask of which values we are blacking out to corrupt the input data
            self.noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                                (self.inputs.shape), n=1, p=1-(self.pct_noise), 
                                dtype=theano.config.floatX)

            # noisy is the input data after it has been corrupted (after some values were masked out by the noise mask)
            # x is a placeholder for the input tensor.
            self.noisy = self.noise * self.inputs
        else:
            self.noisy = self.inputs

        # we are running the calculation of all of the activations (weights * inputs) + biases of this node
        # sigmoid is scaling the resulting activation smoothly from 0 to 1 for our sigmoid neuron
        # which gives us how activated (from 0 to 1) this node is given this input vector
        self.active_hidden = self.activation(T.dot(self.noisy, self.W) + self.b_in)

        # the clever autoencoder part!
        # we transpose the exact same weights matrix so that we can feed the hidden node activations
        # back through the very same weights the inputs were using
        # in order to get our predicted outputs. This is great because we have way fewer dimensions we're working in to optimize,
        # and it's reasonable to assume that the ideal weights should be the same between inputs and hidden nodes
        # moving either forwards or backwards.
        # So here, we're taking the activation (0 to 1) of the hidden node with the transposed weights (so we can map nodes back to the input)
        # which is (activations * transposed weights), and adding the output biases (which are separate from the input biases, I think
        # due to the fact that we scaled the original activations with the sigmoid, so we need to offset them with the biases now
        # then we are scaling the output activations to be smoothly between 0 and 1 with the sigmoid function,
        # which should give us our rating predictions that are between 0.2 and 1.0
        # (is this true? should we rescale the sigmoid in some way to map to the ratings better?)
        # or does the stochastic gradient descent take care of remapping things to the original values of 0.2 to 1.0 well enough on its own?
        # my mental image is worried that the sigmoid is sending too many things to 0.5, which is biasing the predicted ratings to be lower (3 rating maps to 0.6 in input vector)
        
        # need to change the output to go "up" if it is the final output layer
        # then compare the error of the output to the original input layer somehow...
        # but we can worry about that in the cost function
        self.output = self.activation(T.dot(self.active_hidden, self.W.T) + self.b_out)


        # normalization of activations to the {0, 1} range for compatibility with entropy cost function
        if self.activation == T.tanh:
            self.normal_output = (1 + self.output) / 2.
            self.normal_hidden = (1 + self.active_hidden) / 2.
        else:
            self.normal_output = self.output
            self.normal_hidden = self.active_hidden


    def set_cost_and_updates(self, mask=None):
        self.mask = mask

        # entropy is our cost function. it represents how much information was lost.
        # this is applying the entropy cost function to each value of output relative to each value of the uncorrupted original input matrix
        if self.original_input == None:
            self.entropy = -T.sum(self.inputs * T.log(self.normal_output) + 
                            (1 - self.inputs) * T.log(1 - self.normal_output), axis=1)
        else:
            # then compare the error of the output to the original input layer somehow...
            # so instead of inputs vs output, we need to compare active_hidden to original_input
            # active_hidden here is referring to the next "hidden" layer, which is really the output layer (all of the beers)
            self.entropy = -T.sum(self.original_input * T.log(self.normal_hidden) + 
                                (1 - self.original_input) * T.log(1-self.normal_hidden), axis=1)


        # return a cost function, with gradient updates
        # we need to make sure when this is the final output layer that we are passing in the original input mask,
        # to mask out meaningless data
        if self.mask:
            # we're taking (entropy * mask) to ignore the cost where the input data was unknown/meaningless
            self.cost = T.mean(T.dot(self.entropy, self.mask))
        else:
            # all values are meaningful, so fill the mask with 1s
            self.cost = T.mean(self.entropy)
        

        # now we save the current version of the weights matrix, the input biases, and the output biases to represent this state of the neural network
        if self.original_input:
            self.parameters = [self.W, self.b_in]
        else:
            self.parameters = [self.W, self.b_in, self.b_out]

        # calculate the gradient (direction of greatest change of the cost vector) so we can step towards a lower cost in the next iteration
        self.gradients = T.grad(self.cost, self.parameters)

        # make a vector of the new values of each parameter by descending slightly along the gradient (opposite direction of gradient, to move towards lower cost)
        self.updates = []
        # zip is confusing... tuples?
        for param, grad in zip(self.parameters, self.gradients):
            self.updates.append((param, param - self.learning_rate * grad * (1-self.weight_decay)))

        # the cost function is the same as the final output layer cost
        # the only difference is that we need to use updates to update the weights and biases of the ENTIRE NETWORK,
        # not just the current layer... so the gradient function might be different? and the updates function is different... or extended at least...



    def get_output_function(self):
        self.output_function = theano.function([self.inputs], self.output)
        # closure/block function we return from this function (matrix -> matrix)
        return self.output_function


    def get_testing_function(test_data, test_mask, pct_blackout=0.5):
        raise Error("fix me!")

        i, batch_size = T.iscalars('i', 'batch_size')
        self.test_noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.inputs.shape), n=1, p=1-pct_blackout, 
                            dtype=theano.config.floatX)
        self.test_noisy = self.test_noise * self.inputs
        self.test_active_hidden = T.nnet.sigmoid(T.dot(self.test_noisy, self.W) + self.b_in)
        self.test_output = T.nnet.sigmoid(T.dot(self.test_active_hidden, self.W.T) + self.b_out)

        # root mean squared error of unknowns only

        # taking the original input vector's mask of which beers had no input information (no rating)
        # mask out any output predicted ratings where there was no rating of the original beer
        # so we aren't affecting the error factor in dimensions where we don't have any meaningful information in the original input data
        # flattenedOutputVector = dot product ( (mask vector of which items we sent through the network to test, so we only test accuracy of non-inputted answers) with dot product ( inputMask with full output vector ) )
        self.only_originally_unknown = T.dot(1-self.test_noise, T.dot(self.inputs_mask, self.test_output))
        self.test_error = T.pow(T.mean(T.pow(T.dot(self.inputs_mask, self.test_output) - self.inputs, 2)), 0.5)

        self.testing_function = theano.function([i, batch_size], self.test_error, 
                                                givens={self.inputs:        test_data[i:i+batch_size],
                                                        self.inputs_mask:   test_mask[i:i+batch_size]})

        return self.testing_function


    def save(self, f):
        params = {thing.name : thing.get_value() for thing in self.parameters}
        params['n_in'] = self.n_in
        params['n_hidden'] = self.n_hidden
        params['learning_rate'] = self.learning_rate
        params['pct_noise'] = self.pct_noise
        np.savez_compressed(f, **params)

    def load(f):
        stuff = np.load(f)
        return stuff


def beer_dict_from_weights(names, weight_matrix):
    return [{names[i]:weight for i, weight in enumerate(line)} for line in weight_matrix.T]
