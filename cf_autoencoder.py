import theano
import numpy as np

class CFAutoencoder(object):
    def __init__(self, input_tensor, emptiness_tensor, n_in, n_hidden, learning_rate, pct_blackout=0.2, 
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
        self.empties = emptiness_tensor
        self.x = matrixType('x')
        self.x_empty = matrixType('empty')

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

        self.cost = T.mean(T.dot(self.entropy, self.x_empty)

        self.parameters = [self.W, self.b_in, self.b_out]
        self.gradients = T.grad(self.cost, self.parameters)

        self.learning_rate = learning_rate

        self.updates = []
        for param, grad in zip(self.parameters, self.gradients):
            self.updates.append((param, param - self.learning_rate * grad))

        i, batch_size = T.iscalars('i', 'batch_size')
        self.train_step = theano.function([i, batch_size], self.cost, 
                                            updates=self.updates, 
                                            givens={self.x:         self.inputs[i:i+batch_size]
                                                    self.x_empty:   self.empties[i:i+batch_size]})


    def save(self, f):
        with open(f, "wb") as f:
            cPickle.dump([thing.get_value() for thing in self.parameters], f)

def run_epochs(aa, n_epochs):#, save=True):
    if 'n' not in dir(run_epochs):
        run_epochs.n = 0

    if 'costs' not in dir(run_epochs):
        run_epochs.costs = [(0, 999999)]

    start = time.time()
    print time
    for x in xrange(n_epochs):
        run_epochs.n += 1
        costs = epoch(1000, n_train, aa.train_step)
        print "=== epoch {} ===".format(run_epochs.n)
        print "costs: {}".format([line[()] for line in costs])
        print "avg: {}".format(np.mean(costs))
        run_epochs.costs.append((run_epochs.n, np.mean(costs)))

    elapsed = (time.time() - start)
    print "ELAPSED TIME: {}".format(elapsed)

    # if save:
    #     vis = aa.visualize_hidden()
    #     img = Image.fromarray(vis)
    #     summary = "eyes_epoch_{}_{:0.4f}".format(run_epochs.n, run_epochs.costs[-1][1])
    #     img.save("imgs/aa/single_layer/{}.png".format(summary))
    #     aa.save("data/save/aa/single_layer/{}.pkl".format(summary))

    if run_epochs.costs[-2][1] < run_epochs.costs[-1][1]:
        aa = aa.copy(aa.learning_rate * 0.9)