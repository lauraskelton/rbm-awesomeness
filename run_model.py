import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time


N_HIDDEN = 10
LEARNING_RATE = 0.05

data, mask, names = dm.createNDArray()

dm.shuffle_all(data, mask)

eighty = int(len(data) * 0.8)
hundo = len(data)

train = data[:eighty]
train_mask = mask[:eighty]

test = data[eighty:]

shared_train = theano.shared(train, "train_set")
shared_mask = theano.shared(train_mask, "train_mask")

nn = ae.CFAutoencoder(shared_train, shared_mask, data.shape[1], N_HIDDEN, LEARNING_RATE)


def epoch(batch_size_to_use, n_train, theano_function):
    i=0
    costs = []
    while i + batch_size_to_use < n_train:
        costs.append(theano_function(i, batch_size_to_use))
        i += batch_size_to_use

    return costs

def run_epochs(nn, n_epochs, batch_size, n_train):
    if 'n' not in dir(run_epochs):
        run_epochs.n = 0

    if 'costs' not in dir(run_epochs):
        run_epochs.costs = [(0, 999999)]

    start = time.time()
    print time
    for x in xrange(n_epochs):
        run_epochs.n += 1
        costs = epoch(batch_size, n_train, nn.train_step)
        print "=== epoch {} ===".format(run_epochs.n)
        print "costs: {}".format([line[()] for line in costs])
        print "avg: {}".format(np.mean(costs))
        run_epochs.costs.append((run_epochs.n, np.mean(costs)))

    elapsed = (time.time() - start)
    print "ELAPSED TIME: {}".format(elapsed)

run_epochs(nn, 1000, 256, eighty)

def neuron_idx_to_beer_name(names, n):
	return names[n/5]

def make_readable(weights):
	return {neuron_idx_to_beer_name(names, i) : weights[i:i+5] for i in xrange(0, len(weights), 5)}

dicts = [make_readable(row) for row in nn.W.get_value().T]