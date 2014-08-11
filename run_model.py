import datamanager as dm
import autoencoder as ae
import numpy as np

N_HIDDEN = 50
LEARNING_RATE = 0.05

data, mask, names = dm.createNDArray()

dm.shuffle_all(data, mask, names)

sixty = int(len(data) * 0.6)
eighty = int(len(data) * 0.8)
hundo = len(data)

train = data[:sixty]
train_mask = mask[:sixty]

valid = data[sixty:eighty]
test = data[eighty:]

shared_train = theano.shared(train, "train_set")
shared_masks = theano.shared(train_mask, "train_mask")

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
