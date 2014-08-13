import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time


LEARNING_RATE = 0.05

data, mask, names = dm.createNDArray()

dm.shuffle_all(data, mask)

eighty = int(len(data) * 0.8)
hundo = len(data)

train_set = data[:eighty]
train_mask = mask[:eighty]

test_set = data[eighty:]
test_mask = mask[eighty:]

shared_train = theano.shared(train_set, "train_set")
shared_mask = theano.shared(train_mask, "train_mask")

shared_test = theano.shared(test_set, "test_set")
shared_test_mask = theano.shared(test_mask, "test_mask")

nn64 = ae.CFAutoencoder(data.shape[1], 
                        64, LEARNING_RATE, prior_self=None, input_tensor=shared_train, mask_tensor=shared_mask)

nn64_2 = ae.CFAutoencoder(nn64.n_hidden, 
                        32, LEARNING_RATE, prior_self=nn64)

nn64_3 = ae.CFAutoencoder(nn64_2.n_hidden, 
                        16, LEARNING_RATE, prior_self=nn64_2)

nn64_4 = ae.CFAutoencoder(nn64_3.n_hidden, 
                        8, LEARNING_RATE, prior_self=nn64_3)

def epoch(batch_size_to_use, n_train, theano_function):
    i=0
    costs = []
    while i + batch_size_to_use < n_train:
        costs.append(theano_function(i, batch_size_to_use))
        i += batch_size_to_use

    return costs

def run_epochs(nn, n_epochs, batch_size, n_train, new_training=True):
    if 'n' not in dir(run_epochs) or new_training:
        run_epochs.n = 0

    if 'costs' not in dir(run_epochs) or new_training:
        run_epochs.costs = [(0, 999999)]
    
    start = time.time()

    # train for at least this many epochs
    epoch_stop = 50

    while run_epochs.n < epoch_stop:
        run_epochs.n += 1
        costs = epoch(batch_size, n_train, nn.train_step)
        print "=== epoch {} ===".format(run_epochs.n)
        print "costs: {}".format([line[()] for line in costs])
        print "avg: {}".format(np.mean(costs))
        
        # keep training as long as we are improving enough
        if (np.mean(costs) * 1.002) < run_epochs.costs[-1][1]:
            epoch_stop += 1

        run_epochs.costs.append((run_epochs.n, np.mean(costs)))


    elapsed = (time.time() - start)
    print "ELAPSED TIME: {}".format(elapsed)

print "\n\t[Training] 64-hidden node autoencoder:"
run_epochs(nn64, 1000, 256, eighty)

print "\n\t[Training] 64-hidden node autoencoder 2nd Layer:"
run_epochs(nn64_2, 1000, 256, eighty)

print "\n\t[Training] 64-hidden node autoencoder 3rd Layer:"
run_epochs(nn64_3, 1000, 256, eighty)

print "\n\t[Training] 64-hidden node autoencoder 4th Layer:"
run_epochs(nn64_4, 1000, 256, eighty)

def make_readable(weights):
	return {names[i] : weight for i, weight in enumerate(weights)}

# dicts = [make_readable(row) for row in nn.W.get_value().T]

test_size = len(test_set)
test64 = nn64.get_testing_function(shared_test, shared_test_mask)

print "Test error with 64-node network:"
print np.mean(test64(0,test_size))
