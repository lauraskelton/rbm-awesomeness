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

#(self, n_in, n_hidden, learning_rate, prior_self=None, input_tensor=None, mask_tensor=None, pct_blackout=0.2, W=None, b_in=None, b_out=None):
nn10 = ae.CFAutoencoder(data.shape[1], 
                        10, LEARNING_RATE, prior_self=None, input_tensor=shared_train, mask_tensor=shared_mask)
nn50 = ae.CFAutoencoder(data.shape[1], 
                        50, LEARNING_RATE, prior_self=None, input_tensor=shared_train, mask_tensor=shared_mask)

nn10_2 = ae.CFAutoencoder(nn10.n_hidden, 
                        5, LEARNING_RATE, prior_self=nn10)
nn50_2 = ae.CFAutoencoder(nn50.n_hidden, 
                        25, LEARNING_RATE, prior_self=nn50)

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

print "\n\t[Training] 10-hidden node autoencoder:"
run_epochs(nn10, 1000, 256, eighty)
print "\n\t[Training] 50-hidden node autoencoder:"
run_epochs(nn50, 1000, 256, eighty)

print "\n\t[Training] 10-hidden node autoencoder 2nd Layer:"
run_epochs(nn10_2, 1000, 256, eighty)
print "\n\t[Training] 50-hidden node autoencoder 2nd Layer:"
run_epochs(nn50_2, 1000, 256, eighty)

def make_readable(weights):
	return {names[i] : weight for i, weight in enumerate(weights)}

# dicts = [make_readable(row) for row in nn.W.get_value().T]

test_size = len(test_set)
test10 = nn10.get_testing_function(shared_test, shared_test_mask)
test50 = nn50.get_testing_function(shared_test, shared_test_mask)

print "Test error with 10-node network:"
print np.mean(test10(0,test_size))

print "Test error with 50-node network:"
print np.mean(test50(0,test_size))
