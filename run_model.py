import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time

def epoch(batch_size_to_use, n_train, theano_function):
    i=0
    costs = []
    while i + batch_size_to_use < n_train:
        costs.append(theano_function(i, batch_size_to_use))
        i += batch_size_to_use

    return costs

def run_epochs(training_function, batch_size, n_train, min_epochs=50, new_training=True):
    if 'n' not in dir(run_epochs) or new_training:
        run_epochs.n = 0

    if 'costs' not in dir(run_epochs) or new_training:
        run_epochs.costs = [(0, 999999)]
    
    start = time.time()

    # train for at least this many epochs
    epoch_stop = min_epochs

    while run_epochs.n < epoch_stop:
        run_epochs.n += 1
        costs = epoch(batch_size, n_train, training_function)
        print "=== epoch {} ===".format(run_epochs.n)
        print "costs: {}".format([line[()] for line in costs])
        print "avg: {}".format(np.mean(costs))
        
        # keep training as long as we are improving enough
        if (np.mean(costs) * 1.002) < run_epochs.costs[-1][1]:
            epoch_stop += 1

        run_epochs.costs.append((run_epochs.n, np.mean(costs)))


    elapsed = (time.time() - start)
    print "ELAPSED TIME: {}".format(elapsed)


#######################
# prepare the data

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


####################
# prepare the network

x = ae.matrixType('x')
x_mask = ae.matrixType('mask')

nn64_1 = ae.CFAutoencoder(data.shape[1], 64, inputs=x, mask=x_mask)

nn64_2 = ae.CFAutoencoder(nn64_1.n_hidden, 32, inputs=nn64_1.output, mask=x_mask)

nn64_3 = ae.CFAutoencoder(nn64_2.n_hidden, 16, inputs=nn64_2.output, mask=x_mask)

nn64_4 = ae.CFAutoencoder(nn64_3.n_hidden, 8, inputs=nn64_3.output, mask=x_mask)


######################
# train the network

i, batch_size = T.iscalars('i', 'batch_size')

print "\n\t[Training] Layer 1:"
layer1_train = theano.function([i, batch_size], nn64_1.cost, updates=nn64_1.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer1_train, 256, eighty)

nn64_1.set_noise(0)
nn64_1.learning_rate = 0.01
print "\n\t[Tuning] Layer 1:"
layer1_tune = theano.function([i, batch_size], nn64_1.cost, updates=nn64_1.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer1_tune, 256, eighty)


print "\n\t[Training] Layer 2:"
layer2_train = theano.function([i, batch_size], nn64_2.cost, updates=nn64_2.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer2_train, 256, eighty)

nn64_2.set_noise(0)
nn64_2.learning_rate = 0.01
print "\n\t[Tuning] Layer 2:"
layer2_tune = theano.function([i, batch_size], nn64_2.cost, updates=nn64_2.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer2_tune, 256, eighty)


print "\n\t[Training] Layer 3:"
layer3_train = theano.function([i, batch_size], nn64_3.cost, updates=nn64_3.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer3_train, 256, eighty)

nn64_3.set_noise(0)
nn64_3.learning_rate = 0.01
print "\n\t[Tuning] Layer 3:"
layer3_tune = theano.function([i, batch_size], nn64_3.cost, updates=nn64_3.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer3_tune, 256, eighty)


print "\n\t[Training] Layer 4:"
layer4_train = theano.function([i, batch_size], nn64_4.cost, updates=nn64_4.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer4_train, 256, eighty)

nn64_4.set_noise(0)
nn64_4.learning_rate = 0.01
print "\n\t[Tuning] Layer 4:"
layer4_tune = theano.function([i, batch_size], nn64_4.cost, updates=nn64_4.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer4_tune, 256, eighty)

# let's get this thing trained!

