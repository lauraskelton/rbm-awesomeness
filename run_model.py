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

nn64_2 = ae.CFAutoencoder(nn64_1.n_hidden, 32, inputs=nn64_1.active_hidden)

nn64_3 = ae.CFAutoencoder(nn64_2.n_hidden, 16, inputs=nn64_2.active_hidden)

nn64_4 = ae.CFAutoencoder(nn64_3.n_hidden, 8, inputs=nn64_3.active_hidden)


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

######################
# test the network

# need to start with real input data at the input layer, but corrupted
# we already did this in nn64... it should be just nn64_1.noisy. done.

# save mask of how data was corrupted, as well as mask of missing/meaningless input data
# again, we already did this in nn64... it is nn64_1.noise.
# but wait! we called set_noise(0) above, so we need to reset the noise to re-corrupt the inputs.
nn64_1.set_noise(0.5)

# run it through the network to the first hidden layer
# didn't we already do this as well with the exact same data? hmm... this should be nn64_1.active_hidden (= the activations of the first hidden layer)
# no. we have to re-run it through with the noisy input data. (or this happens at the end?? I think? so nn64_1.active_hidden should update automatically to the new noisy inputs.)

# send the activations of the first hidden layer to the second hidden layer
# make sure we are sending the original, uncorrupted activations at this layer, not the noisy ones
# this will happen if we call nn64_2.set_noise(0). which we already did, above.

# send the activations of the second hidden layer to the third hidden layer (nn64_2.active_hidden)
# this should be ok... we already made the third hidden layer dependent on the second one, and set_noise(0) on it 

# send the activations (nn64_3.active_hidden) of the third hidden layer back down through the output function to the second hidden layer
# nn64_4.output - the third layer's output activations (going downward in the stack of autoencoders)
# how do we run the next layer backwards? do we take the output as the input for nn64_3 now?
# maybe we need to run it first to get the activations at the top layer?
nn64_3.input = nn64_4.output # ???

# use the output function to send the (new) activations of the second hidden layer down to the first hidden layer

nn64_2.input = nn64_3.output # ???
# use the output function to send the (new) activations of the first hidden layer down to the input layer
# nn64_2.output should be the predictions of the input data after going through our neural network

# mask out the meaningless data from the neural network's output matrix
# mask out the values we input (the uncorrupted values) from the output matrix (so we are only measuring error on beers that were predicted)
nn64_prediction = T.dot(T.dot(nn64_2.output, nn64_1.mask), 1 - nn64_1.noise)

# use root mean squared error to check how accurate our predictions were when using the entire neural network
nn64_test_error = T.pow(T.mean(T.pow(nn64_prediction - nn64_1.inputs, 2)), 0.5)

print "\n\t[Testing] 64-hidden node autoencoder error: {}".format(nn64_test_error)



