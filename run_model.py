import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time

def epoch(batch_size_to_use, n_train, theano_function):
    i=0
    costs = []
    while i + batch_size_to_use <= n_train:
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

# final output layer
nn64_3 = ae.CFAutoencoder(nn64_2.n_hidden, data.shape[1], inputs=nn64_2.active_hidden, mask=x_mask, original_input=x)

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
                                        givens={x:      shared_train[i:i+batch_size]})

run_epochs(layer2_train, 256, eighty)

nn64_2.set_noise(0)
nn64_2.learning_rate = 0.01
print "\n\t[Tuning] Layer 2:"
layer2_tune = theano.function([i, batch_size], nn64_2.cost, updates=nn64_2.updates,
                                        givens={x:      shared_train[i:i+batch_size]})

run_epochs(layer2_tune, 256, eighty)


print "\n\t[Training] Output Layer:"
layer3_train = theano.function([i, batch_size], nn64_3.cost, updates=nn64_3.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer3_train, 256, eighty)

nn64_3.set_noise(0)
nn64_3.learning_rate = 0.01
print "\n\t[Tuning] Output Layer:"
layer3_tune = theano.function([i, batch_size], nn64_3.cost, updates=nn64_3.updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(layer3_tune, 256, eighty)


######################
# train the entire network


# the cost function is the same as the final output layer cost
# the only difference is that we need to use updates to update the weights and biases of the ENTIRE NETWORK,
# not just the current layer... so the gradient function might be different? and the updates function is different... or extended at least...

# self.parameters = [self.W, self.b_in, self.b_out]

# set gradient to depend on all of the parameters we set above, so that "updates" will update all of the layers' weights and biases

entire_network_params = [nn64_1.W, nn64_1.b_in, nn64_2.W, nn64_2.b_in, nn64_3.W, nn64_3.b_in]
entire_network_gradients = T.grad(nn64_3.cost, entire_network_params)

# make a vector of the new values of each parameter by descending slightly along the gradient (opposite direction of gradient, to move towards lower cost)
nn64_3.learning_rate = 0.05
entire_network_updates = []
for param, grad in zip(entire_network_params, entire_network_gradients):
    entire_network_updates.append((param, param - nn64_3.learning_rate * grad))

print "\n\t[Training] Entire Network:"
entire_network_train = theano.function([i, batch_size], nn64_3.cost, updates=entire_network_updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(entire_network_train, 256, eighty)

nn64_3.set_noise(0)
nn64_3.learning_rate = 0.01
print "\n\t[Tuning] Entire Network:"
entire_network_tune = theano.function([i, batch_size], nn64_3.cost, updates=entire_network_updates,
                                        givens={x:      shared_train[i:i+batch_size],
                                                x_mask: shared_mask[i:i+batch_size]})

run_epochs(entire_network_tune, 256, eighty)

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

# nn64_3.active_hidden should be the predictions of the input data after going through our neural network

# mask out the meaningless data from the neural network's output matrix
# mask out the values we input (the uncorrupted values) from the output matrix (so we are only measuring error on beers that were predicted)
nn64_prediction = T.dot(T.dot(nn64_3.active_hidden, x_mask), 1 - nn64_1.noise)

p = theano.function([x,x_mask], nn64_prediction)

# use root mean squared error to check how accurate our predictions were when using the entire neural network
nn64_test_error = T.mean(T.pow(T.mean(T.pow(nn64_prediction - nn64_1.inputs, 2)), 0.5))

######## How do I actually call the testing function with Theano?

n64_testing_function = theano.function([x,x_mask], nn64_test_error)

cost = []
for i in xrange(0,10):
    cost.append(n64_testing_function(shared_train.get_value(), shared_mask.get_value().T))

print "\n\t[Testing] 64-hidden node autoencoder error: {}".format(T.mean(cost))




