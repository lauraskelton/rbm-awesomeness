import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time
import trainer


#######################
# prepare the data

data, mask, names = dm.createNDArray()

# for negative inputs
# data = dm.haterizeArray(data)

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

input_combined = T.concatenate([x,x_mask], axis=1)

mask_combined = T.concatenate([x_mask,T.zeros_like(x_mask)], axis=1)

####################
# TRAINING WITH WEIGHT DECAY

# refactored_layer = ae.CFAutoencoder(data.shape[1]*2, 16, inputs=input_combined, mask=mask_combined)

# print "\n\t[Training] a network with weight decay!"

# aet = trainer.AETrainer(refactored_layer, refactored_layer.cost, x, shared_train, x_mask=x_mask, shared_mask=shared_mask)

# aet.run_epochs(min_epochs=200, lr_decay=0.1)

# decay_layer.save("hater_nodes")



# #######################################################################################
# 		THIS IS A 256 - 64
# 					\   /
# 					3814
# 			Fancy-pants Network
# #################################


#  
# raise Error("I refactored the Autoencoder/Trainer separation since this code was written. Needs updating.")
# layer1 = ae.CFAutoencoder(data.shape[1]*2, 256, inputs=input_combined, mask=mask_combined, 
#                                 weight_decay=0.0001)
# aet1 = trainer.AETrainer(layer1, x, shared_train, x_mask=x_mask, shared_mask=shared_mask, momentum=0.9)
# aet1.run_epochs(min_epochs=200, lr_decay=0.1)
# layer1.set_noise(0.0)
# layer1.save("layer1_deep")

# layer2 = ae.CFAutoencoder(layer1.n_hidden, 64, inputs=layer1.active_hidden, weight_decay=0.0001)
# aet2 = trainer.AETrainer(layer2, x, shared_train, x_mask=x_mask, shared_mask=shared_mask, momentum=0.9)
# aet2.run_epochs(min_epochs=200, lr_decay=0.1)
# layer2.set_noise(0.0)
# layer2.save("layer2_deep")


# layer1.set_noise(0.5)
# layer3 = ae.CFAutoencoder(layer1.n_hidden + layer2.n_hidden, data.shape[1]*2,
# 							inputs=T.concatenate([layer1.active_hidden, layer2.active_hidden], axis=1),
# 							mask=mask_combined, weight_decay = 0.0001)
# aet3 = trainer.AETrainer(layer3, x, shared_train, x_mask=x_mask, shared_mask=shared_mask, momentum=0.9)
# aet3.run_epochs(min_epochs=200, lr_decay=0.1)
# layer3.save("layer3_deep")


# #############
# # RELOAD & tuning

# layer1 = ae.load("layer1_deep.npz", input_combined, mask_combined)
# layer2 = ae.load("layer2_deep.npz", layer1.active_hidden, mask_combined)
# layer3 = ae.load("layer3_deep.npz", T.concatenate([layer1.active_hidden, layer2.active_hidden], axis=1),
# 					mask=mask_combined, original_input=input_combined)

# layer1.set_noise(0.33)
# layer3.set_noise(0.0)

# for layer in [layer1, layer2, layer3]:
# 	if layer.b_out in layer.parameters:
# 		layer.parameters.remove(layer.b_out)

# tuner = trainer.AETrainer([layer1, layer2, layer3], layer3.cost, 
# 							x, shared_train, x_mask=x_mask, shared_mask=shared_mask)

# tuner.run_epochs(min_epochs=100)

# layer1.save("layer1_tuned")
# layer2.save("layer2_tuned")
# layer3.save("layer3_tuned")

#################################################################################################



# #######################################################################################
# 		THIS IS A 128 - 16 - 3814
# 				Neapolitan Network
# #################################

# layer1 = ae.CFAutoencoder(data.shape[1]*2, 64, inputs=input_combined, mask=mask_combined)
# layer1.set_noise(0.2)
# aet1 = trainer.AETrainer(layer1, layer1.cost, x, shared_train, x_mask=x_mask, shared_mask=shared_mask)
# aet1.run_epochs(min_epochs=100, lr_decay=0.1)


# layer1.set_noise(0.0)
# layer1.save("lono_vanilla1")

# layer2 = ae.CFAutoencoder(layer1.n_hidden, 16, inputs=layer1.active_hidden)
# layer2.set_noise(0.2)
# aet2 = trainer.AETrainer(layer2, layer2.cost, x, shared_train, x_mask=x_mask, shared_mask=shared_mask)
# aet2.run_epochs(min_epochs=100, lr_decay=0.1)
# layer2.set_noise(0.0)
# layer2.save("lono_strawberry2")


# layer3 = ae.CFAutoencoder(layer2.n_hidden, data.shape[1]*2, inputs=layer2.active_hidden, 
# 							mask=mask_combined, original_input=input_combined)
# layer3.set_noise(0.2)
# aet3 = trainer.AETrainer(layer3, layer3.cost, x, shared_train, x_mask=x_mask, shared_mask=shared_mask)

# aet3.run_epochs(min_epochs=100, lr_decay=0.1)
# layer3.save("lono_chocolate3")


# #############
# # RELOAD & tuning

layer1 = ae.load("lono_vanilla1.npz", input_combined)
layer2 = ae.load("lono_strawberry2.npz", layer1.active_hidden)
layer3 = ae.load("lono_chocolate3.npz", layer2.active_hidden, mask=x_mask, original_input=input_combined)

layer1.set_noise(0.2)
layer3.set_noise(0.0)

for layer in [layer1, layer2, layer3]:
	if layer.b_out in layer.parameters:
		layer.parameters.remove(layer.b_out)

tuner = trainer.AETrainer([layer1, layer2, layer3], layer3.cost, x, shared_train, x_mask=x_mask, 
							shared_mask=shared_mask, learning_rate=0.0005, weight_decay=0)

tuner.run_epochs(min_epochs=100, decay_modulo=25, lr_decay=0.2)

layer1.save("lono_vanilla1_t")
layer2.save("lono_strawberry2_t")
layer3.save("lono_chocolate3_t")

#################################################################################################



# nn64_1 = ae.CFAutoencoder(data.shape[1], 64, inputs=x, mask=x_mask)

# nn64_2 = ae.CFAutoencoder(nn64_1.n_hidden, 32, inputs=nn64_1.active_hidden)

# # final output layer
# nn64_3 = ae.CFAutoencoder(nn64_2.n_hidden, data.shape[1], inputs=nn64_2.active_hidden, 
#                             mask=x_mask, original_input=x)

######################
# train the network



# print "\n\t[Training] Layer 1:"
# layer1_train = theano.function([i, batch_size], nn64_1.cost, updates=nn64_1.updates,
#                                         givens={x:      shared_train[i:i+batch_size],
#                                                 x_mask: shared_mask[i:i+batch_size]})

# run_epochs(layer1_train, 256, eighty)

# nn64_1.set_noise(0)
# nn64_1.learning_rate = 0.01
# print "\n\t[Tuning] Layer 1:"
# layer1_tune = theano.function([i, batch_size], nn64_1.cost, updates=nn64_1.updates,
#                                         givens={x:      shared_train[i:i+batch_size],
#                                                 x_mask: shared_mask[i:i+batch_size]})

# run_epochs(layer1_tune, 256, eighty)


# print "\n\t[Training] Layer 2:"
# layer2_train = theano.function([i, batch_size], nn64_2.cost, updates=nn64_2.updates,
#                                         givens={x:      shared_train[i:i+batch_size]})

# run_epochs(layer2_train, 256, eighty)

# nn64_2.set_noise(0)
# nn64_2.learning_rate = 0.01
# print "\n\t[Tuning] Layer 2:"
# layer2_tune = theano.function([i, batch_size], nn64_2.cost, updates=nn64_2.updates,
#                                         givens={x:      shared_train[i:i+batch_size]})

# run_epochs(layer2_tune, 256, eighty)


# print "\n\t[Training] Output Layer:"
# layer3_train = theano.function([i, batch_size], nn64_3.cost, updates=nn64_3.updates,
#                                         givens={x:      shared_train[i:i+batch_size],
#                                                 x_mask: shared_mask[i:i+batch_size]})

# run_epochs(layer3_train, 256, eighty)

# nn64_3.set_noise(0)
# nn64_3.learning_rate = 0.01
# print "\n\t[Tuning] Output Layer:"
# layer3_tune = theano.function([i, batch_size], nn64_3.cost, updates=nn64_3.updates,
#                                         givens={x:      shared_train[i:i+batch_size],
#                                                 x_mask: shared_mask[i:i+batch_size]})

# run_epochs(layer3_tune, 256, eighty)


######################
# train the entire network


# the cost function is the same as the final output layer cost
# the only difference is that we need to use updates to update the weights and biases of the ENTIRE NETWORK,
# not just the current layer... so the gradient function might be different? and the updates function is different... or extended at least...

# self.parameters = [self.W, self.b_in, self.b_out]

# set gradient to depend on all of the parameters we set above, so that "updates" will update all of the layers' weights and biases

# entire_network_params = [nn64_1.W, nn64_1.b_in, nn64_2.W, nn64_2.b_in, nn64_3.W, nn64_3.b_in]
# entire_network_gradients = T.grad(nn64_3.cost, entire_network_params)

# # make a vector of the new values of each parameter by descending slightly along the gradient (opposite direction of gradient, to move towards lower cost)
# nn64_3.learning_rate = 0.05
# entire_network_updates = []
# for param, grad in zip(entire_network_params, entire_network_gradients):
#     entire_network_updates.append((param, param - nn64_3.learning_rate * grad))

# print "\n\t[Training] Entire Network:"
# entire_network_train = theano.function([i, batch_size], nn64_3.cost, updates=entire_network_updates,
#                                         givens={x:      shared_train[i:i+batch_size],
#                                                 x_mask: shared_mask[i:i+batch_size]})

# run_epochs(entire_network_train, 256, eighty)

# nn64_3.set_noise(0)
# nn64_3.learning_rate = 0.01
# print "\n\t[Tuning] Entire Network:"
# entire_network_tune = theano.function([i, batch_size], nn64_3.cost, updates=entire_network_updates,
#                                         givens={x:      shared_train[i:i+batch_size],
#                                                 x_mask: shared_mask[i:i+batch_size]})

# run_epochs(entire_network_tune, 256, eighty)

# let's get this thing trained!

######################
# test the network

# need to start with real input data at the input layer, but corrupted
# we already did this in nn64... it should be just nn64_1.noisy. done.

# save mask of how data was corrupted, as well as mask of missing/meaningless input data
# again, we already did this in nn64... it is nn64_1.noise.
# but wait! we called set_noise(0) above, so we need to reset the noise to re-corrupt the inputs.
# nn64_1.set_noise(0.5)

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
# # mask out the values we input (the uncorrupted values) from the output matrix (so we are only measuring error on beers that were predicted)
# nn64_prediction = T.dot(T.dot(x_mask, T.transpose(nn64_3.active_hidden)), 1 - nn64_1.noise)

# # use root mean squared error to check how accurate our predictions were when using the entire neural network
# nn64_test_error = T.mean(T.pow(T.mean(T.pow(nn64_prediction - nn64_1.inputs, 2)), 0.5))




# layer1_pred = T.dot(T.dot(x_mask, T.transpose(nn64_1.output)), 1-nn64_1.noise)



######## How do I actually call the testing function with Theano?

# n64_testing_function = theano.function([x,x_mask], nn64_test_error)

# cost = []
# for i in xrange(0,10):
#     cost.append(n64_testing_function(shared_train.get_value(), shared_mask.get_value()))

# print "\n\t[Testing] 64-hidden node autoencoder error: {}".format(T.mean(cost))




