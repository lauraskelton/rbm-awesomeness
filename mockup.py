execfile('nodeviz.py')
import nodeviz as nv
import datavisualizer as visualizer
import pandas as pd
beer_weights = visualizer.load_all_beer_weight_ids()
beer_weights_data = pd.DataFrame.from_dict(beer_weights, orient='index')
beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t', index_col='BEER_ID')
beer_data = beer_extra_data.join(beer_weights_data, how='inner')
# W, b_in = visualizer.load_weights_biases("hater_nodes.npz")
# nodeviz = nv.NodeVisualizer(W, b_in, beer_data)


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


def style_vec(style):
	mock = np.mat(beer_data["STYLE_NAME"] == style)
	return mock

def specific_beer_vec(specific_beers):
	mock = np.sum([np.mat(beer_data["BEER"] == beer) for beer in specific_beers], axis=0)
	return mock

def get_favourite_beers(beer_data, data, n):
	if (type(data) == np.matrix) and data.shape == (1, 3814):
		sorted_pairs = sorted(zip(np.array(data.astype(float))[0][:1907], 
											beer_data["BEER"]), key=lambda x: -x[0])
	elif (type(data) == np.ndarray) and data.shape == (3814,):
		sorted_pairs = sorted(zip(data.astype(float), beer_data["BEER"]), key=lambda x: -x[0])
	else:
		print type(data)
		print data.shape
		raise Exception("Figure out why data was passed in weird")
	return [name for score, name in sorted_pairs[:n]]

# nodeviz.activations(style_vec("American IPA"))

full_hater_mode = (np.zeros((1,1907)), np.ones((1,1907)))
lovers_paradise = (np.ones((1,1907)), np.ones((1,1907)))
bug_detection = (np.zeros((1,1907)), np.zeros((1,1907)))

x = ae.matrixType('x')
x_mask = ae.matrixType('mask')
input_combined = T.concatenate([x,x_mask], axis=1)
mask_combined = T.concatenate([x_mask,T.zeros_like(x_mask)], axis=1)

# self.nodeviz = nv.NodeVisualizer(self.W, self.b_in, self.beer_data)
layer1 = ae.load("vanilla1_t.npz", input_combined)
layer2 = ae.load("strawberry2_t.npz", layer1.active_hidden)
layer3 = ae.load("chocolate3_t.npz", layer2.active_hidden, mask=x_mask, original_input=input_combined)

viz = nv.NodeVisualizer([layer1, layer2, layer3], x, x_mask, beer_data)


untuned1 = ae.load("lono_vanilla1_t.npz", input_combined)
untuned2 = ae.load("lono_strawberry2_t.npz", untuned1.active_hidden)
untuned3 = ae.load("lono_chocolate3_t.npz", untuned2.active_hidden, mask=x_mask, original_input=input_combined)

unviz = nv.NodeVisualizer([untuned1, untuned2, untuned3], x, x_mask, beer_data)

def pp(vector):
	print ["{:.4f}".format(r) for r in vector]

def quicktest(n):
	return get_favourite_beers(beer_data, unviz.activations[2](np.mat(train_set[n]), np.mat(train_mask[n]))[0], 25)

pp(viz.activations[1](*full_hater_mode)[0])
pp(viz.activations[1](*lovers_paradise)[0])
pp(viz.activations[1](*bug_detection)[0])

def get_avg_rating(beer):
	idx = names.index(beer)
	r = 0.
	n = 0
	for rating in test_set[:,idx]:
		if rating != 0:
			r += rating
			n += 1

	return (r, n)