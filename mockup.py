execfile('nodeviz.py')
import nodeviz as nv
import datavisualizer as visualizer
import pandas as pd
beer_weights = visualizer.load_all_beer_weight_ids()
beer_weights_data = pd.DataFrame.from_dict(beer_weights, orient='index')
beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t', index_col='BEER_ID')
W, b_in = visualizer.load_weights_biases("hater_nodes.npz")
beer_data = beer_extra_data.join(beer_weights_data, how='inner')
nodeviz = nv.NodeVisualizer(W, b_in, beer_data)
def style_vec(style):
	mock = np.mat(beer_data["STYLE_NAME"] == style)
	return np.concatenate([mock, mock], axis=1)
nodeviz.activations(style_vec("American IPA"))

full_hater_mode = np.concatenate([np.zeros((1,1907)), np.ones((1,1907))], axis=1)
lovers_paradise = np.ones((1,3814))