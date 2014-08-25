execfile('nodeviz.py')
import nodeviz as nv
import datavisualizer as visualizer
import pandas as pd
beer_weights = visualizer.load_all_beer_weight_ids()
beer_weights_data = pd.DataFrame.from_dict(beer_weights, orient='index')
beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t', index_col='BEER_ID')
W, b_in = visualizer.load_weights_biases()
beer_data = beer_extra_data.join(beer_weights_data, how='inner')
nodeviz = nv.NodeVisualizer(W, b_in, beer_data)
mock = nodeviz.mock_vector(cats=["India Pale Ale"])
np.sum(mock)
