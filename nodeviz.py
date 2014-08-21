-import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time
import datavisualizer as vis
import pandas as pd
from pandas import DataFrame as DF
from datavisualizer import rgbMix


beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t')


class NodeVisualizer(object):
	def __init__(self, W, b_in, beer_data):
		self.vec = T.dvector("vec")
		self.ae = ae.CFAutoencoder(W.shape[0], W.shape[1], vec, pct_noise=0, W=W, b_in=b_in)
		self.activations = theano.function([vec], ae.active_hidden)

		self.beer_data = beer_data
		self.beer_data["ABV"] = self.beer_data["ABV"].apply(toFloat)

		self.buckets = {}
		self.buckets["ABV"] = get_buckets(beer_data['ABV'])
		self.buckets["GRAVITY"] = get_buckets(beer_data['GRAVITY'])
		self.buckets["COLOR"] = get_buckets(beer_data['COLOR'])
		self.buckets["IBU"] = get_buckets(beer_data['IBU'])

	def mock_vector(self, cats=None, **kwargs):
		out = np.zeros(len(self.beer_data))

		for metric, bucket in kwargs:
			u, s = self.buckets[metric][bucket]

			delta = self.beer_data[metric].apply(lambda x : gauss(x, u, s/2)).fillna(0)

			max_gauss = gauss(u, u, s/2)

			# maybe add them together and divide at end? This could result in just a bunch of 1s
			out = np.maximum(out, delta / max_gauss)

		if cats:
			for cat in cats:
				delta = self.beer_data["CATEGORY_NAME"] == cat
				out = np.maximum(out, delta)

		return out

	def get_colors(self, cats=None, **kwargs):
		return [rgbMix(value, 1, 0) for value in self.activations(self.mock_vector(cats, kwargs))]


def get_buckets(metric):
	metric = metric.copy()
	metric.sort()
	n = sum(1-metric.apply(np.isnan))
-	step = int(n / 5.)

	out = []
	# first 4 quantiles
	for i in xrange(5):
		d = metric[i*step : (i+1)*step]
		out.append((np.mean(d), np.std(d)))

	return out


def gauss(x,u,s): 
	x = float(x)
	u = float(u)
	s = float(s)
	return np.exp(-(x-u)**2/(2*s)**2)/(s*np.sqrt(2*np.pi))

def toFloat(s):
	try:
		return float(s)
	except:
		return None

