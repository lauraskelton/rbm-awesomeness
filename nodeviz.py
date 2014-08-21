import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time
import datavisualizer as vis
import pandas as pd
from pandas import DataFrame as DF


beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t')


class NodeVisualizer(object):
	def __init__(self, ae, beer_data):
		self.ae = ae
		self.beer_data = beer_data
		self.beer_data["ABV"] = self.beer_data["ABV"].apply(toFloat)

		self.buckets = {}
		self.buckets["ABV"] = get_buckets(beer_data['ABV'])
		self.buckets["GRAVITY"] = get_buckets(beer_data['GRAVITY'])


	def mock_vector(self, **kwargs):
		out = np.zeros(len(self.beer_data))

		for metric, bucket in kwargs:
			u, s = self.buckets[metric][bucket]

			g = self.beer_data[metric].apply(lambda x : gauss(x, u, s/2)).fillna(0)

			max_gauss = gauss(u, u, s/2)
			out = np.maximum(out, g / max_gauss)

		return out

def get_buckets(metric):
	metric = metric.copy()
	metric.sort()
	n = sum(1-metric.apply(np.isnan))
	step = int(n / 5.)

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

