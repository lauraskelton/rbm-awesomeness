# -*- coding: utf-8 -*-
import datamanager as dm
import autoencoder as ae
import numpy as np
import theano
import theano.tensor as T
import time
import datavisualizer as vis
import pandas as pd
from pandas import DataFrame as DF
from datavisualizer import rgbString
import json


beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t')


class NodeVisualizer(object):
	def __init__(self, W, b_in, beer_data):
		self.vec = ae.matrixType("vec")
		self.neuralnet = ae.CFAutoencoder(W.shape[0], W.shape[1], self.vec, pct_noise=0, W=W, b_in=b_in)
		self.activations = theano.function([self.vec], self.neuralnet.active_hidden)

		self.beer_data = beer_data
		self.beer_data["ABV"] = self.beer_data["ABV"].apply(toFloat)

		self.buckets = {}
		self.buckets["ABV"] = get_buckets(beer_data['ABV'])
		self.buckets["GRAVITY"] = get_buckets(beer_data['GRAVITY'])
		self.buckets["COLOR"] = get_buckets(beer_data['COLOR'])
		self.buckets["IBU"] = get_buckets(beer_data['IBU'])

	def mock_vector(self, cats=None, style=None, specific_beer=None, **kwargs):
		out = np.zeros((1, len(self.beer_data)))
		zeros = np.zeros((1, len(self.beer_data)))

		normalizer = 0.0

		for metric, bucket in kwargs.iteritems():
			u, s = self.buckets[metric][bucket]
			delta = np.mat(self.beer_data[metric].apply(lambda x : gauss(x, u, s/2)).fillna(0))

			max_gauss = gauss(u, u, s/2)

			# maybe add them together and divide at end? This could result in just a bunch of 1s
			out += delta / max_gauss
			normalizer += 1.
			# import pdb;pdb.set_trace()

		if cats: # u'🐱🐱🐱' lol laura are these unicode 'cats' ?
			for cat in cats: # u'🐱' haha obviously
				delta = self.beer_data["CATEGORY_NAME"] == cat
				out += delta
				normalizer += 1.

		if style:
			out = style_vec(style) # overrides other vectors

		if specific_beer:
			out = specific_beer_vec(specific_beer) # overrides other vectors


		if normalizer:
			return out/normalizer

		return out

	def get_colors(self, cats=None, style=None, specific_beer=None, **kwargs):
		print cats
		print kwargs
		mock = self.mock_vector(cats, **kwargs)
		# import pdb; pdb.set_trace()
		activations = self.activations(self.mock_vector(cats, **kwargs))[0]
		print activations

		strings = rgbString(activations, activations.max(), activations.min())
		print strings
		return json.dumps({"type":"colors","data":strings})

	def style_vec(style):
		mock = np.mat(self.beer_data["STYLE_NAME"] == style)
		return np.concatenate([mock, mock], axis=1)

	def specific_beer_vec(specific_beer):
		mock = np.mat(beer_data["BEER"] == specific_beer)
		return np.concatenate([mock, mock], axis=1)

	def get_d3_node_data(self):
		# 	var circleData = [{cx:40,cy:60}, {cx:80,cy:60}, {cx:120,cy:60}]
		circleData = []

		# NOTE: need a way to create nodes for multiple layers.
		# What property of self contains the different layers' node activations?
		# All we would need to do is to create a different "cy" for each hidden layer.

		# ALSO: it would be good to take a weights vector and draw lines between the nodes of different layers, for the deep networks.
		
		for i in range(self.neuralnet.n_hidden):
			circleData.append({"cx": ((1+i) * 40),"cy": 60})
		return json.dumps({"type":"circles","data":circleData})

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

