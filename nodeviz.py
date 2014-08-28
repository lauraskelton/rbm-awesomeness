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
from datavisualizer import rgbStrings
import json


beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t')


class NodeVisualizer(object):
	def __init__(self, layers, x, x_mask, beer_data):
		for layer in layers:
			layer.set_noise(0)

		self.layers = layers		
		self.activations = [theano.function([x, x_mask], layer.active_hidden) for layer in self.layers]

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

	# def get_colors(self, cats=None, style=None, specific_beer=None, **kwargs):
	# 	print cats
	# 	print kwargs
	# 	mock = self.mock_vector(cats, **kwargs)
	# 	# import pdb; pdb.set_trace()
	# 	activations = self.activations(self.mock_vector(cats, **kwargs))[0]
	# 	print activations

	# 	strings = rgbString(activations, activations.max(), activations.min())
	# 	print strings
	# 	return json.dumps({"type":"colors","data":strings})

	def style_vec(style):
		mock = np.mat(self.beer_data["STYLE_NAME"] == style)
		return mock

	def specific_beer_vec(specific_beer):
		mock = np.mat(beer_data["BEER"] == specific_beer)
		return mock

	# def get_d3_node_data(self):
	# 	# 	var circleData = [{cx:40,cy:60}, {cx:80,cy:60}, {cx:120,cy:60}]
	# 	circleData = []

	# 	# NOTE: need a way to create nodes for multiple layers.
	# 	# What property of self contains the different layers' node activations?
	# 	# All we would need to do is to create a different "cy" for each hidden layer.

	# 	# ALSO: it would be good to take a weights vector and draw lines between the nodes of different layers, for the deep networks.

	# 	for i in range(self.neuralnet.n_hidden):
	# 		circleData.append({"cx": ((1+i) * 40),"cy": 60})
	# 	return json.dumps({"type":"circles","data":circleData})

	def get_node_colors(self, cats=None, style=None, specific_beer=None, **kwargs):
		print cats
		print kwargs
		mock = self.mock_vector(cats, **kwargs)
		activations_vectors = [activations(self.mock_vector(cats, **kwargs), np.mat(np.ones(1907)))[0] for activations in self.activations]
		print activations_vectors

		# [num for list in x for num in list]
		# [ item for innerlist in outerlist for item in innerlist ]
		# this = [rgbString(activations, activations.max(), activations.min()) for innerlist in activations_vectors for activations in innerlist]
		# import pdb; pdb.set_trace()
		list_of_lists_of_rgbs = [rgbStrings(vector, vector.max(), vector.min()) for vector in activations_vectors]
		rgb_string_list = [rgb_strings for innerlist in list_of_lists_of_rgbs for rgb_strings in innerlist]
		print rgb_string_list

		outString = []
		outString.append("rgb({},{},{})".format(200,0,0))
		outString = outString + rgb_string_list
		return json.dumps({"type":"colors","data":outString})

	def get_d3_node_data_network(self):
		# NOTE: these need to represent 2 different layers in reality.	
		# self.layers -> list of layers??

		#hidden1 = self.neuralnet.n_hidden
		#hidden2 = self.neuralnet.n_hidden
		nodeData = []
		linkData = []
		# NOTE: need a way to create nodes for multiple layers.
		# What property of self contains the different layers' node activations?
		# All we would need to do is to create a different "cy" for each hidden layer.

		# ALSO: it would be good to take a weights vector and draw lines between the nodes of different layers, for the deep networks.
		
		# large "input" node in place of a node for each input beer (way too many- over 3000 inputs actually)
		nodeData.append({"size": 100,"fixed":"true","x":0.5,"y":0}) # this is node 0

		for layer_index, layer in enumerate(self.layers):
			node_count = layer.n_hidden
			#node_count = 64
			if node_count > 80:
				node_size = 1
			elif node_count > 25:
				node_size = 8
			else:
				node_size = 10
			if layer_index == len(self.layers) - 1:
				continue

			node_last_index = len(nodeData) - 1 # this is the id of the final node of the previous layer
			for i in range(node_count):
				# nodes 1 to hidden1 (i+1 is the node number)
				nodeNum = node_last_index + 1 + i # this is the next node id after the existing nodes we have saved
				if layer_index == 0:
					linkData.append({"source":0,"target":nodeNum}) # link the "input" node 0 to every node in the first hidden layer
					
				if layer_index < (len(self.layers) - 1): # not the final hidden layer so link to next layer
					next_node_count = self.layers[layer_index+1].n_hidden # access the next layer so we can make links
					for j in range(next_node_count):
						# nodes hidden1+1 to hidden1+hidden2
						secondNodeNum = node_last_index + node_count + 1 + j # the id of each node in the next layer
						# add a link from this node to each second layer node...
						linkData.append({"source":nodeNum,"target":secondNodeNum})

				nodeData.append({"size": node_size,"fixed":"true","x":float(i + 1)/float(node_count + 1),"y":layer_index + 1})

		return json.dumps({"type":"nodes","data":{"nodes":nodeData,"links":linkData}})

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

