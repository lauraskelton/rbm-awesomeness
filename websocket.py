import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.httpserver

import nodeviz as nv
import datavisualizer as visualizer
import pandas as pd

import autoencoder as ae
import theano
import theano.tensor as T

class WSHandler(tornado.websocket.WebSocketHandler):
	def open(self):
		print 'new connection'
		# self.W, self.b_in = visualizer.load_weights_biases()
		beer_weights = visualizer.load_all_beer_weight_ids()
		beer_weights_data = pd.DataFrame.from_dict(beer_weights, orient='index')
		beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t', index_col='BEER_ID')
		self.beer_data = beer_extra_data.join(beer_weights_data, how='inner')

		x = ae.matrixType('x')
		x_mask = ae.matrixType('mask')
		input_combined = T.concatenate([x,x_mask], axis=1)
		mask_combined = T.concatenate([x_mask,T.zeros_like(x_mask)], axis=1)

		# self.nodeviz = nv.NodeVisualizer(self.W, self.b_in, self.beer_data)
		layer1 = ae.load("vanilla1_t.npz", input_combined)
		layer2 = ae.load("strawberry2_t.npz", layer1.active_hidden)
		layer3 = ae.load("chocolate3_t.npz", layer2.active_hidden, mask=x_mask, original_input=input_combined)

		# NOTE: fix this- chocolate layer to produce most active beers
		self.viz = nv.NodeVisualizer([layer1, layer2, layer3], x, x_mask, self.beer_data)

		# simple D3 circles
		#circleData = self.nodeviz.get_d3_node_data()
		#self.write_message(circleData)
		#print circleData

		# D3 network nodes
		networkData = self.viz.get_d3_node_data_network()
		self.write_message(networkData)
		# print networkData
	
	def on_message(self, message):
		print 'message received %s' % message
		#self.write_message("red blue green")
		
		# send this to nodeviz somehow?
		
		if message.startswith("setBucket"):
			message_array = message.split(' ')[1:]
			buckets_dict = {}
			for i in range(len(message_array)):
				if i % 2 == 0:
					buckets_dict[str(message_array[i])] = int(message_array[i+1])

			# simple D3 circles
			#colors = self.nodeviz.get_colors(**buckets_dict)

			# D3 network nodes
			colors = self.viz.get_node_colors(**buckets_dict)
			self.write_message(colors)

		elif message.startswith("setBeer"):
			beers = message[len("setBeer"):].strip().split(",")
			# clearn up the strings
			beers = [beer.strip() for beer in beers]


			buckets_dict = {"specific_beers" : beers}

			# simple D3 circles
			#colors = self.nodeviz.get_colors(**buckets_dict)

			# D3 network nodes
			colors = self.viz.get_node_colors(**buckets_dict)
			self.write_message(colors)

		else:
			print "I don't know what to do with this message:\n" + message

	def on_close(self):
		print 'connection closed'


application = tornado.web.Application([(r'/ws', WSHandler),])

if __name__ == "__main__":
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(8000)
	tornado.ioloop.IOLoop.instance().start()


