import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.httpserver

import nodeviz as nv
import datavisualizer as visualizer
import pandas as pd

class WSHandler(tornado.websocket.WebSocketHandler):
	def open(self):
		print 'new connection'
		self.W, self.b_in = visualizer.load_weights_biases()
		beer_weights = visualizer.load_all_beer_weight_ids()
		beer_weights_data = pd.DataFrame.from_dict(beer_weights, orient='index')
		beer_extra_data = pd.read_csv('data/beer_data.csv', sep='\t', index_col='BEER_ID')
		self.beer_data = beer_extra_data.join(beer_weights_data, how='inner')
		self.nodeviz = nv.NodeVisualizer(self.W, self.b_in, self.beer_data)
	
	def on_message(self, message):
		print 'message received %s' % message
		self.write_message("red blue green")
		
		# send this to nodeviz somehow?
		message_array = message.split(' ')
		if len(message_array) > 0 and message_array[0] == "setBucket" and len(message_array) % 2 == 1:
			del message_array[0]
			buckets_dict = {}
			for i in range(len(message_array)):
				if i % 2 == 0:
					buckets_dict[str(message_array[i])] = int(message_array[i+1])

			colors = self.nodeviz.get_colors(**buckets_dict)
			self.write_message(' '.join(colors))

	def on_close(self):
		print 'connection closed'


application = tornado.web.Application([(r'/ws', WSHandler),])

if __name__ == "__main__":
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(8000)
	tornado.ioloop.IOLoop.instance().start()


