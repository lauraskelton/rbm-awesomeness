import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.httpserver

import nodeviz as viz
from datavisualizer import load_weights_biases
import pandas as pd

class WSHandler(tornado.websocket.WebSocketHandler):
	def open(self):
		print 'new connection'
	
	def on_message(self, message):
		print 'message received %s' % message
		self.write_message("red blue green")
		
		# send this to nodeviz somehow?
		message_array = message.split(' ')
		if len(message_array) == 3 and message_array[0] == "setBucket":
			viz.set_bucket(int(message_array[1]),int(message_array[2]))

	def on_close(self):
		print 'connection closed'


application = tornado.web.Application([(r'/ws', WSHandler),])

if __name__ == "__main__":
	http_server = tornado.httpserver.HTTPServer(application)
	http_server.listen(8000)
	tornado.ioloop.IOLoop.instance().start()

W, b_in = load_weights_biases
beer_data = pd.read_csv('data/beer_data.csv', sep='	', index_col='BEER_ID')
nodeviz = viz.NodeVisualiser(W, b_in, beer_data)
