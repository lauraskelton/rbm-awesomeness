from PIL import Image,ImageDraw,ImageFilter
import ftplib
import numpy as np
import operator
import cPickle as pickle
from datamanager import createNDArray
import datamanager as dm


def write_d3_data(node_data, scale=500):
	for i in range(len(node_data)):
		outputString = '''				{"cx": '''
		outputString += str(float(i*0.8*scale/2))
		outputString += ''', 
		"cy": '''
		outputString += str(float(0.5)*0.8*scale/2)
		outputString += ''', 
		"radius": 5, 
		"font-size": 10},
		'''
	return outputString

def d3_wrapper(data, scale=500):
	outputString = '''
<!DOCTYPE html>
<html>
	<head>
		<script type="text/javascript" src="http://d3js.org/d3.v2.js"></script>
	</head>
	<body>
		<div id="viz"></div>
		<script type="text/javascript">
			//Beer Data Set
			var beerData = [
			'''
	outputString += write_d3_data(data)

	outputString += '''
			];
			//Create the SVG Viewport
			var svgContainer = d3.select("body").append("svg")
				.attr("width",'''
	outputString += str(scale)
	outputString += ''')
				.attr("height",'''
	outputString += str(scale)
	outputString += ''');

			var foreignObject = svgContainer.selectAll("foreignObject")
				.data(beerData)
				.style("stroke", "gray")
   				.style("fill", "white")
				.enter()
				.append("foreignObject");

			

		</script>
	</body>
</html>
	'''
	return outputString

def makeD3Vis(filename="beerd3vis"):
	out=open("html_tmp/%s.html" % (filename),'w')
	out.write(d3_wrapper(np.random.rand(15)))
	out.close()
	site,username,password = loadFTPCredentials()
	session = ftplib.FTP(site,username,password)
	file = open("html_tmp/%s.html" % (filename),'rb')                  # file to send
	session.storbinary("STOR beermap/%s.html" % (filename), file)     # send the file
	file.close()                                    # close file and FTP
	session.quit()
	print 'gmap uploaded'

def loadFTPCredentials():
	site = ''
	username = ''
	password = ''
	for line in open('data/ftp.txt'):
		(asite,ausername,apassword)=line.split('\t')
		site=str(asite.rstrip())
		username=str(ausername.rstrip())
		password=str(apassword.rstrip())
	return site, username, password
