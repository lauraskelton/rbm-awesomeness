from PIL import Image,ImageDraw,ImageFilter
import ftplib
import numpy as np
import operator

# Load beer style information for each beer
def loadStyle(path='data'):
	# example line from beerstyle.txt would be "Lagunitas IPA\tAmerican IPA" (beer,style)
	styleData={}
	for line in open(path+'/beerstyle.txt'):
		(name,style)=line.split('\t')
		beer=name.rstrip()
		if style.rstrip() == 'NULL':
			styleData[beer]=str('')
		else:
			styleData[beer]=str(style.rstrip())
	return styleData

def writeBeerMapData(beerWeights):
	style = loadStyle()
	output = []
	newBeersArray = sorted(beerWeights.iteritems(), key=operator.itemgetter(1))
	hOffset = 0
	for beer, ratingWeight in newBeersArray[-50:]:
		loc = [vOffset * 0.01, hOffset * 0.01]
		(r,g,b) = getBackgroundColor(ratingWeight, np.max(beerWeights.values()), np.min(beerWeights.values()))
		output.append(outputBeer(loc,beer,style[beer],'#000000','rgb('+str(r)+','+str(g)+','+str(b)+')'))
		hOffset += 1
	return ''.join(output)

def rgbMix(colorValue, maxColorValue, minColorValue, red=255, green=109, blue=87):
	r=red
	g=green
	b=blue
	if colorValue < maxColorValue and colorValue > minColorValue:
		scaling = (maxColorValue - colorValue)/(maxColorValue - minColorValue)
		r=scaleColor(r, scaling)
		g=scaleColor(g, scaling)
		b=scaleColor(b, scaling)
	elif colorValue <= minColorValue:
		r=255
		g=255
		b=255
	return r, g, b

def getBackgroundColor(ratingWeight, maxWeight, minWeight):
	if ratingWeight >= 0:
		backgroundColor = rgbMix(ratingWeight, maxWeight, 0, red=17, green=217, blue=137)
	else:
		backgroundColor = rgbMix(-ratingWeight, -minWeight, 0, red=217, green=87, blue=17)
	return backgroundColor


def scaleColor(colorInput, scaling):
	return int(scaling * (255-colorInput) + colorInput)

def outputBeer(coords,beer,style,textColor,backgroundColor):
	
	return '\t{latitude: '+str(float((coords[0])*15))+', '\
		+'longitude: '+str(float((coords[1])*15))+', '\
			+'title: "'+str(beer)+'", '\
			+'description: "'+str(style.replace("&","and"))+'",'\
			+'textColor: "'+str(textColor)+'", '\
			+'color: "'+str(backgroundColor)+'"},\n'

def generateGoogleJavascript(beerWeights):
	out = []
	out.append('<!DOCTYPE html>\n')
	out.append('<html>\n')
	out.append('\t<head>\n')
	out.append('\t\t<title>Beer Map</title>\n')
	out.append('\t\t<style>\n')
	out.append('\t\t\thtml, body, #map-canvas {\n')
	out.append('\t\t\t\theight: 100%;\n')
	out.append('\t\t\t\tmargin: 0px;\n')
	out.append('\t\t\t\tpadding: 0px\n')
	out.append('\t\t\t}\n')
	out.append('\t\t</style>\n')
	out.append('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
	out.append('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
	out.append('\t\t<script>\n')
	out.append('\n')
	out.append('function CoordMapType() {\n')
	out.append('}\n')
	out.append('\n')
	out.append('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
	out.append('CoordMapType.prototype.maxZoom = 11;\n')
	out.append('CoordMapType.prototype.minZoom = 5;\n')
	out.append('\n')
	out.append('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
	out.append('\tvar div = ownerDocument.createElement(\'div\');\n')
	out.append('\tdiv.innerHTML = \'\';\n')
	out.append('\tdiv.style.width = this.tileSize.width + \'px\';\n')
	out.append('\tdiv.style.height = this.tileSize.height + \'px\';\n')
	out.append('\tdiv.style.fontSize = \'10\';\n')
	out.append('\tdiv.style.borderStyle = \'none\';\n')
	out.append('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
	out.append('\treturn div;\n')
	out.append('};\n')
	out.append('\n')
	out.append('CoordMapType.prototype.name = \'Beer\';\n')
	out.append('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
	out.append('\n')
	out.append('var map;\n')
	out.append('var coordinateMapType = new CoordMapType();\n')
	out.append('\n')
	
	out.append('var beerMapData = [\n')
	out.append(writeBeerMapData(beerWeights))
	out.append('];\n')
	
	out.append('\n')
	out.append('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
	out.append('var zoomStart = 5;\n')

	out.append('\n')
	out.append('function initialize() {\n')
	out.append('\tvar mapOptions = {\n')
	out.append('\t\tzoom: zoomStart,\n')
	out.append('\t\tcenter: mapCenter,\n')
	out.append('\t\tstreetViewControl: false,\n')
	out.append('\t\tmapTypeId: \'coordinate\',\n')
	out.append('\t\tmapTypeControlOptions: {\n')
	out.append('\t\t\tmapTypeIds: [\'coordinate\'],\n')
	out.append('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
	out.append('\t\t}\n')
	out.append('\t};\n')
	out.append('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
	out.append('\n')
	out.append('\tfor (var i=0;i<beerMapData.length;i++)\n')
	out.append('\t{\n')
	out.append('\t\tvar myOptions = {\n')
	out.append('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
	out.append('\t\t\t,boxStyle: {\n')
	out.append('\t\t\t\tborder: "1px solid black"\n')
	out.append('\t\t\t\t,textAlign: "center"\n')
	out.append('\t\t\t\t,fontSize: "6pt"\n')
	out.append('\t\t\t\t,width: "60px"\n')
	out.append('\t\t\t\t,color: beerMapData[i][\'textColor\']\n')
	out.append('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
	out.append('\t\t\t}\n')
	out.append('\t\t\t,disableAutoPan: true\n')
	out.append('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
	out.append('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
	out.append('\t\t\t,closeBoxURL: ""\n')
	out.append('\t\t\t,isHidden: false\n')
	out.append('\t\t\t,pane: "mapPane"\n')
	out.append('\t\t\t,enableEventPropagation: true\n')
	out.append('\t\t};\n')
	out.append('\t\tvar ibLabel = new InfoBox(myOptions);\n')
	out.append('\t\tibLabel.open(map);\n')
	out.append('\t}\n')
	out.append('\n')
	out.append('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
	out.append('}\n')
	out.append('\n')
	out.append('google.maps.event.addDomListener(window, \'load\', initialize);\n')
	out.append('\n')
	out.append('\t\t</script>\n')
	out.append('\t</head>\n')
	out.append('\t<body>\n')
	out.append('\t\t<div id="map-canvas"></div>\n')
	out.append('\t</body>\n')
	out.append('</html>\n')

	return ''.join(out)

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

def makeBeerMap(beerWeights, filename="beernodemap"):
	out=open("data/%s.html" % (filename),'w')
	out.write(generateGoogleJavascript(beerWeights))
	out.close()
	site,username,password = loadFTPCredentials()
	session = ftplib.FTP(site,username,password)
	file = open("data/%s.html" % (filename),'rb')                  # file to send
	session.storbinary("STOR beermap/%s.html" % (filename), file)     # send the file
	file.close()                                    # close file and FTP
	session.quit()
	print 'gmap uploaded'


# This is the function we actually call!!!
def makeAllBeerMaps(allBeerWeights, filename="beernodemap"):
	i = 0
	for beerWeights in allBeerWeights:
		makeBeerMap(beerWeights, filename="{}{}".format(filename,i))
		i += 1


# Create and upload new beer map to server with optional ABV and IBU coloring
def saveAndPushGoogleBeerMap(loc,seedPrefs,style,abv=None,ibu=None):
	
	backgroundColor = {}
	textColor = {}
	
	for beer in seedPrefs:
		
		rA, gA, bA = rgbMix(abv[beer], 10, 4, red=126, green=200, blue=252)
		rI, gI, bI = rgbMix(ibu[beer], 80, 10, red=231, green=252, blue=126)
		
		r=(rA+rI)-255
		g=(gA+gI)-255
		b=(bA+bI)-255
		
		backgroundColor[beer] = 'rgb('+str(r)+','+str(g)+','+str(b)+')'
		textColor[beer] = '#000000'
	
	makeBeerMap(seedPrefs, loc, style, textColor, backgroundColor, filename="beergmapabvibu", showCenter=False, center=None, highlights=None)
