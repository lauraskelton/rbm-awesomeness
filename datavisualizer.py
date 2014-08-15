from PIL import Image,ImageDraw,ImageFilter
import ftplib
import numpy as np
import operator
import cPickle as pickle
from datamanager import createNDArray

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

def writeBeerMapData(allBeerWeights):
	style = loadStyle()
	output = []
	vOffset = 0
	for beerWeights in allBeerWeights:
		newBeersArray = sorted(beerWeights.iteritems(), key=operator.itemgetter(1))
		hOffset = 0
		for beer, ratingWeight in newBeersArray[:10]:
			loc = [vOffset * 0.05, hOffset * 0.04]
			(r,g,b) = getBackgroundColor(ratingWeight, np.max(beerWeights.values()), np.min(beerWeights.values()))
			output.append(outputBeer(loc,beer,style[beer],'#000000','rgb('+str(r)+','+str(g)+','+str(b)+')'))
			hOffset += 1
		for beer, ratingWeight in newBeersArray[-10:]:
			loc = [vOffset * 0.05, hOffset * 0.04]
			(r,g,b) = getBackgroundColor(ratingWeight, np.max(beerWeights.values()), np.min(beerWeights.values()))
			output.append(outputBeer(loc,beer,style[beer],'#000000','rgb('+str(r)+','+str(g)+','+str(b)+')'))
			hOffset += 1
		vOffset += 1
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

def generateGoogleJavascript(allBeerWeights):
	return_string = '''
<!DOCTYPE html>
<html>
	<head>
		<title>Beer Map</title>
		<style>
			html, body, #map-canvas {
				height: 100%;
				margin: 0px;
				padding: 0px
			}
		</style>
		<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>
		<script type="text/javascript" src="../beermap/infobox.js"></script>
		<script>

function CoordMapType() {
}

CoordMapType.prototype.tileSize = new google.maps.Size(256,256);
CoordMapType.prototype.maxZoom = 11;
CoordMapType.prototype.minZoom = 5;

CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {
	var div = ownerDocument.createElement('div');
	div.innerHTML = '';
	div.style.width = this.tileSize.width + 'px';
	div.style.height = this.tileSize.height + 'px';
	div.style.fontSize = '10';
	div.style.borderStyle = 'none';
	div.style.backgroundColor = '#E5E3DF';
	return div;
};

CoordMapType.prototype.name = 'Beer';
CoordMapType.prototype.alt = 'Beer Map Type';

var map;
var coordinateMapType = new CoordMapType();

var beerMapData = [
'''
	return_string += writeBeerMapData(allBeerWeights)
	return_string += '''
];

var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);
var zoomStart = 5;

function initialize() {
	var mapOptions = {
		zoom: zoomStart,
		center: mapCenter,
		streetViewControl: false,
		mapTypeId: 'coordinate',
		mapTypeControlOptions: {
			mapTypeIds: ['coordinate'],
			style: google.maps.MapTypeControlStyle.DROPDOWN_MENU
		}
	};
	map = new google.maps.Map(document.getElementById('map-canvas'),mapOptions);

	for (var i=0;i<beerMapData.length;i++)
	{
		var myOptions = {
			content: beerMapData[i]['title']+"<br/>("+beerMapData[i]['description']+")"
			,boxStyle: {
				border: "1px solid black"
				,textAlign: "center"
				,fontSize: "6pt"
				,width: "60px"
				,color: beerMapData[i]['textColor']
				,backgroundColor: beerMapData[i]['color']
			}
			,disableAutoPan: true
			,pixelOffset: new google.maps.Size(-25, 0)
			,position: new google.maps.LatLng(beerMapData[i]['latitude'], beerMapData[i]['longitude'])
			,closeBoxURL: ""
			,isHidden: false
			,pane: "mapPane"
			,enableEventPropagation: true
		};
		var ibLabel = new InfoBox(myOptions);
		ibLabel.open(map);
	}

map.mapTypes.set('coordinate', coordinateMapType);
}

google.maps.event.addDomListener(window, 'load', initialize);

		</script>
	</head>
	<body>
		<div id="map-canvas"></div>
	</body>
</html>
	'''
	return return_string

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

def makeBeerMap(allBeerWeights, filename="beernodemap"):
	out=open("data/%s.html" % (filename),'w')
	out.write(generateGoogleJavascript(allBeerWeights))
	out.close()
	site,username,password = loadFTPCredentials()
	session = ftplib.FTP(site,username,password)
	file = open("data/%s.html" % (filename),'rb')                  # file to send
	session.storbinary("STOR beermap/%s.html" % (filename), file)     # send the file
	file.close()                                    # close file and FTP
	session.quit()
	print 'gmap uploaded'

def beer_dict_from_weights(names, weight_matrix):
    return [{names[i]:weight for i, weight in enumerate(line)} for line in weight_matrix.T]


# This is the function we actually call!!!
def makeAllBeerMaps(filename="beernodemap"):
	trainedWeights = np.load("tuned_12.npz")['W']
	trainingArray, bitMaskArray, filteredBeerNamesArray = createNDArray()
	allBeerWeights = beer_dict_from_weights(filteredBeerNamesArray, trainedWeights)
	makeBeerMap(allBeerWeights, filename=filename)


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
