from PIL import Image,ImageDraw,ImageFilter
from scipy import stats
import math
from math import sqrt
import copy
import operator
import pickle
import random
import ftplib
import numpy as np

def loadBeerChooser(path='data'):
    
    # Get beer names
    beers={}
    for line in open(path+'/u.item'):
        (id,name)=line.split('\t')
        beers[id]=name.rstrip()
    
    # Load data
    prefs={}
    for line in open(path+'/u.data'):
        (user,beerid,rating)=line.split(',')
        prefs.setdefault(user,{})
        prefs[user][beers[beerid]]=float(rating)
    return prefs

def loadTopPrefs():
    # Reload data from save point
    itemPrefs=pickle.load(open("pickle/itemprefs.p","rb"))
    controversy = loadControversy()
    topPrefs = createControversyTopPrefs(controversy,itemPrefs, 900)
    return topPrefs

def loadControversy(path='data'):
    controversy={}
    
    for line in open(path+'/beercontroversy.txt'):
        (beer,contro)=line.split('\t')
        controversy[beer.rstrip()]=contro
    return controversy

def createControversyTopPrefs(controversy,itemPrefs,numBeers):
    topBeers={}
    topPrefs={}
    sorted_controversy=sorted(controversy.iteritems(), key=operator.itemgetter(1))
    sorted_controversy.reverse()
    for x in range(numBeers):
        beer=sorted_controversy[x][0]
        topBeers[beer]=beer
    for topBeer in topBeers:
        topPrefs[topBeer]={}
        topPrefs[topBeer]=itemPrefs[topBeer]
    return topPrefs

def transformPrefs(itemPrefs):
    userPrefs={}
    for beer in itemPrefs:
        for person in itemPrefs[beer]:
            userPrefs.setdefault(person,{})
            # Flip person and beer
            userPrefs[person][beer]=itemPrefs[beer][person]
    return userPrefs

def createNDArray(topPrefs, numTesting = 100):
    userPrefs = transformPrefs(topPrefs)
    beersArray = sorted(topPrefs.iteritems(), key=operator.itemgetter(0))
    usersArray = sorted(userPrefs.iteritems(), key=operator.itemgetter(0))
    
    # pull out some subset of users to use as validation data
    random.shuffle(usersArray)
    
    # 2 dimensional array of ones
    trainingArray = np.ones((len(usersArray)-numTesting, len(beersArray)))
    
    # 2 dimensional array of 0.5's (meaningless "average" data)
    for x in np.nditer(trainingArray, op_flags=['readwrite']):
        x[...] = 0.5 * x
    
    
    trainingResults = trainingArray
    for j in range(0,len(usersArray)-numTesting):
        for k in range(0,len(beersArray)):
            if beersArray[k][0] in userPrefs[usersArray[j][0]]:
                trainingResults[j][k] = userPrefs[usersArray[j][0]][beersArray[k][0]]
                if random.random() < 0.5:
                    trainingArray[j][k] = trainingResults[j][k]
    
    testingArray = np.ones((numTesting, len(beersArray)))
    for x in np.nditer(testingArray, op_flags=['readwrite']):
        x[...] = 0.5 * x
    
    m = 0
    testingResults = testingArray
    for j in range(len(usersArray)-numTesting,len(usersArray)):
        # print "index"
        #print j
        #print "counter"
        #print m
        for k in range(0,len(beersArray)):
            if beersArray[k][0] in userPrefs[usersArray[j][0]]:
                testingResults[m][k] = userPrefs[usersArray[j][0]][beersArray[k][0]]
                if random.random() < 0.5:
                    testingArray[m][k] = testingResults[m][k]
        #        print testingArray[m][k]
        m += 1
    return trainingArray, trainingResults, testingArray, testingResults

def createNDArray(topPrefs, numTesting = 100):
    userPrefs = transformPrefs(topPrefs)
    beersArray = sorted(topPrefs.iteritems(), key=operator.itemgetter(0))
    usersArray= sorted(userPrefs.iteritems(), key=operator.itemgetter(0))
    random.shuffle(usersArray)
    trainingArray = np.ones((len(usersArray)-numTesting, len(beersArray)))
    for x in np.nditer(trainingArray, op_flags=['readwrite']):
        x[...] = 0.5 * x
    
    trainingResults = trainingArray
    for j in range(0,len(usersArray)-numTesting):
        for k in range(0,len(beersArray)):
            if beersArray[k][0] in userPrefs[usersArray[j][0]]:
                trainingResults[j][k] = userPrefs[usersArray[j][0]][beersArray[k][0]]
                if random.random() < 0.5:
                    trainingArray[j][k] = trainingResults[j][k]

    testingArray = np.ones((numTesting, len(beersArray)))
    for x in np.nditer(testingArray, op_flags=['readwrite']):
        x[...] = 0.5 * x

    m = 0
    testingResults = testingArray
    for j in range(len(usersArray)-numTesting,len(usersArray)):
        # print "index"
        #print j
        #print "counter"
        #print m
        for k in range(0,len(beersArray)):
            if beersArray[k][0] in userPrefs[usersArray[j][0]]:
                testingResults[m][k] = userPrefs[usersArray[j][0]][beersArray[k][0]]
                if random.random() < 0.5:
                    testingArray[m][k] = testingResults[m][k]
        #        print testingArray[m][k]
        m += 1
    return trainingArray, trainingResults, testingArray, testingResults

def loadBeerNetData():
    topPrefs = loadTopPrefs()
    trainingArray, trainingResults, testingArray, testingResults = createNDArray(topPrefs)
    trainingData = (trainingArray, trainingResults)
    testData = (testingArray, testingResults)
    return trainingData, testData

def getBeersArray():
    topPrefs = loadTopPrefs()
    beersArray = sorted(topPrefs.iteritems(), key=operator.itemgetter(0))
    return beersArray


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.
        
        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.
        
        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.
        
        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""
    tr_d, te_d = loadBeerNetData()
    training_inputs = [np.reshape(x, (900, 1)) for x in tr_d[0]]
    training_results = [np.reshape(x, (900, 1)) for x in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, (900, 1)) for x in te_d[0]]
    test_results = [np.reshape(x, (900, 1)) for x in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return (training_data, test_data)

def getInputWeightsForHiddenNodes(net, numInputs, numHidden):
    # weights between layer 0 (inputs layer ie beers) and layer 1 (hidden layer)
    

    # w[0][90] is the weight for the connection from beer 90 to hidden node 0
    # We are looking for the beers that are the most highly connected to node 0 to "describe" that node
    # Or, we are looking for a dictionary of the strength of the connection from each beer to each node?

    beersArray = getBeersArray()
    weight = {}
    
    for j in range(0,numHidden):
        weight.setdefault(j,{})
        for k in range(0,numInputs):
            weight[j][beersArray[k][0]] = net.weights[0][j][k]
    return weight

def printAllWeightsArray(weight, numHidden):
    weightArray = []
    for j in range(0,numHidden):
        print ' '
        print ' '
        print 'Hidden Node %d' % j
        print ' '
        w = getNodeWeightArray(weight[j])
        for x in range(0,10):
            print '\t\t%s' % (w[x][0])

def getNodeWeightArray(weightDict):
    weightArray = sorted(weightDict.iteritems(), key=operator.itemgetter(1))
    weightArray.reverse()
    return weightArray

"""
def getTopBeersForNode(weight,node,numTop):
    nodeBeers = {}
    weightArray = getWeightArray(weight[node])

    for x in range(numTop):
        weightArray[x][0]


"""

import beermapping as bmap

# Create and upload new beer map for hidden node
def beerMapNode(loc,beers,style,weights,node):
    backgroundColor = {}
    textColor = {}
    relevantBeers = {}
    
    weightArray = getNodeWeightArray(weights)
    maxColorValue = weightArray[0][1]
    
    for x in range(0,25):
        beer = weightArray[x][0]
        relevantBeers[beer] = beers[beer]
    
    weightArray.reverse()
    minColorValue = weightArray[0][1] * (-1)

    for x in range(0,25):
        beer = weightArray[x][0]
        relevantBeers[beer] = beers[beer]
    
    for beer in relevantBeers:
        if weights[beer] >= 0:
            r, g, b = bmap.rgbMix(weights[beer], maxColorValue, 0, red=71, green=245, blue=155)
        elif weights[beer] < 0:
            r, g, b = bmap.rgbMix(weights[beer]*(-1), minColorValue, 0, red=245, green=123, blue=71)
        backgroundColor[beer] = 'rgb('+str(r)+','+str(g)+','+str(b)+')'
        textColor[beer] = '#000000'
    
    bmap.makeBeerMap(relevantBeers, loc, style, textColor, backgroundColor, filename="beergmapnode%d" % (node), showCenter=False, center=None, highlights=None)

def beerMapAllNodes(weight,numNodes):
    loc,itemPrefs,realsim,realdist,abv,ibu,gravity,style = bmap.loadBeerData()
    
    beers = loadTopPrefs()
    
    for j in range(0,numNodes):
        beerMapNode(loc,beers,style,weight[j],j)
        print 'node %d done' % (j)

import network

def evaluateTest(Network, test_data):
    """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
    
    test_results = [(Network.feedforward(x), y) for (x,y) in test_data]
    
    errorterm = 0.000
    m = 0.000
    for j in range (0,900):
        for k in range (0,100):
            if test_results[k][1][j] != 0.5:
                m += 0.5
            if pow((test_results[k][0][j] - test_results[k][1][j]),2) >= 0.16:
                # guess was incorrect by more than one star
                errorterm += 1
    #m += 1
    if m > 0:
        errorterm = errorterm/m
        errorterm = 1-errorterm
    
    return errorterm










