import math
import operator
import numpy as np
import random

def shuffle_all(*args):
    idx = range(len(args[0]))
    random.shuffle(idx)
    for arg in args:
        assert(len(arg) == len(args[0]))
    return [[arg[i] for i in idx] for arg in args]

def loadBeerChooser(path='data'):
    
    # Get beer names
    beers={}
    for line in open(path+'/u.item'):
        (id,name)=line.split('\t')
        beers[id]=name.rstrip()
    
    # Load data
    userPrefs={}
    for line in open(path+'/u.data'):
        (user,beerid,rating)=line.split(',')
        userPrefs.setdefault(user,{})
        userPrefs[user][beers[beerid]]=float(rating)
    return userPrefs

def transformPrefs(userPrefs):
    itemPrefs={}
    for person in userPrefs:
        for beer in userPrefs[person]:
            itemPrefs.setdefault(beer,{})
            # Flip person and beer
            itemPrefs[beer][person]=userPrefs[person][beer]
    return itemPrefs

def createNDArray():
    userPrefs = loadBeerChooser()
    itemPrefs = transformPrefs(userPrefs)
    beersArray = sorted(itemPrefs.iteritems(), key=operator.itemgetter(0))
    usersArray = sorted(userPrefs.iteritems(), key=operator.itemgetter(0))
    
    # 2 dimensional array of zeros, each beer gets 1 zero in the vector
    trainingArray = np.zeros((len(usersArray), len(beersArray)))
    bitMaskArray = np.zeros((len(usersArray), len(beersArray)))

    filteredBeerNamesArray = []
    for k in range(0,len(beersArray)):
        # add beer name to names array in correct order
        filteredBeerNamesArray.append(beersArray[k][0])

     # Copy all of the ratings to the appropriate place in the vector
    for j in range(0,len(usersArray)):
        for k in range(0,len(beersArray)):
            if beersArray[k][0] in userPrefs[usersArray[j][0]]:
                rating = userPrefs[usersArray[j][0]][beersArray[k][0]]
                trainingArray[j][k] = rating
                bitMaskArray[j][k] = 1

    return trainingArray, bitMaskArray, filteredBeerNamesArray


    def createNDArrayOld():
    userPrefs = loadBeerChooser()
    itemPrefs = transformPrefs(userPrefs)
    beersArray = sorted(itemPrefs.iteritems(), key=operator.itemgetter(0))
    usersArray = sorted(userPrefs.iteritems(), key=operator.itemgetter(0))
    
    # 2 dimensional array of zeros, each beer gets 5 zeros in the vector, one for each rating level (1...5)
    trainingArray = np.zeros((len(usersArray), len(beersArray) * 5))
    bitMaskArray = np.zeros((len(usersArray), len(beersArray) * 5))

    filteredBeerNamesArray = []
    for k in range(0,len(beersArray)):
        # add beer name to names array in correct order
        filteredBeerNamesArray.append(beersArray[k][0])

     # Copy all of the ratings to the appropriate place in the vector
    for j in range(0,len(usersArray)):
        for k in range(0,len(beersArray)):
            if beersArray[k][0] in userPrefs[usersArray[j][0]]:
                rating = userPrefs[usersArray[j][0]][beersArray[k][0]]
                rating = (rating * 5) - 1
                trainingArray[j][(k * 5) + rating] = 1
                for r in range(0,5):
                    bitMaskArray[j][(k * 5) + r] = 1

    return trainingArray, bitMaskArray, filteredBeerNamesArray
¡










