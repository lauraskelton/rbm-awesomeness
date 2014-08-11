from PIL import Image,ImageDraw,ImageFilter
from scipy import stats
import math
from math import sqrt
import copy
import operator
import pickle
import random
import ftplib

# Call this function to add every single beer to the beer cluster map with no pauses
# (note: with lots of beers and user ratings (I have over 200,000) this takes a very long time to run)
def clusterAllBeers():
    
    loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist=createInitialBeerClusters()
    
    totalBeersToAdd=len(allItemPrefs)-len(seedPrefs)
    
    loc,seedPrefs=addMoreBeersToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,limit=totalBeersToAdd)
    
    return loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist


# Call this function to create the initial beer map we will add other beers to
def createInitialBeerClusters(path='data'):
    # Load beer names from file
    beers={}
    
    # example line from beernames.txt would be "967\tLagunitas IPA" (beerid,name)
    for line in open(path+'/beernames.txt'):
        (beerid,name)=line.split('\t')
        beers[beerid]=name.rstrip()
    
    # Load all beer ratings data from file
    prefs={}
    for line in open(path+'/beerratings.txt'):
        (user,beerid,rating)=line.split(',')
        prefs.setdefault(user,{})
        prefs[user][beers[beerid]]=float(rating)
    
    allItemPrefs={}
    for user in prefs:
        for beer in prefs[user]:
            allItemPrefs.setdefault(beer,{})
            
            # Flip beer and user
            allItemPrefs[beer][user]=prefs[user][beer]
    
    # Create a dictionary of each beer's similarity to every other beer
    realsim={}
    c=0
    for beer in allItemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(allItemPrefs))
        
        # Find the similarity of every beer to this one
        realsim[beer]={}
        for otherBeer in allItemPrefs:
            realsim[beer][otherBeer]=float(sim_pearson_correlation(allItemPrefs,beer,otherBeer))
    
    # Create a dictionary of each beer's distance from every other beer
    realdist={}
    c=0
    for beer in allItemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(allItemPrefs))
        
        # Find the distance from every beer to this one
        realdist[beer]={}
        for otherBeer in allItemPrefs:
            realdist[beer][otherBeer]=float(sim_pearson_distance(allItemPrefs,beer,otherBeer))
    
    # Find the most controversial beers (they are the most significant beers for creating a beer similarities diagram)
    controversy={}
    c=0
    for beer in allItemPrefs:
        controSum=0
        num=0
        for otherBeer in allItemPrefs:
            # The most controversial beers (highly predictive beers) are those that have the strongest overall correlation values
            # with other beers, either positively or negatively. So, if you love this beer, that tells me a lot about which other
            # beers you'll love and/or which beers you'll hate.
            controSum+=abs(realsim[beer][otherBeer])
            num+=1
        if num>0:
            controversy[beer]=controSum/num
        else:
            controversy[beer]=0
    
    # Gets a sorted array in descending order of all beers in order of how controversial they are
    # (which means, how much information their ratings give you about what a user's other beer preferences are)
    sorted_controversy=sorted(controversy.iteritems(), key=operator.itemgetter(1))
    sorted_controversy.reverse()
    
    # Get a dictionary of the 30 most controversial beers and their user ratings to seed the clustering algorithm
    # (The clustering error is lower if we start with a few beers and find their relative positions,
    # then add more beers to that initial diagram one by one)
    seedBeers={}
    seedPrefs={}
    
    # Get names of the 30 most controversial beers
    for x in range(30):
        beer=sorted_controversy[x][0]
        seedBeers[beer]=beer
    
    # Get ratings data for each of the 30 most controversial beers from the dictionary of all beer ratings data we loaded earlier
    for seedBeer in seedBeers:
        seedPrefs[seedBeer]={}
        seedPrefs[seedBeer]=allItemPrefs[seedBeer]
    
    # Randomly initialize the starting points of the beer diagram locations for the seed beers in 2D
    loc={}
    for beer in seedPrefs:
        loc[beer]=[random.random(),random.random()]
    
    # Move each beer incrementally to approximate the distances the beers should be from each other
    # based on their pearson distance calculation (how similar their ratings are)
    loc=decreaseClusterError(loc,realdist,seedPrefs,rate=0.01,tries=200000)
    
    # Move each beer one at a time to the weighted centerpoint of all of the most similar beers
    # Then run the clustering algorithm again so that the resulting distances on the graph
    # are close to the calculated pearson distances between each beer
    # Repeat this over all of the beers a few times in case a beer jumps to another part of the graph
    for m in range(3):
        for beer in seedPrefs:
            tmpPrefs=copy.deepcopy(seedPrefs)
            del tmpPrefs[beer]
            del loc[beer]
            loc=addBeerAndRecluster(tmpPrefs,allItemPrefs,realdist,realsim,loc,beer)
    
    # Create an image of the beer map and check to make sure that the clustering is working, and it makes sense
    draw2d(loc,seedPrefs,png="initial_beer_clusters.png",scale=2000)
    
    # Good idea to save since this takes awhile to calculate
    pickle.dump(loc, open("pickle/initial_loc.p","wb"))
    pickle.dump(seedPrefs, open("pickle/initial_seedprefs.p","wb"))
    pickle.dump(allItemPrefs, open("pickle/initial_allitemprefs.p","wb"))
    pickle.dump(sorted_controversy, open("pickle/initial_sorted_controversy.p","wb"))
    pickle.dump(realsim, open("pickle/initial_realsim.p","wb"))
    pickle.dump(realdist, open("pickle/initial_realdist.p","wb"))
    
    return loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist

# Keep calling this function to add new beers a few at a time
def addMoreBeersToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,limit=10):
    
    for x in range(limit):
        loc,seedPrefs=addBeerToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc)
        print "added beer %d / %d" % (x+1,limit)
    return loc,seedPrefs





# private methods--------------------------------------------------------------------------------->

def addBeerToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc):
    
    # Get the next index of the beer to add to the cluster
    # (This is the next most controversial beer on the list)
    newIndex=len(seedPrefs)
    
    if newIndex < len(allItemPrefs):
        beer=sorted_controversy[newIndex][0]
        
        addBeerAndRecluster(seedPrefs,allItemPrefs,realdist,realsim,loc,beer)
        
        # Draw an image of the current beer graph to see where this beer landed
        if newIndex<100:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=2000)
        elif newIndex<200:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=3000)
        elif newIndex<400:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=4000)
        elif newIndex<550:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=5000)
        elif newIndex<700:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=6000)
        elif newIndex<850:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=7000)
        else:
            draw2d(loc,seedPrefs,png="%d_beer_clusters.png" % (newIndex),scale=8000)
        
        # Save the data structures just in case
        pickle.dump(loc, open("pickle/%d_loc.p" % (newIndex),"wb"))
        pickle.dump(seedPrefs, open("pickle/%d_seedprefs.p" % (newIndex),"wb"))
    else:
        print 'all beers have been added'
    
    return loc,seedPrefs

def addBeerAndRecluster(seedPrefs,itemPrefs,realdist,realsim,loc,beer):
    
    # adjust only this beer's position to match closest other seed beers
    locSecondary={}
    locSecondary[beer]={}
    
    secondaryPrefs={}
    secondaryPrefs[beer]={}
    secondaryPrefs[beer]=itemPrefs[beer]
    
    locSecondary[beer]=[0.0,0.0]
    totalSim=0
    
    # Sorted array of most similar beers to this one
    simArray=sorted(realsim[beer].iteritems(), key=operator.itemgetter(1))
    simArray.reverse()
    for x in range(len(simArray)):
        # See if this is a seed beer
        simBeer=simArray[x][0]
        if simBeer==beer: continue
        if simBeer in seedPrefs:
            if realsim[beer][simBeer]>0:
                # weighted average of all similar beer locations for better starting point
                # instead of the random starting point, will decrease the overall error of the beer diagram
                locSecondary[beer][0]+=(loc[simBeer][0])*realsim[beer][simBeer]
                locSecondary[beer][1]+=(loc[simBeer][1])*realsim[beer][simBeer]
                
                totalSim+=realsim[beer][simBeer]
            else: break
    
    if totalSim>0:
        locSecondary[beer][0]=(locSecondary[beer][0])/totalSim
        locSecondary[beer][1]=(locSecondary[beer][1])/totalSim
    
    locSecondary=decreaseClusterErrorForThisBeer(locSecondary,realdist,secondaryPrefs,seedPrefs,loc,rate=0.01,tries=10000)
    
    seedPrefs[beer]={}
    seedPrefs[beer]=itemPrefs[beer]
    loc[beer]={}
    loc[beer]=[locSecondary[beer][0],locSecondary[beer][1]]
    
    loc=decreaseClusterError(loc,realdist,seedPrefs,rate=0.01,tries=500)
    
    return loc

# testclustering.decreaseClusterError(loc,newRealDist,seedPrefs,rate=0.01,tries=10000)

def decreaseClusterError(loc,realdist,itemPrefs,rate=0.01,tries=10000):
    
    fakedist={}
    grad={}
    for beer in itemPrefs:
        fakedist[beer]={}
        for otherBeer in itemPrefs:
            fakedist[beer][otherBeer]=[0.0]
    
    for m in range(0,tries):
        # Calculate the distance between each beer when they are at these coordinates ("fakedist")
        for beer in itemPrefs:
            for otherBeer in itemPrefs:
                # calculate current distance by square root of sum of the squares of the changes in x and y coordinates
                fakedist[beer][otherBeer]=sqrt(sum([pow(loc[beer][x]-loc[otherBeer][x],2) for x in range(len(loc[beer]))]))
        
        # Move points
        for beer in itemPrefs:
            grad[beer]=[0.0,0.0]
        
        totalerror=0
        num=0
        for beerK in itemPrefs:
            for beerJ in itemPrefs:
                if beerK==beerJ: continue
                if realdist[beerK][beerJ]==-1: continue
                
                # The errorterm is the difference between this current distance and what the distance should be, in ratio to the current distance
                if fakedist[beerK][beerJ]!=0:
                    errorterm=(fakedist[beerK][beerJ]-realdist[beerK][beerJ])/fakedist[beerK][beerJ]
                else:
                    errorterm=1
                
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                if fakedist[beerK][beerJ]!=0:
                    grad[beerK][0]+=(loc[beerJ][0]-loc[beerK][0])*errorterm
                    grad[beerK][1]+=(loc[beerJ][1]-loc[beerK][1])*errorterm
                
                # Keep track of the total error
                totalerror+=pow(errorterm,2)
                num+=1
        if num>0:
            totalerror = sqrt(totalerror)/num
        
        # Status updates for large datasets
        if (m+1)%500==0:
            print "%d / %d with error: %f" % (m+1,tries,totalerror)
        
        # Move each of the points by the learning rate times the gradient
        for beerK in itemPrefs:
            if num>0:
                loc[beerK][0]+=rate*(grad[beerK][0]/num)
                loc[beerK][1]+=rate*(grad[beerK][1]/num)
    return loc

def decreaseClusterErrorForThisBeer(loc,realdist,secondaryPrefs,seedPrefs,locSeed,rate=0.01,tries=50):
    
    fakedist={}
    grad={}
    for beer in secondaryPrefs:
        fakedist[beer]={}
        for otherBeer in seedPrefs:
            fakedist[beer][otherBeer]=[0.0]
    
    lasterror=None
    for m in range(0,tries):
        # Calculate the distance between each beer when they are at these coordinates ("fakedist")
        for beer in secondaryPrefs:
            for otherBeer in seedPrefs:
                # calculate current distance by square root of sum of the squares of the changes in x and y coordinates
                fakedist[beer][otherBeer]=sqrt(sum([pow(loc[beer][x]-locSeed[otherBeer][x],2) for x in range(len(loc[beer]))]))
        
        # Move points
        for beer in secondaryPrefs:
            grad[beer]=[0.0,0.0]
        
        totalerror=0
        num=0
        for beerK in secondaryPrefs:
            for beerJ in seedPrefs:
                if beerK==beerJ: continue
                if realdist[beerK][beerJ]==-1: continue
                
                # The errorterm is the difference between this current distance and what the distance should be, in ratio to the current distance
                if fakedist[beerK][beerJ]!=0:
                    errorterm=(fakedist[beerK][beerJ]-realdist[beerK][beerJ])/fakedist[beerK][beerJ]
                else:
                    errorterm=1
                
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                if fakedist[beerK][beerJ]!=0:
                    grad[beerK][0]+=(locSeed[beerJ][0]-loc[beerK][0])*errorterm
                    grad[beerK][1]+=(locSeed[beerJ][1]-loc[beerK][1])*errorterm
                
                # Keep track of the total error
                totalerror+=pow(errorterm,2)
                num+=1
        if num>0:
            totalerror = sqrt(totalerror)/num
        
        # Status updates for large datasets
        #if (m+1)%10000==0:
            #print "%d / %d with error: %f" % (m,tries,totalerror)
            #print "error: %f" % (totalerror)
        
        # Move each of the points by the learning rate times the gradient
        for beerK in secondaryPrefs:
            if num>0:
                loc[beerK][0]+=rate*(grad[beerK][0]/num)
                loc[beerK][1]+=rate*(grad[beerK][1]/num)
    
    return loc

def draw2d(loc,itemPrefs,png='l_beerclusters2d.png', scale=2500):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        draw.text((x,y),beer,(0,0,0))
    img.save('images/'+png,'PNG')

def sim_pearson_distance(itemPrefs,beer1name,beer2name):
    
    beer1Data = itemPrefs[beer1name]
    beer2Data = itemPrefs[beer2name]
    x = []
    y = []
    
    for user in beer1Data:
        if beer2Data.has_key(user):
            x.append(beer1Data[user])
            y.append(beer2Data[user])
    
    # Find the number of elements
    n=len(x)
    
    # If they have no ratings in common, return -1 (which we will ignore in our distance calculations later)
    if n<2: return -1
    
    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        # return -1 (which we will ignore in our distance calculations later)
        return -1
    # Greater correlation means smaller distance apart, and distances should be positive (pearson_correlation ranges from -1 to 1)
    return 1-pearson_correlation

def sim_pearson_correlation(itemPrefs,beer1name,beer2name):
    
    beer1Data = itemPrefs[beer1name]
    beer2Data = itemPrefs[beer2name]
    x = []
    y = []
    
    for user in beer1Data:
        if beer2Data.has_key(user):
            x.append(beer1Data[user])
            y.append(beer2Data[user])
    
    # Find the number of elements
    n=len(x)
    
    # If they have no ratings in common, return 0 (they are not correlated)
    if n<2: return 0
    
    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        # return 0 (they are not correlated)
        return 0
    
    return pearson_correlation

def sim_pearson_distance_orig(itemPrefs,beer1name,beer2name):
    
    beer1Data = itemPrefs[beer1name]
    beer2Data = itemPrefs[beer2name]
    x = []
    y = []
    
    for user in beer1Data:
        if beer2Data.has_key(user):
            x.append(beer1Data[user])
            y.append(beer2Data[user])
    
    # Find the number of elements
    n=len(x)
    
    # If they have no ratings in common, return -1 (which we will ignore in our distance calculations later)
    if n==0: return -1
    
    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        # return -1 (which we will ignore in our distance calculations later)
        return -1
    # Greater correlation means smaller distance apart, and distances should be positive (pearson_correlation ranges from -1 to 1)
    return 1-pearson_correlation

def sim_pearson_correlation_orig(itemPrefs,beer1name,beer2name):
    
    beer1Data = itemPrefs[beer1name]
    beer2Data = itemPrefs[beer2name]
    x = []
    y = []
    
    for user in beer1Data:
        if beer2Data.has_key(user):
            x.append(beer1Data[user])
            y.append(beer2Data[user])
    
    # Find the number of elements
    n=len(x)
    
    # If they have no ratings in common, return 0 (they are not correlated)
    if n==0: return 0
    
    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        # return 0 (they are not correlated)
        return 0
    
    return pearson_correlation

# TEMPORARY BUG TESTING -------------------------------------------------------------------------------------------------->

# Call this function to create the initial beer map we will add other beers to
def testingCreateInitialBeerClusters(path='data'):
    # Load beer names from file
    beers={}
    
    # example line from beernames.txt would be "967\tLagunitas IPA" (beerid,name)
    for line in open(path+'/beernames.txt'):
        (beerid,name)=line.split('\t')
        beers[beerid]=name.rstrip()
    
    # Load all beer ratings data from file
    prefs={}
    for line in open(path+'/beerratings.txt'):
        (user,beerid,rating)=line.split(',')
        prefs.setdefault(user,{})
        prefs[user][beers[beerid]]=float(rating)
    
    allItemPrefs={}
    for user in prefs:
        for beer in prefs[user]:
            allItemPrefs.setdefault(beer,{})
            
            # Flip beer and user
            allItemPrefs[beer][user]=prefs[user][beer]
    
    # Create a dictionary of each beer's similarity to every other beer
    realsim={}
    c=0
    for beer in allItemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(allItemPrefs))
        
        # Find the similarity of every beer to this one
        realsim[beer]={}
        for otherBeer in allItemPrefs:
            realsim[beer][otherBeer]=float(sim_pearson_correlation(allItemPrefs,beer,otherBeer))
    
    # Create a dictionary of each beer's distance from every other beer
    realdist={}
    c=0
    for beer in allItemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(allItemPrefs))
        
        # Find the distance from every beer to this one
        realdist[beer]={}
        for otherBeer in allItemPrefs:
            realdist[beer][otherBeer]=float(sim_pearson_distance(allItemPrefs,beer,otherBeer))
    
    # Find the most controversial beers (they are the most significant beers for creating a beer similarities diagram)
    controversy={}
    c=0
    for beer in allItemPrefs:
        controSum=0
        num=0
        for otherBeer in allItemPrefs:
            # The most controversial beers (highly predictive beers) are those that have the strongest overall correlation values
            # with other beers, either positively or negatively. So, if you love this beer, that tells me a lot about which other
            # beers you'll love and/or which beers you'll hate.
            controSum+=abs(realsim[beer][otherBeer])
            num+=1
        if num>0:
            controversy[beer]=controSum/num
        else:
            controversy[beer]=0
    
    # Gets a sorted array in descending order of all beers in order of how controversial they are
    # (which means, how much information their ratings give you about what a user's other beer preferences are)
    sorted_controversy=sorted(controversy.iteritems(), key=operator.itemgetter(1))
    sorted_controversy.reverse()
    
    # Get a dictionary of the 30 most controversial beers and their user ratings to seed the clustering algorithm
    # (The clustering error is lower if we start with a few beers and find their relative positions,
    # then add more beers to that initial diagram one by one)
    seedBeers={}
    seedPrefs={}
    
    # Get names of the 30 most controversial beers
    for x in range(30):
        beer=sorted_controversy[x][0]
        seedBeers[beer]=beer
    
    # Get ratings data for each of the 30 most controversial beers from the dictionary of all beer ratings data we loaded earlier
    for seedBeer in seedBeers:
        seedPrefs[seedBeer]={}
        seedPrefs[seedBeer]=allItemPrefs[seedBeer]
    
    # Randomly initialize the starting points of the beer diagram locations for the seed beers in 2D
    loc={}
    for beer in seedPrefs:
        loc[beer]=[random.random(),random.random()]
    
    # Move each beer incrementally to approximate the distances the beers should be from each other
    # based on their pearson distance calculation (how similar their ratings are)
    loc=decreaseClusterError(loc,realdist,seedPrefs,rate=0.01,tries=200000)
    
    return loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist

def testingMoreCreateInitialBeerClusters(loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist):
    
    # Move each beer one at a time to the weighted centerpoint of all of the most similar beers
    # Then run the clustering algorithm again so that the resulting distances on the graph
    # are close to the calculated pearson distances between each beer
    # Repeat this over all of the beers a few times in case a beer jumps to another part of the graph
    for m in range(3):
        for beer in seedPrefs:
            tmpPrefs=copy.deepcopy(seedPrefs)
            del tmpPrefs[beer]
            del loc[beer]
            loc=addBeerAndRecluster(tmpPrefs,allItemPrefs,realdist,realsim,loc,beer)
    
    # Create an image of the beer map and check to make sure that the clustering is working, and it makes sense
    draw2d(loc,seedPrefs,png="initial_beer_clusters.png",scale=2000)
    
    # Good idea to save since this takes awhile to calculate
    pickle.dump(loc, open("pickle/initial_loc.p","wb"))
    pickle.dump(seedPrefs, open("pickle/initial_seedprefs.p","wb"))
    pickle.dump(allItemPrefs, open("pickle/initial_allitemprefs.p","wb"))
    pickle.dump(sorted_controversy, open("pickle/initial_sorted_controversy.p","wb"))
    pickle.dump(realsim, open("pickle/initial_realsim.p","wb"))
    pickle.dump(realdist, open("pickle/initial_realdist.p","wb"))
    
    return loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist

# testclustering.testingRejiggerClusterMap(loc,seedPrefs,allItemPrefs,sorted_controversy,newRealSim,newRealDist,style,abv,ibu)

def testingRejiggerClusterMap(loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist,style,abv,ibu,start=0):
    
    # Move each beer one at a time to the weighted centerpoint of all of the most similar beers
    # Then run the clustering algorithm again so that the resulting distances on the graph
    # are close to the calculated pearson distances between each beer
    # Repeat this over all of the beers a few times in case a beer jumps to another part of the graph
    c=start
    for x in range(len(seedPrefs)-start):
        tmpPrefs=copy.deepcopy(seedPrefs)
        beer=sorted_controversy[x+start][0]
        del tmpPrefs[beer]
        del loc[beer]
        loc=addBeerAndRecluster(tmpPrefs,allItemPrefs,realdist,realsim,loc,beer)
    
        # Create an image of the beer map and check to make sure that the clustering is working, and it makes sense
        #draw2d(loc,seedPrefs,png=("rejigger3/rejigger_%d_clusters_" % c)+str(beer)+".png",scale=3000)
        saveAndPushGoogleBeerMapABVIBU(loc,seedPrefs,style,abv,ibu)
        
        # Good idea to save since this takes awhile to calculate
        pickle.dump(loc, open(("pickle/rejigger3/rejigger_%d_loc.p" % c),"wb"))
        pickle.dump(seedPrefs, open(("pickle/rejigger3/rejigger_%d_seedprefs.p" % c),"wb"))
        print ("rejiggered beer %d / %d - " % (c,len(seedPrefs)))+str(beer)
        c+=1

    return loc,seedPrefs

def createNewSimDist(allItemPrefs):
    # Create a dictionary of each beer's similarity to every other beer
    realsim={}
    c=0
    for beer in allItemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(allItemPrefs))
        
        # Find the similarity of every beer to this one
        realsim[beer]={}
        for otherBeer in allItemPrefs:
            realsim[beer][otherBeer]=float(sim_pearson_correlation(allItemPrefs,beer,otherBeer))
    
    # Create a dictionary of each beer's distance from every other beer
    realdist={}
    c=0
    for beer in allItemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(allItemPrefs))
        
        # Find the distance from every beer to this one
        realdist[beer]={}
        for otherBeer in allItemPrefs:
            realdist[beer][otherBeer]=float(sim_pearson_distance(allItemPrefs,beer,otherBeer))
    return realsim,realdist

def testingSaveNewSim(newRealSim,newRealDist):
    # Good idea to save since this takes awhile to calculate
    pickle.dump(newRealSim, open("pickle/testing_newRealSim.p","wb"))
    pickle.dump(newRealDist, open("pickle/testing_newRealDist.p","wb"))

def testingRejiggerBeer(beer,loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist,style,abv,ibu):
    
    # Move each beer one at a time to the weighted centerpoint of all of the most similar beers
    # Then run the clustering algorithm again so that the resulting distances on the graph
    # are close to the calculated pearson distances between each beer
    # Repeat this over all of the beers a few times in case a beer jumps to another part of the graph
    tmpPrefs=copy.deepcopy(seedPrefs)
    del tmpPrefs[beer]
    del loc[beer]
    loc=addBeerAndRecluster(tmpPrefs,allItemPrefs,realdist,realsim,loc,beer)
    
    # Create an image of the beer map and check to make sure that the clustering is working, and it makes sense
    draw2d(loc,seedPrefs,png=("rejigger2/rejigger_clusters_")+str(beer)+".png",scale=3000)
    saveAndPushGoogleBeerMapABVIBU(loc,seedPrefs,style,abv,ibu)
    
    # Good idea to save since this takes awhile to calculate
    pickle.dump(loc, open(("pickle/rejigger2/rejigger_loc_"+str(beer)+".p"),"wb"))
    pickle.dump(seedPrefs, open(("pickle/rejigger2/rejigger_seedprefs_"+str(beer)+".p"),"wb"))
    print ("rejiggered beer "+str(beer))
    
    return loc,seedPrefs

def testingSaveAndDraw(loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist):
    # Create an image of the beer map and check to make sure that the clustering is working, and it makes sense
    draw2d(loc,seedPrefs,png="testing_beer_clusters.png",scale=2000)
    
    # Good idea to save since this takes awhile to calculate
    pickle.dump(loc, open("pickle/testing_loc.p","wb"))
    pickle.dump(seedPrefs, open("pickle/testing_seedprefs.p","wb"))
    pickle.dump(allItemPrefs, open("pickle/testing_allitemprefs.p","wb"))
    pickle.dump(sorted_controversy, open("pickle/testing_sorted_controversy.p","wb"))
    pickle.dump(realsim, open("pickle/testing_realsim.p","wb"))
    pickle.dump(realdist, open("pickle/testing_realdist.p","wb"))

def testingLoadFromSave():
    
    # Reload data from save point to try again
    loc=pickle.load(open("pickle/727_loc.p","rb"))
    seedPrefs=pickle.load(open("pickle/727_seedprefs.p","rb"))
    #allItemPrefs=pickle.load(open("pickle/testing_allitemprefs.p","rb"))
    #sorted_controversy=pickle.load(open("pickle/testing_sorted_controversy.p","rb"))
    return loc,seedPrefs#,allItemPrefs,sorted_controversy

def loadABV(path='data'):
    # example line from beerabv.txt would be "Lagunitas IPA\t5.7" (beer,abv)
    abvData={}
    for line in open(path+'/beerabv.txt'):
        (name,abv)=line.split('\t')
        beer=name.rstrip()
        abvData[beer]=float(abv)
    return abvData

def loadIBU(path='data'):
    # example line from beeribu.txt would be "Lagunitas IPA\t46" (beer,ibu)
    ibuData={}
    for line in open(path+'/beeribu.txt'):
        (name,ibu)=line.split('\t')
        beer=name.rstrip()
        if ibu.rstrip() == 'NULL':
            ibuData[beer]=float(0)
        else:
            ibuData[beer]=float(ibu)
    return ibuData

def loadGravity(path='data'):
    # example line from beergravity.txt would be "Lagunitas IPA\t1.014" (beer,gravity)
    gravityData={}
    for line in open(path+'/beergravity.txt'):
        (name,gravity)=line.split('\t')
        beer=name.rstrip()
        if gravity.rstrip() == 'NULL':
            gravityData[beer]=float(0)
        else:
            gravityData[beer]=float(gravity)
    return gravityData

def loadControversy(path='data'):
    # example line from beergravity.txt would be "Lagunitas IPA\t1.014" (beer,gravity)
    controversyData={}
    for line in open(path+'/beercontroversy.txt'):
        (name,controversy)=line.split('\t')
        beer=name.rstrip()
        if controversy.rstrip() == 'NULL':
            controversyData[beer]=float(0)
        else:
            controversyData[beer]=float(controversy)
    return controversyData

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

def loadColor(path='data'):
    # example line from beercolor.txt would be "Lagunitas IPA\t11" (beer,color)
    colorData={}
    for line in open(path+'/beercolor.txt'):
        (name,color)=line.split('\t')
        beer=name.rstrip()
        if color.rstrip() == 'NULL':
            colorData[beer]=float(0)
        else:
            colorData[beer]=float(color)
    return colorData

def loadBittersweet(abv,gravity,ibu,seedPrefs):
    ogData={}
    for beer in seedPrefs:
        if abv[beer] != 0 and gravity[beer] != 0:
            ogData[beer]=((abv[beer]/131)+gravity[beer]-1)*1000
        else:
            ogData[beer]=0
    bittersweetData={}
    for beer in seedPrefs:
        if ibu[beer] != 0 and ogData[beer] != 0:
            print beer
            print "ibu: %f" % (ibu[beer])
            print "ogData: %f" % (ogData[beer])
            print "gravity: %f" % (gravity[beer])
            print "abv: %f" % (abv[beer])
            bittersweetData[beer]= ibu[beer]/ogData[beer]
        else:
            bittersweetData[beer]=0
    return bittersweetData

def loadSweet(bittersweet,gravity,itemPrefs):
    sweetData={}
    for beer in itemPrefs:
        bitterScale=0
        if bittersweet[beer] < 1 and bittersweet[beer]>0.25:
            bitterScale=float(1-((bittersweet[beer]-0.25)/0.75))
        elif bittersweet[beer] <= 0.25:
            bitterScale=1
        gravityScale=1
        if gravity[beer] < 1.017 and gravity[beer] > 1.010:
            gravityScale=float((gravity[beer]-1.010)/0.007)
        elif gravity[beer] <= 1.010:
            gravityScale=0
        print beer
        print "bitterScale: %f" % (bitterScale)
        print "gravityScale: %f" % (gravityScale)
        sweetData[beer]=bitterScale*gravityScale
        print "sweet: %f" % (sweetData[beer])
    return sweetData

def loadIndex(sorted_controversy,allItemPrefs):
    index={}
    for x in range(len(sorted_controversy)):
        beer=sorted_controversy[x][0]
        index[beer]=str(x)
    return index

def loadSweetNew(ibu,gravity,itemPrefs):
    sweetData={}
    for beer in itemPrefs:
        ibuScale=0
        if ibu[beer] < 80 and ibu[beer]>10:
            ibuScale=float(1-((ibu[beer]-10)/70))
        elif ibu[beer] <= 10:
            ibuScale=1
        gravityScale=1
        if gravity[beer] < 1.017 and gravity[beer] > 1.010:
            gravityScale=float((gravity[beer]-1.010)/0.007)
        elif gravity[beer] <= 1.010:
            gravityScale=0
        print beer
        print "ibuScale: %f" % (ibuScale)
        print "gravityScale: %f" % (gravityScale)
        sweetData[beer]=ibuScale*gravityScale
        print "sweet: %f" % (sweetData[beer])
    return sweetData

def drawWithBittersweet(loc,itemPrefs,bittersweet,png='bittersweet_beer_map_red.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        lightness=255
        if bittersweet[beer] < 1 and bittersweet[beer]>0.25:
            lightness=int(((bittersweet[beer]-0.25)/0.75)*255)
            darkness=int((1-((bittersweet[beer]-0.25)/0.75))*255)
        elif bittersweet[beer] <= 0.25:
            darkness=255
            lightness=0
        if bittersweet[beer] != 0:
            draw.text((x,y),beer,(lightness,darkness,darkness))
    img.save('images/'+png,'PNG')

def drawWithSweet(loc,itemPrefs,sweet,png='sweet_beer_map_red.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        lightness=255
        if sweet[beer] < 0.9 and sweet[beer] > 0.1:
            lightness=int(((sweet[beer]-0.1)/0.8)*255)
            darkness=int((1-((sweet[beer]-0.1)/0.8))*255)
        elif sweet[beer] <= 0.1:
            darkness=255
            lightness=0
        if sweet[beer] != 0:
            draw.text((x,y),beer,(lightness,darkness,darkness))
    img.save('images/'+png,'PNG')

def drawWithABVSweet(loc,seedPrefs,abv,gravity,png='abv_sweet_beer_map.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    highSweet={}
    for beer in seedPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        
        highSweet[beer]=0
        if abv[beer] != 0:
            if gravity[beer] != 0:
                highSweet[beer]=float(gravity[beer]/abv[beer])
        else:
            highSweet[beer]=0
        
        r=126
        g=200
        b=252
        if highSweet[beer] < 0.254 and highSweet[beer] > 0.101:
            r=int(((0.153-(highSweet[beer]-0.101))/0.153)*(255-r)+r)
            g=int(((0.153-(highSweet[beer]-0.101))/0.153)*(255-g)+g)
            b=int(((0.153-(highSweet[beer]-0.101))/0.153)*(255-b)+b)
        elif highSweet[beer] <= 0.101:
            r=255
            g=255
            b=255
        if highSweet[beer] != 0:
            if highSweet[beer] != 0:
                draw.text((x,y),beer+': '+str(abv[beer]),(r,g,b))
    img.save('images/'+png,'PNG')

def drawWithIBU(loc,itemPrefs,ibu,png='ibu_beer_map_red.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        lightness=255
        if ibu[beer] < 80 and ibu[beer]>10:
            lightness=int(((ibu[beer]-10)/70)*255)
            darkness=int((1-((ibu[beer]-10)/70))*255)
        elif ibu[beer] <= 10:
            darkness=255
            lightness=0
        if ibu[beer] != 0:
            draw.text((x,y),beer,(lightness,darkness,darkness))
    img.save('images/'+png,'PNG')

def drawWithABV(loc,itemPrefs,abv,png='abv_beer_map_red.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        lightness=255
        if abv[beer] < 9 and abv[beer] > 3:
            lightness=int(((abv[beer]-3)/6)*255)
            darkness=int((1-((abv[beer]-3)/6))*255)
        elif abv[beer] <= 3:
            darkness=255
            lightness=0
        if abv[beer] != 0:
            draw.text((x,y),beer,(lightness,darkness,darkness))
    img.save('images/'+png,'PNG')

def drawWithABVCircles(loc,itemPrefs,abv,png='abv_beer_map_image.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    r=100
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.7)*scale/2
        y=((loc[beer][1]+0.5)*0.7)*scale/2
        darkness=0
        lightness=255
        if abv[beer] < 9 and abv[beer] > 3:
            lightness=int(((abv[beer]-3)/6)*255)
            darkness=int((1-((abv[beer]-3)/6))*255)
        elif abv[beer] <= 3:
            darkness=255
            lightness=0
        if abv[beer] != 0:
            #draw.text((x,y),beer,(lightness,darkness,darkness))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(lightness,darkness,darkness))
    img=img.filter(ImageFilter.GaussianBlur(radius=100))
    img.save('images/'+png,'PNG')

def drawWithColor(loc,itemPrefs,color,png='color_beer_map.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        hexColor="#050B0A"
        if color[beer]==29:
            hexColor="#100B0A"
        elif color[beer]==28:
            hexColor="#120D0C"
        elif color[beer]==27:
            hexColor="#16100F"
        elif color[beer]==26:
            hexColor="#19100F"
        elif color[beer]==25:
            hexColor="#231716"
        elif color[beer]==24:
            hexColor="#261716"
        elif color[beer]==23:
            hexColor="#361F1B"
        elif color[beer]==22:
            hexColor="#4A2727"
        elif color[beer]==21:
            hexColor="#4E2A0C"
        elif color[beer]==20:
            hexColor="#5D341A"
        elif color[beer]==19:
            hexColor="#6B3A1E"
        elif color[beer]==18:
            hexColor="#7C452D"
        elif color[beer]==17:
            hexColor="#8D4C32"
        elif color[beer]==16:
            hexColor="#985336"
        elif color[beer]==15:
            hexColor="#A85839"
        elif color[beer]==14:
            hexColor="#B26033"
        elif color[beer]==13:
            hexColor="#BC6733"
        elif color[beer]==12:
            hexColor="#BF7138"
        elif color[beer]==11:
            hexColor="#C17A37"
        elif color[beer]==10:
            hexColor="#BE823A"
        elif color[beer]==9:
            hexColor="#BE8C3A"
        elif color[beer]==8:
            hexColor="#C1963C"
        elif color[beer]==7:
            hexColor="#CDAA37"
        elif color[beer]==6:
            hexColor="#D5BC26"
        elif color[beer]==5:
            hexColor="#E0D01B"
        elif color[beer]==4:
            hexColor="#EAE615"
        elif color[beer] <= 3:
            hexColor="#F6F513"
        if color[beer] != 0:
            draw.text((x,y),beer,hexColor)
    img.save('images/'+png,'PNG')

def drawWithGravity(loc,itemPrefs,gravity,png='gravity_beer_map_red.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        lightness=255
        if gravity[beer] < 1.017 and gravity[beer] > 1.010:
            lightness=int(((gravity[beer]-1.010)/0.007)*255)
            darkness=int((1-((gravity[beer]-1.010)/0.007))*255)
        elif gravity[beer] <= 1.010:
            darkness=255
            lightness=0
        if gravity[beer] != 0:
            draw.text((x,y),beer,(lightness,darkness,darkness))
    img.save('images/'+png,'PNG')

def drawWithABVBW(loc,itemPrefs,abv,png='abv_beer_map.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        if abv[beer] < 12:
            darkness=int((1-(abv[beer]/12))*255)
        draw.text((x,y),beer,(darkness,darkness,darkness))
    img.save('images/'+png,'PNG')

def drawWithIBUBW(loc,itemPrefs,ibu,png='ibu_beer_map.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        darkness=0
        if ibu[beer] < 100:
            darkness=int((1-(ibu[beer]/100))*255)
        draw.text((x,y),beer,(darkness,darkness,darkness))
    img.save('images/'+png,'PNG')

def saveKML(loc,seedPrefs):
    out=open('data/beerkml.kml','w')
    out.write('<?xml version="1.0" encoding="UTF-8"?>')
    out.write('\n<kml xmlns="http://www.opengis.net/kml/2.2">')
    out.write('\n\t<Document>')
    out.write('\n\t\t<name>Beer Map</name>')
    out.write('\n\t\t<open>1</open>')
    out.write('\n\t\t<description>Map of Beer Similarities</description>')
    out.write('\n\t\t<Style id="redPaddle">')
    out.write('\n\t\t\t<IconStyle>')
    out.write('\n\t\t\t\t<Icon>')
    out.write('\n\t\t\t\t\t<href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href>')
    out.write('\n\t\t\t\t</Icon>')
    out.write('\n\t\t\t</IconStyle>')
    out.write('\n\t\t</Style>')
    for beer in seedPrefs:
        out.write('\n\t\t<Placemark>')
        out.write('\n\t\t\t<name>'+str(beer)+'</name>')
        out.write('\n\t\t\t<styleUrl>#redPaddle</styleUrl>')
        out.write('\n\t\t\t<open>1</open>')
        out.write('\n\t\t\t<Point>')
        out.write('\n\t\t\t\t<coordinates>')
        out.write(str(float((loc[beer][0])*15))+','+str(float((loc[beer][1])*(-15)))+',0')
        out.write('</coordinates>')
        out.write('\n\t\t\t</Point>')
        out.write('\n\t\t</Placemark>')
    out.write('\n\t</Document>')
    out.write('\n</kml>')

def saveKMLStyle(loc,seedPrefs,style):
    out=open('data/beerkml.kml','w')
    out.write('<?xml version="1.0" encoding="UTF-8"?>')
    out.write('\n<kml xmlns="http://www.opengis.net/kml/2.2">')
    out.write('\n\t<Document>')
    out.write('\n\t\t<name>Beer Map</name>')
    out.write('\n\t\t<open>1</open>')
    out.write('\n\t\t<description>Map of Beer Similarities</description>')
    out.write('\n\t\t<Style id="redPaddle">')
    out.write('\n\t\t\t<IconStyle>')
    out.write('\n\t\t\t\t<Icon>')
    out.write('\n\t\t\t\t\t<href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href>')
    out.write('\n\t\t\t\t</Icon>')
    out.write('\n\t\t\t</IconStyle>')
    out.write('\n\t\t</Style>')
    for beer in seedPrefs:
        out.write('\n\t\t<Placemark>')
        out.write('\n\t\t\t<name>'+str(beer)+'</name>')
        out.write('\n\t\t\t<description>'+str(style[beer].replace("&","&amp;"))+'</description>')
        out.write('\n\t\t\t<styleUrl>#redPaddle</styleUrl>')
        out.write('\n\t\t\t<open>1</open>')
        out.write('\n\t\t\t<Point>')
        out.write('\n\t\t\t\t<coordinates>')
        out.write(str(float((loc[beer][0])*15))+','+str(float((loc[beer][1])*(-15)))+',0')
        out.write('</coordinates>')
        out.write('\n\t\t\t</Point>')
        out.write('\n\t\t</Placemark>')
    out.write('\n\t</Document>')
    out.write('\n</kml>')

def saveHeatmapABV(loc,seedPrefs,abv):
    out=open('data/beerheat.txt','w')
    out.write('var abvHeatMapData = [\n')
    for beer in seedPrefs:
        if abv[beer]>7:
            out.write('{location: new google.maps.LatLng('+str(float((loc[beer][1])*(-15)))+', '+str(float((loc[beer][0])*(15)))+'), weight: '+str(abv[beer])+'}')
            out.write(',')
            out.write('\n')
    out.write('];')

def saveJSStyle(loc,seedPrefs,style):
    out=open('data/beermapdata.txt','w')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
        out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
        out.write('title: "'+str(beer)+'", ')
        out.write('description: "'+str(style[beer].replace("&","and"))+'"},')
        out.write('\n')
    out.write('];')

def saveD3(loc,seedPrefs,style):
    out=open('data/beerd3data.txt','w')
    out.write('var beerData = [\n')
    for beer in seedPrefs:
        out.write('\t{"cx": '+str(float((loc[beer][0]+0.5)*0.8*2000/2))+', ')
        out.write('"cy": '+str(float((loc[beer][1]+0.5)*0.8*2000/2))+', ')
        out.write('"radius": 5, ')
        out.write('"font-size": 10, ')
        out.write('"title": "'+str(beer)+'", ')
        out.write('"description": "('+str(style[beer].replace("&","and"))+')"},')
        out.write('\n')
    out.write('];')

def loadFTPCredentials():
    site = ''
    username = ''
    password = ''
    for line in open('data/ftp.txt'):
        (asite,ausername,apassword)=line.split('\t')
        site=asite
        username=ausername
        password=apassword
    return site, username, password

def saveAndPushD3(loc,seedPrefs,style,scale=2000):
    out=open('data/beerd3map.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<script type="text/javascript" src="http://mbostock.github.com/d3/d3.js"></script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="viz"></div>\n')
    out.write('\t\t<script type="text/javascript">\n')
    out.write('\t\t\t//Beer Data Set\n')
    out.write('\t\t\tvar beerData = [\n')
    for beer in seedPrefs:
        out.write('\t\t\t\t{"cx": '+str(float((loc[beer][0]+0.5)*0.8*scale/2))+', ')
        out.write('"cy": '+str(float((loc[beer][1]+0.5)*0.8*scale/2))+', ')
        out.write('"radius": 5, ')
        out.write('"font-size": 10, ')
        out.write('"title": "'+str(beer)+'", ')
        out.write('"description": "('+str(style[beer].replace("&","and"))+')"},')
        out.write('\n')
    out.write('\t\t\t];\n')
    out.write('\t\t\t//Create the SVG Viewport\n')
    out.write('\t\t\tvar svgContainer = d3.select("body").append("svg")\n')
    out.write('\t\t\t\t.attr("width",'+str(scale)+')\n')
    out.write('\t\t\t\t.attr("height",'+str(scale)+');\n')
    out.write('\n')
    out.write('\t\t\tvar foreignObject = svgContainer.selectAll("foreignObject")\n')
    out.write('\t\t\t\t.data(beerData)\n')
    out.write('\t\t\t\t.enter()\n')
    out.write('\t\t\t\t.append("foreignObject");\n')
    out.write('\n')
    out.write('\t\t\tvar foreignObjectLabels = foreignObject\n')
    out.write('\t\t\t\t.attr("x", function(d) { return d.cy; })\n')
    out.write('\t\t\t\t.attr("y", function(d) { return '+str(scale)+'-d.cx; })\n')
    out.write('\t\t\t\t.attr({width: 80, height: 40})\n')
    out.write('\t\t\t\t.append("xhtml:body")\n')
    out.write('\t\t\t\t.append("xhtml:div")\n')
    out.write('\t\t\t\t.style({\n')
    out.write('\t\t\t\t\t"font-size": "6px",\n')
    out.write('\t\t\t\t\t"background-color": "white",\n')
    out.write('\t\t\t\t\t"border": "1px solid black",\n')
    out.write('\t\t\t\t\t"text-align": "center"\n')
    out.write('\t\t\t\t\t})\n')
    out.write('\t\t\t\t.html(function (d) { return d.title+"<br/>"+d.description; })\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    out.close()

    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beerd3map.html','rb')                  # file to send
    session.storbinary('STOR beermap/beerd3map.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'd3 uploaded'

def drawWithABVBlue(loc,seedPrefs,abv,png='abv_beer_map_color.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in seedPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        r=126
        g=208
        b=252
        if abv[beer] < 10 and abv[beer] > 4:
            r=int(((6-(abv[beer]-4))/6)*(255-r)+r)
            g=int(((6-(abv[beer]-4))/6)*(255-g)+g)
            b=int(((6-(abv[beer]-4))/6)*(255-b)+b)
        elif abv[beer] <= 4:
            r=255
            g=255
            b=255
        if abv[beer] != 0:
            draw.text((x,y),beer+': '+str(abv[beer]),(r,g,b))
    img.save('images/'+png,'PNG')

def drawWithIBUBlue(loc,seedPrefs,ibu,png='ibu_beer_map_color.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in seedPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        r=126
        g=252
        b=145
        if ibu[beer] < 80 and ibu[beer] > 10:
            r=int(((70-(ibu[beer]-10))/70)*(255-r)+r)
            g=int(((70-(ibu[beer]-10))/70)*(255-g)+g)
            b=int(((70-(ibu[beer]-10))/70)*(255-b)+b)
        elif ibu[beer] <= 10:
            r=255
            g=255
            b=255
        if ibu[beer] != 0:
            draw.text((x,y),beer+': '+str(ibu[beer]),(r,g,b))
    img.save('images/'+png,'PNG')

def drawWithABVIBUBlue(loc,seedPrefs,abv,ibu,png='abv_ibu_beer_map_color.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in seedPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        rA=126
        gA=200
        bA=252
        if abv[beer] < 10 and abv[beer] > 4:
            rA=int(((6-(abv[beer]-4))/6)*(255-rA)+rA)
            gA=int(((6-(abv[beer]-4))/6)*(255-gA)+gA)
            bA=int(((6-(abv[beer]-4))/6)*(255-bA)+bA)
        elif abv[beer] <= 4:
            rA=255
            gA=255
            bA=255
        rI=231
        gI=252
        bI=126
        if ibu[beer] < 80 and ibu[beer] > 10:
            rI=int(((70-(ibu[beer]-10))/70)*(255-rI)+rI)
            gI=int(((70-(ibu[beer]-10))/70)*(255-gI)+gI)
            bI=int(((70-(ibu[beer]-10))/70)*(255-bI)+bI)
        elif ibu[beer] <= 10:
            rI=255
            gI=255
            bI=255
        r=(rA+rI)/2
        g=(gA+gI)/2
        b=(bA+bI)/2
        if abv[beer] != 0:
            if ibu[beer] != 0:
                draw.text((x,y),beer+': '+str(abv[beer])+', '+str(abv[beer])+'IBU',(r,g,b))
    img.save('images/'+png,'PNG')

def drawWithABVIBUSweetBlue(loc,seedPrefs,abv,ibu,sweet,png='abv_ibu_sweet_beer_map_color.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in seedPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        # abv
        rA=126
        gA=200
        bA=252
        if abv[beer] < 10 and abv[beer] > 4:
            rA=int(((6-(abv[beer]-4))/6)*(255-rA)+rA)
            gA=int(((6-(abv[beer]-4))/6)*(255-gA)+gA)
            bA=int(((6-(abv[beer]-4))/6)*(255-bA)+bA)
        elif abv[beer] <= 4:
            rA=255
            gA=255
            bA=255
        # ibu
        rI=231
        gI=252
        bI=126
        if ibu[beer] < 80 and ibu[beer] > 10:
            rI=int(((70-(ibu[beer]-10))/70)*(255-rI)+rI)
            gI=int(((70-(ibu[beer]-10))/70)*(255-gI)+gI)
            bI=int(((70-(ibu[beer]-10))/70)*(255-bI)+bI)
        elif ibu[beer] <= 10:
            rI=255
            gI=255
            bI=255
        # sweet
        rS=218
        gS=126
        bS=252
        if sweet[beer] < 0.9 and sweet[beer] > 0.1:
            rS=int(((0.8-(sweet[beer]-0.1))/0.8)*(255-rS)+rS)
            gS=int(((0.8-(sweet[beer]-0.1))/0.8)*(255-gS)+gS)
            bS=int(((0.8-(sweet[beer]-0.1))/0.8)*(255-bS)+bS)
        elif sweet[beer] <= 0.1:
            rS=255
            gS=255
            bS=255
        # average color
        if abv[beer] != 0:
            if ibu[beer] != 0:
                if sweet[beer] != 0:
                    r=(rA+rI+rS)/3
                    g=(gA+gI+gS)/3
                    b=(bA+bI+bS)/3
                else:
                    r=(rA+rI)/2
                    g=(gA+gI)/2
                    b=(bA+bI)/2
            else:
                if sweet[beer] != 0:
                    r=(rA+rS)/2
                    g=(gA+gS)/2
                    b=(bA+bS)/2
                else:
                    r=rA
                    g=gA
                    b=bA
        else:
            if ibu[beer] != 0:
                if sweet[beer] != 0:
                    r=(rI+rS)/2
                    g=(gI+gS)/2
                    b=(bI+bS)/2
                else:
                    r=rI
                    g=gI
                    b=bI
            else:
                if sweet[beer] != 0:
                    r=rS
                    g=gS
                    b=bS
                else:
                    r=255
                    g=255
                    b=255
        draw.text((x,y),beer+': '+str(abv[beer])+', '+str(abv[beer])+'IBU',(r,g,b))
    img.save('images/'+png,'PNG')

def drawWithColor(loc,itemPrefs,color,png='color_beer_map.png', scale=3000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=((loc[beer][0]+0.5)*0.8)*scale/2
        y=((loc[beer][1]+0.5)*0.8)*scale/2
        hexColor="#050B0A"
        if color[beer]==29:
            hexColor="#100B0A"
        elif color[beer]==28:
            hexColor="#120D0C"
        elif color[beer]==27:
            hexColor="#16100F"
        elif color[beer]==26:
            hexColor="#19100F"
        elif color[beer]==25:
            hexColor="#231716"
        elif color[beer]==24:
            hexColor="#261716"
        elif color[beer]==23:
            hexColor="#361F1B"
        elif color[beer]==22:
            hexColor="#4A2727"
        elif color[beer]==21:
            hexColor="#4E2A0C"
        elif color[beer]==20:
            hexColor="#5D341A"
        elif color[beer]==19:
            hexColor="#6B3A1E"
        elif color[beer]==18:
            hexColor="#7C452D"
        elif color[beer]==17:
            hexColor="#8D4C32"
        elif color[beer]==16:
            hexColor="#985336"
        elif color[beer]==15:
            hexColor="#A85839"
        elif color[beer]==14:
            hexColor="#B26033"
        elif color[beer]==13:
            hexColor="#BC6733"
        elif color[beer]==12:
            hexColor="#BF7138"
        elif color[beer]==11:
            hexColor="#C17A37"
        elif color[beer]==10:
            hexColor="#BE823A"
        elif color[beer]==9:
            hexColor="#BE8C3A"
        elif color[beer]==8:
            hexColor="#C1963C"
        elif color[beer]==7:
            hexColor="#CDAA37"
        elif color[beer]==6:
            hexColor="#D5BC26"
        elif color[beer]==5:
            hexColor="#E0D01B"
        elif color[beer]==4:
            hexColor="#EAE615"
        elif color[beer] <= 3:
            hexColor="#F6F513"
        if color[beer] != 0:
            draw.text((x,y),beer,hexColor)
    img.save('images/'+png,'PNG')

def saveAndPushGoogleBeerMapColor(loc,seedPrefs,style,color):
    out=open('data/beergmapcolor.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map - Color</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 11;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        
        if color[beer]>0:
            out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
            out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
            out.write('title: "'+str(beer)+'", ')
            out.write('description: "'+str(style[beer].replace("&","and"))+'",')
            
            textColor="#FFFFFF"
            hexColor="#050B0A"
            if color[beer]==29:
                hexColor="#100B0A"
                textColor="#FFFFFF"
            elif color[beer]==28:
                hexColor="#120D0C"
                textColor="#FFFFFF"
            elif color[beer]==27:
                hexColor="#16100F"
                textColor="#FFFFFF"
            elif color[beer]==26:
                hexColor="#19100F"
                textColor="#FFFFFF"
            elif color[beer]==25:
                hexColor="#231716"
                textColor="#FFFFFF"
            elif color[beer]==24:
                hexColor="#261716"
                textColor="#FFFFFF"
            elif color[beer]==23:
                hexColor="#361F1B"
                textColor="#FFFFFF"
            elif color[beer]==22:
                hexColor="#4A2727"
                textColor="#FFFFFF"
            elif color[beer]==21:
                hexColor="#4E2A0C"
                textColor="#FFFFFF"
            elif color[beer]==20:
                hexColor="#5D341A"
                textColor="#FFFFFF"
            elif color[beer]==19:
                hexColor="#6B3A1E"
                textColor="#FFFFFF"
            elif color[beer]==18:
                hexColor="#7C452D"
                textColor="#FFFFFF"
            elif color[beer]==17:
                hexColor="#8D4C32"
                textColor="#000000"
            elif color[beer]==16:
                hexColor="#985336"
                textColor="#000000"
            elif color[beer]==15:
                hexColor="#A85839"
                textColor="#000000"
            elif color[beer]==14:
                hexColor="#B26033"
                textColor="#000000"
            elif color[beer]==13:
                hexColor="#BC6733"
                textColor="#000000"
            elif color[beer]==12:
                hexColor="#BF7138"
                textColor="#000000"
            elif color[beer]==11:
                hexColor="#C17A37"
                textColor="#000000"
            elif color[beer]==10:
                hexColor="#BE823A"
                textColor="#000000"
            elif color[beer]==9:
                hexColor="#BE8C3A"
                textColor="#000000"
            elif color[beer]==8:
                hexColor="#C1963C"
                textColor="#000000"
            elif color[beer]==7:
                hexColor="#CDAA37"
                textColor="#000000"
            elif color[beer]==6:
                hexColor="#D5BC26"
                textColor="#000000"
            elif color[beer]==5:
                hexColor="#E0D01B"
                textColor="#000000"
            elif color[beer]==4:
                hexColor="#EAE615"
                textColor="#000000"
            elif color[beer] <= 3:
                hexColor="#F6F513"
                textColor="#000000"
            
            out.write('textColor: "'+textColor+'", ')
            out.write('color: "'+hexColor+'"},')
            out.write('\n')

    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,color: beerMapData[i][\'textColor\']\n')
    out.write('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    
    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmapcolor.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmapcolor.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def saveAndPushGoogleBeerMapABVIBU(loc,seedPrefs,style,abv,ibu):
    out=open('data/beergmapabvibu.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 11;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
        out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
        out.write('title: "'+str(beer)+'", ')
        out.write('description: "'+str(style[beer].replace("&","and"))+'",')
        rA=126
        gA=200
        bA=252
        if abv[beer] < 10 and abv[beer] > 4:
            rA=int(((6-(abv[beer]-4))/6)*(255-rA)+rA)
            gA=int(((6-(abv[beer]-4))/6)*(255-gA)+gA)
            bA=int(((6-(abv[beer]-4))/6)*(255-bA)+bA)
        elif abv[beer] <= 4:
            rA=255
            gA=255
            bA=255
        rI=231
        gI=252
        bI=126
        if ibu[beer] < 80 and ibu[beer] > 10:
            rI=int(((70-(ibu[beer]-10))/70)*(255-rI)+rI)
            gI=int(((70-(ibu[beer]-10))/70)*(255-gI)+gI)
            bI=int(((70-(ibu[beer]-10))/70)*(255-bI)+bI)
        elif ibu[beer] <= 10:
            rI=255
            gI=255
            bI=255
        r=(rA+rI)-255
        g=(gA+gI)-255
        b=(bA+bI)-255
        out.write('color: "rgb('+str(r)+','+str(g)+','+str(b)+')"},')
        out.write('\n')
    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    
    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmapabvibu.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmapabvibu.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def saveAndPushGoogleBeerMapIBU(loc,seedPrefs,style,ibu):
    out=open('data/beergmapibu.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map - IBU - International Bittering Units</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 9;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        if ibu[beer] > 0:
            out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
            out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
            out.write('title: "'+str(beer)+'", ')
            out.write('description: "'+str(style[beer].replace("&","and"))+'",')
            r=126
            g=252
            b=145
            if ibu[beer] < 80 and ibu[beer] > 10:
                r=int(((70-(ibu[beer]-10))/70)*(255-r)+r)
                g=int(((70-(ibu[beer]-10))/70)*(255-g)+g)
                b=int(((70-(ibu[beer]-10))/70)*(255-b)+b)
            elif ibu[beer] <= 10:
                r=255
                g=255
                b=255
            out.write('color: "rgb('+str(r)+','+str(g)+','+str(b)+')"},')
            out.write('\n')
    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    
    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmapibu.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmapibu.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def saveAndPushGoogleBeerMapABV(loc,seedPrefs,style,abv):
    out=open('data/beergmapabv.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map - ABV</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 9;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        if abv[beer] > 0:
            out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
            out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
            out.write('title: "'+str(beer)+'", ')
            out.write('description: "'+str(style[beer].replace("&","and"))+'",')
            r=126
            g=208
            b=252
            if abv[beer] < 10 and abv[beer] > 4:
                r=int(((6-(abv[beer]-4))/6)*(255-r)+r)
                g=int(((6-(abv[beer]-4))/6)*(255-g)+g)
                b=int(((6-(abv[beer]-4))/6)*(255-b)+b)
            elif abv[beer] <= 4:
                r=255
                g=255
                b=255
            out.write('color: "rgb('+str(r)+','+str(g)+','+str(b)+')"},')
            out.write('\n')
    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    
    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmapabv.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmapabv.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def saveAndPushGoogleBeerMapControversy(loc,seedPrefs,style,controversy):
    out=open('data/beergmapcontroversy.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map - Controversy</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 9;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        if controversy[beer] > 0:
            out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
            out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
            out.write('title: "'+str(beer)+'", ')
            out.write('description: "'+str(style[beer].replace("&","and"))+'",')
            r=222
            g=79
            b=35
            if controversy[beer] < 0.3 and controversy[beer] > 0.15:
                r=int(((0.15-(controversy[beer]-0.15))/0.15)*(255-r)+r)
                g=int(((0.15-(controversy[beer]-0.15))/0.15)*(255-g)+g)
                b=int(((0.15-(controversy[beer]-0.15))/0.15)*(255-b)+b)
            elif controversy[beer] <= 0.15:
                r=255
                g=255
                b=255
            out.write('color: "rgb('+str(r)+','+str(g)+','+str(b)+')"},')
            out.write('\n')
    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    
    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmapcontroversy.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmapcontroversy.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def saveAndPushGoogleBeerMapGravity(loc,seedPrefs,style,gravity):
    out=open('data/beergmapgravity.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map - Specific Gravity</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 9;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        
        if gravity[beer] > 0:
            out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
            out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
            out.write('title: "'+str(beer)+'", ')
            out.write('description: "'+str(style[beer].replace("&","and"))+'",')
            
            r=126
            g=208
            b=252
            if gravity[beer] < 1.017 and gravity[beer] > 1.010:
                r=int(((0.007-(gravity[beer]-1.010))/0.007)*(255-r)+r)
                g=int(((0.007-(gravity[beer]-1.010))/0.007)*(255-g)+g)
                b=int(((0.007-(gravity[beer]-1.010))/0.007)*(255-b)+b)
            elif gravity[beer] <= 1.010:
                r=255
                g=255
                b=255
            out.write('color: "rgb('+str(r)+','+str(g)+','+str(b)+')"},')
            out.write('\n')

    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,backgroundColor: beerMapData[i][\'color\']\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')
    
    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmapgravity.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmapgravity.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def saveAndPushGoogleBeerMap(loc,seedPrefs,style):
    out=open('data/beergmap.html','w')
    out.write('<!DOCTYPE html>\n')
    out.write('<html>\n')
    out.write('\t<head>\n')
    out.write('\t\t<title>Beer Map</title>\n')
    out.write('\t\t<style>\n')
    out.write('\t\t\thtml, body, #map-canvas {\n')
    out.write('\t\t\t\theight: 100%;\n')
    out.write('\t\t\t\tmargin: 0px;\n')
    out.write('\t\t\t\tpadding: 0px\n')
    out.write('\t\t\t}\n')
    out.write('\t\t</style>\n')
    out.write('\t\t<script src="https://maps.googleapis.com/maps/api/js?v=3.exp&sensor=false"></script>\n')
    out.write('\t\t<script type="text/javascript" src="../beermap/infobox.js"></script>\n')
    out.write('\t\t<script>\n')
    out.write('\n')
    out.write('function CoordMapType() {\n')
    out.write('}\n')
    out.write('\n')
    out.write('CoordMapType.prototype.tileSize = new google.maps.Size(256,256);\n')
    out.write('CoordMapType.prototype.maxZoom = 9;\n')
    out.write('CoordMapType.prototype.minZoom = 5;\n')
    out.write('\n')
    out.write('CoordMapType.prototype.getTile = function(coord, zoom, ownerDocument) {\n')
    out.write('\tvar div = ownerDocument.createElement(\'div\');\n')
    out.write('\tdiv.innerHTML = \'\';\n')
    out.write('\tdiv.style.width = this.tileSize.width + \'px\';\n')
    out.write('\tdiv.style.height = this.tileSize.height + \'px\';\n')
    out.write('\tdiv.style.fontSize = \'10\';\n')
    out.write('\tdiv.style.borderStyle = \'none\';\n')
    out.write('\tdiv.style.backgroundColor = \'#E5E3DF\';\n')
    out.write('\treturn div;\n')
    out.write('};\n')
    out.write('\n')
    out.write('CoordMapType.prototype.name = \'Beer\';\n')
    out.write('CoordMapType.prototype.alt = \'Beer Map Type\';\n')
    out.write('\n')
    out.write('var map;\n')
    out.write('var coordinateMapType = new CoordMapType();\n')
    out.write('\n')
    out.write('var beerMapData = [\n')
    for beer in seedPrefs:
        out.write('\t{latitude: '+str(float((loc[beer][0])*15))+', ')
        out.write('longitude: '+str(float((loc[beer][1])*15))+', ')
        out.write('title: "'+str(beer)+'", ')
        out.write('description: "'+str(style[beer].replace("&","and"))+'"},')
        out.write('\n')
    out.write('];\n')
    out.write('\n')
    out.write('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
    out.write('\n')
    out.write('function initialize() {\n')
    out.write('\tvar mapOptions = {\n')
    out.write('\t\tzoom: 5,\n')
    out.write('\t\tcenter: mapCenter,\n')
    out.write('\t\tstreetViewControl: false,\n')
    out.write('\t\tmapTypeId: \'coordinate\',\n')
    out.write('\t\tmapTypeControlOptions: {\n')
    out.write('\t\t\tmapTypeIds: [\'coordinate\'],\n')
    out.write('\t\t\tstyle: google.maps.MapTypeControlStyle.DROPDOWN_MENU\n')
    out.write('\t\t}\n')
    out.write('\t};\n')
    out.write('\tmap = new google.maps.Map(document.getElementById(\'map-canvas\'),mapOptions);\n')
    out.write('\n')
    out.write('\tfor (var i=0;i<beerMapData.length;i++)\n')
    out.write('\t{\n')
    out.write('\t\tvar myOptions = {\n')
    out.write('\t\t\tcontent: beerMapData[i][\'title\']+"<br/>("+beerMapData[i][\'description\']+")"\n')
    out.write('\t\t\t,boxStyle: {\n')
    out.write('\t\t\t\tborder: "1px solid black"\n')
    out.write('\t\t\t\t,textAlign: "center"\n')
    out.write('\t\t\t\t,fontSize: "6pt"\n')
    out.write('\t\t\t\t,width: "60px"\n')
    out.write('\t\t\t\t,backgroundColor: "#fff"\n')
    out.write('\t\t\t}\n')
    out.write('\t\t\t,disableAutoPan: true\n')
    out.write('\t\t\t,pixelOffset: new google.maps.Size(-25, 0)\n')
    out.write('\t\t\t,position: new google.maps.LatLng(beerMapData[i][\'latitude\'], beerMapData[i][\'longitude\'])\n')
    out.write('\t\t\t,closeBoxURL: ""\n')
    out.write('\t\t\t,isHidden: false\n')
    out.write('\t\t\t,pane: "mapPane"\n')
    out.write('\t\t\t,enableEventPropagation: true\n')
    out.write('\t\t};\n')
    out.write('\t\tvar ibLabel = new InfoBox(myOptions);\n')
    out.write('\t\tibLabel.open(map);\n')
    out.write('\t}\n')
    out.write('\n')
    out.write('map.mapTypes.set(\'coordinate\', coordinateMapType);\n')
    out.write('}\n')
    out.write('\n')
    out.write('google.maps.event.addDomListener(window, \'load\', initialize);\n')
    out.write('\n')
    out.write('\t\t</script>\n')
    out.write('\t</head>\n')
    out.write('\t<body>\n')
    out.write('\t\t<div id="map-canvas"></div>\n')
    out.write('\t</body>\n')
    out.write('</html>\n')

    out.close()
    
    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beergmap.html','rb')                  # file to send
    session.storbinary('STOR beermap/beergmap.html', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def outputForceJson(seedPrefs,realsim,index,sorted_controversy):
    out=open('data/beerforce.json','w')
    out.write('{\n')
    out.write('\t"nodes":[\n')
    c=0
    for x in range(len(seedPrefs)):
        beer=sorted_controversy[x][0]
        if c>0:
            out.write(',\n')
        out.write('\t\t{"name":"'+str(beer)+'",')
        out.write('"group":1}')
        c+=1
    out.write('\n\t],\n')
    
    out.write('\t"links":[\n')
    # Sorted array of most similar beers to this one
    simArray=sorted(realsim[beer].iteritems(), key=operator.itemgetter(1))
    simArray.reverse()
    c=0
    for x in range(len(seedPrefs)):
        beer=sorted_controversy[x][0]
        for y in range(len(simArray)):
            # See if this is a seed beer
            simBeer=simArray[y][0]
            if simBeer==beer: continue
            if simBeer in seedPrefs:
                if realsim[beer][simBeer]>0.88:
                    if index[beer]<index[simBeer]:
                        # beers are similar, so add a link between them
                        if c>0:
                            out.write(',\n')
                        out.write('\t\t{"source":'+index[beer]+',')
                        out.write('"target":'+index[simBeer]+',')
                        out.write('"value":'+str(realsim[beer][simBeer])+'}')
                        c+=1
#else: continue
    out.write('\n\t]\n')
    out.write('}')
    out.close()

    session = ftplib.FTP(loadFTPCredentials())
    file = open('data/beerforce.json','rb')                  # file to send
    session.storbinary('STOR beermap/beerforce.json', file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'json uploaded'


# Keep calling this function to add new beers a few at a time

# loc,seedPrefs=testclustering.addMoreBeersToClusterMapGmap(sorted_controversy,seedPrefs,allItemPrefs,newRealDist,newRealSim,loc,style,abv,ibu,limit=100)
def addMoreBeersToClusterMapGmap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,style,abv,ibu,limit=10):
    
    for x in range(limit):
        loc,seedPrefs=addBeerToClusterMapGmap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,style,abv,ibu)
        print "added beer %d / %d" % (x+1,limit)
    return loc,seedPrefs

def addBeerToClusterMapGmap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,style,abv,ibu):
    
    # Get the next index of the beer to add to the cluster
    # (This is the next most controversial beer on the list)
    newIndex=len(seedPrefs)
    
    if newIndex < len(allItemPrefs):
        beer=sorted_controversy[newIndex][0]
        
        addBeerAndRecluster(seedPrefs,allItemPrefs,realdist,realsim,loc,beer)
        saveAndPushGoogleBeerMapABVIBU(loc,seedPrefs,style,abv,ibu)
        
        # Save the data structures just in case
        pickle.dump(loc, open("pickle/%d_loc.p" % (newIndex),"wb"))
        pickle.dump(seedPrefs, open("pickle/%d_seedprefs.p" % (newIndex),"wb"))
    else:
        print 'all beers have been added'
    
    return loc,seedPrefs

