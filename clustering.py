from PIL import Image,ImageDraw
from scipy import stats
import math
from math import sqrt
import copy
import operator
import pickle
import random

# Adds every single beer to the beer cluster map with no pauses
# (note: with lots of beers and user ratings (I have over 200,000) this takes a very long time to run)
def clusterAllBeers():

    loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist=createInitialBeerClusters()
    
    totalBeersToAdd=len(allItemPrefs)-len(seedPrefs)

    loc,seedPrefs=addMoreBeersToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,limit=totalBeersToAdd)

    return loc,seedPrefs,allItemPrefs,sorted_controversy,realsim,realdist


# Creates the initial beer map we will add other beers to
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

# Add new beers a few at a time
def addMoreBeersToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc,limit=10):
    
    for x in range(limit):
        loc,seedPrefs=addBeerToClusterMap(sorted_controversy,seedPrefs,allItemPrefs,realdist,realsim,loc)
        print "added beer %d / %d" % (x+1,limit)
    return loc,seedPrefs






# internal methods--------------------------------------------------------------------------------->

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
    
    locSecondary=decreaseClusterErrorForThisBeer(locSecondary,realdist,secondaryPrefs,seedPrefs,loc,rate=0.01,tries=100000)
    
    seedPrefs[beer]={}
    seedPrefs[beer]=itemPrefs[beer]
    loc[beer]={}
    loc[beer]=[locSecondary[beer][0],locSecondary[beer][1]]
    
    loc=decreaseClusterError(loc,realdist,seedPrefs,rate=0.01,tries=10000)
    
    return loc

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
        if (m+1)%10000==0:
            print "%d / %d with error: %f" % (m,tries,totalerror)
        
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
        if (m+1)%10000==0:
            print "%d / %d with error: %f" % (m,tries,totalerror)
        
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
    if n==0: return -1
    
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
    if n==0: return 0
    
    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        # return 0 (they are not correlated)
        return 0
    
    return pearson_correlation
