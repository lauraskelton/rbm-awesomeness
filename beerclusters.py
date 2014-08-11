from PIL import Image,ImageDraw
import codecs

def readfile(filename):
    lines=[line for line in file(filename)]
    
    # First line is the column titles
    colnames=lines[0].strip().split('\t')[1:]
    rownames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split('\t')
        # First column in each row is the rowname
        rownames.append(p[0])
        # The data for this row is the remainder of the row
        data.append([float(x) for x in p[1:]])
    return rownames, colnames, data

from math import sqrt
def pearson(v1,v2):
    # Simple sums
    sum1=sum(v1)
    sum2=sum(v2)
    
    # Sums of the squares
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])
    
    # Sum of the products
    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])
    
    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0
    
    return 1.0-num/den

class bicluster:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance

def hcluster(rows,distance=pearson):
    distances={}
    currentclustid=-1
    
    # Clusters are initially just the rows
    clust=[bicluster(rows[i],id=i) for i in range(len(rows))]
    
    while len(clust)>1:
        lowestpair=(0,1)
        closest=distance(clust[0].vec,clust[1].vec)
        
        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                # distances is the cache of distance calculations
                if (clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)
                
                d=distances[(clust[i].id,clust[j].id)]
                
                if d<closest:
                    closest=d
                    lowestpair=(i,j)
        
        # calculate the average of two clusters
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]
        
        # create the new cluster
        newcluster=bicluster(mergevec,left=clust[lowestpair[0]],right=clust[lowestpair[1]],distance=closest,id=currentclustid)
        
        # cluster ids that weren't in the original set are negative
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)
    
    return clust[0]

def printclust(clust,labels=None,n=0):
    # indent to make a hierarchy layout
    for i in range(n): print ' ',
    if clust.id<0:
        # negative id means that this is branch
        print '-'
    else:
        # positive id means that this is an endpoint
        if labels==None: print clust.id
        else: print labels[clust.id]
    
    # now print the right and left branches
    if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)

def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left==None and clust.right==None: return 1
    
    # Otherwise the height is the sum of the heights of each branch
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left==None and clust.right==None: return 0
    
    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance

def drawdendrogram(clust,labels,png='clusters.png'):
    # height and width
    h=getheight(clust)*20
    w=3000
    depth=getdepth(clust)
    
    # width is fized, so scale distances accordingly
    scaling=float(w-150)/depth
    
    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)
    
    draw.line((0,h/2,10,h/2),fill=(255,0,0))
    
    # Draw the first node
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(png,'PNG')

def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=clust.distance*scaling
        # Vertical line from this cluster to children
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        
        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))
        
        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))
        
        # Call the function to draw the left and right nodes
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        # If this is an endpoint, draw the item label
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))

def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow=[data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

import random

def kcluster(rows,distance=pearson,k=4):
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) for i in range(len(rows[0]))]
    
    # Create k randomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]
    
    lastmatches=None
    for t in range(100):
        print 'Iteration %d' % t
        bestmatches=[[] for i in range(k)]
        
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row): bestmatch=i
            bestmatches[bestmatch].append(j)
        
        # If the results are the same as last time, this is complete
        if bestmatches==lastmatches: break
        lastmatches=bestmatches
        
        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
    
    return bestmatches

def printkclust(clust,labels=None,k=0):
    # Print out the list of clusters and their members
    for j in range(k):
        print j
        for r in clust[j]:
            print labels[r]

def tanimoto(v1,v2):
    c1,c2,shr=0,0,0
    
    for i in range(len(v1)):
        if v1[i]!=0: c1+=1 # in v1
        if v2[i]!=0: c2+=1 # in v2
        if v1[i]!=0 and v2[i]!=0: shr+=1 # in both
    
    return 1.0-(float(shr)/(c1+c2-shr))

def scaledown(data,distance=pearson,rate=0.01):
    n=len(data)
    
    # The real distances between every pair of items
    realdist=[[distance(data[i],data[j]) for j in range(n)] for i in range(0,n)]
    
    outersum=0.0
    
    # Randomly initialize the starting points of the locations in 2D
    loc=[[random.random(),random.random()] for i in range(n)]
    fakedist=[[0.0 for j in range(n)] for i in range(n)]
    
    lasterror=None
    for m in range(0,1000):
        # Find projected distances
        for i in range(n):
            for j in range(n):
                fakedist[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2) for x in range(len(loc[i]))]))
        
        # Move points
        grad=[[0.0,0.0] for i in range(n)]
        
        totalerror=0
        for k in range(n):
            for j in range(n):
                if j==k: continue
                # The error is percent difference between the distances
                if realdist[j][k]!=0:
                    errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]
                else:
                    errorterm=0
                
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                if fakedist[j][k]!=0:
                    grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
                    grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm
                else:
                    grad[k][0]+=0
                    grad[k][1]+=0
                
                # Keep track of the total error
                totalerror+=abs(errorterm)
        print totalerror
        
        # If the answer got worse by moving the points, we are done
        if lasterror and lasterror<totalerror: break
        lasterror=totalerror
        
        # Move each of the points by the learning rate times the gradient
        for k in range(n):
            loc[k][0]-=rate*grad[k][0]
            loc[k][1]-=rate*grad[k][1]
    
    return loc

def draw2d(data,labels,png='mds2d.png'):
    img=Image.new('RGB',(3000,3000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1500
        y=(data[i][1]+0.5)*1500
        draw.text((x,y),labels[i],(0,0,0))
    img.save(png,'PNG')

from scipy import stats
import math
def sim_pearson_dist(prefs,p1,p2):
    
    beer1 = prefs[p1]
    beer2 = prefs[p2]
    x = []
    y = []

    for user in beer1:
        if beer2.has_key(user):
            x.append(beer1[user])
            y.append(beer2[user])

    # Find the number of elements
    n=len(x)

    # If they have no ratings in common, return 0
    if n==0: return -1

    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        #print 'is nan'
        #print p1
        #print p2
        return -1
    
    return 1-pearson_correlation

def sim_pearson_normal(prefs,p1,p2):
    
    beer1 = prefs[p1]
    beer2 = prefs[p2]
    x = []
    y = []
    
    for user in beer1:
        if beer2.has_key(user):
            x.append(beer1[user])
            y.append(beer2[user])
    
    # Find the number of elements
    n=len(x)
    
    # If they have no ratings in common, return 0
    if n==0: return 0
    
    pearson_correlation=stats.pearsonr(x, y)[0]
    
    if math.isnan(pearson_correlation):
        #print 'is nan'
        #print p1
        #print p2
        return 0
    
    return pearson_correlation

def loadItemPrefs():
    prefs=loadLatestBeerChooser()
    itemPrefs=transformPrefs(prefs)
    return itemPrefs

def findControversy(itemPrefs):
    controversy={}
    c=0
    for beer in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d -------------------------------" % (c,len(itemPrefs))
        
        controSum=0
        num=0
        for otherBeer in itemPrefs:
            controSum+=abs(sim_pearson_normal(itemPrefs,beer,otherBeer))
            num+=1
        if num>0:
            controversy[beer]=controSum/num
        else:
            controversy[beer]=0
        print beer
        print controversy[beer]

    return controversy

def findMeaningfulBeers(controversy):
    invertedControversy=transformControversy(controversy)
    sortedList=sorted(invertedControversy)
    print sortedList
    result=[]
    for i in range(len(sortedList)):
        result[i]=invertedControversy[sortedList[i]]
    print result
    return result


def transformControversy(controversy):
    result={}
    for beer in controversy:
        result[controversy[beer]]=beer
    return result

def calculateAllBeerSimilarities(itemPrefs):
    # Create a dictionary of items showing distance to every other item
    realsim={}
    
    c=0
    for beer in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(itemPrefs))
        
        # Find the distance from every item to this one
        realsim[beer]={}
        for otherBeer in itemPrefs:
            #if otherBeer == beer: continue
            realsim[beer][otherBeer]=float(sim_pearson_normal(itemPrefs,beer,otherBeer))
    
    return realsim


def sim_pearson_dist_old(prefs,p1,p2):
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item]=1
    
    # Find the number of elements
    n=len(si)
    
    # If they have no ratings in common, return 0
    if n==0: return 0
    
    # Add up all the preferences
    sum1=sum([(prefs[p1][it]*5) for it in si])
    sum2=sum([(prefs[p2][it]*5) for it in si])
    
    # Sum up the squares
    sum1Sq=sum([pow((prefs[p1][it]*5),2) for it in si])
    sum2Sq=sum([pow((prefs[p2][it]*5),2) for it in si])
    
    # Sum up the products
    pSum=sum([(prefs[p1][it]*5)*(prefs[p2][it]*5) for it in si])
    
    # Calculate Pearson score
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0
    
    r=float(1.0-num/den)
    
    return r
#return sum1Sq,pow(sum1,2),n

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def loadLatestBeerChooser(path='data'):
    
    # Get beer names
    beers={}
    for line in open(path+'/u.item'):
    #inputFile=codecs.open(path+'/u.item','r', 'utf-8')
    #for line in inputFile:
        (id,name)=line.split('\t')
        beers[id]=name.rstrip()
    
    # Load data
    prefs={}
    for line in open(path+'/u.data'):
        (user,beerid,rating)=line.split(',')
        prefs.setdefault(user,{})
        prefs[user][beers[beerid]]=float(rating)
    return prefs

def loadSeedBeers(path='data'):
    
    # Get beer names
    beers={}
    for line in open(path+'/u.seedbeers'):
        (id,name)=line.split('\t')
        beers[id]=name.rstrip()
    
    # Load data
    itemPrefs={}
    for line in open(path+'/u.data'):
        (user,beerid,rating)=line.split(',')
        if beerid in beers:
        # This is a rating of one of our seed beers
            beer = beers[beerid]
            itemPrefs.setdefault(beer,{})
            itemPrefs[beer][user]=float(rating)

    return itemPrefs

def loadSecondaryBeers(path='data'):
    
    # Get secondary beer names
    beersSecondary={}
    for line in open(path+'/u.secondarybeers'):
        (id,name)=line.split('\t')
        beersSecondary[id]=name.rstrip()
    
    # Get seed beer names
    beersSeed={}
    for line in open(path+'/u.seedbeers'):
        (id,name)=line.split('\t')
        beersSeed[id]=name.rstrip()
    
    # Get all beer names
    beers={}
    for line in open(path+'/u.item'):
        (id,name)=line.split('\t')
        beers[id]=name.rstrip()
    
    # Load data
    secondaryPrefs={}
    seedPrefs={}
    allItemPrefs={}
    for line in open(path+'/u.data'):
        (user,beerid,rating)=line.split(',')
        beer = beers[beerid]
        allItemPrefs.setdefault(beer,{})
        allItemPrefs[beer][user]=float(rating)
        if beerid in beersSeed:
            # This is a rating of one of our seed beers
            seedPrefs.setdefault(beer,{})
            seedPrefs[beer][user]=float(rating)
        if beerid in beersSecondary:
            # This is a rating of one of our secondary beers
            secondaryPrefs.setdefault(beer,{})
            secondaryPrefs[beer][user]=float(rating)

    locSeed=pickle.load(open("data/locsaveseed.p","rb"))
    
    return secondaryPrefs,seedPrefs,allItemPrefs,locSeed

def calculateSecondaryBeerDistancesRecluster(secondaryPrefs,seedPrefs,loc,locSeed):
    # Create a dictionary of items showing distance to every other item
    realdist={}
    beerPrefs={}
    locBeer={}
    
    for beer in seedPrefs:
        
        # This is one of our seed beers
        beerPrefs.setdefault(beer,{})
        for user in seedPrefs[beer]:
            beerPrefs[beer][user]=seedPrefs[beer][user]
        
        locBeer.setdefault(beer,{})
        for x in range(len(locSeed[beer])):
            locBeer[beer][x]=locSeed[beer][x]

    for beer in secondaryPrefs:
    
        # This is one of our secondary beers
        beerPrefs.setdefault(beer,{})
        for user in secondaryPrefs[beer]:
            beerPrefs[beer][user]=secondaryPrefs[beer][user]

        locBeer.setdefault(beer,{})
        for x in range(len(loc[beer])):
            locBeer[beer][x]=loc[beer][x]

    # Now we have just one prefs and one loc with all of the beers to work with

    c=0
    for beer in beerPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(beerPrefs))
        
        # Find the distance from every item to this one
        realdist[beer]={}
        for otherBeer in beerPrefs:
            realdist[beer][otherBeer]=float(sim_pearson_dist(beerPrefs,beer,otherBeer))
    
    return locBeer,realdist,beerPrefs

def calculateSecondaryBeerDistances(secondaryPrefs,seedPrefs,allItemPrefs):
    # Create a dictionary of items showing distance to every other item
    realdist={}
    
    c=0
    for addedBeer in secondaryPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(secondaryPrefs))
        
        # Find the distance from every item to this one
        realdist[addedBeer]={}
        for seedBeer in seedPrefs:
            #if otherBeer == beer: continue
            realdist[addedBeer][seedBeer]=float(sim_pearson_dist(allItemPrefs,addedBeer,seedBeer))
    
    return realdist

def calculateAllBeerDistances(itemPrefs):
    # Create a dictionary of items showing distance to every other item
    realdist={}
    
    c=0
    for beer in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(itemPrefs))

        # Find the distance from every item to this one
        realdist[beer]={}
        for otherBeer in itemPrefs:
            #if otherBeer == beer: continue
            realdist[beer][otherBeer]=float(sim_pearson_dist(itemPrefs,beer,otherBeer))

    return realdist

def beerDistanceCalc():
    prefs = loadLatestBeerChooser()
    itemPrefs = transformPrefs(prefs)
    realdist = calculateAllBeerDistances(itemPrefs)
    return prefs,itemPrefs,realdist


def generateBeerDistances():

    prefs=loadLatestBeerChooser()
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)

    realdist=calculateAllBeerDistances(itemPrefs)
    # realdist[beer][otherBeer]=[pearson_distance between beer and otherBeer]

    return itemPrefs,realdist

def saveBeerDistances(prefs):
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    
    realdist=calculateAllBeerDistances(itemPrefs)
    # realdist[beer][otherBeer]=[pearson_distance between beer and otherBeer]

#out=codecs.open('realdist.txt','a', 'utf-8')
    out=open('cachedist.txt','w')
    # column labels
    out.write('name')
    for eachBeer in itemPrefs:
        out.write('\t'+eachBeer)
    out.write('\n')

    c=0
    for beer in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0:
            print "%d / %d" % (c,len(itemPrefs))
                
        # row label
        out.write(beer)
        
        for otherBeer in itemPrefs:
            distancestring="%f" % (realdist[beer][otherBeer])
            out.write('\t'+distancestring)
        out.write('\n')

def saveControversy(controversy):

    out=open('data/beercontroversy.txt','w')
    for beer in controversy:
        c=0
        for item in beer:
            out.write(str(item))
            if c==0:
                out.write('\t')
            else:
                out.write('\n')
            c+=1

def loadControversy(path='data'):
    controversy={}
    
    for line in open(path+'/beercontroversy.txt'):
        (beer,contro)=line.split('\t')
        controversy[beer.rstrip()]=contro
    return controversy

def createControversySeedPrefs(sorted_controversy,itemPrefs):
    seedBeers={}
    seedPrefs={}
    for x in range(30):
        beer=sorted_controversy[x][0]
        seedBeers[beer]=beer
    for seedBeer in seedBeers:
        seedPrefs[seedBeer]={}
        seedPrefs[seedBeer]=itemPrefs[seedBeer]
    return seedPrefs

import operator
import pickle
def addBeerToControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc):
    
    # get the next index of the beer to add to the cluster
    newIndex=len(seedPrefs)

    # Save the existing data structures
    folderPath="data/controversy/clusters/"
    pickle.dump(seedPrefs, open(folderPath+("%dseedprefssave.p" % (newIndex)),"wb"))
    pickle.dump(loc, open(folderPath+("%dlocsave.p" % (newIndex)),"wb"))
    
    # adjust only this beer's position to match existing seed beers
    
    locSecondary={}
    beer=sorted_controversy[newIndex][0]
    locSecondary[beer]={}
    locSecondary[beer]=[random.random(),random.random()]
    
    secondaryPrefs={}
    secondaryPrefs[beer]={}
    secondaryPrefs[beer]=itemPrefs[beer]
    
    # Sorted array of most similar beers to this one
    simArray=sorted(realsim[beer].iteritems(), key=operator.itemgetter(1))
    simArray.reverse()
    for x in range(len(simArray)):
        # See if this is a seed beer
        simBeer=simArray[x]
        if simBeer==beer: continue
        if simBeer in seedPrefs:
            if realsim[beer][simBeer]>0:
                print beer
                print simBeer
                print realsim[beer][simBeer]
                # Use the similar beer's location as this beer's starting location
                # instead of random starting location
                locSecondary[beer]=[loc[simBeer][0],loc[simBeer][1]]
                break

    locSecondary=beerDecreaseErrorSecondary(locSecondary,realdist,secondaryPrefs,seedPrefs,loc,rate=0.01,tries=100000)

    seedPrefs[beer]={}
    seedPrefs[beer]=itemPrefs[beer]
    loc[beer]={}
    loc[beer]=[locSecondary[beer][0],locSecondary[beer][1]]
    
    draw2dBeer(loc,seedPrefs,png="contropng/middle/l_contro_clusters_%d_before.png" % (newIndex),scale=2000)
    
    loc=beerDecreaseError(loc,realdist,seedPrefs,rate=0.01,tries=100000)
    
    draw2dBeer(loc,seedPrefs,png="contropng/l_contro_clusters_%d.png" % (newIndex),scale=2000)

    return loc,seedPrefs

def addManyBeersToControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc,limit=10):
    
    for x in range(10):
        loc,seedPrefs=addBeerToControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc)
    
    return loc,seedPrefs

def moveBeerToBeerAndReControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc,beer,toBeer):
    
    # adjust only this beer's position to match existing seed beers
    
    locSecondary={}
    locSecondary[beer]={}
    locSecondary[beer]=[loc[toBeer][0],loc[toBeer][0]]
    
    secondaryPrefs={}
    secondaryPrefs[beer]={}
    secondaryPrefs[beer]=itemPrefs[beer]
    
    del seedPrefs[beer]
    del loc[beer]
    
    locSecondary=beerDecreaseErrorSecondary(locSecondary,realdist,secondaryPrefs,seedPrefs,loc,rate=0.01,tries=100000)
    
    seedPrefs[beer]={}
    seedPrefs[beer]=itemPrefs[beer]
    loc[beer]={}
    loc[beer]=[locSecondary[beer][0],locSecondary[beer][1]]
    
    draw2dBeer(loc,seedPrefs,png="contropng/middle/l_contro_clusters_redo_before.png",scale=2000)
    
    loc=beerDecreaseError(loc,realdist,seedPrefs,rate=0.01,tries=100000)
    
    draw2dBeer(loc,seedPrefs,png="contropng/l_contro_clusters_redo.png",scale=2000)
    
    return loc,seedPrefs

def testMoveBeerAndReControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc,beer):
    
    # adjust only this beer's position to match existing seed beers
    
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
    
    #print [row for row in simArray if row[1]==1]
    
    for x in range(len(simArray)):
        # See if this is a seed beer
        simBeer=simArray[x][0]
        if simBeer==beer: continue
        if simBeer in seedPrefs:
            if realsim[beer][simBeer]>0:
                print beer
                print simBeer
                print realsim[beer][simBeer]
                print loc[simBeer]
                # Use the similar beer's location as this beer's starting location
                # instead of random starting location
                #locSecondary[beer]=[loc[simBeer][0],loc[simBeer][1]]
                
                # weighted average of all similar beer locations for better starting point
                locSecondary[beer][0]+=(loc[simBeer][0])*realsim[beer][simBeer]
                locSecondary[beer][1]+=(loc[simBeer][1])*realsim[beer][simBeer]
                
                totalSim+=realsim[beer][simBeer]
            else: break
    if totalSim>0:
        locSecondary[beer][0]=(locSecondary[beer][0])/totalSim
        locSecondary[beer][1]=(locSecondary[beer][1])/totalSim

    print locSecondary[beer]
    print totalSim

def moveBeerAndReControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc,beer):
    
    # adjust only this beer's position to match existing seed beers
    
    locSecondary={}
    locSecondary[beer]={}
    
    secondaryPrefs={}
    secondaryPrefs[beer]={}
    secondaryPrefs[beer]=itemPrefs[beer]
    
    del seedPrefs[beer]
    del loc[beer]

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
                #print simBeer
                #print realsim[beer][simBeer]
                # Use the similar beer's location as this beer's starting location
                # instead of random starting location
                #locSecondary[beer]=[loc[simBeer][0],loc[simBeer][1]]
                
                # weighted average of all similar beer locations for better starting point
                locSecondary[beer][0]+=(loc[simBeer][0])*realsim[beer][simBeer]
                locSecondary[beer][1]+=(loc[simBeer][1])*realsim[beer][simBeer]
                
                totalSim+=realsim[beer][simBeer]
            else: break

    if totalSim>0:
        locSecondary[beer][0]=(locSecondary[beer][0])/totalSim
        locSecondary[beer][1]=(locSecondary[beer][1])/totalSim
    
    print beer
    print locSecondary
    print totalSim
    
    locSecondary=beerDecreaseErrorSecondary(locSecondary,realdist,secondaryPrefs,seedPrefs,loc,rate=0.01,tries=100000)
    
    seedPrefs[beer]={}
    seedPrefs[beer]=itemPrefs[beer]
    loc[beer]={}
    loc[beer]=[locSecondary[beer][0],locSecondary[beer][1]]
    
    draw2dBeer(loc,seedPrefs,png="contropng/recluster/middle/"+beer+"_before.png",scale=2000)
    
    loc=beerDecreaseError(loc,realdist,seedPrefs,rate=0.01,tries=10000)
    
    draw2dBeer(loc,seedPrefs,png="contropng/recluster/"+beer+".png",scale=2000)
    
    return loc

import copy
import pickle
def moveAllBeersAndReControCluster(sorted_controversy,seedPrefs,itemPrefs,realdist,realsim,loc):
    
    for beer in seedPrefs:
        tmpPrefs=copy.deepcopy(seedPrefs)
        loc=moveBeerAndReControCluster(sorted_controversy,tmpPrefs,itemPrefs,realdist,realsim,loc,beer)
    
        # Save the existing data structures
        folderPath="data/controversy/reclusters/"
        pickle.dump(seedPrefs, open(folderPath+beer+"seedprefssave.p","wb"))
        pickle.dump(loc, open(folderPath+beer+"locsave.p","wb"))
    return loc

def saveControversy(controversy):
    
    out=open('data/beercontroversy.txt','w')
    for beer in controversy:
        c=0
        for item in beer:
            out.write(str(item))
            if c==0:
                out.write('\t')
            else:
                out.write('\n')
            c+=1

def loadSavedDistances(filename='cachedist.txt'):
    
    #need to define this
    print 'not done yet'

def beerScaledownAll():

    itemPrefs,realdist=generateBeerDistances()
    # realdist is the real distances between every pair of items

    beerScaledown(realdist,itemPrefs,rate=0.01,tries=50)


def beerScaledown(realdist,itemPrefs):
    
    outersum=0.0

    # Randomly initialize the starting points of the locations in 2D
    loc={}
    fakedist={}
    for beer in itemPrefs:
        loc[beer]=[random.random(),random.random()]
        fakedist[beer]={}
        for otherBeer in itemPrefs:
            fakedist[beer][otherBeer]=[0.0]

    lasterror=None
    # Find projected distances
    for beer in itemPrefs:
        for otherBeer in itemPrefs:
            # Absolute value of the difference between x1 and x2 and y1 and y2
            fakedist[beer][otherBeer]=sqrt(sum([pow(loc[beer][x]-loc[otherBeer][x],2) for x in range(len(loc[beer]))]))

    totalerror=0
    num=0
    for k in itemPrefs:
        for j in itemPrefs:
            if k==j: continue
            if realdist[k][j]==-1: continue

            # The error is percent difference between the distances
            if fakedist[k][j]!=0:
                errorterm=(fakedist[k][j]-realdist[k][j])/fakedist[k][j]
            else:
                errorterm=1

            # Keep track of the total error
            #totalerror+=abs(errorterm)
            totalerror+=pow(errorterm,2)
            num+=1

    if num>0:
        totalerror = sqrt(totalerror)/num
    print totalerror

    return loc

def beerDecreaseError(loc,realdist,itemPrefs,rate=0.01,tries=50):
    
    fakedist={}
    grad={}
    for beer in itemPrefs:
        fakedist[beer]={}
        for otherBeer in itemPrefs:
            fakedist[beer][otherBeer]=[0.0]
    
    lasterror=None
    for m in range(0,tries):
        # Find projected distances
        for beer in itemPrefs:
            for otherBeer in itemPrefs:
                # Absolute value of the difference between x1 and x2 and y1 and y2
                fakedist[beer][otherBeer]=sqrt(sum([pow(loc[beer][x]-loc[otherBeer][x],2) for x in range(len(loc[beer]))]))
        
        # Move points
        for beer in itemPrefs:
            grad[beer]=[0.0,0.0]
        
        totalerror=0
        num=0
        for k in itemPrefs:
            for j in itemPrefs:
                if k==j: continue
                if realdist[k][j]==-1: continue
                
                # The error is percent difference between the distances
                if fakedist[k][j]!=0:
                    errorterm=(fakedist[k][j]-realdist[k][j])/fakedist[k][j]
                else:
                    errorterm=1
                
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                if fakedist[k][j]!=0:
                    grad[k][0]+=(loc[j][0]-loc[k][0])*errorterm
                    grad[k][1]+=(loc[j][1]-loc[k][1])*errorterm
                
                # Keep track of the total error
                totalerror+=pow(errorterm,2)
                num+=1
        if num>0:
            totalerror = sqrt(totalerror)/num
        
        # Status updates for large datasets
        if (m+1)%10000==0:
            print "%d / %d with error: %f" % (m,tries,totalerror)

#if lasterror and lasterror < totalerror: break
#lasterror = totalerror


        # Move each of the points by the learning rate times the gradient
        for beerK in itemPrefs:
            if num>0:
                loc[beerK][0]+=rate*(grad[beerK][0]/num)
                loc[beerK][1]+=rate*(grad[beerK][1]/num)
    
    return loc

def beerScaledownSecondary(realdist,secondaryPrefs,seedPrefs,locSeed):
    
    outersum=0.0
    
    # Randomly initialize the starting points of the locations in 2D
    loc={}
    fakedist={}
    grad={}
    for addedBeer in secondaryPrefs:
        loc[addedBeer]=[random.random(),random.random()]
        fakedist[addedBeer]={}
        for seedBeer in seedPrefs:
            fakedist[addedBeer][seedBeer]=[0.0]
    
    # Find projected distances
    for addedBeer in secondaryPrefs:
        for seedBeer in seedPrefs:
            # Absolute value of the difference between x1 and x2 and y1 and y2
            fakedist[addedBeer][seedBeer]=sqrt(sum([pow(loc[addedBeer][x]-locSeed[seedBeer][x],2) for x in range(len(loc[addedBeer]))]))
    
    totalerror=0
    num=0
    for k in secondaryPrefs:
        for j in seedPrefs:
            if k==j: continue
            if realdist[k][j]==-1: continue
            
            # The error is percent difference between the distances
            if fakedist[k][j]!=0:
                errorterm=(fakedist[k][j]-realdist[k][j])/fakedist[k][j]
            else:
                errorterm=1
            
            # Keep track of the total error
            totalerror+=pow(errorterm,2)
            num+=1
    
    if num>0:
        totalerror = sqrt(totalerror)/num
    print totalerror

    return loc

def beerDecreaseErrorSecondary(loc,realdist,secondaryPrefs,seedPrefs,locSeed,rate=0.01,tries=50):
    
    fakedist={}
    grad={}
    for addedBeer in secondaryPrefs:
        fakedist[addedBeer]={}
        for seedBeer in seedPrefs:
            fakedist[addedBeer][seedBeer]=[0.0]
    
    lasterror=None
    for m in range(0,tries):
        # Find projected distances
        for addedBeer in secondaryPrefs:
            for seedBeer in seedPrefs:
                # Absolute value of the difference between x1 and x2 and y1 and y2
                fakedist[addedBeer][seedBeer]=sqrt(sum([pow(loc[addedBeer][x]-locSeed[seedBeer][x],2) for x in range(len(loc[addedBeer]))]))
        
        # Move points
        for addedBeer in secondaryPrefs:
            grad[addedBeer]=[0.0,0.0]
        
        totalerror=0
        num=0
        for k in secondaryPrefs:
            for j in seedPrefs:
                if k==j: continue
                if realdist[k][j]==-1: continue
                
                # The error is percent difference between the distances
                if fakedist[k][j]!=0:
                    errorterm=(fakedist[k][j]-realdist[k][j])/fakedist[k][j]
                else:
                    errorterm=1
                
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                if fakedist[k][j]!=0:
                    grad[k][0]+=(locSeed[j][0]-loc[k][0])*errorterm
                    grad[k][1]+=(locSeed[j][1]-loc[k][1])*errorterm
                
                # Keep track of the total error
                totalerror+=pow(errorterm,2)
                num+=1
        if num>0:
            totalerror = sqrt(totalerror)/num
        
        # Status updates for large datasets
        if (m+1)%10000==0:
            print "%d / %d with error: %f" % (m,tries,totalerror)

#if lasterror and lasterror < totalerror: break
#lasterror = totalerror
        
        
        # Move each of the points by the learning rate times the gradient
        for beerK in secondaryPrefs:
            if num>0:
                loc[beerK][0]+=rate*(grad[beerK][0]/num)
                loc[beerK][1]+=rate*(grad[beerK][1]/num)

    return loc

def beerShiftAllClustersSecondary(loc,realdist,secondaryPrefs,seedPrefs,locSeed,rate=0.0001,tries=50):
    
    fakedist={}
    grad={}
    for addedBeer in secondaryPrefs:
        fakedist[addedBeer]={}
        for seedBeer in seedPrefs:
            fakedist[addedBeer][seedBeer]=[0.0]
    
    lowesterror=None
    lasterror=None
    for m in range(0,tries):
        # Find projected distances
        for addedBeer in secondaryPrefs:
            for seedBeer in seedPrefs:
                # Absolute value of the difference between x1 and x2 and y1 and y2
                fakedist[addedBeer][seedBeer]=sqrt(sum([pow(loc[addedBeer][x]-locSeed[seedBeer][x],2) for x in range(len(loc[addedBeer]))]))
        
        # Move points
        for addedBeer in secondaryPrefs:
            grad[addedBeer]=[0.0,0.0]
        
        totalerror=0
        num=0
        for k in secondaryPrefs:
            for j in seedPrefs:
                if k==j: continue
                if realdist[k][j]==-1: continue
                
                # The error is percent difference between the distances
                if fakedist[k][j]!=0:
                    errorterm=(fakedist[k][j]-realdist[k][j])/fakedist[k][j]
                else:
                    errorterm=1
                
                # Each point needs to be moved away from or towards the other
                # point in proportion to how much error it has
                if fakedist[k][j]!=0:
                    grad[k][0]+=(locSeed[j][0]-loc[k][0])*errorterm
                    grad[k][1]+=(locSeed[j][1]-loc[k][1])*errorterm
                
                # Keep track of the total error
                totalerror+=pow(errorterm,2)
                num+=1
        if num>0:
            totalerror = sqrt(totalerror)/num
        print totalerror
        
        #if lasterror and lasterror < totalerror: break
        #  lasterror = totalerror
        
        
        # Move each of the points by the learning rate times the gradient
        for beerK in secondaryPrefs:
            loc[beerK][0]+=rate*grad[beerK][0]
            loc[beerK][1]+=rate*grad[beerK][1]
    
    return loc

def draw2dBeerReclusterFix(loc,beerPrefs,seedPrefs,png='l_recluster_beerclusters2d_1.png', scale=8000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in beerPrefs:
        xfix=(loc[beer][0]/200000000) + 0.5
        yfix=(loc[beer][1]/200000000) + 0.5
        x=xfix*scale
        y=yfix*scale
        if beer in seedPrefs:
            draw.text((x,y),beer,(255,0,0))
        else:
            draw.text((x,y),beer,(0,0,0))
    img.save(png,'PNG')

def draw2dBeerRecluster(loc,beerPrefs,seedPrefs,png='l_recluster_beerclusters2d_1.png', scale=8000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in beerPrefs:
        x=((loc[beer][0]+0.5)*0.9)*scale/2
        y=((loc[beer][1]+0.5)*0.9)*scale/2
        if beer in seedPrefs:
            draw.text((x,y),beer,(255,0,0))
        else:
            draw.text((x,y),beer,(0,0,0))
    img.save(png,'PNG')

def draw2dBeerSecondary(loc,locSeed,secondaryPrefs,seedPrefs,png='l_secondary_beerclusters2d.png', scale=2000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for seedBeer in seedPrefs:
        x=(locSeed[seedBeer][0]+0.5)*scale/2
        y=(locSeed[seedBeer][1]+0.5)*scale/2
        draw.text((x,y),seedBeer,(255,0,0))
    for addedBeer in secondaryPrefs:
        x=(loc[addedBeer][0]+0.5)*scale/2
        y=(loc[addedBeer][1]+0.5)*scale/2
        draw.text((x,y),addedBeer,(0,0,0))
    img.save(png,'PNG')

def draw2dBeer(loc,itemPrefs,png='l_beerclusters2d.png', scale=2000):
    img=Image.new('RGB',(scale,scale),(255,255,255))
    draw=ImageDraw.Draw(img)
    for beer in itemPrefs:
        x=(loc[beer][0]+0.5)*scale/2
        y=(loc[beer][1]+0.5)*scale/2
        draw.text((x,y),beer,(0,0,0))
    img.save(png,'PNG')

import pickle

def pickleMyStuff(realdist,itemPrefs):
    pickle.dump(realdist, open("realdistsave.p","wb"))
    pickle.dump(itemPrefs, open("itemprefssave.p","wb"))

def unpickleMyStuff():
    realdist=pickle.load(open("realdistsave.p","rb"))
    itemPrefs=pickle.load(open("itemprefssave.p","rb"))
    return realdist,itemPrefs

def pickleMyPoints(loc,realdist,itemPrefs):
    pickle.dump(loc, open("locsave.p","wb"))
    pickle.dump(realdist, open("realdistsave.p","wb"))
    pickle.dump(itemPrefs, open("itemprefssave.p","wb"))

def unpickleMyPoints():
    loc=pickle.load(open("locsave.p","rb"))
    realdist=pickle.load(open("realdistsave.p","rb"))
    itemPrefs=pickle.load(open("itemprefssave.p","rb"))
    return loc,realdist,itemPrefs

def pickleSecondary(loc,locSeed,realdist,secondaryPrefs,seedPrefs,allItemPrefs):
    pickle.dump(loc, open("pickle/locsave.p","wb"))
    pickle.dump(locSeed, open("pickle/locseedsave.p","wb"))
    pickle.dump(realdist, open("pickle/realdistsave.p","wb"))
    pickle.dump(secondaryPrefs, open("pickle/secondaryprefssave.p","wb"))
    pickle.dump(seedPrefs, open("pickle/seedprefssave.p","wb"))
    pickle.dump(allItemPrefs, open("pickle/allitemprefssave.p","wb"))

def unpickleSecondary():
    loc=pickle.load(open("pickle/locsave.p","rb"))
    locSeed=pickle.load(open("pickle/locseedsave.p","rb"))
    realdist=pickle.load(open("pickle/realdistsave.p","rb"))
    secondaryPrefs=pickle.load(open("pickle/secondaryprefssave.p","rb"))
    seedPrefs=pickle.load(open("pickle/seedprefssave.p","rb"))
    allItemPrefs=pickle.load(open("pickle/allitemprefssave.p","rb"))
    return loc,locSeed,realdist,secondaryPrefs,seedPrefs,allItemPrefs

def pickleRecluster(loc,realdist,beerPrefs,seedPrefs):
    pickle.dump(loc, open("pickle/locsave.p","wb"))
    pickle.dump(realdist, open("pickle/realdistsave.p","wb"))
    pickle.dump(beerPrefs, open("pickle/beerprefssave.p","wb"))
    pickle.dump(seedPrefs, open("pickle/seedprefssave.p","wb"))

def unpickleRecluster():
    loc=pickle.load(open("pickle/locsave.p","rb"))
    realdist=pickle.load(open("pickle/realdistsave.p","rb"))
    beerPrefs=pickle.load(open("pickle/beerprefssave.p","rb"))
    seedPrefs=pickle.load(open("pickle/seedprefssave.p","rb"))
    return loc,realdist,beerPrefs,seedPrefs

def tryAllThisAgain():
    loc,itemPrefs=unpickleMyStuff()
    draw2dBeer(loc,itemPrefs,jpeg='l_beerclusters2d_3.png')