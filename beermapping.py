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

# Draw the beer clusters
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

# Beer Mapping -------------------------------------------------------------------------------------------------------------------->

# Calculate beer similarity and beer distance between all beers
def loadDistAndSim(allItemPrefs):
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
    pickle.dump(realsim, open("pickle/realsim.p","wb"))
    pickle.dump(realdist, open("pickle/realdist.p","wb"))
    return realsim,realdist

# Load ABV data for each beer
def loadABV(path='data'):
    # example line from beerabv.txt would be "Lagunitas IPA\t5.7" (beer,abv)
    abvData={}
    for line in open(path+'/beerabv.txt'):
        (name,abv)=line.split('\t')
        beer=name.rstrip()
        abvData[beer]=float(abv)
    return abvData

# Load IBU data for each beer
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

# Load Specific Gravity data for each beer (relates to how sweet/thick the beer is)
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

def loadBeerData():
    # Reload data from save point
    loc=pickle.load(open("pickle/964_loc.p","rb"))
    itemPrefs=pickle.load(open("pickle/964_seedprefs.p","rb"))
    realsim=pickle.load(open("pickle/realsim.p","rb"))
    realdist=pickle.load(open("pickle/realdist.p","rb"))
    abv=loadABV()
    ibu=loadIBU()
    gravity=loadGravity()
    style=loadStyle()
    
    return loc,itemPrefs,realsim,realdist,abv,ibu,gravity,style

def findBeerCenter():
    loc,itemPrefs,realsim,realdist,abv,ibu,gravity,style=loadBeerData()
    
    beer1=raw_input("Enter a beer you like: ")
    beer2=raw_input("Enter another beer you like: ")
    beer3=raw_input("Enter one more beer you like: ")
    
    locCenter={}
    locCenter[0]=(loc[beer1][0]+loc[beer2][0]+loc[beer3][0])/3
    locCenter[1]=(loc[beer1][1]+loc[beer2][1]+loc[beer3][1])/3
    
    highlightBeers={}
    highlightBeers[beer1]=1
    highlightBeers[beer2]=1
    highlightBeers[beer3]=1

    beerMapYourPrefs(loc,itemPrefs,style,center=locCenter,highlights=highlightBeers)

def findBeerCenterFast(beer1,beer2,beer3):
    loc,itemPrefs,realsim,realdist,abv,ibu,gravity,style=loadBeerData()
    
    locCenter={}
    locCenter[0]=(loc[beer1][0]+loc[beer2][0]+loc[beer3][0])/3
    locCenter[1]=(loc[beer1][1]+loc[beer2][1]+loc[beer3][1])/3
    
    highlightBeers={}
    highlightBeers[beer1]=1
    highlightBeers[beer2]=1
    highlightBeers[beer3]=1
    
    beerMapYourPrefs(loc,itemPrefs,style,center=locCenter,highlights=highlightBeers)

def findBeerCenterFour(beer1,beer2,beer3,beer4):
    loc,itemPrefs,realsim,realdist,abv,ibu,gravity,style=loadBeerData()
    
    locCenter={}
    locCenter[0]=(loc[beer1][0]+loc[beer2][0]+loc[beer3][0]+loc[beer4][0])/4
    locCenter[1]=(loc[beer1][1]+loc[beer2][1]+loc[beer3][1]+loc[beer4][1])/4
    
    highlightBeers={}
    highlightBeers[beer1]=1
    highlightBeers[beer2]=1
    highlightBeers[beer3]=1
    highlightBeers[beer4]=1
    
    beerMapYourPrefs(loc,itemPrefs,style,center=locCenter,highlights=highlightBeers)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Universal create and upload new beer map to server
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

def makeBeerMap(beerPrefs, loc, style, textColor, backgroundColor, filename="beergmap", showCenter=False, center=None, highlights=None):
    out=open("data/%s.html" % (filename),'w')
    out.write(generateGoogleJavascript(beerPrefs, loc, style, textColor, backgroundColor, showCenter=showCenter, center=center, highlights=highlights))
    out.close()
    site,username,password = loadFTPCredentials()
    session = ftplib.FTP(site,username,password)
    file = open("data/%s.html" % (filename),'rb')                  # file to send
    session.storbinary("STOR beermap/%s.html" % (filename), file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print 'gmap uploaded'

def writeBeerMapData(beerPrefs, loc, style, textColor, backgroundColor, showCenter=False, center=None, highlights=None):
    output = []
    for beer in beerPrefs:
        if highlights!=None and highlights.has_key(beer):
            # skip this beer so we can add it last to be on top
            print beer
        else:
            output.append(outputBeer(loc[beer],beer,style[beer],textColor[beer],backgroundColor[beer]))

    if highlights!=None:
        # add this beer last so it shows up on top
        for beer in highlights:
            output.append(outputBeer(loc[beer],beer,style[beer],textColor[beer],backgroundColor[beer]))
        
        if showCenter==True and center != None:
            output.append(outputBeer(center,"YOU","Center of Preference","#000000","rgb(255,46,46)"))
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

def scaleColor(colorInput, scaling):
    return int(scaling * (255-colorInput) + colorInput)

def outputBeer(coords,beer,style,textColor,backgroundColor):
    
    return '\t{latitude: '+str(float((coords[0])*15))+', '\
        +'longitude: '+str(float((coords[1])*15))+', '\
            +'title: "'+str(beer)+'", '\
            +'description: "'+str(style.replace("&","and"))+'",'\
            +'textColor: "'+str(textColor)+'", '\
            +'color: "'+str(backgroundColor)+'"},\n'

def generateGoogleJavascript(beerPrefs, loc, style, textColor, backgroundColor, showCenter=False, center=None, highlights=None):
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
    out.append(writeBeerMapData(beerPrefs, loc, style, textColor, backgroundColor, showCenter=showCenter, center=center, highlights=highlights))
    out.append('];\n')
    
    out.append('\n')
    if center==None:
        out.append('var mapCenter = new google.maps.LatLng(11.9807276122, 4.99151301465);\n')
        out.append('var zoomStart = 5;\n')
    else:
        out.append('var mapCenter = new google.maps.LatLng('+str(float((center[0])*15))+', '+str(float((center[1])*15))+');\n')
        out.append('var zoomStart = 6;\n')
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


#
#
#
#
#
#
#
#

##

#
#
#
#
#




# Create and upload new beer map to server with your beer preferences based on 3 beers
def beerMapYourPrefs(loc,seedPrefs,style,center=None,highlights=None):
    
    backgroundColor = {}
    textColor = {}
    
    for beer in seedPrefs:
        if highlights!=None and highlights.has_key(beer):
            # color this beer red to highlight it
            backgroundColor[beer] = 'rgb(255,46,46)'
        else:
            if highlights!=None:
                gravity = 0
                for otherBeer in highlights:
                    distance = sqrt(sum([pow(loc[beer][x]-loc[otherBeer][x],2) for x in range(len(loc[beer]))]))
                    gravity += 1/(pow(distance,2))
                gravity = gravity/len(highlights)
                r, g, b = rgbMix(gravity, 16, 1, red=255, green=109, blue=87)
                backgroundColor[beer] = 'rgb('+str(r)+','+str(g)+','+str(b)+')'
            else:
                backgroundColor[beer] = 'rgb(255,255,255)'
        textColor[beer] = '#000000'
    
    if center != None:
        makeBeerMap(seedPrefs, loc, style, textColor, backgroundColor, filename="beergmapuser", showCenter=True, center=center, highlights=highlights)
    else:
        makeBeerMap(seedPrefs, loc, style, textColor, backgroundColor, filename="beergmapuser", showCenter=False, center=center, highlights=highlights)











