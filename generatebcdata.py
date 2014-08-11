import recommendations
import clusters

def allBeerDistances(itemPrefs,beer,distance=pearson):

    scores=[distance(itemPrefs,beer,other),other) for other in itemPrefs if other!=beer]
    
    # Sort the list so the closest matches appear at the top
    scores.sort()
    return scores

def calculateAllBeerDistances(prefs):
    # Create a dictionary of items showing distance to every other item
    result={}

    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
        # Status updates for large datasets
        c+=1
        if c%100==0: print "%d / %d" % (c,len(itemPrefs))
        # Find the distance from every item to this one
        scores=allBeerDistances(itemPrefs,item,distance=pearson)
        result[item]=scores
    return result


def beerhcluster(rows, distance=pearson):
    
    prefs=recommendations.loadLatestBeerChooser()
    beersim=recommendations.calculateAllBeerDistances(prefs)
    # beersim[thisBeer]=[(distance,otherBeer) for otherBeer in prefs if otherBeer!=thisBeer]

    distances={}
    currentclustid=-1

    # Clusters are initially just the rows
    clust=[bicluster(rows[i],id=i) for i in range(len(rows))]

    beersim=get
    while len(clust)>1:
        lowestpair=(0,1)
        closest=distance(clust[0].vec,clust[1].vec)

