nstddelta = 5
nstddensity = 0
import numpy as np
dim =2

class Point:
    def randomize_pos(self):
        self.pos = np.random.uniform(-size,size,dim)
    def __init__(self,i,pos=np.zeros(dim)):
        self.i = i
        self.neighbors = set()
        self.pos = pos
        self.totforce = np.zeros(dim)
        self.cluster = 0
        self.visited = False
        self.noise = False
#specifics for FIDECE
        self.density = 0
        self.delta = [100000,self]
        self.cluster_2 = 0
        self.hborder = False
        self.clcenter = False

class Cluster:
    def __init__(self,i):
        self.i = i
        self.center = Point
        self.border = set()
        self.hborder = Point
        self.core = set()
        self.points = set()


def FIDEPE(points,dist,cutoff_d,nstddelta=nstddelta,nstddensity=nstddensity):

#calculating local densities
    for p1 in points:
        for p2 in points:
            if dist[p1.i,p2.i] < cutoff_d:
                p1.density +=1

#calculating minimum distances to overdense points
    for p1 in points:
        for p2 in points:
            if p2.density > p1.density and p1.delta[0] > dist[p1.i,p2.i]:
                p1.delta = [dist[p1.i,p2.i],p2]
        if p1.delta[0] == 100000:#if it is the point with the highest density
            p1.delta = [max(dist[p1.i,:]),p1]

#identifying cluster centers
    clusters=[]
    stddelta=np.std([p.delta[0] for p in points])
    stddensity=np.std([p.density for p in points])
    avgdelta=np.mean([p.delta[0] for p in points])
    avgdensity=np.mean([p.density for p in points])
    i=0
    for p in points:
        if avgdelta+nstddelta*stddelta < p.delta[0] and avgdensity+nstddensity*stddensity < p.density:
            i+=1
            p.cluster_2 = i
            cluster = Cluster(i)
            cluster.points.add(p)
            clusters.append(cluster)
    #find unclassified points and append them to a cluster
    if len(clusters) == 0:
       return

#find unclassified points and append them to a cluster
    query=0
    while query == 0:
        query=1
        for p in points:
            query = query*p.cluster_2
            if p.cluster_2 == 0 and p.delta[1].cluster_2 > 0:
                p.cluster_2=p.delta[1].cluster_2
