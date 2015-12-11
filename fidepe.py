import numpy as np
import random as rn
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import sys
from scipy.linalg import norm
from operator import attrgetter

eps = 1
MinPts = 5
dim = 2
global dist
#for FIDEPE
cutoff_d= 3
nstddelta=5
nstddensity=0


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
#calculating minimun distances to overdense points
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
    #print stddelta,avgdelta,stddensity,avgdensity
    i=0
    for p in points:
        if avgdelta+nstddelta*stddelta < p.delta[0] and avgdensity+nstddensity*stddensity < p.density:
            i+=1
            cluster=Cluster(i)
            cluster.center = p
            clusters.append(cluster)
            p.cluster_2 = i
            cluster.points.add(p)
    #find unclassified points and append them to a cluster
    query=0
    while query == 0:
        query=1
        for p in points:
            query = query*p.cluster_2
            if p.cluster_2 == 0 and p.delta[1].cluster_2 > 0:
                cluster.points.add(p)
                p.cluster_2=p.delta[1].cluster_2
                '''
# mark points as noise
    dc = 1 # test value
    # find border points
    for cluster in clusters:
        for p in cluster.points:
            print p.i
            regionQuery(p,dc)
            if any((p1.cluster_2 != p.cluster_2 for p1 in p.neighbors)):
                cluster.border.add(p)
        print "/n"
        print [a.i for a in cluster.border]
    #print [cluster.border.i for cluster in clusters]
       # highest density in border
        if len(cluster.border) == 0:
            for p1 in cluster.points:
                p1.cluster_2 = 0
                p1.noise = True
            cluster.points = {}
            continue

        bord_density_max = max((p1.density for p1 in cluster.border))
        points2 = cluster.points.copy()
        for p1 in cluster.points:
            if p1.density > bord_density_max:
                cluster.core.add(p1)
                points2.remove(p1)
            else:
                points2.remove(p1)
                p1.cluster_2 = 0
                p1.noise = True
        cluster.points = points2
        '''
    posi=np.array([(p.delta[0],p.density) for p in points])
    plt.scatter(posi[:,0],posi[:,1])
    plt.show()
