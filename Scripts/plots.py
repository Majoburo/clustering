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
cutoff_d= 3
nstddelta = 5
nstddensity = 0

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

# try to get two clusters
'''for i in range(0,int(len(points)/3)):
    points[i].pos = np.array([-20,-20])+np.random.rand(2)
for i in range(int(2 * len(points)/3),len(points)):
    points[i].pos = np.array([30,20])+np.random.rand(2)
'''
def NewtonVolve(timesteps):
    f = open('positions.txt','w')
    for t in range(0,timesteps):
        global dist
        dist=dist*0
        force = np.zeros(dim)
        # move particles
        for p in points:
            p.pos += p.totforce * G_eff
        # compute forces and distances
        for p1 in points:
            for p2 in points:
                if p2 != p1 and dist[p2.i,p1.i] == 0:
                    dist[p2.i,p1.i] = norm(p1.pos-p2.pos)
                    dist[p1.i,p2.i] = dist[p2.i,p1.i]
                if dist[p2.i,p1.i] > 0.1:
                    force = (p2.pos-p1.pos)/(dist[p2.i,p1.i]**3)
                    p1.totforce += force
    for p1 in points:
        f.write('%s %s \n' %(p1.pos[0],p1.pos[1]))
    f.close()

def regionQuery(p, eps=eps):
    for p2 in points:
        if dist[p.i,p2.i] < eps:
            p.neighbors.add(p2)

# DBSCAN2 , following original paper http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.71.1980
def DBSCAN2(eps,MinPts):
    j=1
    for p in points:
        if p.cluster == 0 and not p.noise:
            if expandCluster2(p,j,eps,MinPts):
                j += 1
def expandCluster2(p,j,eps,MinPts):
    regionQuery(p,eps)
    seeds = p.neighbors
    if len(seeds) < MinPts:
        p.cluster = 0
        p.noise = True
        return False
    else:
        for seed in seeds:
            seed.cluster = j
        seeds.remove(p)
        while len(seeds) > 0 :
            p1 = seeds.pop()
            regionQuery(p1,eps)
            if len(p1.neighbors) >= MinPts:
                for p2 in p1.neighbors:
                    if p2.cluster == 0:
                        if not p2.noise:
                            seeds.add(p2)
                        p2.cluster = j
        return True

from fidepe2 import FIDEPE

# try clustering a real dataset

datasets = ['Aggregation',#'spiral','Aggregation','jain','Compound'] #'D31'
right_cluster_num = [3,7,3,5] # 31
import os


for dataset in datasets:
    posfname = '{0}_positions.txt'.format(dataset)
    if not os.path.exists(posfname):
        print("Didn't see test data locally, downloading...")
        URL = "https://cs.joensuu.fi/sipu/datasets/{0}.txt".format(dataset)
        import requests
        r = requests.get(URL,verify=False)
        with open(posfname,'wb') as f:
            f.write(r.content)
        print('downloaded test data')
    f = open(posfname,'r')
    data = np.loadtxt(f)
    f.close()

#    fig = plt.figure()
#    fig.suptitle(dataset)
#    plt.scatter(data[:,0],data[:,1],c=data[:,2],s=100)
#    plt.show()
#    plt.close()
#    continue

    points=[]
    for i in range(data.shape[0]):
        points.append(Point(i,pos=data[i,:]))
    print("Loaded data for " + str(data.shape[0]) + " particles")
    print("Computing distances")
    dist = np.zeros((data.shape[0],data.shape[0]))
    distfname = '{0}_dist.txt'.format(dataset)
    if not os.path.exists(distfname):
        # recompute dist
        for p1 in points:
            for p2 in points:
                if p2 != p1 and dist[p2.i,p1.i] == 0:
                    dist[p2.i,p1.i] = norm(p1.pos-p2.pos)
                    dist[p1.i,p2.i] = dist[p2.i,p1.i]
        with open(distfname,'wb') as f:
            np.savetxt(f,dist)
    else:
        print('found saved distance matrix, loading')
        with open(distfname,'r') as f:
            dist = np.loadtxt(f)
    print("clustering")

    pos = np.array([p.pos for p in points])


    max_stddelta = 7
    max_cutoff_d = 10
    data = np.zeros((max_cutoff_d,max_stddelta))
#    max_eps = 8.5
#    max_minpts = 15
#    data = np.zeros((2*max_eps-2,max_minpts-2),dtype=int)

    with open('contour.txt','w') as f:
        i=0
        for e in np.arange(0,max_cutoff_d):
        #for e in np.arange(1,max_eps,0.5):
            j=0
            for d in np.arange(0,max_stddelta):
            #for d in np.arange(2,max_minpts):
                for p in points:
                    p.neighbors = set([])
                    p.cluster = 0
                    p.noise = False
                    p.density = 0
                    p.delta = [100000,0]
                    p.cluster_2 = 0
                    p.hborder = False
                    p.clcenter = False

                #DBSCAN2(e,d)
                FIDEPE(points,dist,e,d,nstddensity)
                C = [p.cluster_2 for p in points]
                #fig = plt.figure()
                #fig.suptitle('cutoff_d: {0}, Clusters: {1}, stddelta: {2},stddensity {3}'.format(cutoff_d,len(set(C)),d,nstddensity), fontsize=14, fontweight='bold')
                data[i,j] = len(set(C))
                f.write("{} {} {}\n".format(e,d,data[i,j]))
                #plt.scatter(pos[:,0],pos[:,1],c=C)
                #plt.savefig('frame{0:02d}'.format(i))
                #plt.close()
                j+=1
            i+=1
    #plt.contour(np.arange(1,max_cutoff_d),np.arange(0,max_stddelta),data)
    #plt.xlabel('stddelta'); plt.ylabel('Number of Clusters')
    #plt.savefig('fidepe_plots/{0}_contour'.format(dataset))
    #plt.close()

    #os.system('convert -delay 100 -loop 0 *.png fidepe_plots/{0}.gif; rm frame*.png'.format(dataset))
    #os.system('gnuplot -e "passedtitle=\'{0}\'" contour.gnu'.format(dataset))
    os.system('mv contour.txt ../plots/fidepe/{0}.dat'.format(dataset))
    #os.system('mv plot.ps ../plots/fidepe/{0}_contour.ps'.format(dataset))
    #os.system('ps2pdf ../plots/fidepe/{0}_contour.ps ../plots/fidepe/{0}_contour.pdf'.format(dataset))
    #os.system('pdfcrop ../plots/fidepe/{0}_contour.pdf'.format(dataset))
