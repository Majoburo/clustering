import numpy as np
import random as rn
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import sys
from scipy.linalg import norm
from operator import attrgetter

eps = 800
MinPts = 5
dim = 2
global dist
cutoff_d=800

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
        self.delta = (100000,)

class Cluster:
    def __init__(self,i):
        selt.i = i
        self.center = Point
        self.border = []
        self.hborder = Point
        self.core = []

def DBSCAN():
    j=1
   # pdb.set_trace()
    for p in points:
        if not p.visited:
            p.visited = True
            regionQuery(p)
            if len(p.neighbors) < MinPts:
                p.noise = True
            else:
                expandCluster(p,j)
                j+=1

def expandCluster(p,j):
   # p.cluster = j
    neighbors2 = p.neighbors
    neighbors3 = neighbors2
    while any((p1.visited == False for p1 in neighbors2)):
        neighbors3 = neighbors2
        for p1 in neighbors3:
            if not p1.visited:
                p1.visited = True
                regionQuery(p1)
                if len(p1.neighbors) >= MinPts:
                    neighbors2 = p.neighbors|p1.neighbors
    for p3 in neighbors2:
        if p3.cluster == 0 and p3.noise == False:
            p3.cluster = j

def regionQuery(p):
    for p2 in points:
        if dist[p.i,p2.i] < eps:
            p.neighbors.add(p2)

# DBSCAN2 , following original paper http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.71.1980
def DBSCAN2():
    j=1
    for p in points:
        if p.cluster == 0 and not p.noise:
            if expandCluster2(p,j):
                j += 1
def expandCluster2(p,j):
    regionQuery(p)
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
            regionQuery(p1)
            if len(p1.neighbors) >= MinPts:
                for p2 in p1.neighbors:
                    if p2.cluster == 0:
                        if not p2.noise:
                            seeds.add(p2)
                        p2.cluster = j
        return True



'''
def update_plot(i,C,poss,scatt):
    scatt.set_offsets(poss[i][:])
    scatt.set_array(C)
    return scatt,
poss = NewtonVolve(pos,dist)
import pdb
pdb.set_trace()
scatt = plt.scatter(poss[0][:,0],poss[0][:,1],c=C)
fig = plt.figure()
ani = FuncAnimation(fig,update_plot,frames=range(timesteps),fargs=(C,poss,scatt))
plt.show()
cutoff_d=4
'''

def FIDEPE(dist,cutoff_d):
    for p1 in points:
        for p2 in points:
            if (dist[p1.i,p2.i]-cutoff_d) < 0:
                p1.density +=1

    for p1 in points:
        for p2 in points:
            if p2.density > p1.density and p1.delta[0] > dist[p1.i,p2.i]:
                p1.delta = (dist[p1.i,p2.i],p2.i)
        if p1.delta[0] == 100000:#if it is the point with the highest density
            p1.delta = (max(dist[p1.i,:]),p1)
    '''
    stddelta=np.std(p.delta[0] for p in points)
    stddensity=np.std(p.density for p in points)
    avgdelta=np.mean(p.delta[0] for p in points)
    avgdensity=np.mean(p.density for p in points)
    
    for p1 in points:
        if avgdelta+
    #if p1.delta[0] and p1.density are high
    #    clusters.append(Cluster(i))
    for p1 in points:
        p1.cluster_2 = p2.delta[1]

#find border
    for p1 in points:
        for p2 in points:
             if (dist[p1,p2]-cutoff_d) < 0 and p1.cluster_2 != p2.cluster_2:
                cluster_2.border= True

    for p1 in range(0,npart):
        for p2 in range(0,npart):
            if p1.cluster_2 == p2.cluster_2 and p1.border:
                clusterdensity = max(p2.density,p1.density)

    for p1 in range(0,npart):
        for p2 in range(0,npart):
            if p1.cluster_2 <= p2.cluster_2:
                clusterdensity
'''
    posi=np.array([(p.delta[0],p.density) for p in points])
    plt.scatter(posi[:,0],posi[:,1])
    plt.show()
#NewtonVolve(timesteps)
'''
pos = np.array([p.pos for p in points])
#C = np.array([p.i,p.cluster for p in points])
plt.scatter(pos[:,0],pos[:,1])#,c=C)
plt.show()
'''
#DBSCAN2()

#pos = np.array([p.pos for p in points])
#C = [p.cluster for p in points]
#print(C)
#plt.scatter(pos[:,0],pos[:,1],c=C,s=100)
#plt.show()



# try clustering a real dataset
import os.path
if not os.path.exists('positions.txt'):
    print("Didn't see test data locally, downloading...")
    URL = "https://cs.joensuu.fi/sipu/datasets/a1.txt"
    import requests
    r = requests.get(URL,verify=False)
    with open('a1.txt','wb') as f:
        f.write(r.content)
    print('downloaded test data')
f = open('positions.txt','r')
data = np.loadtxt(f)
f.close()

points=[]
for i in range(data.shape[0]):
    points.append(Point(i,pos=data[i,:]))
print("Loaded data for " + str(data.shape[0]) + " particles")
print("Computing distances")
dist = np.zeros((data.shape[0],data.shape[0]))
if not os.path.exists('dist.txt'):
    # recompute dist
    for p1 in points:
        for p2 in points:
            if p2 != p1 and dist[p2.i,p1.i] == 0:
                dist[p2.i,p1.i] = norm(p1.pos-p2.pos)
                dist[p1.i,p2.i] = dist[p2.i,p1.i]
    with open('dist.txt','wb') as f:
        np.savetxt(f,dist)
else:
    print('found saved distance matrix, loading')
    with open('dist.txt','r') as f:
        dist = np.loadtxt(f)
print("clustering")
FIDEPE(dist,cutoff_d)


'''DBSCAN2()

C = [p.cluster for p in points]
plt.scatter(data[:,0],data[:,1],c=C,s=20)
plt.show()
'''
