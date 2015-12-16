import numpy as np
import random as rn
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import sys
from scipy.linalg import norm

print("\n---Welcome human---")
size = int(sys.argv[1]) #input("Please state the size of your square world: ")
npart = int(sys.argv[2]) # input("How many particles populate your world? ")
eps = float(sys.argv[3])
cutoff_d = 3
timesteps = 20
MinPts = 3
dim = 2
dist = np.zeros((npart,npart))
G_eff = 5e-3

print("Let me Newton-volve your system...")

class Point:
    def randomize_pos(self):
        self.pos = np.random.uniform(-size,size,dim)
    def __init__(self,i):
        self.i = i
        self.neighbors = []
        self.pos = np.zeros(dim)
        self.totforce = np.zeros(dim)
        self.cluster = 0
        self.visited = False
        self.visitedN = False
        self.noise = False
#specifics for FIDECE
        self.density = 0
        self.delta = (1000,)

class Cluster:
    def __init__(self,i):
        selt.i = i
        self.center = Point
        self.border = []
        self.hborder = Point
        self.core = []

points = [Point(i) for i in range(npart)]
for p in points:
    p.randomize_pos()

# try to get two clusters
for i in range(0,len(points)/3):
    points[i].pos = np.array([-20,-20])+np.random.rand(2)
for i in range(2 * len(points)/3,len(points)):
    points[i].pos = np.array([30,20])+np.random.rand(2)

def NewtonVolve(timesteps):
    for t in range(0,timesteps):
		# compute forces
		global dist
		dist=dist*0
		force = np.zeros(dim)
		for p1 in points:
			for p2 in points:
				if p2 != p1 and dist[p2.i,p1.i] == 0:
					dist[p2.i,p1.i] = norm(p1.pos-p2.pos)
					dist[p1.i,p2.i] = dist[p2.i,p1.i]
                if dist[p2.i,p1.i] > 0.1:
					force = (p2.pos-p1.pos)/(dist[p2.i,p1.i]**3)
					p1.totforce += force
		# move particles
		for p in points:
			p.pos += p.totforce * G_eff

import pdb
def DBSCAN():
    j=1
    pdb.set_trace()
    for p in points:
        if not p.visited:
            p.visited = True
            regionQuery(p)
            if len(p.neighbors) < MinPts:
                p.noise = True
            else:
                p.cluster = j
            	expandCluster(p,j)
def expandCluster(p,j):
    neighbors2 = []
    for p1 in p.neighbors:
        if not p1.visitedN:
            p1.visitedN = True
            regionQuery(p1)
            if len(p1.neighbors) >= MinPts:
                neighbors2 = list(set(p.neighbors)&set(p1.neighbors))

    for p1 in p.neighbors: p1.visitedN = False

    p.neighbors = neighbors2
    for p1 in neighbors2:
        if p1.cluster == 0 and p1.noise == False:
            p1.cluster == j


def regionQuery(p):
    for p2 in points:
        if dist[p.i,p2.i] < eps:
            p.neighbors.append(p2)

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
'''
def FIDEPE(dist,cutoff_d):
	for p1 in points:
		for p2 in points:
					if (norm(p1.pos-p2.pos)-cutoff_d) < 0:
				p1.density +=1
			
	for p1 in points:
		for p2 in points:
			if p2.density > p1.density:
						dist = norm(p1.pos-p2.pos)
				if dist < p1.delta[0]:
					p1.delta = dist,p2
		if p1.delta[0] == 1000:#if it is the point with the highest density
			p1.delta = max(dist[p1,:]),p1
	#if p1.delta[0] and p1.density are high
	#	clusters.append(Cluster(i))
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

NewtonVolve(timesteps)
'''
pos = np.array([p.pos for p in points])
#C = np.array([p.i,p.cluster for p in points])
plt.scatter(pos[:,0],pos[:,1])#,c=C)
plt.show()
'''
DBSCAN()

pos = np.array([p.pos for p in points])
C = [p.cluster for p in points]
print(C)
plt.scatter(pos[:,0],pos[:,1],c=C,s=100)
plt.show()

