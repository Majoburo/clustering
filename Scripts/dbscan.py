def DBSCAN():
    j=1
    for p in points:
        if p.cluster == 0 and not p.noise:
            if expandCluster(p,j):
                j += 1

def expandCluster(p,j):
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

