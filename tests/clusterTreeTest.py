from clustertree import ClusterTree, RegularCuboid
from cluster import Cluster
import numpy as np
import random


lim1 = 2**4
lim2 = 2**3
lim3 = 2**2
link_num = 4
points1 = [np.array([float(i) / lim1]) for i in xrange(lim1)]
links1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
points2 = [np.array([float(i) / lim1, float(j) / lim2]) for i in xrange(lim1) for j in xrange(lim2)]
links2 = [[points2[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 2)) for x in xrange(link_num)]]
          for j in xrange(lim2) for i in xrange(lim1)]
points3 = [np.array([float(i) / lim1, float(j) / lim2, float(k) / lim3]) for i in xrange(lim1) for j in
           xrange(lim2)
           for k in xrange(lim3)]
links3 = [
    [points3[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 1) * (lim3 - 1)) for x in xrange(link_num)]]
    for k in xrange(lim3) for j in xrange(lim2) for i in xrange(lim1)]
cluster1 = Cluster(points1, links1)
cluster2 = Cluster(points2, links2)
cluster3 = Cluster(points3, links3)
rc1 = RegularCuboid(cluster1)
rc2 = RegularCuboid(cluster2)
rc3 = RegularCuboid(cluster3)
ct1 = ClusterTree(rc1, 2)
ct2 = ClusterTree(rc2, 2)
ct3 = ClusterTree(rc3, 2)
