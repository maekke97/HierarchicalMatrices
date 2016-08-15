from clustertree import ClusterTree, RegularCuboid, BlockClusterTree, export
from cluster import Cluster
import numpy as np
import random


lim1 = 2**6
lim2 = 2**6
lim3 = 2**6
link_num = 2
# points1 = [np.array([float(i) / lim1]) for i in xrange(lim1)]
# links1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
# points2 = [np.array([float(i) / lim1, float(j) / lim2]) for i in xrange(lim1) for j in xrange(lim2)]
# links2 = [[points2[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 2)) for x in xrange(link_num)]]
#           for j in xrange(lim2) for i in xrange(lim1)]
points3 = [np.array([float(i) / lim1, float(j) / lim2, float(k) / lim3]) for i in xrange(lim1) for j in
           xrange(lim2)
           for k in xrange(lim3)]
links3 = [
    [points3[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 1) * (lim3 - 1)) for x in xrange(link_num)]]
    for k in xrange(lim3) for j in xrange(lim2) for i in xrange(lim1)]
# ipoints1 = [np.array([i]) for i in xrange(lim1)]
# ilinks1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
# ipoints2 = [np.array([i, j]) for i in xrange(lim1) for j in xrange(lim2)]
# ilinks2 = [[points2[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 2)) for x in xrange(link_num)]]
#           for j in xrange(lim2) for i in xrange(lim1)]
# ipoints3 = [np.array([i, j, k]) for i in xrange(lim1) for j in
#            xrange(lim2)
#            for k in xrange(lim3)]
# ilinks3 = [
#     [points3[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 1) * (lim3 - 1)) for x in xrange(link_num)]]
#     for k in xrange(lim3) for j in xrange(lim2) for i in xrange(lim1)]

# cluster1 = Cluster(points1, links1)
# cluster2 = Cluster(points2, links2)
cluster3 = Cluster(points3, links3)
# rc1 = RegularCuboid(cluster1)
# rc2 = RegularCuboid(cluster2)
rc3 = RegularCuboid(cluster3)
# ct1 = ClusterTree(rc1, 1)
# ct2 = ClusterTree(rc2, 1)
ct3 = ClusterTree(rc3, 1)
# icluster1 = Cluster(ipoints1, ilinks1)
# icluster2 = Cluster(ipoints2, ilinks2)
# icluster3 = Cluster(ipoints3, ilinks3)
# irc1 = RegularCuboid(icluster1)
# irc2 = RegularCuboid(icluster2)
# irc3 = RegularCuboid(icluster3)
# ict1 = ClusterTree(irc1, 1)
# ict2 = ClusterTree(irc2, 1)
# ict3 = ClusterTree(irc3, 1)
# bct1 = BlockClusterTree(ct1, ct1)
# ibct1 = BlockClusterTree(ict1, ict1)
# bct2 = BlockClusterTree(ct2, ct2)
# ibct2 = BlockClusterTree(ict2, ict2)
bct3 = BlockClusterTree(ct3, ct3)
# ibct3 = BlockClusterTree(ict3, ict3)
