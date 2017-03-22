import math
import sys

import numpy as np

from block_cluster_tree import build_block_cluster_tree
from cluster import Cluster
from cluster_tree import build_cluster_tree
from grid import Grid
from splitable import RegularCuboid

sys.setrecursionlimit(200)


link_num = 2
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
# lim1 = 2 ** 8
# points1 = [np.array([float(i) / lim1]) for i in xrange(lim1)]
# links1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
# grid1 = Grid(points1, links1)
# cluster1 = Cluster(grid1)
# rc1 = RegularCuboid(cluster1)
# ct1 = build_cluster_tree(rc1)
# ct1.export('dot', 'out1.dot')
# bct1 = build_block_cluster_tree(ct1)
# bct1.plot('plot_test1.png')

lim2 = 2 ** 8
# points2left = [np.array([0, float(i)/lim2]) for i in xrange(lim2)]
# points2top = [np.array([float(i)/lim2, 1]) for i in xrange(lim2)]
# points2right = [np.array([1, float(i)/lim2]) for i in xrange(lim2, 0, -1)]
# points2bottom = [np.array([float(i)/lim2, 0]) for i in xrange(lim2, 0, -1)]
# points2 = points2left + points2top + points2right + points2bottom

points2 = [np.array([math.cos(2 * math.pi * i / lim2), math.sin(2 * math.pi * i / lim2)]) for i in xrange(lim2)]

links2 = [[points2[i + 1]] for i in xrange(len(points2) - 1)]
links2.append([points2[0]])

grid2 = Grid(points2, links2)
cluster2 = Cluster(grid2)
rc2 = RegularCuboid(cluster2, )
ct2 = build_cluster_tree(rc2)
ct2.export('dot', 'out2.dot')
bct2 = build_block_cluster_tree(ct2)
# bct2.export('dot', 'out2.dot')
bct2.plot('plotUnitCircle256.png', face_color='#ffffff')

# lim3 = 2 ** 2
# points3 = [np.array([float(i) / lim3, float(j) / lim3, float(k) / lim3]) for i in xrange(lim3) for j in
#            xrange(lim3) for k in xrange(lim3)]
# links3 = [[points3[l] for l in [random.randint(0, (lim3 - 1) ** 3) for x in xrange(link_num)]]
#           for k in xrange(lim3) for j in xrange(lim3) for i in xrange(lim3)]
# grid3 = Grid(points3, links3)
# cluster3 = Cluster(grid3)
# rc3 = RegularCuboid(cluster3)
# ct3 = build_cluster_tree(rc3)
# bct3 = build_block_cluster_tree(ct3)
# bct3.plot('plot_test3.png')
# icluster1 = Cluster(ipoints1, ilinks1)
# icluster2 = Cluster(ipoints2, ilinks2)
# icluster3 = Cluster(ipoints3, ilinks3)
# irc1 = RegularCuboid(icluster1)
# irc2 = RegularCuboid(icluster2)
# irc3 = RegularCuboid(icluster3)
# ict1 = ClusterTree(irc1, 1)
# ict2 = ClusterTree(irc2, 1)
# ict3 = ClusterTree(irc3, 1)
# ibct1 = BlockClusterTree(ict1, ict1)
# ibct2 = BlockClusterTree(ict2, ict2)

# bct3.plot()
# ibct3 = BlockClusterTree(ict3, ict3)
