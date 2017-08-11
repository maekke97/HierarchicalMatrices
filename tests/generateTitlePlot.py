import math
from HierMat import *

lim2 = 2 ** 8
points2 = [(math.cos(2 * math.pi * (float(i)/lim2)),
            math.sin(2 * math.pi * (float(i)/lim2)))
           for i in xrange(lim2)]

links2 = [[points2[i + 1]] for i in xrange(len(points2) - 1)]
links2.append([points2[0]])

grid2 = Grid(points2, links2)
cluster2 = Cluster(grid2)
rc2 = RegularCuboid(cluster2, )
ct2 = build_cluster_tree(rc2)
bct2 = build_block_cluster_tree(ct2)
plot(bct2, 'plotUnitCircle256.png', face_color='#ffffff')
