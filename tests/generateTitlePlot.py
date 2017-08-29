import math
from HierMat import *
import numpy

lim = 2 ** 8
points = [(math.cos(2 * math.pi * (float(i) / lim)),
           math.sin(2 * math.pi * (float(i) / lim)))
          for i in xrange(lim)]

supports = {points[index]: (points[index-1], points[index + 1]) for index in xrange(1, lim-1)}
supports[points[0]] = (points[-1], points[1])
supports[points[-1]] = (points[-2], points[0])

grid = Grid(points, supports)
cluster = Cluster(grid)
reg_cub = RegularCuboid(cluster)
cluster_tree = build_cluster_tree(reg_cub)
block_cluster_tree = build_block_cluster_tree(cluster_tree)
plot(block_cluster_tree, 'plotUnitCircle256.png', face_color='#ffffff')


class TriCuboid(Cuboid):
    def split(self, axis=None):
        if axis:
            index = axis
        else:
            # determine dimension in which to restructure
            index = numpy.argmax(abs(self.high_corner - self.low_corner))
        # determine value at splitting point
        split1 = self.low_corner[index] + (self.high_corner[index] - self.low_corner[index]) / 3
        split2 = self.low_corner[index] + 2 * (self.high_corner[index] - self.low_corner[index]) / 3
        low_corner1 = numpy.array(self.low_corner)
        low_corner2 = numpy.array(self.low_corner)
        low_corner3 = numpy.array(self.low_corner)
        low_corner2[index] = split1
        low_corner3[index] = split2
        high_corner1 = numpy.array(self.high_corner)
        high_corner2 = numpy.array(self.high_corner)
        high_corner3 = numpy.array(self.high_corner)
        high_corner1[index] = split1
        high_corner2[index] = split2
        return TriCuboid(low_corner1, high_corner1), TriCuboid(low_corner2, high_corner2), TriCuboid(low_corner3,
                                                                                                     high_corner3)


class TriRegularCuboid(RegularCuboid):
    def split(self):
        cub1, cub2, cub3 = self.cuboid.split()
        indices1 = []
        indices2 = []
        indices3 = []
        for index in self.cluster.indices:
            if self.cluster.grid.points[index] in cub1:
                indices1.append(index)
            elif self.cluster.grid.points[index] in cub2:
                indices2.append(index)
            else:
                indices3.append(index)
        outs = []
        if len(indices1) > 0:
            cluster1 = Cluster(self.cluster.grid, indices1)
            tri_rc1 = TriRegularCuboid(cluster1, cub1)
            outs.append(tri_rc1)
        if len(indices2) > 0:
            cluster2 = Cluster(self.cluster.grid, indices2)
            tri_rc2 = TriRegularCuboid(cluster2, cub2)
            outs.append(tri_rc2)
        if len(indices3) > 0:
            cluster3 = Cluster(self.cluster.grid, indices3)
            tri_rc3 = TriRegularCuboid(cluster3, cub3)
            outs.append(tri_rc3)
        return outs


tri_cub = TriCuboid((-1, -1), (1, 1))
tri_reg_cub = TriRegularCuboid(cluster, tri_cub)
tri_cluster_tree = build_cluster_tree(tri_reg_cub)
export(tri_cluster_tree, form='dot', out_file='tri_ct.dot')
tri_block_cluster_tree = build_block_cluster_tree(tri_cluster_tree)
plot(tri_block_cluster_tree, 'plotUnitCircleTri256.png')



