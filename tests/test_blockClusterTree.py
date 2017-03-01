import random
from unittest import TestCase

import numpy

from block_cluster_tree import BlockClusterTree
from cluster import Cluster
from cluster_tree import ClusterTree
from cuboid import Cuboid
from grid import Grid
from splitable import RegularCuboid
from utils import admissible


class TestBlockClusterTree(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim1 = 16
        cls.lim2 = 8
        cls.lim3 = 4
        cls.link_num = 4
        cls.points1 = [numpy.array([float(i) / cls.lim1]) for i in xrange(cls.lim1)]
        cls.links1 = [[cls.points1[l] for l in [random.randint(0, cls.lim1 - 1) for x in xrange(cls.link_num)]]
                      for i in xrange(cls.lim1)]
        cls.points2 = [numpy.array([float(i) / cls.lim2, float(j) / cls.lim2])
                       for i in xrange(cls.lim2) for j in xrange(cls.lim2)]
        cls.links2 = [[cls.points2[l] for l in [random.randint(0, cls.lim2 ** 2 - 1) for x in xrange(cls.link_num)]]
                      for j in xrange(cls.lim2) for i in xrange(cls.lim2)]
        cls.points3 = [numpy.array([float(i) / cls.lim3, float(j) / cls.lim3, float(k) / cls.lim3])
                       for i in xrange(cls.lim3) for j in xrange(cls.lim3) for k in xrange(cls.lim3)]
        cls.links3 = [[cls.points3[l] for l in [random.randint(0, cls.lim3 ** 3 - 1) for x in xrange(cls.link_num)]]
                      for k in xrange(cls.lim3) for j in xrange(cls.lim3) for i in xrange(cls.lim3)]
        cls.grid1 = Grid(cls.points1, cls.links1)
        cls.grid2 = Grid(cls.points2, cls.links2)
        cls.grid3 = Grid(cls.points3, cls.links3)
        cls.cluster1 = Cluster(cls.grid1)
        cls.cluster2 = Cluster(cls.grid2)
        cls.cluster3 = Cluster(cls.grid3)
        cls.cub1 = Cuboid(numpy.array([0]), numpy.array([1]))
        cls.cub2 = Cuboid(numpy.array([0, 0]), numpy.array([1, 1]))
        cls.cub3 = Cuboid(numpy.array([0, 0, 0]), numpy.array([1, 1, 1]))
        cls.rc1 = RegularCuboid(cls.cluster1)
        cls.rc2 = RegularCuboid(cls.cluster2)
        cls.rc3 = RegularCuboid(cls.cluster3)
        cls.ct1 = ClusterTree(cls.rc1, 1)
        cls.ct2 = ClusterTree(cls.rc2, 1)
        cls.ct3 = ClusterTree(cls.rc3, 1)
        cls.bct1 = BlockClusterTree(cls.ct1, cls.ct1, admissible_function=admissible)
        cls.bct2 = BlockClusterTree(cls.ct2, cls.ct2, admissible_function=admissible)
        cls.bct3 = BlockClusterTree(cls.ct3, cls.ct3, admissible_function=admissible)

    def test_depth(self):
        self.fail()

    def test_to_list(self):
        self.fail()

    def test_export(self):
        self.fail()
