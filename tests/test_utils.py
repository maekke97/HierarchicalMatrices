from unittest import TestCase
from utils import BlockClusterTree, ClusterTree, RegularCuboid, Cluster, Grid, admissible
import numpy
import random


class TestUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim1 = 16
        cls.lim2 = 4
        cls.lim3 = 2
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
        cls.rc1 = RegularCuboid(cls.cluster1)
        cls.rc2 = RegularCuboid(cls.cluster2)
        cls.rc3 = RegularCuboid(cls.cluster3)
        cls.ct1 = ClusterTree(cls.rc1, 1)
        cls.ct2 = ClusterTree(cls.rc2, 1)
        cls.ct3 = ClusterTree(cls.rc3, 1)
        cls.bct1 = BlockClusterTree(cls.ct1, cls.ct1, admissible_function=admissible)
        cls.bct2 = BlockClusterTree(cls.ct2, cls.ct2, admissible_function=admissible)
        cls.bct3 = BlockClusterTree(cls.ct3, cls.ct3, admissible_function=admissible)
    
    def test_setup1(self):
        self.assertIsInstance(self.grid1, Grid)
        self.assertIsInstance(self.cluster1, Cluster)
        self.assertIsInstance(self.ct1, ClusterTree)
        self.assertIsInstance(self.bct1, BlockClusterTree)

    def test_setup2(self):
        self.assertIsInstance(self.cluster2, Cluster)

    def test_setup3(self):
        self.assertIsInstance(self.cluster3, Cluster)

    def test_diameter1(self):
        check1 = numpy.linalg.norm(numpy.array([float(self.lim1 - 1) / self.lim1]))
        self.assertEquals(self.cluster1.diameter(), check1)

    def test_diameter2(self):
        check2 = numpy.linalg.norm(numpy.array([float(self.lim2 - 1) / self.lim2, float(self.lim2 - 1) / self.lim2]))
        self.assertEquals(self.cluster2.diameter(), check2)

    def test_diameter3(self):
        check3 = numpy.linalg.norm(numpy.array([float(self.lim3 - 1) / self.lim3, float(self.lim3 - 1) / self.lim3,
                                                float(self.lim3 - 1) / self.lim3]))
        self.assertEquals(self.cluster3.diameter(), check3)

    def test_distance1(self):
        dist_points1 = [numpy.array([2 + float(i) / self.lim1]) for i in xrange(self.lim1)]
        dist_links1 = [[dist_points1[l] for l in [random.randint(0, self.lim1 - 1) for x in xrange(self.link_num)]]
                       for i in xrange(self.lim1)]
        dist_grid1 = Grid(dist_points1, dist_links1)
        dist_cluster1 = Cluster(dist_grid1)
        dist_check1 = numpy.linalg.norm(numpy.array([2 - float(self.lim1 - 1) / self.lim1]))
        self.assertEquals(self.cluster1.distance(dist_cluster1), dist_check1)

    def test_distance2(self):
        dist_points2 = [numpy.array([2 + float(i) / self.lim2, 2 + float(j) / self.lim2])
                        for i in xrange(self.lim2) for j in xrange(self.lim2)]
        dist_links2 = [[dist_points2[l] for l in [random.randint(0, (self.lim2 - 1) ** 2)
                                                  for x in xrange(self.link_num)]]
                       for j in xrange(self.lim2) for i in xrange(self.lim2)]
        dist_grid2 = Grid(dist_points2, dist_links2)
        dist_cluster2 = Cluster(dist_grid2)
        dist_check2 = numpy.linalg.norm(numpy.array([2 - float(self.lim2 - 1) / self.lim2,
                                                     2 - float(self.lim2 - 1) / self.lim2]))
        self.assertEquals(self.cluster2.distance(dist_cluster2), dist_check2)

    def test_distance3(self):
        dist_points3 = [numpy.array([2 + float(i) / self.lim3, 2 + float(j) / self.lim3, 2 + float(k) / self.lim3])
                        for i in xrange(self.lim3) for j in xrange(self.lim3) for k in xrange(self.lim3)]
        dist_links3 = [[dist_points3[l] for l in [random.randint(0, (self.lim3 - 1) ** 3)
                                                  for x in xrange(self.link_num)]]
                       for k in xrange(self.lim3) for j in xrange(self.lim3) for i in xrange(self.lim3)]
        dist_grid3 = Grid(dist_points3, dist_links3)
        dist_cluster3 = Cluster(dist_grid3)
        dist_check3 = numpy.linalg.norm(numpy.array([2 - float(self.lim3 - 1) / self.lim3,
                                                     2 - float(self.lim3 - 1) / self.lim3,
                                                     2 - float(self.lim3 - 1) / self.lim3]))
        self.assertEquals(self.cluster3.distance(dist_cluster3), dist_check3)

    def test_depth1(self):
        self.assertEqual(self.ct1.depth(), 4)

    def test_depth2(self):
        self.assertEqual(self.ct2.depth(), 4)

    def test_depth3(self):
        self.assertEqual(self.ct3.depth(), 3)

