import random
from unittest import TestCase

import numpy

from cluster import Cluster
from grid import Grid


class TestCluster(TestCase):
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

    def test_init(self):
        self.assertEqual(type(self.cluster1), Cluster)
        self.assertEqual(type(self.cluster2), Cluster)
        self.assertEqual(type(self.cluster3), Cluster)

    def test_getitem(self):
        self.assertEqual(self.cluster1[0], self.grid1[0])
        self.assertTrue(numpy.array_equal(self.cluster2[-1], self.grid2[-1]))
        check = random.randint(0, self.lim3 ** 3)
        self.assertTrue(numpy.array_equal(self.cluster3[check], self.points3[check]))

    def test_repr(self):
        check = "<Cluster object with grid {0} and indices {1}>".format(self.grid1, range(len(self.grid1)))
        self.assertEqual(self.cluster1.__repr__(), check)
        check = "<Cluster object with grid {0} and indices {1}>".format(self.grid2, range(len(self.grid2)))
        self.assertEqual(self.cluster2.__repr__(), check)
        check = "<Cluster object with grid {0} and indices {1}>".format(self.grid3, range(len(self.grid3)))
        self.assertEqual(self.cluster3.__repr__(), check)

    def test_iterator(self):
        grid_check = [p for p in self.cluster1]
        self.assertEqual(self.cluster1.grid.points, grid_check)
        grid_check = [p for p in self.cluster2]
        self.assertEqual(self.cluster2.grid.points, grid_check)
        grid_check = [p for p in self.cluster3]
        self.assertEqual(self.cluster3.grid.points, grid_check)

    def test_len(self):
        self.assertEqual(len(self.cluster1), self.lim1)
        self.assertEqual(len(self.cluster2), self.lim2 ** 2)
        self.assertEqual(len(self.cluster3), self.lim3 ** 3)

    def test_eq(self):
        self.assertEqual(self.cluster1, self.cluster1)
        self.assertNotEqual(self.cluster1, self.cluster2)
        self.assertEqual(self.cluster2, self.cluster2)
        self.assertNotEqual(self.cluster2, self.cluster3)
        self.assertEqual(self.cluster3, self.cluster3)
        self.assertNotEqual(self.cluster3, self.cluster1)

    def test_dim(self):
        self.assertEqual(self.cluster1.dim(), 1)
        self.assertEqual(self.cluster2.dim(), 2)
        self.assertEqual(self.cluster3.dim(), 3)

    def test_diameter(self):
        check1 = numpy.linalg.norm(numpy.array([float(self.lim1 - 1) / self.lim1]))
        self.assertEquals(self.cluster1.diameter(), check1)
        check2 = numpy.linalg.norm(numpy.array([float(self.lim2 - 1) / self.lim2, float(self.lim2 - 1) / self.lim2]))
        self.assertEquals(self.cluster2.diameter(), check2)
        check3 = numpy.linalg.norm(numpy.array([float(self.lim3 - 1) / self.lim3, float(self.lim3 - 1) / self.lim3,
                                                float(self.lim3 - 1) / self.lim3]))
        self.assertEquals(self.cluster3.diameter(), check3)

    def test_distance(self):
        dist_points1 = [numpy.array([2 + float(i) / self.lim1]) for i in xrange(self.lim1)]
        dist_links1 = [[dist_points1[l] for l in [random.randint(0, self.lim1 - 1) for x in xrange(self.link_num)]]
                       for i in xrange(self.lim1)]
        dist_grid1 = Grid(dist_points1, dist_links1)
        dist_cluster1 = Cluster(dist_grid1)
        dist_check1 = numpy.linalg.norm(numpy.array([2 - float(self.lim1 - 1) / self.lim1]))
        self.assertEquals(self.cluster1.distance(dist_cluster1), dist_check1)
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
