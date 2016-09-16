import random
from unittest import TestCase

import numpy

from utils import BlockClusterTree, ClusterTree, RegularCuboid, Cuboid, Cluster, Grid, admissible


class TestUtils(TestCase):
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
    
    def test_setup1(self):
        self.assertIsInstance(self.grid1, Grid)
        self.assertIsInstance(self.cluster1, Cluster)
        self.assertIsInstance(self.cub1, Cuboid)
        self.assertIsInstance(self.rc1, RegularCuboid)
        self.assertIsInstance(self.ct1, ClusterTree)
        self.assertIsInstance(self.bct1, BlockClusterTree)

    def test_setup2(self):
        self.assertIsInstance(self.grid2, Grid)
        self.assertIsInstance(self.cluster2, Cluster)
        self.assertIsInstance(self.cub2, Cuboid)
        self.assertIsInstance(self.rc2, RegularCuboid)
        self.assertIsInstance(self.ct2, ClusterTree)
        self.assertIsInstance(self.bct2, BlockClusterTree)

    def test_setup3(self):
        self.assertIsInstance(self.grid3, Grid)
        self.assertIsInstance(self.cluster3, Cluster)
        self.assertIsInstance(self.cub3, Cuboid)
        self.assertIsInstance(self.rc3, RegularCuboid)
        self.assertIsInstance(self.ct3, ClusterTree)
        self.assertIsInstance(self.bct3, BlockClusterTree)

    def test_iterator1(self):
        grid_check = [p for p in self.grid1]
        self.assertEqual(self.grid1.points, grid_check)
        cluster_check = [c for c in self.cluster1]
        self.assertEqual([self.cluster1.grid[i] for i in self.cluster1.indices], cluster_check)

    def test_iterator2(self):
        grid_check = [p for p in self.grid2]
        self.assertEqual(self.grid2.points, grid_check)
        cluster_check = [c for c in self.cluster2]
        self.assertEqual([self.cluster2.grid[i] for i in self.cluster2.indices], cluster_check)

    def test_iterator3(self):
        grid_check = [p for p in self.grid3]
        self.assertEqual(self.grid3.points, grid_check)
        cluster_check = [c for c in self.cluster3]
        self.assertEqual([self.cluster3.grid[i] for i in self.cluster3.indices], cluster_check)

    def test_diameter1(self):
        check1 = numpy.linalg.norm(numpy.array([float(self.lim1 - 1) / self.lim1]))
        self.assertEquals(self.cluster1.diameter(), check1)
        self.assertEqual(self.cub1.diameter(), 1)
        self.assertEqual(self.rc1.diameter(), check1)

    def test_diameter2(self):
        check2 = numpy.linalg.norm(numpy.array([float(self.lim2 - 1) / self.lim2, float(self.lim2 - 1) / self.lim2]))
        self.assertEquals(self.cluster2.diameter(), check2)
        self.assertEqual(self.cub2.diameter(), numpy.sqrt(2))
        self.assertEqual(self.rc2.diameter(), check2)

    def test_diameter3(self):
        check3 = numpy.linalg.norm(numpy.array([float(self.lim3 - 1) / self.lim3, float(self.lim3 - 1) / self.lim3,
                                                float(self.lim3 - 1) / self.lim3]))
        self.assertEquals(self.cluster3.diameter(), check3)
        self.assertEqual(self.cub3.diameter(), numpy.sqrt(3))
        self.assertEqual(self.rc3.diameter(), check3)

    def test_distance1(self):
        dist_points1 = [numpy.array([2 + float(i) / self.lim1]) for i in xrange(self.lim1)]
        dist_links1 = [[dist_points1[l] for l in [random.randint(0, self.lim1 - 1) for x in xrange(self.link_num)]]
                       for i in xrange(self.lim1)]
        dist_grid1 = Grid(dist_points1, dist_links1)
        dist_cluster1 = Cluster(dist_grid1)
        dist_check1 = numpy.linalg.norm(numpy.array([2 - float(self.lim1 - 1) / self.lim1]))
        self.assertEquals(self.cluster1.distance(dist_cluster1), dist_check1)
        dist_cub1 = Cuboid(numpy.array([2]), numpy.array([3]))
        self.assertEqual(self.cub1.distance(dist_cub1), 1)
        dist_rc1 = RegularCuboid(dist_cluster1)
        self.assertEqual(self.rc1.distance(dist_rc1), dist_check1)

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
        dist_cub2 = Cuboid(numpy.array([2, 2]), numpy.array([3, 3]))
        self.assertEqual(self.cub1.distance(dist_cub2), numpy.sqrt(2))
        dist_rc2 = RegularCuboid(dist_cluster2)
        self.assertEqual(self.rc2.distance(dist_rc2), dist_check2)

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
        dist_cub3 = Cuboid(numpy.array([2, 2, 2]), numpy.array([3, 3, 3]))
        self.assertEqual(self.cub1.distance(dist_cub3), numpy.sqrt(3))
        dist_rc3 = RegularCuboid(dist_cluster3)
        self.assertEqual(self.rc3.distance(dist_rc3), dist_check3)

    def test_depth1(self):
        self.assertEqual(self.ct1.depth(), numpy.log2(self.lim1))
        self.assertEqual(self.bct1.depth(), numpy.log2(self.lim1))

    def test_depth2(self):
        self.assertEqual(self.ct2.depth(), numpy.log2(self.lim2 ** 2))
        self.assertEqual(self.bct2.depth(), numpy.log2(self.lim2 ** 2))

    def test_depth3(self):
        self.assertEqual(self.ct3.depth(), numpy.log2(self.lim3 ** 3))
        self.assertEqual(self.bct3.depth(), numpy.log2(self.lim3 ** 3))

    def test_dim1(self):
        self.assertEqual(self.grid1.dim(), 1)
        self.assertEqual(self.cluster1.dim(), 1)

    def test_dim2(self):
        self.assertEqual(self.grid2.dim(), 2)
        self.assertEqual(self.cluster2.dim(), 2)

    def test_dim3(self):
        self.assertEqual(self.grid3.dim(), 3)
        self.assertEqual(self.cluster3.dim(), 3)

    def test_length1(self):
        self.assertEqual(len(self.grid1), self.lim1)
        self.assertEqual(len(self.cluster1), self.lim1)
        self.assertEqual(len(self.rc1), self.lim1)

    def test_length2(self):
        self.assertEqual(len(self.grid2), self.lim2**2)
        self.assertEqual(len(self.cluster2), self.lim2**2)
        self.assertEqual(len(self.rc2), self.lim2**2)

    def test_length3(self):
        self.assertEqual(len(self.grid3), self.lim3**3)
        self.assertEqual(len(self.cluster3), self.lim3**3)
        self.assertEqual(len(self.rc3), self.lim3**3)
