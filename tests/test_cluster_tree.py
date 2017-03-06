import math
import random
from unittest import TestCase

import numpy

from cluster import Cluster
from cluster_tree import ClusterTree, build_cluster_tree
from cuboid import Cuboid
from grid import Grid
from splitable import RegularCuboid
from utils import load


class TestClusterTree(TestCase):
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
        cls.ct1 = build_cluster_tree(cls.rc1)
        cls.ct2 = build_cluster_tree(cls.rc2)
        cls.ct3 = build_cluster_tree(cls.rc3)

    def test_init(self):
        test = ClusterTree(self.rc1)
        self.assertIsInstance(test, ClusterTree)
        self.assertIsInstance(self.ct1, ClusterTree)
        self.assertIsInstance(self.ct2, ClusterTree)
        self.assertIsInstance(self.ct3, ClusterTree)

    def test_repr(self):
        test = ClusterTree(self.rc1)
        test_str = "<ClusterTree at level 0 without children>"
        self.assertEqual(test.__repr__(), test_str)
        test.sons.append(ClusterTree(self.rc1, level=1))
        test_str = "<ClusterTree at level 0 with children [<ClusterTree at level 1 without children>]>"
        self.assertEqual(test.__repr__(), test_str)

    def test_len(self):
        self.assertEqual(len(self.ct1), self.lim1)
        self.assertEqual(len(self.ct2), self.lim2 ** 2)
        self.assertEqual(len(self.ct3), self.lim3 ** 3)

    def test_str(self):
        test_fill = ",".join([str(p) for p in self.points1])
        test = "ClusterTree at level 0 with content:\n{0}".format(test_fill)
        self.assertEqual(str(self.ct1), test)

    def test_getitem(self):
        self.assertTrue(numpy.array_equal(self.ct1[0], self.rc1[0]))
        self.assertTrue(numpy.array_equal(self.ct2[-1], self.rc2[-1]))
        self.assertTrue(numpy.array_equal(self.ct3[0], self.rc3[0]))

    def test_get_index(self):
        self.assertEqual(self.ct1.get_index(0), self.cluster1.get_index(0))
        self.assertEqual(self.ct1.get_index(-1), self.cluster1.get_index(-1))
        self.assertEqual(self.ct2.get_index(0), self.cluster2.get_index(0))
        self.assertEqual(self.ct2.get_index(-1), self.cluster2.get_index(-1))
        self.assertEqual(self.ct3.get_index(0), self.cluster3.get_index(0))
        self.assertEqual(self.ct3.get_index(-1), self.cluster3.get_index(-1))

    def test_get_grid_item(self):
        self.assertEqual(self.ct1.get_grid_item(0), self.grid1[0])
        self.assertEqual(self.ct1.get_grid_item(-1), self.grid1[-1])
        self.assertTrue(numpy.array_equal(self.ct2.get_grid_item(0), self.grid2[0]))
        self.assertTrue(numpy.array_equal(self.ct2.get_grid_item(-1), self.grid2[-1]))
        self.assertTrue(numpy.array_equal(self.ct3.get_grid_item(0), self.grid3[0]))
        self.assertTrue(numpy.array_equal(self.ct3.get_grid_item(-1), self.grid3[-1]))

    def test_eq(self):
        self.assertEqual(self.ct1, self.ct1)
        self.assertNotEqual(self.ct1, self.ct2)
        self.assertEqual(self.ct2, self.ct2)
        self.assertNotEqual(self.ct2, self.ct3)
        self.assertEqual(self.ct3, self.ct3)
        self.assertNotEqual(self.ct3, self.ct1)

    def test_to_list(self):
        self.assertEqual(len(self.ct1.to_list()), 2)
        self.assertEqual(len(self.ct2.to_list()), 2)
        self.assertEqual(len(self.ct3.to_list()), 2)

    def test_export(self):
        out_file_xml = 'test_EI_1.xml'
        out_file_dot = 'test_EI_1.dot'
        out_file_bin = 'test_EI_1.bin'
        self.ct1.export('xml', out_file_xml)
        self.ct1.export('dot', out_file_dot)
        self.ct1.export('bin', out_file_bin)
        test_ct = load(out_file_bin)
        self.assertEqual(self.ct1, test_ct)
        out_file_xml = 'test_EI_2.xml'
        out_file_dot = 'test_EI_2.dot'
        out_file_bin = 'test_EI_2.bin'
        self.ct2.export('xml', out_file_xml)
        self.ct2.export('dot', out_file_dot)
        self.ct2.export('bin', out_file_bin)
        test_ct = load(out_file_bin)
        self.assertEqual(self.ct2, test_ct)
        out_file_xml = 'test_EI_3.xml'
        out_file_dot = 'test_EI_3.dot'
        out_file_bin = 'test_EI_3.bin'
        self.ct3.export('xml', out_file_xml)
        self.ct3.export('dot', out_file_dot)
        self.ct3.export('bin', out_file_bin)
        test_ct = load(out_file_bin)
        self.assertEqual(self.ct3, test_ct)
        self.assertRaises(NotImplementedError, self.ct1.export, 'test', out_file_bin)

    def test_depth(self):
        self.assertEqual(self.ct1.depth(), math.log(self.lim1, 2))
        self.assertEqual(self.ct2.depth(), math.log(self.lim2 ** 2, 2))
        self.assertEqual(self.ct3.depth(), math.log(self.lim3 ** 3, 2))

    def test_diameter(self):
        check = numpy.linalg.norm(numpy.array([float(self.lim1 - 1) / self.lim1]))
        self.assertEqual(self.ct1.diameter(), check)
        check = numpy.linalg.norm(numpy.array([float(self.lim2 - 1) / self.lim2, float(self.lim2 - 1) / self.lim2]))
        self.assertEqual(self.ct2.diameter(), check)
        check = numpy.linalg.norm(numpy.array([float(self.lim3 - 1) / self.lim3, float(self.lim3 - 1) / self.lim3,
                                               float(self.lim3 - 1) / self.lim3]))
        self.assertEqual(self.ct3.diameter(), check)

    def test_distance(self):
        dist_check = numpy.linalg.norm(numpy.array([2 - float(self.lim1 - 1) / self.lim1]))
        dist_points = [numpy.array([2 + float(i) / self.lim1]) for i in xrange(self.lim1)]
        dist_links = [[dist_points[l] for l in [random.randint(0, self.lim1 - 1) for x in xrange(self.link_num)]]
                      for i in xrange(self.lim1)]
        dist_grid = Grid(dist_points, dist_links)
        dist_cluster = Cluster(dist_grid)
        dist_rc = RegularCuboid(dist_cluster)
        dist_ct = ClusterTree(dist_rc, 1)
        self.assertEqual(self.ct1.distance(dist_ct), dist_check)
        dist_points = [numpy.array([2 + float(i) / self.lim2, 2 + float(j) / self.lim2])
                       for i in xrange(self.lim2) for j in xrange(self.lim2)]
        dist_links = [[dist_points[l] for l in [random.randint(0, (self.lim2 - 1) ** 2)
                                                for x in xrange(self.link_num)]]
                      for j in xrange(self.lim2) for i in xrange(self.lim2)]
        dist_grid = Grid(dist_points, dist_links)
        dist_cluster = Cluster(dist_grid)
        dist_check = numpy.linalg.norm(numpy.array([2 - float(self.lim2 - 1) / self.lim2,
                                                    2 - float(self.lim2 - 1) / self.lim2]))
        dist_rc = RegularCuboid(dist_cluster)
        dist_ct = ClusterTree(dist_rc, 1)
        self.assertEqual(self.ct2.distance(dist_ct), dist_check)
        dist_points = [numpy.array([2 + float(i) / self.lim3, 2 + float(j) / self.lim3, 2 + float(k) / self.lim3])
                       for i in xrange(self.lim3) for j in xrange(self.lim3) for k in xrange(self.lim3)]
        dist_links = [[dist_points[l] for l in [random.randint(0, (self.lim3 - 1) ** 3)
                                                for x in xrange(self.link_num)]]
                      for k in xrange(self.lim3) for j in xrange(self.lim3) for i in xrange(self.lim3)]
        dist_grid = Grid(dist_points, dist_links)
        dist_cluster = Cluster(dist_grid)
        dist_check = numpy.linalg.norm(numpy.array([2 - float(self.lim3 - 1) / self.lim3,
                                                    2 - float(self.lim3 - 1) / self.lim3,
                                                    2 - float(self.lim3 - 1) / self.lim3]))
        dist_rc = RegularCuboid(dist_cluster)
        dist_ct = ClusterTree(dist_rc, 1)
        self.assertEqual(self.ct3.distance(dist_ct), dist_check)
