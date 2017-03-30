from unittest import TestCase

import numpy
import random
import math
import os
import matplotlib.figure

from HierMat.block_cluster_tree import BlockClusterTree, build_block_cluster_tree
from HierMat.cluster import Cluster
from HierMat.cluster_tree import build_cluster_tree
from HierMat.cuboid import Cuboid
from HierMat.utils import admissible, load
from HierMat.grid import Grid
from HierMat.splitable import RegularCuboid


class TestBlockClusterTree(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim1 = 8
        cls.lim2 = 3
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
        cls.bct1 = build_block_cluster_tree(cls.ct1, right_cluster_tree=cls.ct1, admissible_function=admissible)
        cls.bct2 = build_block_cluster_tree(cls.ct2, right_cluster_tree=cls.ct2, admissible_function=admissible)
        cls.bct3 = build_block_cluster_tree(cls.ct3, right_cluster_tree=cls.ct3, admissible_function=admissible)

    def test_init(self):
        test = BlockClusterTree(self.rc1, right_clustertree=self.rc1)
        self.assertIsInstance(test, BlockClusterTree)
        test = BlockClusterTree(self.rc1, right_clustertree=self.rc1, sons=[self.rc2])
        self.assertIsInstance(test, BlockClusterTree)
        test = build_block_cluster_tree(self.ct1)
        self.assertIsInstance(test, BlockClusterTree)
        self.assertIsInstance(self.bct1, BlockClusterTree)
        self.assertIsInstance(self.bct2, BlockClusterTree)
        self.assertIsInstance(self.bct3, BlockClusterTree)

    def test_repr(self):
        test_str = "<BlockClusterTree at level 0>"
        test = BlockClusterTree(self.ct1, right_clustertree=self.ct1)
        self.assertEqual(test.__repr__(), test_str)
        test_str = "<BlockClusterTree at level 0 with children [<BlockClusterTree at level 1>]>"
        test = BlockClusterTree(self.ct1, right_clustertree=self.ct1,
                                sons=[BlockClusterTree(self.ct1, right_clustertree=self.ct1, level=1)])
        self.assertEqual(test.__repr__(), test_str)

    def test_eq(self):
        self.assertEqual(self.bct1, self.bct1)
        self.assertFalse(self.bct1 == self.bct2)
        self.assertEqual(self.bct2, self.bct2)
        self.assertFalse(self.bct2 == self.bct3)
        self.assertEqual(self.bct3, self.bct3)
        self.assertFalse(self.bct3 == self.bct1)

    def test_ne(self):
        self.assertNotEqual(self.bct1, self.bct2)
        self.assertFalse(self.bct1 != self.bct1)
        self.assertNotEqual(self.bct2, self.bct3)
        self.assertFalse(self.bct2 != self.bct2)
        self.assertNotEqual(self.bct3, self.bct1)
        self.assertFalse(self.bct3 != self.bct3)

    def test_depth(self):
        self.assertEqual(self.bct1.depth(), math.log(self.lim1, 2))
        self.assertEqual(self.bct2.depth(), math.ceil(math.log(self.lim2, 2)*2))
        self.assertEqual(self.bct3.depth(), math.log(self.lim3, 2)*3)

    def test_to_list(self):
        self.assertEqual(len(self.bct1.to_list()), 2)
        self.assertEqual(len(self.bct2.to_list()), 2)
        self.assertEqual(len(self.bct3.to_list()), 2)

    def test_shape(self):
        self.assertEqual(self.bct1.shape(), (self.lim1, self.lim1))
        self.assertEqual(self.bct2.shape(), (self.lim2**2, self.lim2**2))
        self.assertEqual(self.bct3.shape(), (self.lim3**3, self.lim3**3))

    def test_export(self):
        out_string = 'test_EI_bct1.'
        forms = ['xml', 'dot', 'bin']
        for form in forms:
            self.bct1.export(form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.bct1)
        out_string = 'test_EI_bct2.'
        for form in forms:
            self.bct2.export(form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.bct2)
        out_string = 'test_EI_bct3.'
        for form in forms:
            self.bct3.export(form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.bct3)
        self.assertRaises(NotImplementedError, self.bct1.export, form='bla')

    def test_plot(self):
        out_string = 'test_plot_bct1'
        self.bct1.plot(out_string)
        self.assertTrue(os.path.exists(out_string))
        out_string = 'test_plot_bct2'
        self.bct2.plot(out_string, ticks=True)
        self.assertTrue(os.path.exists(out_string))
        out_string = 'test_plot_bct3'
        self.bct3.plot(out_string)
        self.assertTrue(os.path.exists(out_string))
        fig = self.bct3.plot(ticks=True)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    @classmethod
    def tearDownClass(cls):
        out_string = 'test_EI_bct{0}.'
        plot_out = 'test_plot_bct'
        forms = ['xml', 'dot', 'bin']
        try:
            for i in xrange(3):
                for form in forms:
                    os.remove(out_string.format(i+1)+form)
                os.remove(plot_out + str(i+1))
        except OSError:
            pass
