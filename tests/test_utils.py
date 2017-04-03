from unittest import TestCase

import matplotlib.figure
import numpy
import random
import os

from HierMat.block_cluster_tree import BlockClusterTree, build_block_cluster_tree
from HierMat.cluster import Cluster
from HierMat.cluster_tree import ClusterTree, build_cluster_tree, admissible
from HierMat.cuboid import Cuboid
from HierMat.utils import load, export, plot
from HierMat.grid import Grid
from HierMat.splitable import RegularCuboid


class TestUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim1 = 16
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
        cls.ct1 = build_cluster_tree(cls.rc1, 1)
        cls.ct2 = build_cluster_tree(cls.rc2, 1)
        cls.ct3 = build_cluster_tree(cls.rc3, 1)
        cls.bct1 = build_block_cluster_tree(cls.ct1, cls.ct1, admissible_function=admissible)
        cls.bct2 = build_block_cluster_tree(cls.ct2, cls.ct2, admissible_function=admissible)
        cls.bct3 = build_block_cluster_tree(cls.ct3, cls.ct3, admissible_function=admissible)

    def test_export_import(self):
        out_string = 'test_EI_bct1.'
        forms = ['xml', 'dot', 'bin']
        for form in forms:
            export(self.bct1, form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.bct1)
        out_string = 'test_EI_bct2.'
        for form in forms:
            export(self.bct2, form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.bct2)
        out_string = 'test_EI_bct3.'
        for form in forms:
            export(self.bct3, form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.bct3)
        self.assertRaises(NotImplementedError, export, self.bct1, form='bla')
        out_string = 'test_EI_ct1.'
        for form in forms:
            export(self.ct1, form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.ct1)
        out_string = 'test_EI_ct2.'
        for form in forms:
            export(self.ct2, form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.ct2)
        out_string = 'test_EI_ct3.'
        for form in forms:
            export(self.ct3, form, out_file=out_string + form)
            self.assertTrue(os.path.exists(out_string + form))
        check = load(out_string + 'bin')
        self.assertEqual(check, self.ct3)
        self.assertRaises(NotImplementedError, export, self.ct1, form='bla')

    def test_plot(self):
        out_string = 'test_plot_bct1'
        plot(self.bct1, out_string)
        self.assertTrue(os.path.exists(out_string))
        out_string = 'test_plot_bct2'
        plot(self.bct2, out_string, ticks=True)
        self.assertTrue(os.path.exists(out_string))
        out_string = 'test_plot_bct3'
        plot(self.bct3, out_string)
        self.assertTrue(os.path.exists(out_string))
        fig = plot(self.bct3, ticks=True)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    @classmethod
    def tearDownClass(cls):
        out_string_bct = 'test_EI_bct{0}.'
        out_string_ct = 'test_EI_ct{0}.'
        plot_out = 'test_plot_bct'
        forms = ['xml', 'dot', 'bin']
        try:
            for i in xrange(3):
                for form in forms:
                    os.remove(out_string_bct.format(i + 1) + form)
                    os.remove(out_string_ct.format(i + 1) + form)
                os.remove(plot_out + str(i + 1))
        except OSError:
            pass
