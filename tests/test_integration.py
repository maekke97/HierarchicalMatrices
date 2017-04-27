import random
from unittest import TestCase

import numpy
import os

from HierMat.block_cluster_tree import build_block_cluster_tree
from HierMat.cluster import Cluster
from HierMat.cluster_tree import admissible, build_cluster_tree
from HierMat.cuboid import Cuboid
from HierMat.grid import Grid
from HierMat.hmat import build_hmatrix
from HierMat.rmat import RMat
from HierMat.splitable import RegularCuboid
from HierMat.examples.model_1d import model_1d


class TestIntegration(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim1 = 16
        cls.lim2 = 4
        cls.lim3 = 4
        cls.link_num = 4
        cls.points1 = [(float(i) / cls.lim1,) for i in xrange(cls.lim1)]
        cls.links1 = {p: [cls.points1[l] for l in [random.randint(0, cls.lim1 - 1) for x in xrange(cls.link_num)]]
                      for p in cls.points1}
        cls.points2 = [(float(i) / cls.lim2, float(j) / cls.lim2)
                       for i in xrange(cls.lim2) for j in xrange(cls.lim2)]
        cls.links2 = {p: [cls.points2[l] for l in [random.randint(0, cls.lim2 ** 2 - 1) for x in xrange(cls.link_num)]]
                      for p in cls.points2}
        cls.points3 = [(float(i) / cls.lim3, float(j) / cls.lim3, float(k) / cls.lim3)
                       for i in xrange(cls.lim3) for j in xrange(cls.lim3) for k in xrange(cls.lim3)]
        cls.links3 = {p: [cls.points3[l] for l in [random.randint(0, cls.lim3 ** 3 - 1) for x in xrange(cls.link_num)]]
                      for p in cls.points3}
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
        cls.ct1 = build_cluster_tree(cls.rc1, 4)
        cls.ct2 = build_cluster_tree(cls.rc2, 4)
        cls.ct3 = build_cluster_tree(cls.rc3, 4)
        cls.bct1 = build_block_cluster_tree(cls.ct1, cls.ct1, admissible_function=admissible)
        cls.bct2 = build_block_cluster_tree(cls.ct2, cls.ct2, admissible_function=admissible)
        cls.bct3 = build_block_cluster_tree(cls.ct3, cls.ct3, admissible_function=admissible)

        def build_full(x):
            return numpy.matrix(numpy.ones(x.shape()))

        def build_rmat(x):
            return RMat(numpy.matrix(numpy.ones((x.shape()[0], 1))),
                        numpy.matrix(numpy.ones((x.shape()[1], 1))),
                        max_rank=1)
        cls.hmat1 = build_hmatrix(cls.bct1, build_rmat, build_full)
        cls.hmat2 = build_hmatrix(cls.bct2, build_rmat, build_full)
        cls.hmat3 = build_hmatrix(cls.bct3, build_rmat, build_full)

    def test_setup(self):
        check = self.hmat1 * self.hmat2
        out = check.to_matrix()
        self.assertAlmostEqual(numpy.linalg.norm(out), numpy.linalg.norm(16*numpy.matrix(numpy.ones((16, 16)))))
        check2 = self.hmat2.transpose() * self.hmat1.transpose()
        tcheck = check.transpose()
        self.assertAlmostEqual(tcheck.norm(), check2.norm())

    def test_model_1d(self):
        self.assertTrue(model_1d(n=2**6, max_rank=1, n_min=1))

    @classmethod
    def tearDownClass(cls):
        created_files = ['gallmat_full.txt', 'hmat_full.txt', 'hmat.bin']
        try:
            for f in created_files:
                os.remove(f)
        except OSError:
            pass
