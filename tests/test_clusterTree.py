from unittest import TestCase
from clustertree import ClusterTree, RegularCuboid
from cluster import Cluster
import numpy as np
import random


class TestClusterTree(TestCase):
    @classmethod
    def setUpClass(cls):
        lim1 = 8
        lim2 = 8
        lim3 = 8
        link_num = 4
        points1 = [np.array([float(i) / lim1]) for i in xrange(lim1)]
        links1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
        points2 = [np.array([float(i) / lim1, float(j) / lim2]) for i in xrange(lim1) for j in xrange(lim2)]
        links2 = [[points2[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 2)) for x in xrange(link_num)]]
                  for j in xrange(lim2) for i in xrange(lim1)]
        points3 = [np.array([float(i) / lim1, float(j) / lim2, float(k) / lim3]) for i in xrange(lim1) for j in
                   xrange(lim2)
                   for k in xrange(lim3)]
        links3 = [
            [points3[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 1) * (lim3 - 1)) for x in xrange(link_num)]]
            for k in xrange(lim3) for j in xrange(lim2) for i in xrange(lim1)]
        cluster1 = Cluster(points1, links1)
        cluster2 = Cluster(points2, links2)
        cluster3 = Cluster(points3, links3)
        cls.rc1 = RegularCuboid(cluster1)
        cls.rc2 = RegularCuboid(cluster2)
        cls.rc3 = RegularCuboid(cluster3)
        cls.ct1 = ClusterTree(cls.rc1, 1)
        cls.ct2 = ClusterTree(cls.rc2, 1)
        cls.ct3 = ClusterTree(cls.rc3, 1)

    def test_depth(self):
        self.assertEqual(self.ct1.depth(), 3)
        self.assertEqual(self.ct2.depth(), 6)
        self.assertEqual(self.ct3.depth(), 9)

    def test_export(self):
        out1 = self.ct1.export()
        self.assertNotEqual(out1, "")
