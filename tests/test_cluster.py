import unittest
from cluster import Cluster
import numpy as np
import random


class TestCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lim1 = 5
        lim2 = 5
        lim3 = 5
        link_num = 5
        points1 = [np.array([float(i)/lim1]) for i in xrange(lim1)]
        links1 = [[points1[l] for l in [random.randint(0, lim1-1) for x in xrange(link_num)]] for i in xrange(lim1)]
        points2 = [np.array([float(i)/lim1, float(j)/lim2]) for i in xrange(lim1) for j in xrange(lim2)]
        links2 = [[points2[l] for l in [random.randint(0, (lim1-1)*(lim2-2)) for x in xrange(link_num)]]
                  for j in xrange(lim2) for i in xrange(lim1)]
        points3 = [np.array([float(i)/lim1, float(j)/lim2, float(k)/lim3]) for i in xrange(lim1) for j in xrange(lim2)
                   for k in xrange(lim3)]
        links3 = [[points3[l] for l in [random.randint(0, (lim1-1)*(lim2-1)*(lim3-1)) for x in xrange(link_num)]]
                  for k in xrange(lim3) for j in xrange(lim2) for i in xrange(lim1)]
        cls.cluster1 = Cluster(points1, links1)
        cls.cluster2 = Cluster(points2, links2)
        cls.cluster3 = Cluster(points3, links3)
        cls.check1 = np.linalg.norm(np.array([float(lim1-1)/lim1]))
        cls.check2 = np.linalg.norm(np.array([float(lim1-1)/lim1, float(lim2-1)/lim2]))
        cls.check3 = np.linalg.norm(np.array([float(lim1-1)/lim1, float(lim2-1)/lim2, float(lim3-1)/lim3]))
        dist_points1 = [np.array([2+float(i) / lim1]) for i in xrange(lim1)]
        dist_links1 = [[dist_points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]]
                       for i in xrange(lim1)]
        dist_points2 = [np.array([2+float(i) / lim1, 2+float(j) / lim2]) for i in xrange(lim1) for j in xrange(lim2)]
        dist_links2 = [[dist_points2[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 2)) for x in xrange(link_num)]]
                  for j in xrange(lim2) for i in xrange(lim1)]
        dist_points3 = [np.array([2+float(i) / lim1, 2+float(j) / lim2, 2+float(k) / lim3])
                        for i in xrange(lim1) for j in xrange(lim2) for k in xrange(lim3)]
        dist_links3 = [[dist_points3[l] for l in [random.randint(0, (lim1 - 1) * (lim2 - 1) * (lim3 - 1))
                                                  for x in xrange(link_num)]]
                       for k in xrange(lim3) for j in xrange(lim2) for i in xrange(lim1)]
        cls.dist_cluster1 = Cluster(dist_points1, dist_links1)
        cls.dist_cluster2 = Cluster(dist_points2, dist_links2)
        cls.dist_cluster3 = Cluster(dist_points3, dist_links3)
        cls.dist_check1 = np.linalg.norm(np.array([2-float(lim1 - 1) / lim1]))
        cls.dist_check2 = np.linalg.norm(np.array([2-float(lim1 - 1) / lim1, 2-float(lim2 - 1) / lim2]))
        cls.dist_check3 = np.linalg.norm(np.array([2-float(lim1 - 1) / lim1,
                                                   2-float(lim2 - 1) / lim2, 2-float(lim3 - 1) / lim3]))

    def test_setup(self):
        self.assertIsInstance(self.cluster1, Cluster)
        self.assertIsInstance(self.cluster2, Cluster)
        self.assertIsInstance(self.cluster3, Cluster)

    def test__diameter(self):
        self.assertEquals(self.cluster1.diam, self.check1)
        self.assertEquals(self.cluster2.diam, self.check2)
        self.assertEquals(self.cluster3.diam, self.check3)

    def test_distance(self):
        self.assertEquals(self.cluster1.distance(self.dist_cluster1), self.dist_check1)
        self.assertEquals(self.cluster2.distance(self.dist_cluster2), self.dist_check2)
        self.assertEquals(self.cluster3.distance(self.dist_cluster3), self.dist_check3)
