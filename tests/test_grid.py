import random
from unittest import TestCase

import numpy

from grid import Grid


class TestGrid(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lim1 = 16
        cls.lim2 = 4
        cls.lim3 = 4
        cls.link_num = 2
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

    def test_init(self):
        self.assertEqual(type(self.grid1), Grid)
        self.assertEqual(type(self.grid2), Grid)
        self.assertEqual(type(self.grid3), Grid)
        self.assertRaises(ValueError, Grid, self.points1, self.links2)

    def test_len(self):
        self.assertEqual(len(self.grid1), self.lim1)
        self.assertEqual(len(self.grid2), self.lim2 ** 2)
        self.assertEqual(len(self.grid3), self.lim3 ** 3)

    def test_getitem(self):
        self.assertEqual(self.grid1[0], self.points1[0])
        self.assertTrue(numpy.array_equal(self.grid2[-1], self.points2[-1]))
        check = random.randint(0, self.lim3 ** 3)
        self.assertTrue(numpy.array_equal(self.grid3[check], self.points3[check]))

    def test_get_point(self):
        self.assertTrue(numpy.array_equal(self.grid1.get_point(0), self.points1[0]))
        self.assertTrue(numpy.array_equal(self.grid1.get_point(-1), self.points1[-1]))
        self.assertTrue(numpy.array_equal(self.grid2.get_point(0), self.points2[0]))
        self.assertTrue(numpy.array_equal(self.grid2.get_point(-1), self.points2[-1]))
        self.assertTrue(numpy.array_equal(self.grid3.get_point(0), self.points3[0]))
        self.assertTrue(numpy.array_equal(self.grid3.get_point(-1), self.points3[-1]))

    def test_get_link(self):
        self.assertTrue(numpy.array_equal(self.grid1.get_link(0), self.links1[0]))
        self.assertTrue(numpy.array_equal(self.grid1.get_link(-1), self.links1[-1]))
        self.assertTrue(numpy.array_equal(self.grid2.get_link(0), self.links2[0]))
        self.assertTrue(numpy.array_equal(self.grid2.get_link(-1), self.links2[-1]))
        self.assertTrue(numpy.array_equal(self.grid3.get_link(0), self.links3[0]))
        self.assertTrue(numpy.array_equal(self.grid3.get_link(-1), self.links3[-1]))

    def test_iterator(self):
        iterator = self.grid1.__iter__()
        iteriter = iterator.__iter__()
        self.assertTrue(iterator == iteriter)
        iterator = self.grid2.__iter__()
        iteriter = iterator.__iter__()
        self.assertTrue(iterator == iteriter)
        iterator = self.grid3.__iter__()
        iteriter = iterator.__iter__()
        self.assertTrue(iterator == iteriter)
        grid_check = [p for p in self.grid1]
        self.assertEqual(self.grid1.points, grid_check)
        grid_check = [p for p in self.grid2]
        self.assertEqual(self.grid2.points, grid_check)
        grid_check = [p for p in self.grid3]
        self.assertEqual(self.grid3.points, grid_check)

    def test_eq(self):
        self.assertEqual(self.grid1, self.grid1)
        self.assertNotEqual(self.grid1, self.grid2)
        self.assertEqual(self.grid2, self.grid2)
        self.assertNotEqual(self.grid2, self.grid3)
        self.assertEqual(self.grid3, self.grid3)
        self.assertNotEqual(self.grid3, self.grid1)

    def test_dim(self):
        self.assertEqual(self.grid1.dim(), 1)
        self.assertEqual(self.grid2.dim(), 2)
        self.assertEqual(self.grid3.dim(), 3)

    def test_plot(self):
        self.grid2.plot()
