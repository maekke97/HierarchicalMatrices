import math
from unittest import TestCase

import numpy as np
from cuboid import *

from utils import minimal_cuboid, Cluster


class TestCuboid(TestCase):
    def setUp(self):
        ll1 = np.array([0], float)
        ll2 = np.array([0, 0], float)
        ll3 = np.array([0, 0, 0], float)
        ur1 = np.array([5], float)
        ur2 = np.array([5, 5], float)
        ur3 = np.array([5, 5, 5], float)
        split1 = np.array([2.5], float)
        split11 = np.array([1.25], float)
        split12 = np.array([3.75], float)
        split2l = np.array([2.5, 0], float)
        split2r = np.array([2.5, 5], float)
        split3l = np.array([2.5, 0, 0], float)
        split3r = np.array([2.5, 5, 5], float)
        self.cub1 = Cuboid(ll1, ur1)
        self.cub2 = Cuboid(ll2, ur2)
        self.cub3 = Cuboid(ll3, ur3)
        self.split1l = Cuboid(ll1, split1)
        self.split1r = Cuboid(split1, ur1)
        self.split2l = Cuboid(ll2, split2r)
        self.split2r = Cuboid(split2l, ur2)
        self.split3l = Cuboid(ll3, split3r)
        self.split3r = Cuboid(split3l, ur3)
        self.minimal1 = Cuboid(ll1, ur1)
        self.cluster1 = Cluster([np.array(i) for i in xrange(6)], [[np.array(i) for i in (1, 2)] for j in xrange(6)])
        self.minimal2 = Cuboid(ll2, ur2)
        self.cluster2 = Cluster([np.array([x, y]) for x in xrange(6) for y in xrange(6)],
                                [[np.array([i, j]) for j in xrange(3)] for i in xrange(36)])
        self.minimal3 = Cuboid(ll3, ur3)
        self.cluster3 = Cluster([np.array([0, 0, 0]), np.array([5, 5, 5])], [[np.array(2)] for i in xrange(2)])
        self.control_cuboid1 = Cuboid(np.array([6], float), np.array([8], float))
        self.control_cuboid2 = Cuboid(np.array([6, 6], float), np.array([8, 8], float))
        self.control_cuboid3 = Cuboid(np.array([6, 6, 6], float), np.array([8, 8, 8], float))

    def test_init(self):
        test = Cuboid([0.5, 1, 1.5], [1, 2, 2])
        self.assertIsInstance(test, Cuboid)

    def test_minimal_cuboid(self):
        self.assertEqual(self.minimal1, minimal_cuboid(self.cluster1))
        self.assertEqual(self.minimal2, minimal_cuboid(self.cluster2))
        self.assertEqual(self.minimal3, minimal_cuboid(self.cluster3))

    def test_half(self):
        split1l, split1r = self.cub1.half()
        self.assertEqual(split1l, self.split1l)
        self.assertEqual(split1r, self.split1r)
        split2l, split2r = self.cub2.half()
        self.assertEqual(split2l, self.split2l)
        self.assertEqual(split2r, self.split2r)
        split3l, split3r = self.cub3.half()
        self.assertEqual(split3l, self.split3l)
        self.assertEqual(split3r, self.split3r)

    def test_diameter(self):
        self.assertEquals(self.cub1.diameter(), 5.)
        self.assertEquals(self.cub2.diameter(), math.sqrt(50.))
        self.assertEquals(self.cub3.diameter(), math.sqrt(75.))

    def test_distance(self):
        self.assertEquals(self.cub1.distance(self.control_cuboid1), 1.)
        self.assertEquals(self.cub2.distance(self.control_cuboid2), math.sqrt(2))
        self.assertEquals(self.cub3.distance(self.control_cuboid3), math.sqrt(3))
        self.assertEquals(self.control_cuboid1.distance(self.cub1),
                          self.cub1.distance(self.control_cuboid1))
        self.assertEquals(self.control_cuboid2.distance(self.cub2),
                          self.cub2.distance(self.control_cuboid2))
        self.assertEquals(self.control_cuboid3.distance(self.cub3),
                          self.cub3.distance(self.control_cuboid3))
