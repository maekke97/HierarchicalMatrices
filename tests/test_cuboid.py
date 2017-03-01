from unittest import TestCase

import numpy

from cuboid import Cuboid


class TestCuboid(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cub1 = Cuboid(numpy.array([0]), numpy.array([1]))
        cls.cub2 = Cuboid(numpy.array([0, 0]), numpy.array([1, 1]))
        cls.cub3 = Cuboid(numpy.array([0, 0, 0]), numpy.array([1, 1, 1]))

    def test_init(self):
        self.assertEqual(type(self.cub1), Cuboid)
        self.assertEqual(type(self.cub2), Cuboid)
        self.assertEqual(type(self.cub3), Cuboid)

    def test_eq(self):
        self.assertEqual(self.cub1, self.cub1)
        self.assertNotEqual(self.cub1, self.cub2)
        self.assertEqual(self.cub2, self.cub2)
        self.assertNotEqual(self.cub2, self.cub3)
        self.assertEqual(self.cub3, self.cub3)
        self.assertNotEqual(self.cub3, self.cub1)

    def test_contains(self):
        self.assertTrue(numpy.array([0.5]) in self.cub1)
        self.assertFalse(numpy.array([1.1]) in self.cub1)
        self.assertTrue(numpy.array([0.1, 0.9]) in self.cub2)
        self.assertFalse(numpy.array([-0.1, 0.9]) in self.cub2)
        self.assertTrue(numpy.array([0, 1, 0]) in self.cub3)
        self.assertFalse(numpy.array([2, 0, 1]) in self.cub3)
        self.assertRaises(ValueError, self.cub3.__contains__, numpy.array([1, 1]))

    def test_repr(self):
        check = "Cuboid({0},{1})".format(str(self.cub1.low_corner), str(self.cub1.high_corner))
        self.assertEqual(check, self.cub1.__repr__())
        check = "Cuboid({0},{1})".format(str(self.cub2.low_corner), str(self.cub2.high_corner))
        self.assertEqual(check, self.cub2.__repr__())
        check = "Cuboid({0},{1})".format(str(self.cub3.low_corner), str(self.cub3.high_corner))
        self.assertEqual(check, self.cub3.__repr__())

    def test_str(self):
        check = "Cuboid with:\n\tlow corner: {0},\n\thigh corner{1}.".format(str(self.cub1.low_corner),
                                                                             str(self.cub1.high_corner))
        self.assertEqual(check, str(self.cub1))
        check = "Cuboid with:\n\tlow corner: {0},\n\thigh corner{1}.".format(str(self.cub2.low_corner),
                                                                             str(self.cub2.high_corner))
        self.assertEqual(check, str(self.cub2))
        check = "Cuboid with:\n\tlow corner: {0},\n\thigh corner{1}.".format(str(self.cub3.low_corner),
                                                                             str(self.cub3.high_corner))
        self.assertEqual(check, str(self.cub3))

    def test_split(self):
        left_cub = Cuboid(numpy.array([0]), numpy.array([0.5]))
        right_cub = Cuboid(numpy.array([0.5]), numpy.array([1]))
        left_split, right_split = self.cub1.split()
        self.assertEqual(left_cub, left_split)
        self.assertEqual(right_cub, right_split)
        left_cub0 = Cuboid(numpy.array([0, 0]), numpy.array([0.5, 1]))
        right_cub0 = Cuboid(numpy.array([0.5, 0]), numpy.array([1, 1]))
        left_cub1 = Cuboid(numpy.array([0, 0]), numpy.array([1, 0.5]))
        right_cub1 = Cuboid(numpy.array([0, 0.5]), numpy.array([1, 1]))
        left_split0, right_split0 = self.cub2.split()
        left_split1, right_split1 = self.cub2.split(1)
        self.assertEqual(left_cub0, left_split0)
        self.assertEqual(right_cub0, right_split0)
        self.assertEqual(left_cub1, left_split1)
        self.assertEqual(right_cub1, right_split1)
        left_cub0 = Cuboid(numpy.array([0, 0, 0]), numpy.array([0.5, 1, 1]))
        right_cub0 = Cuboid(numpy.array([0.5, 0, 0]), numpy.array([1, 1, 1]))
        left_cub1 = Cuboid(numpy.array([0, 0, 0]), numpy.array([1, 0.5, 1]))
        right_cub1 = Cuboid(numpy.array([0, 0.5, 0]), numpy.array([1, 1, 1]))
        left_cub2 = Cuboid(numpy.array([0, 0, 0]), numpy.array([1, 1, 0.5]))
        right_cub2 = Cuboid(numpy.array([0, 0, 0.5]), numpy.array([1, 1, 1]))
        left_split0, right_split0 = self.cub3.split()
        left_split1, right_split1 = self.cub3.split(1)
        left_split2, right_split2 = self.cub3.split(2)
        self.assertEqual(left_cub0, left_split0)
        self.assertEqual(right_cub0, right_split0)
        self.assertEqual(left_cub1, left_split1)
        self.assertEqual(right_cub1, right_split1)
        self.assertEqual(left_cub2, left_split2)
        self.assertEqual(right_cub2, right_split2)

    def test_diameter(self):
        self.assertEqual(self.cub1.diameter(), 1)
        self.assertEqual(self.cub2.diameter(), numpy.sqrt(2))
        self.assertEqual(self.cub3.diameter(), numpy.sqrt(3))

    def test_distance(self):
        dist_cub1 = Cuboid(numpy.array([2]), numpy.array([3]))
        self.assertEqual(self.cub1.distance(dist_cub1), 1)
        dist_cub2 = Cuboid(numpy.array([2, 2]), numpy.array([3, 3]))
        self.assertEqual(self.cub1.distance(dist_cub2), numpy.sqrt(2))
        dist_cub3 = Cuboid(numpy.array([2, 2, 2]), numpy.array([3, 3, 3]))
        self.assertEqual(self.cub1.distance(dist_cub3), numpy.sqrt(3))
