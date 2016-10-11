from unittest import TestCase

import numpy

from Hmat import RMat


class TestRMat(TestCase):
    def test_setup(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat = RMat(left_block, right_block, 3)
        self.assertIsInstance(rmat, RMat)

    def test_add(self):
        left1 = numpy.matrix([[1], [2], [3]])
        left2 = numpy.matrix([[2], [3], [4]])
        add_left = numpy.matrix([[1, 2], [2, 3], [3, 4]])
        right1 = numpy.matrix([[5], [6], [7]])
        right2 = numpy.matrix([[4], [5], [6]])
        add_right = numpy.matrix([[5, 4], [6, 5], [7, 6]])
        rmat1 = RMat(left1, right1, 1)
        rmat2 = RMat(left2, right2, 1)
        addmat = RMat(add_left, add_right, 2)
        self.assertEqual(rmat1 + rmat2, addmat)
