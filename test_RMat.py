from unittest import TestCase

import numpy

from Rmat import RMat


class TestRMat(TestCase):
    def test_setup(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat = RMat(left_block, right_block, 3)
        self.assertIsInstance(rmat, RMat)

    def test_initExceptions(self):
        left_block = numpy.matrix([[1, 2], [2, 3], [3, 4]])
        right_block = numpy.matrix([[1], [2], [3]])
        self.assertRaises(ValueError, RMat, left_block, right_block, 2)
        self.assertRaises(ValueError, RMat, left_block, left_block, 1)

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

    def test_toMatrix(self):
        left = numpy.matrix([[1], [2], [3]])
        right = numpy.matrix([[2], [3], [4]])
        rmat = RMat(left, right, 1)
        check_mat = numpy.matrix([[2, 3, 4], [4, 6, 8], [6, 9, 12]])
        self.assertTrue(numpy.array_equal(rmat.to_matrix(), check_mat))

    def test_reduced_qr_decomp(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat = RMat(left_block, right_block, 3)
        red_rmat = rmat.reduce(2)
        res_a = numpy.matrix([[30.52901104, -3.60036909],
                              [31.80619274, 2.24569436],
                              [32.3747334, 1.18885057]
                              ])
        res_b = numpy.matrix([[0.16125158, -0.98446933],
                              [0.85768556, 0.17458427],
                              [0.488235, 0.01845182]
                              ])
        res = RMat(res_a, res_b, 2)
        self.assertAlmostEqual(red_rmat, res)
