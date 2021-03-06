from unittest import TestCase

import numpy

from HierMat.rmat import RMat


class TestRMat(TestCase):
    def test_setup(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat = RMat(left_block, right_block)
        self.assertIsInstance(rmat, RMat)
        rmat = RMat(left_block, right_block, 1)
        self.assertIsInstance(rmat, RMat)
        rmat = RMat(left_block, right_block, 4)
        self.assertIsInstance(rmat, RMat)

    def test_eq(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat1 = RMat(left_block, right_block)
        left_block = numpy.matrix([[3, 2, 1], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 1, 4], [4, 3, 2], [3, 4, 2]])
        rmat2 = RMat(left_block, right_block)
        self.assertEqual(rmat1, rmat1)
        self.assertFalse(rmat1 == rmat2)

    def test_ne(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat1 = RMat(left_block, right_block)
        left_block = numpy.matrix([[3, 2, 1], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 1, 4], [4, 3, 2], [3, 4, 2]])
        rmat2 = RMat(left_block, right_block)
        self.assertNotEqual(rmat1, rmat2)
        self.assertFalse(rmat1 != rmat1)

    def test_initExceptions(self):
        left_block = numpy.matrix([[1, 2], [2, 3], [3, 4]])
        right_block = numpy.matrix([[1], [2], [3]])
        self.assertRaises(ValueError, RMat, left_block, right_block, 2)
        self.assertRaises(ValueError, RMat, left_block)
        self.assertRaises(ValueError, RMat, left_block, max_rank=0)

    def test_from_matrix(self):
        mat = numpy.matrix(numpy.ones((3, 3)))
        check = RMat(mat, max_rank=1)
        self.assertAlmostEqual(check.norm(), numpy.linalg.norm(mat))
        check = RMat(mat, max_rank=4)
        self.assertAlmostEqual(check.norm(), numpy.linalg.norm(mat))

    def test_str(self):
        check = '''Rank-k matrix with left block:
[[1]
 [2]
 [3]]
and right block:
[[4]
 [5]
 [6]]'''
        rmat = RMat(numpy.matrix([[1], [2], [3]]), numpy.matrix([[4], [5], [6]]), 1)
        self.assertEqual(str(rmat), check)

    def test_repr(self):
        check = '<RMat with left_mat: matrix([[1],[2],[3]]), right_mat: matrix([[4],[5],[6]]) and max_rank: 1>'
        rmat = RMat(numpy.matrix([[1], [2], [3]]), numpy.matrix([[4], [5], [6]]), 1)
        self.assertEqual(rmat.__repr__(), check)

    def test_add(self):
        left1 = numpy.matrix([[1], [2], [3]])
        left2 = numpy.matrix([[2], [3], [4]])
        add_left = numpy.matrix([[1, 2], [2, 3], [3, 4]])
        right1 = numpy.matrix([[5], [6], [7]])
        right2 = numpy.matrix([[4], [5], [6]])
        add_right = numpy.matrix([[5, 4], [6, 5], [7, 6]])
        rmat1 = RMat(left1, right1)
        rmat2 = RMat(left2, right2)
        add_mat = RMat(add_left, add_right)
        res = rmat1 + rmat2
        self.assertEqual(res, add_mat)
        self.assertAlmostEqual(abs(rmat1 + rmat2), abs(rmat2 + rmat1), 6)
        self.assertRaises(ValueError, rmat1.__add__, numpy.ones((2, 4)))
        self.assertRaises(ValueError, rmat1.__add__, 1)
        res = rmat1 + numpy.matrix(numpy.ones((3, 3)))
        check = rmat1.to_matrix() + numpy.matrix(numpy.ones((3, 3)))
        self.assertEqual(res.norm(), numpy.linalg.norm(check))
        self.assertRaises(NotImplementedError, rmat1.__add__, 'bla')
        self.assertRaises(NotImplementedError, rmat1.__add__, numpy.ones((3, 3)))

    def test_radd_(self):
        left1 = numpy.matrix([[1], [2], [3]])
        right1 = numpy.matrix([[5], [6], [7]])
        rmat1 = RMat(left1, right1)
        addend = numpy.matrix(numpy.ones((3, 3)))
        res = rmat1.__radd__(addend)
        check = rmat1.to_matrix() + numpy.matrix(numpy.ones((3, 3)))
        self.assertEqual(res.norm(), numpy.linalg.norm(check))

    def test_neg(self):
        left1 = numpy.matrix([[1], [2], [3]])
        right1 = numpy.matrix([[2], [3], [4]])
        rmat_pos = RMat(left1, right1, 1)
        rmat_neg = RMat(-left1, right1, 1)
        self.assertEqual(-rmat_pos, rmat_neg)

    def test_minus(self):
        left1 = numpy.matrix([[1], [2], [3]])
        left2 = numpy.matrix([[2], [3], [4]])
        sub_left = numpy.matrix([[1, -2], [2, -3], [3, -4]])
        right1 = numpy.matrix([[5], [6], [7]])
        right2 = numpy.matrix([[4], [5], [6]])
        sub_right = numpy.matrix([[5, 4], [6, 5], [7, 6]])
        rmat1 = RMat(left1, right1, 1)
        rmat2 = RMat(left2, right2, 1)
        sub_mat = RMat(sub_left, sub_right, 2)
        minus = rmat1 - rmat2
        res = numpy.matrix([[-3, -4, -5], [-2, -3, -4], [-1, -2, -3]])
        self.assertEqual(minus, sub_mat.reduce(1))

    def test_abs(self):
        left = numpy.matrix([[1], [2], [3]])
        right = numpy.matrix([[4], [5], [6]])
        rmat = RMat(left, right, 1)
        self.assertAlmostEqual(abs(rmat), 32.832910319, places=8)

    def test_mul(self):
        left1 = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right1 = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat1 = RMat(left1, right1, 3)
        left2 = numpy.matrix([[1, 2], [2, 2], [4, 3]])
        right2 = numpy.matrix([[2, 3], [1, 5], [5, 1]])
        rmat2 = RMat(left2, right2, 2)
        left3 = numpy.matrix([[1], [2], [3], [4]])
        right3 = numpy.matrix([[4], [5], [6], [7]])
        rmat3 = RMat(left3, right3, 1)
        left4 = numpy.matrix([[1], [2], [3]])
        right4 = numpy.matrix([[4], [5], [6]])
        rmat4 = RMat(left4, right4, 1)
        res1 = rmat1 * rmat2
        res2 = rmat4 * rmat2
        self.assertEqual(res1.max_rank, 3)
        self.assertEqual(res2.max_rank, 2)
        self.assertRaises(ValueError, rmat2.__mul__, rmat3)
        self.assertRaises(ValueError, rmat1.__mul__, rmat3)
        mat = numpy.matrix(numpy.ones((3, 1)))
        res = RMat(left1, mat.T * right1, 3)
        self.assertEqual(res, rmat1 * mat)
        self.assertRaises(ValueError, rmat1.__rmul__, mat)
        mat = numpy.matrix(numpy.ones((1, 3)))
        check = mat * left1 * right1.T
        self.assertTrue(numpy.array_equal(mat * rmat1, check))
        mat = numpy.matrix(numpy.ones((4, 1)))
        self.assertRaises(ValueError, rmat1.__mul__, mat)
        mat = numpy.ones((3, 1))
        res = left1 * (right1.T * mat)
        self.assertTrue(numpy.array_equal(res, rmat1 * mat))
        mat = numpy.ones((4, 1))
        self.assertRaises(ValueError, rmat1.__mul__, mat)
        self.assertRaises(NotImplementedError, rmat1.__mul__, 'a')
        self.assertRaises(NotImplementedError, rmat1.__rmul__, 'a')
        self.assertEqual(rmat1 * 1, rmat1)
        self.assertEqual(1 * rmat1, rmat1)

    def test_norm(self):
        left = numpy.matrix([[1], [2], [3]])
        right = numpy.matrix([[2], [3], [4]])
        rmat = RMat(left, right, 1)
        self.assertEqual(rmat.norm(), abs(rmat))

    def test_toMatrix(self):
        left = numpy.matrix([[1], [2], [3]])
        right = numpy.matrix([[2], [3], [4]])
        rmat = RMat(left, right, 1)
        check_mat = numpy.matrix([[2, 3, 4], [4, 6, 8], [6, 9, 12]])
        self.assertTrue(numpy.array_equal(rmat.to_matrix(), check_mat))

    def test_reduce(self):
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
        self.assertAlmostEqual(red_rmat, res, places=6)

    def test_form_add(self):
        left1 = numpy.matrix([[1, 2], [2, 2], [4, 3]])
        right1 = numpy.matrix([[2, 3], [1, 5], [5, 1]])
        left2 = numpy.matrix([[4, 1, 3], [5, 1, 1], [1, 3, 1]])
        right2 = numpy.matrix([[1, 1, 1], [2, 1, 3], [5, 1, 5]])
        rmat1 = RMat(left1, right1)
        rmat2 = RMat(left2, right2)
        added = rmat1 + rmat2
        added2 = rmat2 + rmat1
        self.assertEqual(added.reduce(2), rmat1.form_add(rmat2, 2))
        self.assertEqual(added.reduce(1), rmat1.form_add(rmat2, 1))
        self.assertEqual(added2.reduce(2), rmat2.form_add(rmat1, 2))
        self.assertEqual(added2.reduce(1), rmat2.form_add(rmat1, 1))
        rmat1 = RMat(left1, right1, 2)
        rmat2 = RMat(left2, right2, 3)
        added = rmat1 + rmat2
        self.assertEqual(added, rmat1.form_add(rmat2))

    def test_type(self):
        left1 = numpy.matrix([[1, 2], [2, 2], [4, 3]])
        right1 = numpy.matrix([[2, 3], [1, 5], [5, 1]])
        rmat1 = RMat(left1, right1, 2)
        self.assertEqual(type(rmat1), RMat)

    def test_split(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat = RMat(left_block, right_block)
        check1 = RMat(numpy.matrix([[1, 2, 3]]), numpy.matrix([[2, 3, 4]]))
        check2 = RMat(numpy.matrix([[3, 2, 1], [2, 3, 1]]), numpy.matrix([[4, 3, 2], [3, 4, 2]]))
        check3 = RMat(numpy.matrix([[1, 2, 3], [3, 2, 1]]), numpy.matrix([[4, 3, 2], [3, 4, 2]]))
        self.assertEqual(rmat.split(0, 0, 1, 1), check1)
        self.assertEqual(rmat.split(1, 1, 3, 3), check2)
        self.assertEqual(rmat.split(0, 1, 2, 3), check3)

    def test_transpose(self):
        left_block = numpy.matrix([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
        right_block = numpy.matrix([[2, 3, 4], [4, 3, 2], [3, 4, 2]])
        rmat = RMat(left_block, right_block)
        self.assertEqual(rmat.transpose(), RMat(left_mat=right_block, right_mat=left_block))
        trans = rmat.transpose()
        full = rmat.to_matrix()
        self.assertTrue(numpy.array_equal(trans.to_matrix(), full.transpose()))
