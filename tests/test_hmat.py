from unittest import TestCase

import numpy

from HierMat.hmat import HMat
from HierMat.rmat import RMat


class TestHmat(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content1 = numpy.matrix(numpy.zeros((3, 4)))
        cls.content2 = numpy.matrix(numpy.zeros((3, 2)))
        cls.content3 = numpy.matrix(numpy.zeros((4, 2)))
        cls.content4 = numpy.matrix(numpy.zeros((4, 4)))
        cls.hmat1 = HMat(content=cls.content1, shape=(3, 4), root_index=(0, 0))
        cls.hmat2 = HMat(content=cls.content2, shape=(3, 2), root_index=(0, 4))
        cls.hmat3 = HMat(content=cls.content3, shape=(4, 2), root_index=(3, 0))
        cls.hmat4 = HMat(content=cls.content4, shape=(4, 4), root_index=(3, 2))
        cls.hmat = HMat(blocks=[cls.hmat1, cls.hmat2, cls.hmat3, cls.hmat4], shape=(7, 6), root_index=(0, 0))
        cls.content21 = numpy.matrix(numpy.ones((3, 4)))
        cls.hmat21 = HMat(content=cls.content21, shape=(3, 4), root_index=(0, 0))
        cls.hmat20 = HMat(blocks=[cls.hmat21], shape=(3, 4), root_index=(0, 0))
        cls.hmat_lvl2 = HMat(blocks=[cls.hmat20, cls.hmat2, cls.hmat3, cls.hmat4], shape=(7, 6), root_index=(0, 0))
        cls.cblock1 = numpy.matrix(numpy.ones((3, 2)))
        cls.cblock2 = numpy.matrix(numpy.ones((3, 1)))
        cls.cblock3 = numpy.matrix(numpy.ones((3, 3)))
        cls.cblock4 = numpy.matrix(numpy.ones((2, 2)))
        cls.cblock5 = numpy.matrix(numpy.ones((2, 1)))
        cls.cblock6 = numpy.matrix(numpy.ones((2, 3)))
        cls.cmat1 = HMat(content=cls.cblock1, shape=(3, 2), root_index=(0, 0))
        cls.cmat2 = HMat(content=cls.cblock2, shape=(3, 1), root_index=(0, 2))
        cls.cmat3 = HMat(content=cls.cblock3, shape=(3, 3), root_index=(0, 3))
        cls.cmat4 = HMat(content=cls.cblock4, shape=(2, 2), root_index=(3, 0))
        cls.cmat5 = HMat(content=cls.cblock5, shape=(2, 1), root_index=(3, 2))
        cls.cmat6 = HMat(content=cls.cblock6, shape=(2, 3), root_index=(3, 3))
        cls.consistent1 = HMat(blocks=[cls.cmat1, cls.cmat2, cls.cmat3, cls.cmat4, cls.cmat5, cls.cmat6],
                               shape=(5, 6), root_index=(0, 0))
        cls.cmat1T = HMat(content=cls.cblock1.T, shape=(2, 3), root_index=(0, 0))
        cls.cmat2T = HMat(content=cls.cblock2.T, shape=(1, 3), root_index=(2, 0))
        cls.cmat3T = HMat(content=cls.cblock3.T, shape=(3, 3), root_index=(3, 0))
        cls.cmat4T = HMat(content=cls.cblock4.T, shape=(2, 2), root_index=(0, 3))
        cls.cmat5T = HMat(content=cls.cblock5.T, shape=(1, 2), root_index=(2, 3))
        cls.cmat6T = HMat(content=cls.cblock6.T, shape=(3, 2), root_index=(3, 3))
        cls.consistent2 = HMat(blocks=[cls.cmat1T, cls.cmat2T, cls.cmat3T, cls.cmat4T, cls.cmat5T, cls.cmat6T],
                               shape=(6, 5), root_index=(0, 0))

    def test_get_item(self):
        self.assertEqual(self.consistent1[0], self.cmat1)
        self.assertEqual(self.consistent1[0, 0], self.cmat1)
        self.assertEqual(self.consistent2[4], self.cmat5T)
        self.assertEqual(self.consistent2[2, 3], self.cmat5T)

    def test_determine_block_structure(self):
        check = {(0, 0): (3, 4), (0, 4): (3, 2), (3, 0): (4, 2), (3, 2): (4, 4)}
        self.assertEqual(check, self.hmat._block_structure())

    def test_is_consistent(self):
        self.assertFalse(self.hmat.is_consistent())
        self.assertTrue(self.hmat1.is_consistent())
        self.assertTrue(self.consistent1.is_consistent())
        self.assertTrue(self.consistent2.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), root_index=(1, 1))
        fail = HMat(blocks=[fail1], shape=(3, 2), root_index=(0, 0))
        self.assertFalse(fail.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), root_index=(1, 1))
        fail2 = HMat(content=numpy.matrix(numpy.ones((2, 3))), shape=(2, 3), root_index=(1, 3))
        fail = HMat(blocks=[fail1, fail2], shape=(3, 5), root_index=(1, 1))
        self.assertFalse(fail.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), root_index=(0, 0))
        fail = HMat(blocks=[fail1], shape=(2, 2), root_index=(0, 0))
        self.assertFalse(fail.is_consistent())

    def test_column_sequence(self):
        check = [2, 1, 3]
        self.assertEqual(self.consistent1.column_sequence(), check)
        self.assertRaises(Warning, self.hmat.column_sequence)
        self.assertEqual(self.hmat1.column_sequence(), [4])
        check = [3, 2]
        self.assertEqual(self.consistent2.column_sequence(), check)

    def test_row_sequence(self):
        check = [3, 2]
        self.assertEqual(self.consistent1.row_sequence(), check)
        self.assertRaises(Warning, self.hmat.row_sequence)
        self.assertEqual(self.hmat1.row_sequence(), [3])
        check = [2, 1, 3]
        self.assertEqual(self.consistent2.row_sequence(), check)

    def test_eq(self):
        self.assertEqual(self.hmat1, self.hmat1)
        self.assertEqual(self.hmat, self.hmat)
        self.assertEqual(self.hmat_lvl2, self.hmat_lvl2)
        self.assertFalse(self.hmat2 == self.hmat1)
        self.assertFalse(self.hmat20 == self.hmat1)
        self.assertFalse(HMat(content=self.content1, shape=(4, 4), root_index=(0, 0)) == self.hmat1)
        self.assertFalse(HMat(content=self.content1, shape=(3, 4), root_index=(1, 0)) == self.hmat1)
        self.assertFalse(HMat(blocks=[self.hmat1], shape=(3, 4), root_index=(0, 0)) == self.hmat20)
        self.assertFalse(HMat(content=numpy.ones((3, 4)), shape=(3, 4), root_index=(0, 0)) == self.hmat1)
        rmat = RMat(self.content1, self.content1)
        hmat = HMat(content=rmat, shape=(3, 3), root_index=(0, 0))
        rmat2 = RMat(self.content2, self.content2)
        hmat2 = HMat(content=rmat2, shape=(3, 3), root_index=(0, 0))
        self.assertFalse(hmat == hmat2)
        self.assertFalse(hmat == rmat)

    def test_neq(self):
        self.assertNotEqual(self.hmat2, self.hmat1)
        self.assertNotEqual(self.hmat20, self.hmat1)

    def test_add(self):
        addend1 = HMat(content=numpy.matrix(numpy.ones((3, 4))), shape=(3, 4), root_index=(0, 0))
        addend2 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), root_index=(0, 4))
        addend3 = HMat(content=numpy.matrix(numpy.ones((4, 2))), shape=(4, 2), root_index=(3, 0))
        addend4 = HMat(content=numpy.matrix(numpy.ones((4, 4))), shape=(4, 4), root_index=(3, 2))
        addend_hmat = HMat(blocks=[addend1, addend2, addend3, addend4], shape=(7, 6), root_index=(0, 0))
        res = addend_hmat + self.hmat
        self.assertEqual(res, addend_hmat)
        self.assertRaises(ValueError, addend1.__add__, addend2)
        addend = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), root_index=(0, 0))
        self.assertRaises(ValueError, addend.__add__, addend2)
        addend = HMat(content=numpy.matrix(numpy.ones((7, 6))), shape=(7, 6), root_index=(0, 0))
        self.assertRaises(ValueError, addend.__add__, addend_hmat)
        self.assertRaises(NotImplementedError, addend_hmat.__add__, 'bla')
        self.assertRaises(NotImplementedError, addend_hmat.__add__, numpy.ones((7, 6)))
        addend_hmat = HMat(blocks=[addend1, addend2, addend3], shape=(7, 6), root_index=(0, 0))
        self.assertRaises(ValueError, self.hmat.__add__, addend_hmat)
        check = HMat(content=numpy.matrix(2 * numpy.ones((3, 4))), shape=(3, 4), root_index=(0, 0))
        res = addend1 + numpy.matrix(numpy.ones((3, 4)))
        self.assertEqual(res, check)
        self.assertRaises(ValueError, addend1._add_matrix, numpy.matrix(numpy.ones((3, 2))))
        rmat = RMat(numpy.matrix(numpy.ones((3, 2))), numpy.matrix(numpy.ones((3, 2))), 2)
        hmat = HMat(content=rmat, shape=(3, 3), root_index=(0, 0))
        self.assertRaises(NotImplementedError, hmat.__add__, rmat)
        mat = numpy.matrix(numpy.zeros((7, 6)))
        res = addend_hmat + mat
        self.assertEqual(addend_hmat, res)
        mat = numpy.matrix(numpy.ones((7, 6)))
        res = addend_hmat + mat
        check = 2 * addend_hmat
        self.assertEqual(check, res)

    def test_repr(self):
        check = '<HMat with {content}>'.format(content=self.hmat_lvl2.blocks)
        self.assertEqual(self.hmat_lvl2.__repr__(), check)

    def test_to_matrix(self):
        block1 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), root_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), root_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), root_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), root_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), root_index=(0, 0))
        res = numpy.matrix([numpy.arange(i, i + 10) for i in xrange(1, 11)])
        self.assertTrue(numpy.array_equal(hmat.to_matrix(), res))
        self.assertTrue(numpy.array_equal(self.hmat.to_matrix(), numpy.zeros((7, 6))))
        check_lvl2 = numpy.zeros((7, 6))
        check_lvl2[0:3, 0:4] = 1
        self.assertTrue(numpy.array_equal(self.hmat_lvl2.to_matrix(), check_lvl2))

    def test_mul(self):
        self.assertRaises(NotImplementedError, self.hmat_lvl2.__mul__, 'bla')

    def test_rmul(self):
        self.assertRaises(NotImplementedError, self.hmat.__rmul__, self.hmat)
        self.assertEqual(2 * self.hmat, self.hmat * 2)

    def test_mul_with_vector(self):
        block1 = numpy.matrix([numpy.arange(i, i+5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), root_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), root_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), root_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), root_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), root_index=(0, 0))
        check = numpy.array([[i] for i in xrange(55, 150, 10)])
        res = hmat._mul_with_vector(numpy.ones((10, 1)))
        self.assertTrue(numpy.array_equal(check, res))
        self.assertRaises(ValueError, hmat._mul_with_vector, numpy.ones((11, 1)))

    def test_mul_with_matrix(self):
        block1 = numpy.matrix([numpy.arange(i, i+5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), root_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), root_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), root_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), root_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), root_index=(0, 0))
        check = numpy.matrix([[i]*10 for i in xrange(55, 150, 10)])
        res = hmat._mul_with_matrix(numpy.matrix(numpy.ones((10, 10))))
        self.assertTrue(numpy.array_equal(check, res))
        mult = numpy.matrix(numpy.ones((10, 10)))
        res = hmat * mult
        check = numpy.matrix([[i] * 10 for i in xrange(55, 150, 10)])
        self.assertTrue(numpy.array_equal(res, check))
        self.assertRaises(ValueError, hmat._mul_with_matrix, numpy.ones((11, 10)))
        self.assertRaises(ValueError, hmat._mul_with_matrix, numpy.ones((9, 11)))

    def test_mul_with_int(self):
        block1 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), root_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), root_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), root_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), root_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), root_index=(0, 0))
        check = numpy.matrix([[i for i in xrange(j, j+10)] for j in xrange(1, 11)])
        res = hmat * 1
        self.assertTrue(numpy.array_equal(res.to_matrix(), check))
        check *= 2
        res = hmat * 2
        self.assertTrue(numpy.array_equal(res.to_matrix(), check))

    def test_mul_with_rmat(self):
        rmat = RMat(self.content1, self.content1)
        hmat = HMat(content=rmat, shape=(3, 3), root_index=(0, 0))
        prod = hmat * rmat
        self.assertEqual(prod, hmat)
        rmat = RMat(self.content3, self.content3)
        prod = self.hmat1 * rmat
        self.assertTrue(numpy.array_equal(prod.to_matrix(), self.hmat1.to_matrix()))
        rmat = RMat(numpy.matrix(numpy.ones((7, 3))), numpy.matrix(numpy.ones((6, 3))))
        self.assertRaises(TypeError, self.hmat_lvl2.__mul__, rmat)

    def test_mul_with_hmat(self):
        self.assertRaises(ValueError, self.hmat.__mul__, self.hmat)
        self.assertRaises(ValueError, self.hmat1.__mul__, self.hmat3)
        hmat = HMat(content=self.content3, shape=(4, 2), root_index=(0, 0))
        check = HMat(content=numpy.matrix(numpy.zeros((3, 2))), shape=(3, 2), root_index=(0, 0))
        self.assertEqual(self.hmat1 * hmat, check)
        rmat = RMat(numpy.matrix(numpy.ones((3, 1))), right_mat=numpy.matrix(numpy.ones((3, 1))))
        hmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), root_index=(0, 3))
        hmat1 = HMat(content=rmat, shape=(3, 3), root_index=(3, 0))
        check_rmat = RMat(numpy.matrix(3*numpy.ones((3, 1))), right_mat=numpy.matrix(numpy.ones((3, 1))))
        check = HMat(content=check_rmat, shape=(3, 3), root_index=(0, 0))
        self.assertEqual(hmat * hmat1, check)
        res1 = HMat(content=numpy.matrix(6 * numpy.ones((3, 3))), shape=(3, 3), root_index=(0, 0))
        res2 = HMat(content=numpy.matrix(6 * numpy.ones((3, 2))), shape=(3, 2), root_index=(0, 3))
        res3 = HMat(content=numpy.matrix(6 * numpy.ones((2, 3))), shape=(2, 3), root_index=(3, 0))
        res4 = HMat(content=numpy.matrix(6 * numpy.ones((2, 2))), shape=(2, 2), root_index=(3, 3))
        res = HMat(blocks=[res1, res2, res3, res4], shape=(5, 5), root_index=(0, 0))
        check = self.consistent1 * self.consistent2
        self.assertEqual(check, res)
        hmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), root_index=(0, 0))
        hmat2 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), root_index=(0, 3))
        hmat_1 = HMat(blocks=[hmat, hmat2], shape=(3, 5), root_index=(0, 0))
        hmat3 = HMat(content=numpy.matrix(numpy.ones((2, 3))), shape=(2, 3), root_index=(0, 0))
        hmat4 = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), root_index=(2, 0))
        hmat_2 = HMat(blocks=[hmat3, hmat4], shape=(5, 3), root_index=(0, 0))
        self.assertRaises(ValueError, hmat_1.__mul__, hmat_2)
