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
        self.assertRaises(NotImplementedError, self.hmat_lvl2.__mul__, self.hmat_lvl2)

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
        res = hmat._mul_with_matrix(numpy.matrix(numpy.ones((10, 15))))
        check = numpy.matrix([[i] * 15 for i in xrange(55, 150, 10)])
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
