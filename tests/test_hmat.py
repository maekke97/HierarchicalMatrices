import random
from unittest import TestCase

import numpy

from HierMat.block_cluster_tree import build_block_cluster_tree
from HierMat.cluster import Cluster
from HierMat.cluster_tree import build_cluster_tree, admissible
from HierMat.grid import Grid
from HierMat.hmat import HMat, build_hmatrix
from HierMat.rmat import RMat
from HierMat.splitable import RegularCuboid


class TestHmat(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content1 = numpy.matrix(numpy.zeros((3, 4)))
        cls.content2 = numpy.matrix(numpy.zeros((3, 2)))
        cls.content3 = numpy.matrix(numpy.zeros((4, 2)))
        cls.content4 = numpy.matrix(numpy.zeros((4, 4)))
        cls.hmat1 = HMat(content=cls.content1, shape=(3, 4), parent_index=(0, 0))
        cls.hmat2 = HMat(content=cls.content2, shape=(3, 2), parent_index=(0, 4))
        cls.hmat3 = HMat(content=cls.content3, shape=(4, 2), parent_index=(3, 0))
        cls.hmat4 = HMat(content=cls.content4, shape=(4, 4), parent_index=(3, 2))
        cls.hmat = HMat(blocks=[cls.hmat1, cls.hmat2, cls.hmat3, cls.hmat4], shape=(7, 6), parent_index=(0, 0))
        cls.content21 = numpy.matrix(numpy.ones((3, 4)))
        cls.hmat21 = HMat(content=cls.content21, shape=(3, 4), parent_index=(0, 0))
        cls.hmat20 = HMat(blocks=[cls.hmat21], shape=(3, 4), parent_index=(0, 0))
        cls.hmat_lvl2 = HMat(blocks=[cls.hmat20, cls.hmat2, cls.hmat3, cls.hmat4], shape=(7, 6), parent_index=(0, 0))
        cls.cblock1 = numpy.matrix(numpy.ones((3, 2)))
        cls.cblock2 = numpy.matrix(numpy.ones((3, 1)))
        cls.cblock3 = numpy.matrix(numpy.ones((3, 3)))
        cls.cblock4 = numpy.matrix(numpy.ones((2, 2)))
        cls.cblock5 = numpy.matrix(numpy.ones((2, 1)))
        cls.cblock6 = numpy.matrix(numpy.ones((2, 3)))
        cls.cmat1 = HMat(content=cls.cblock1, shape=(3, 2), parent_index=(0, 0))
        cls.cmat2 = HMat(content=cls.cblock2, shape=(3, 1), parent_index=(0, 2))
        cls.cmat3 = HMat(content=cls.cblock3, shape=(3, 3), parent_index=(0, 3))
        cls.cmat4 = HMat(content=cls.cblock4, shape=(2, 2), parent_index=(3, 0))
        cls.cmat5 = HMat(content=cls.cblock5, shape=(2, 1), parent_index=(3, 2))
        cls.cmat6 = HMat(content=cls.cblock6, shape=(2, 3), parent_index=(3, 3))
        cls.consistent1 = HMat(blocks=[cls.cmat1, cls.cmat2, cls.cmat3, cls.cmat4, cls.cmat5, cls.cmat6],
                               shape=(5, 6), parent_index=(0, 0))
        cls.cmat1T = HMat(content=cls.cblock1.T, shape=(2, 3), parent_index=(0, 0))
        cls.cmat2T = HMat(content=cls.cblock2.T, shape=(1, 3), parent_index=(2, 0))
        cls.cmat3T = HMat(content=cls.cblock3.T, shape=(3, 3), parent_index=(3, 0))
        cls.cmat4T = HMat(content=cls.cblock4.T, shape=(2, 2), parent_index=(0, 3))
        cls.cmat5T = HMat(content=cls.cblock5.T, shape=(1, 2), parent_index=(2, 3))
        cls.cmat6T = HMat(content=cls.cblock6.T, shape=(3, 2), parent_index=(3, 3))
        cls.consistent2 = HMat(blocks=[cls.cmat1T, cls.cmat2T, cls.cmat3T, cls.cmat4T, cls.cmat5T, cls.cmat6T],
                               shape=(6, 5), parent_index=(0, 0))

    def test_abs_norm(self):
        self.assertEqual(abs(self.cmat3), 3.0)
        self.assertEqual(abs(self.cmat4), 2.0)
        self.assertEqual(abs(self.consistent1), 5.4772255750516612)
        rmat1 = RMat(numpy.matrix(numpy.ones((3, 1))), numpy.matrix(numpy.ones((3, 1))), max_rank=1)
        hmat1 = HMat(content=rmat1, shape=(3, 3), parent_index=(0, 0))
        hmat2 = HMat(content=rmat1, shape=(3, 3), parent_index=(0, 3))
        hmat3 = HMat(content=rmat1, shape=(3, 3), parent_index=(3, 0))
        hmat4 = HMat(content=rmat1, shape=(3, 3), parent_index=(3, 3))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(6, 6), parent_index=(0, 0))
        self.assertEqual(abs(hmat), 6.0)
        self.assertEqual(abs(hmat), hmat.norm())
        self.assertEqual(abs(hmat), hmat.norm('fro'))
        self.assertRaises(NotImplementedError, hmat.norm, 2)

    def test_get_item(self):
        self.assertEqual(self.consistent1[0], self.cmat1)
        self.assertEqual(self.consistent1[0, 0], self.cmat1)
        self.assertEqual(self.consistent2[4], self.cmat5T)
        self.assertEqual(self.consistent2[2, 3], self.cmat5T)

    def test_determine_block_structure(self):
        check = {(0, 0): (3, 4), (0, 4): (3, 2), (3, 0): (4, 2), (3, 2): (4, 4)}
        self.assertEqual(check, self.hmat.block_structure())

    def test_is_consistent(self):
        self.assertFalse(self.hmat.is_consistent())
        self.assertTrue(self.hmat1.is_consistent())
        self.assertTrue(self.consistent1.is_consistent())
        self.assertTrue(self.consistent2.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(1, 1))
        fail = HMat(blocks=[fail1], shape=(3, 2), parent_index=(0, 0))
        self.assertFalse(fail.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 0))
        fail2 = HMat(content=numpy.matrix(numpy.ones((2, 3))), shape=(2, 3), parent_index=(0, 3))
        fail = HMat(blocks=[fail1, fail2], shape=(3, 5), parent_index=(0, 0))
        self.assertFalse(fail.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 0))
        fail = HMat(blocks=[fail1], shape=(2, 2), parent_index=(0, 0))
        self.assertFalse(fail.is_consistent())
        fail1 = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        fail2 = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 3))
        fail3 = HMat(content=numpy.matrix(numpy.ones((2, 3))), shape=(2, 3), parent_index=(0, 6))
        fail = HMat(blocks=[fail1, fail2, fail3], shape=(3, 9), parent_index=(0, 0))
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
        self.assertFalse(HMat(content=self.content1, shape=(4, 4), parent_index=(0, 0)) == self.hmat1)
        self.assertFalse(HMat(content=self.content1, shape=(3, 4), parent_index=(1, 0)) == self.hmat1)
        self.assertFalse(HMat(blocks=[self.hmat1], shape=(3, 4), parent_index=(0, 0)) == self.hmat20)
        self.assertFalse(HMat(content=numpy.ones((3, 4)), shape=(3, 4), parent_index=(0, 0)) == self.hmat1)
        rmat = RMat(self.content1, self.content1)
        hmat = HMat(content=rmat, shape=(3, 3), parent_index=(0, 0))
        rmat2 = RMat(self.content2, self.content2)
        hmat2 = HMat(content=rmat2, shape=(3, 3), parent_index=(0, 0))
        self.assertFalse(hmat == hmat2)
        self.assertFalse(hmat == rmat)

    def test_neq(self):
        self.assertNotEqual(self.hmat2, self.hmat1)
        self.assertNotEqual(self.hmat20, self.hmat1)

    def test_add(self):
        addend1 = HMat(content=numpy.matrix(numpy.ones((3, 4))), shape=(3, 4), parent_index=(0, 0))
        addend2 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 4))
        addend3 = HMat(content=numpy.matrix(numpy.ones((4, 2))), shape=(4, 2), parent_index=(3, 0))
        addend4 = HMat(content=numpy.matrix(numpy.ones((4, 4))), shape=(4, 4), parent_index=(3, 2))
        addend_hmat = HMat(blocks=[addend1, addend2, addend3, addend4], shape=(7, 6), parent_index=(0, 0))
        splitter_mat = HMat(content=numpy.matrix(numpy.zeros((7, 6))), shape=(7, 6), parent_index=(0, 0))
        self.assertEqual(addend_hmat + splitter_mat, addend_hmat)
        self.assertEqual(splitter_mat + addend_hmat, addend_hmat)
        res = addend_hmat + self.hmat
        self.assertEqual(res, addend_hmat)
        self.assertRaises(ValueError, addend1.__add__, addend2)
        addend = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 0))
        self.assertRaises(ValueError, addend.__add__, addend2)
        self.assertRaises(NotImplementedError, addend_hmat.__add__, 'bla')
        self.assertRaises(NotImplementedError, addend_hmat.__add__, numpy.ones((7, 6)))
        addend_hmat = HMat(content=numpy.matrix(1), shape=(1, 1), parent_index=(0, 0))
        self.assertEqual(addend_hmat + 0, addend_hmat)
        addend_hmat = HMat(blocks=[addend1, addend2, addend3], shape=(7, 6), parent_index=(0, 0))
        self.assertRaises(ValueError, self.hmat.__add__, addend_hmat)
        check = HMat(content=numpy.matrix(2 * numpy.ones((3, 4))), shape=(3, 4), parent_index=(0, 0))
        res = addend1 + numpy.matrix(numpy.ones((3, 4)))
        self.assertEqual(res, check)
        self.assertRaises(ValueError, addend1._add_matrix, numpy.matrix(numpy.ones((3, 2))))
        rmat = RMat(numpy.matrix(numpy.ones((3, 2))), numpy.matrix(numpy.ones((3, 2))), 2)
        hmat = HMat(content=rmat, shape=(3, 3), parent_index=(0, 0))
        self.assertRaises(NotImplementedError, hmat.__add__, rmat)
        mat = numpy.matrix(numpy.zeros((7, 6)))
        res = addend_hmat + mat
        self.assertEqual(addend_hmat, res)
        mat = numpy.matrix(numpy.ones((7, 6)))
        res = addend_hmat + mat
        check = 2 * addend_hmat
        self.assertEqual(check, res)
        left = RMat(numpy.matrix(numpy.zeros((3, 1))), numpy.matrix(numpy.zeros((3, 1))), max_rank=1)
        left_mat = HMat(content=left, shape=(3, 3), parent_index=(0, 0))
        addend = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        self.assertAlmostEqual(numpy.linalg.norm((addend + left_mat).to_matrix()), numpy.linalg.norm(addend.to_matrix()))

    def test_radd(self):
        addend1 = HMat(content=numpy.matrix(numpy.ones((3, 4))), shape=(3, 4), parent_index=(0, 0))
        addend2 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 4))
        addend3 = HMat(content=numpy.matrix(numpy.ones((4, 2))), shape=(4, 2), parent_index=(3, 0))
        addend4 = HMat(content=numpy.matrix(numpy.ones((4, 4))), shape=(4, 4), parent_index=(3, 2))
        addend_hmat = HMat(blocks=[addend1, addend2, addend3, addend4], shape=(7, 6), parent_index=(0, 0))
        mat = numpy.matrix(numpy.zeros((7, 6)))
        res = addend_hmat.__radd__(mat)
        self.assertEqual(addend_hmat, res)

    def test_neg(self):
        negcmat1 = HMat(content=-self.cblock1, shape=(3, 2), parent_index=(0, 0))
        negcmat2 = HMat(content=-self.cblock2, shape=(3, 1), parent_index=(0, 2))
        negcmat3 = HMat(content=-self.cblock3, shape=(3, 3), parent_index=(0, 3))
        negcmat4 = HMat(content=-self.cblock4, shape=(2, 2), parent_index=(3, 0))
        negcmat5 = HMat(content=-self.cblock5, shape=(2, 1), parent_index=(3, 2))
        negcmat6 = HMat(content=-self.cblock6, shape=(2, 3), parent_index=(3, 3))
        negconsistent1 = HMat(blocks=[negcmat1, negcmat2, negcmat3, negcmat4, negcmat5, negcmat6],
                              shape=(5, 6), parent_index=(0, 0))
        self.assertEqual(-self.cmat1, negcmat1)
        self.assertEqual(-self.consistent1, negconsistent1)

    def test_minus(self):
        zercmat1 = HMat(content=numpy.matrix(numpy.zeros((3, 2))), shape=(3, 2), parent_index=(0, 0))
        zercmat2 = HMat(content=numpy.matrix(numpy.zeros((3, 1))), shape=(3, 1), parent_index=(0, 2))
        zercmat3 = HMat(content=numpy.matrix(numpy.zeros((3, 3))), shape=(3, 3), parent_index=(0, 3))
        zercmat4 = HMat(content=numpy.matrix(numpy.zeros((2, 2))), shape=(2, 2), parent_index=(3, 0))
        zercmat5 = HMat(content=numpy.matrix(numpy.zeros((2, 1))), shape=(2, 1), parent_index=(3, 2))
        zercmat6 = HMat(content=numpy.matrix(numpy.zeros((2, 3))), shape=(2, 3), parent_index=(3, 3))
        zerconsistent1 = HMat(blocks=[zercmat1, zercmat2, zercmat3, zercmat4, zercmat5, zercmat6],
                              shape=(5, 6), parent_index=(0, 0))
        self.assertEqual(self.cmat1 - self.cmat1, zercmat1)
        self.assertEqual(self.consistent1 - self.consistent1, zerconsistent1)
        res = self.hmat - self.hmat
        self.assertTrue(numpy.array_equal(res.to_matrix(), numpy.matrix(numpy.zeros((7, 6)))))

    def test_repr(self):
        check = '<HMat with {content}>'.format(content=self.hmat_lvl2.blocks)
        self.assertEqual(self.hmat_lvl2.__repr__(), check)

    def test_to_matrix(self):
        block1 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), parent_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), parent_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), parent_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), parent_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), parent_index=(0, 0))
        res = numpy.matrix([numpy.arange(i, i + 10) for i in xrange(1, 11)])
        self.assertTrue(numpy.array_equal(hmat.to_matrix(), res))
        self.assertTrue(numpy.array_equal(self.hmat.to_matrix(), numpy.zeros((7, 6))))
        check_lvl2 = numpy.zeros((7, 6))
        check_lvl2[0:3, 0:4] = 1
        self.assertTrue(numpy.array_equal(self.hmat_lvl2.to_matrix(), check_lvl2))
        rmat = RMat(numpy.matrix(numpy.ones((3, 1))), numpy.matrix(numpy.ones((3, 1))))
        hmat1 = HMat(content=rmat, shape=(3, 3), parent_index=(0, 0))
        hmat2 = HMat(content=rmat, shape=(3, 3), parent_index=(0, 3))
        hmat3 = HMat(content=rmat, shape=(3, 3), parent_index=(3, 0))
        hmat4 = HMat(content=rmat, shape=(3, 3), parent_index=(3, 3))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(6, 6), parent_index=(0, 0))
        self.assertTrue(numpy.array_equal(hmat.to_matrix(), numpy.matrix(numpy.ones((6, 6)))))
        blocks1 = [HMat(content=rmat, shape=(3, 3), parent_index=(i, j))
                   for i in xrange(0, 4, 3) for j in xrange(0, 4, 3)]
        block_mat1 = HMat(blocks=blocks1, shape=(6, 6), parent_index=(0, 0))
        block_mat2 = HMat(blocks=blocks1, shape=(6, 6), parent_index=(6, 0))
        block_mat3 = HMat(blocks=blocks1, shape=(6, 6), parent_index=(0, 6))
        block_mat4 = HMat(blocks=blocks1, shape=(6, 6), parent_index=(6, 6))
        outer_block = HMat(blocks=[block_mat1, block_mat2, block_mat3, block_mat4], shape=(12, 12), parent_index=(0, 0))
        self.assertTrue(numpy.array_equal(outer_block.to_matrix(), numpy.matrix(numpy.ones((12, 12)))))

    def test_mul(self):
        self.assertRaises(NotImplementedError, self.hmat_lvl2.__mul__, 'bla')

    def test_rmul(self):
        self.assertRaises(NotImplementedError, self.hmat.__rmul__, self.hmat)
        self.assertEqual(2 * self.hmat, self.hmat * 2)

    def test_mul_with_vector(self):
        block1 = numpy.matrix([numpy.arange(i, i+5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), parent_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), parent_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), parent_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), parent_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), parent_index=(0, 0))
        check = numpy.array([[i] for i in xrange(55, 150, 10)])
        res = hmat._mul_with_vector(numpy.ones((10, 1)))
        self.assertTrue(numpy.array_equal(check, res))
        self.assertRaises(ValueError, hmat._mul_with_vector, numpy.ones((11, 1)))

    def test_mul_with_matrix(self):
        block1 = numpy.matrix([numpy.arange(i, i+5) for i in xrange(1, 6)])
        block2 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(6, 11)])
        block4 = numpy.matrix([numpy.arange(i, i + 5) for i in xrange(11, 16)])
        hmat1 = HMat(content=block1, shape=(5, 5), parent_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), parent_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), parent_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), parent_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), parent_index=(0, 0))
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
        hmat1 = HMat(content=block1, shape=(5, 5), parent_index=(0, 0))
        hmat2 = HMat(content=block2, shape=(5, 5), parent_index=(5, 0))
        hmat3 = HMat(content=block2, shape=(5, 5), parent_index=(0, 5))
        hmat4 = HMat(content=block4, shape=(5, 5), parent_index=(5, 5))
        hmat = HMat(blocks=[hmat1, hmat2, hmat3, hmat4], shape=(10, 10), parent_index=(0, 0))
        check = numpy.matrix([[i for i in xrange(j, j+10)] for j in xrange(1, 11)])
        res = hmat * 1
        self.assertTrue(numpy.array_equal(res.to_matrix(), check))
        check *= 2
        res = hmat * 2
        self.assertTrue(numpy.array_equal(res.to_matrix(), check))

    def test_mul_with_rmat(self):
        rmat = RMat(self.content1, self.content1)
        hmat = HMat(content=rmat, shape=(3, 3), parent_index=(0, 0))
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
        hmat = HMat(content=self.content3, shape=(4, 2), parent_index=(0, 0))
        check = HMat(content=numpy.matrix(numpy.zeros((3, 2))), shape=(3, 2), parent_index=(0, 0))
        self.assertEqual(self.hmat1 * hmat, check)
        rmat = RMat(numpy.matrix(numpy.ones((3, 1))), right_mat=numpy.matrix(numpy.ones((3, 1))))
        hmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 3))
        hmat1 = HMat(content=rmat, shape=(3, 3), parent_index=(3, 0))
        check_rmat = RMat(numpy.matrix(3*numpy.ones((3, 1))), right_mat=numpy.matrix(numpy.ones((3, 1))))
        check = HMat(content=3*numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        self.assertEqual(hmat * hmat1, check)
        blocks = [HMat(content=numpy.matrix(1), shape=(1, 1), parent_index=(i, j)) for i in xrange(3) for j in xrange(3)]
        block_mat = HMat(blocks=blocks, shape=(3, 3), parent_index=(0, 0))
        hmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        hmat1 = HMat(content=rmat, shape=(3, 3), parent_index=(0, 0))
        self.assertTrue(numpy.array_equal((hmat1 * block_mat).to_matrix(), (3*hmat1).to_matrix()))
        self.assertTrue(numpy.array_equal((block_mat * hmat1).to_matrix(), (3 * hmat1).to_matrix()))
        res1 = HMat(content=numpy.matrix(6 * numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        res2 = HMat(content=numpy.matrix(6 * numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 3))
        res3 = HMat(content=numpy.matrix(6 * numpy.ones((2, 3))), shape=(2, 3), parent_index=(3, 0))
        res4 = HMat(content=numpy.matrix(6 * numpy.ones((2, 2))), shape=(2, 2), parent_index=(3, 3))
        res = HMat(blocks=[res1, res2, res3, res4], shape=(5, 5), parent_index=(0, 0))
        check = self.consistent1 * self.consistent2
        self.assertEqual(check, res)
        hmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        hmat2 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 3))
        hmat_1 = HMat(blocks=[hmat, hmat2], shape=(3, 5), parent_index=(0, 0))
        hmat3 = HMat(content=numpy.matrix(numpy.ones((2, 3))), shape=(2, 3), parent_index=(0, 0))
        hmat4 = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(2, 0))
        hmat_2 = HMat(blocks=[hmat3, hmat4], shape=(5, 3), parent_index=(0, 0))
        self.assertRaises(ValueError, hmat_1.__mul__, hmat_2)

    def test_split(self):
        self.assertRaises(NotImplementedError, self.hmat.restructure, self.hmat.block_structure())
        check = HMat(content='bla', shape=(2, 2), parent_index=(0, 0))
        self.assertRaises(NotImplementedError, check.restructure, {(0, 0): (1, 1)})
        splitter = HMat(content=RMat(numpy.matrix(numpy.ones((2, 1))), numpy.matrix(numpy.ones((2, 1)))),
                        shape=(2, 2), parent_index=(0, 0))
        check_blocks = [HMat(content=RMat(numpy.matrix(numpy.ones((1, 1))), numpy.matrix(numpy.ones((1, 1)))),
                             shape=(1, 1), parent_index=(i, j)) for i in xrange(2) for j in xrange(2)]
        check = HMat(blocks=check_blocks, shape=(2, 2), parent_index=(0, 0))
        self.assertEqual(splitter.restructure(check.block_structure()), check)

    def test_transpose(self):
        trans = self.hmat.transpose()
        full = self.hmat.to_matrix()
        self.assertTrue(numpy.array_equal(trans.to_matrix(), full.transpose()))
        hmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        hmat2 = HMat(content=numpy.matrix(numpy.ones((3, 2))), shape=(3, 2), parent_index=(0, 3))
        hmat_1 = HMat(blocks=[hmat, hmat2], shape=(3, 5), parent_index=(0, 0))
        thmat = HMat(content=numpy.matrix(numpy.ones((3, 3))), shape=(3, 3), parent_index=(0, 0))
        thmat2 = HMat(content=numpy.matrix(numpy.ones((2, 3))), shape=(2, 3), parent_index=(3, 0))
        thmat_1 = HMat(blocks=[thmat, thmat2], shape=(5, 3), parent_index=(0, 0))
        self.assertEqual(hmat_1.transpose(), thmat_1)

    def test_inv(self):
        self.assertRaises(numpy.linalg.LinAlgError, self.hmat1.inv)
        mat = numpy.matrix(numpy.eye(5))
        hmat = HMat(content=mat, shape=(5, 5), parent_index=(0, 0))
        self.assertEqual(hmat.inv(), hmat)
        zmat = numpy.matrix(numpy.zeros((5, 5)))
        hmat11 = HMat(content=mat, shape=(5, 5), parent_index=(0, 0))
        hmat12 = HMat(content=zmat, shape=(5, 5), parent_index=(0, 5))
        hmat21 = HMat(content=zmat, shape=(5, 5), parent_index=(5, 0))
        hmat22 = HMat(content=mat, shape=(5, 5), parent_index=(5, 5))
        hmat = HMat(blocks=[hmat11, hmat12, hmat21, hmat22], shape=(10, 10), parent_index=(0, 0))
        self.assertEqual(hmat, hmat.inv())
        hmat = hmat * 2
        self.assertEqual(0.25 * hmat, hmat.inv())
        hmat33 = HMat(content=mat, shape=(5, 5), parent_index=(10, 10))
        hmat13 = HMat(content=zmat, shape=(5, 5), parent_index=(0, 10))
        hmat31 = HMat(content=zmat, shape=(5, 5), parent_index=(10, 0))
        hmat23 = HMat(content=zmat, shape=(5, 5), parent_index=(5, 10))
        hmat32 = HMat(content=zmat, shape=(5, 5), parent_index=(10, 5))
        hmat = HMat(blocks=[hmat11, hmat12, hmat21, hmat22, hmat13, hmat31, hmat23, hmat32, hmat33],
                    shape=(15, 15), parent_index=(0, 0))
        hmat_inv = hmat.inv()
        self.assertEqual(hmat.norm(), hmat_inv.norm())
        rmat = RMat(left_mat=numpy.matrix(numpy.ones((3, 1))), right_mat=numpy.matrix(numpy.ones((3, 1))))
        rhmat = HMat(content=rmat, shape=(3, 3), parent_index=(0, 0))
        self.assertRaises(NotImplementedError, rhmat.inv)

    def test_build_hmatrix(self):
        full_func = lambda x: numpy.matrix(numpy.ones(x.shape()))
        block_func = lambda x: RMat(numpy.matrix(numpy.ones((x.shape()[0], 1))),
                                    numpy.matrix(numpy.ones((x.shape()[1], 1))),
                                    max_rank=1)
        lim1 = 2
        link_num = 4
        points1 = [numpy.array([float(i) / lim1]) for i in xrange(lim1)]
        links1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
        grid1 = Grid(points1, links1)
        cluster1 = Cluster(grid1)
        rc1 = RegularCuboid(cluster1)
        ct1 = build_cluster_tree(rc1, max_leaf_size=4)
        bct1 = build_block_cluster_tree(ct1, right_cluster_tree=ct1, admissible_function=admissible)
        hmat = build_hmatrix(bct1, generate_rmat_function=block_func, generate_full_matrix_function=block_func)
        check_rmat = RMat(numpy.matrix(numpy.ones((2, 1))), numpy.matrix(numpy.ones((2, 1))), max_rank=1)
        check = HMat(content=check_rmat, shape=(2, 2), parent_index=(0, 0))
        self.assertEqual(hmat, check)
        lim1 = 8
        link_num = 4
        points1 = [numpy.array([float(i) / lim1]) for i in xrange(lim1)]
        links1 = [[points1[l] for l in [random.randint(0, lim1 - 1) for x in xrange(link_num)]] for i in xrange(lim1)]
        grid1 = Grid(points1, links1)
        cluster1 = Cluster(grid1)
        rc1 = RegularCuboid(cluster1)
        ct1 = build_cluster_tree(rc1, max_leaf_size=4)
        bct1 = build_block_cluster_tree(ct1, right_cluster_tree=ct1, admissible_function=admissible)
        hmat = build_hmatrix(bct1, generate_rmat_function=block_func, generate_full_matrix_function=block_func)
        check_rmat = RMat(numpy.matrix(numpy.ones((4, 1))), numpy.matrix(numpy.ones((4, 1))), max_rank=1)
        check1 = HMat(content=check_rmat, shape=(4, 4), parent_index=(0, 0))
        check2 = HMat(content=check_rmat, shape=(4, 4), parent_index=(0, 4))
        check3 = HMat(content=check_rmat, shape=(4, 4), parent_index=(4, 0))
        check4 = HMat(content=check_rmat, shape=(4, 4), parent_index=(4, 4))
        check = HMat(blocks=[check1, check2, check3, check4], shape=(8, 8), parent_index=(0, 0))
        self.assertEqual(hmat, check)
        bct1 = build_block_cluster_tree(ct1, right_cluster_tree=ct1, admissible_function=lambda x, y: True)
        hmat = build_hmatrix(bct1, generate_rmat_function=block_func, generate_full_matrix_function=block_func)
        self.assertIsInstance(hmat, HMat)
