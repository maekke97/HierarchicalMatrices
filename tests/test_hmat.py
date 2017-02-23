from unittest import TestCase

import numpy

from Hmat import HMat


class TestHmat(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content1 = numpy.matrix(numpy.zeros((3, 4)))
        cls.content2 = numpy.matrix(numpy.zeros((3, 2)))
        cls.content3 = numpy.matrix(numpy.zeros((4, 2)))
        cls.content4 = numpy.matrix(numpy.zeros((4, 4)))
        cls.content21 = numpy.matrix(numpy.ones((3, 4)))
        cls.hmat1 = HMat(content=cls.content1, shape=(3, 4), parent_index=(0, 0))
        cls.hmat21 = HMat(content=cls.content21, shape=(3, 4), parent_index=(0, 0))
        cls.hmat20 = HMat(blocks=[cls.hmat21], shape=(3, 4), parent_index=(0, 0))
        cls.hmat2 = HMat(content=cls.content2, shape=(3, 2), parent_index=(0, 4))
        cls.hmat3 = HMat(content=cls.content3, shape=(4, 2), parent_index=(3, 0))
        cls.hmat4 = HMat(content=cls.content4, shape=(4, 4), parent_index=(3, 2))
        cls.hmat = HMat(blocks=[cls.hmat1, cls.hmat2, cls.hmat3, cls.hmat4], shape=(7, 6), parent_index=(0, 0))
        cls.hmat_lvl2 = HMat(blocks=[cls.hmat20, cls.hmat2, cls.hmat3, cls.hmat4], shape=(7, 6), parent_index=(0, 0))

    def test_to_matrix(self):
        self.assertTrue(numpy.array_equal(self.hmat.to_matrix(), numpy.zeros((7, 6))))
        check_lvl2 = numpy.zeros((7, 6))
        check_lvl2[0:3, 0:4] = 1
        self.assertTrue(numpy.array_equal(self.hmat_lvl2.to_matrix(), check_lvl2))

    def test_content_mul(self):
        res = self.hmat21.content_mul(numpy.ones((4, 1)))
        self.assertTrue(numpy.array_equal(res, 4 * numpy.ones((3, 1))))
