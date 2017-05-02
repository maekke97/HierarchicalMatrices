from unittest import TestCase
from HierMat import RMat, HMat
import numpy


class TestMath(TestCase):
    def test_rmat_vec_mul(self):
        left = numpy.matrix(numpy.random.rand(3, 1))
        right = numpy.matrix(numpy.random.rand(3, 1))
        rmat = RMat(left, right, 1)
        x = numpy.random.rand(3, 1)
        check1 = rmat.to_matrix() * x
        check2 = rmat * x
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=14)
        left = numpy.matrix(numpy.random.rand(5, 2))
        right = numpy.matrix(numpy.random.rand(4, 2))
        rmat = RMat(left, right, 2)
        x = numpy.random.rand(4, 1)
        check1 = rmat.to_matrix() * x
        check2 = rmat * x
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=14)

    def test_rmat_add_exact(self):
        left1 = numpy.matrix(numpy.random.rand(3, 1))
        right1 = numpy.matrix(numpy.random.rand(3, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(3, 1))
        right2 = numpy.matrix(numpy.random.rand(3, 1))
        rmat2 = RMat(left2, right2)
        check1 = rmat1.to_matrix() + rmat2.to_matrix()
        check2 = (rmat1 + rmat2).to_matrix()
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=14)
        left1 = numpy.matrix(numpy.random.rand(5, 1))
        right1 = numpy.matrix(numpy.random.rand(4, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(5, 1))
        right2 = numpy.matrix(numpy.random.rand(4, 1))
        rmat2 = RMat(left2, right2)
        check1 = rmat1.to_matrix() + rmat2.to_matrix()
        check2 = (rmat1 + rmat2).to_matrix()
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=14)

    def test_rmat_add_subtract_exact(self):
        left1 = numpy.matrix(numpy.random.rand(3, 1))
        right1 = numpy.matrix(numpy.random.rand(3, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(3, 1))
        right2 = numpy.matrix(numpy.random.rand(3, 1))
        rmat2 = RMat(left2, right2)
        step1 = rmat1 + rmat2
        check1 = step1 - rmat2
        check2 = step1 - rmat1
        self.assertAlmostEqual(check1.norm(), rmat1.norm(), places=14)
        self.assertAlmostEqual(check2.norm(), rmat2.norm(), places=14)
        left1 = numpy.matrix(numpy.random.rand(5, 1))
        right1 = numpy.matrix(numpy.random.rand(4, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(5, 1))
        right2 = numpy.matrix(numpy.random.rand(4, 1))
        rmat2 = RMat(left2, right2)
        step1 = rmat1 + rmat2
        check1 = step1 - rmat2
        check2 = step1 - rmat1
        self.assertAlmostEqual(check1.norm(), rmat1.norm(), places=14)
        self.assertAlmostEqual(check2.norm(), rmat2.norm(), places=14)

    def test_hmat_vec_mul(self):
        blocks = []
        for i in xrange(2):
            for j in xrange(2):
                blocks.append(HMat(content=numpy.matrix(numpy.random.rand(2, 2)),
                                   shape=(2, 2),
                                   parent_index=(2*i, 2*j)
                                   )
                              )
        hmat = HMat(blocks=blocks, shape=(4, 4), parent_index=(0, 0))
        x = numpy.random.rand(4, 1)
        check1 = hmat * x
        hmat_full = hmat.to_matrix()
        check2 = hmat_full * x
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=14)
