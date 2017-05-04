from unittest import TestCase
from HierMat import RMat, HMat
import numpy


class TestMath(TestCase):
    precision = 14

    def test_rmat_vec_mul(self):
        left = numpy.matrix(numpy.random.rand(3, 1))
        right = numpy.matrix(numpy.random.rand(3, 1))
        rmat = RMat(left, right, 1)
        x = numpy.random.rand(3, 1)
        check1 = rmat.to_matrix() * x
        check2 = rmat * x
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=self.precision)
        left = numpy.matrix(numpy.random.rand(5, 2))
        right = numpy.matrix(numpy.random.rand(4, 2))
        rmat = RMat(left, right, 2)
        x = numpy.random.rand(4, 1)
        check1 = rmat.to_matrix() * x
        check2 = rmat * x
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=self.precision)

    def test_rmat_add_exact(self):
        left1 = numpy.matrix(numpy.random.rand(3, 1))
        right1 = numpy.matrix(numpy.random.rand(3, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(3, 1))
        right2 = numpy.matrix(numpy.random.rand(3, 1))
        rmat2 = RMat(left2, right2)
        check1 = rmat1.to_matrix() + rmat2.to_matrix()
        check2 = (rmat1 + rmat2).to_matrix()
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=self.precision)
        left1 = numpy.matrix(numpy.random.rand(5, 1))
        right1 = numpy.matrix(numpy.random.rand(4, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(5, 1))
        right2 = numpy.matrix(numpy.random.rand(4, 1))
        rmat2 = RMat(left2, right2)
        check1 = rmat1.to_matrix() + rmat2.to_matrix()
        check2 = (rmat1 + rmat2).to_matrix()
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=self.precision)

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
        self.assertAlmostEqual(check1.norm(), rmat1.norm(), places=self.precision)
        self.assertAlmostEqual(check2.norm(), rmat2.norm(), places=self.precision)
        left1 = numpy.matrix(numpy.random.rand(5, 1))
        right1 = numpy.matrix(numpy.random.rand(4, 1))
        rmat1 = RMat(left1, right1)
        left2 = numpy.matrix(numpy.random.rand(5, 1))
        right2 = numpy.matrix(numpy.random.rand(4, 1))
        rmat2 = RMat(left2, right2)
        step1 = rmat1 + rmat2
        check1 = step1 - rmat2
        check2 = step1 - rmat1
        self.assertAlmostEqual(check1.norm(), rmat1.norm(), places=self.precision)
        self.assertAlmostEqual(check2.norm(), rmat2.norm(), places=self.precision)
        res = rmat1 + rmat1 - 2 * rmat1
        self.assertAlmostEqual(res.norm(), 0, places=self.precision)

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
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=self.precision)

    def test_hmat_add_subtract(self):
        blocks = []
        for i in xrange(3):
            for j in xrange(3):
                blocks.append(HMat(content=numpy.matrix(numpy.random.rand(2, 2)),
                                   shape=(2, 2),
                                   parent_index=(2 * i, 2 * j)
                                   )
                              )
        hmat = HMat(blocks=blocks, shape=(6, 6), parent_index=(0, 0))
        res = hmat - hmat
        self.assertAlmostEqual(res.norm(), 0, places=self.precision)
        res = hmat + hmat - 2 * hmat
        self.assertAlmostEqual(res.norm(), 0, places=self.precision)
        x = numpy.random.rand(6, 1)
        check1 = (hmat + hmat) * x
        check2 = hmat * x + hmat * x
        self.assertAlmostEqual(numpy.linalg.norm(check1 - check2), 0, places=self.precision)

    def test_hmat_multiplication(self):
        blocks1 = []
        for i in xrange(2):
            for j in xrange(2):
                blocks1.append(HMat(content=numpy.matrix(numpy.random.rand(2, 2)),
                                    shape=(2, 2),
                                    parent_index=(2 * i, 2 * j)
                                    )
                               )
        hmat1 = HMat(blocks=blocks1, shape=(4, 4), parent_index=(0, 0))
        x = numpy.random.rand(4, 1)
        res1 = (hmat1 * hmat1) * x
        res2 = hmat1 * (hmat1 * x)
        self.assertAlmostEqual(numpy.linalg.norm(res1 - res2), 0, places=self.precision)

    def test_hmat_inversion(self):
        blocks1 = []
        for i in xrange(2):
            for j in xrange(2):
                blocks1.append(HMat(content=numpy.matrix(numpy.random.rand(2, 2)),
                                    shape=(2, 2),
                                    parent_index=(2 * i, 2 * j)
                                    )
                               )
        hmat1 = HMat(blocks=blocks1, shape=(4, 4), parent_index=(0, 0))
        res = hmat1 * hmat1.inv()
        check = numpy.matrix(numpy.eye(4))
        self.assertAlmostEqual(numpy.linalg.norm(res.to_matrix() - check), 0, places=12)
        blocks2 = []
        for i in xrange(2):
            for j in xrange(2):
                blocks2.append(HMat(content=numpy.matrix(numpy.random.rand(3, 3)),
                                    shape=(3, 3),
                                    parent_index=(3 * i, 3 * j)
                                    )
                               )
        hmat2 = HMat(blocks=blocks2, shape=(6, 6), parent_index=(0, 0))
        res = hmat2 * hmat2.inv()
        check = numpy.matrix(numpy.eye(6))
        self.assertAlmostEqual(numpy.linalg.norm(res.to_matrix() - check), 0, places=12)
        blocks3 = []
        for i in xrange(3):
            for j in xrange(3):
                blocks3.append(HMat(content=numpy.matrix(numpy.random.rand(3, 3)),
                                    shape=(3, 3),
                                    parent_index=(3 * i, 3 * j)
                                    )
                               )
        hmat3 = HMat(blocks=blocks3, shape=(9, 9), parent_index=(0, 0))
        x = numpy.random.rand(9, 1)
        y = hmat3 * x
        z = hmat3.inv() * y
        self.assertAlmostEqual(numpy.linalg.norm(x - z), 0, places=11)
        # blocks4 = []
        # for outer_i in xrange(2):
        #     for outer_j in xrange(2):
        #         inner_blocks = []
        #         for i in xrange(3):
        #             for j in xrange(3):
        #                 inner_blocks.append(HMat(content=numpy.matrix(numpy.random.rand(3, 3)),
        #                                          shape=(3, 3),
        #                                          parent_index=(3 * i, 3 * j)
        #                                          )
        #                                     )
        #         blocks4.append(HMat(blocks=inner_blocks, shape=(9, 9), parent_index=(9 * outer_i, 9 * outer_j)))
        # hmat4 = HMat(blocks=blocks4, shape=(18, 18), parent_index=(0, 0))
        # x = numpy.random.rand(18, 1)
        # y = hmat4 * x
        # z = hmat4.inv() * y
        # self.assertAlmostEqual(numpy.linalg.norm(x - z), 0, places=12)
