import numpy


class RMat(object):
    """Rank-k matrix"""

    def __init__(self, left_mat, right_mat, max_rank):
        """Build Rank-k matrix

        Arguments:
            left_mat: numpy.matrix n x k
            right_mat: numpy.matrix m x k
            max_rank: integer k_max
        """
        # check input
        left_shape = left_mat.shape
        right_shape = right_mat.shape
        if left_shape != right_shape:
            raise ValueError("left_mat and right_mat must have same shape")
        if min(left_shape) > max_rank:
            raise ValueError("rank of matrices is to large")
        self.k_max = max_rank
        self.left_mat = left_mat
        self.right_mat = right_mat

    def __repr__(self):
        left_str = self.left_mat.__repr__().replace('\n', '').replace(' ', '')
        right_str = self.right_mat.__repr__().replace('\n', '').replace(' ', '')
        out_str = '<RMat with left_mat: {0}, right_mat: {1} and max_rank: {2}>'.format(left_str,
                                                                                       right_str,
                                                                                       self.k_max)
        return out_str

    def __str__(self):
        left_str = str(self.left_mat)
        right_str = str(self.right_mat)
        out_str = 'Rank-k matrix with left block:\n{0}\nand right block:\n{1}'.format(left_str, right_str)
        return out_str

    def __eq__(self, other):
        """Check for equality"""
        left_eq = numpy.array_equal(self.left_mat, other.left_mat)
        right_eq = numpy.array_equal(self.right_mat, other.right_mat)
        return left_eq and right_eq and self.k_max == other.k_max

    def __add__(self, other):
        """Add two Rank-k-matrices

        Resulting matrix will have higher rank
        """
        new_left = numpy.concatenate([self.left_mat, other.left_mat], axis=1)
        new_right = numpy.concatenate([self.right_mat, other.right_mat], axis=1)
        new_k = self.k_max + other.k_max
        return RMat(new_left, new_right, new_k)

    def __sub__(self, other):
        """Subtract two Rank-k-matrices"""
        return self + (-other)

    def __neg__(self):
        """Unary minus"""
        new_k = self.k_max
        new_left = numpy.matrix(-self.left_mat)
        new_right = numpy.matrix(self.right_mat)
        return RMat(new_left, new_right, new_k)

    def __abs__(self):
        """Frobenius-norm"""
        return numpy.linalg.norm(self.left_mat * self.right_mat.T)

    def norm(self, type):
        pass

    def __mul__(self, other):
        i, r1 = self.left_mat.shape
        j, r1 = self.right_mat.shape
        j2, r2 = other.left_mat.shape
        k, r2 = other.right_mat.shape
        if j != j2:
            raise ValueError('Dimension miss-match')
        cost1 = 2 * r1 * r2 * (i + j) - r2 * (i + r1)
        cost2 = 2 * r1 * r2 * (j + k) - r1 * (k + r2)
        if cost2 >= cost1:
            return RMat(self.left_mat * (self.right_mat.T * other.left_mat), other.right_mat, r1 * r2)
        else:
            return RMat(self.left_mat, other.right_mat * (other.left_mat.T * self.right_mat), r1 * r2)

    def form_add(self, other, rank=None):
        """Formatted addition of self and other, i.e. addition and reduction to rank.

        If rank is omitted, reduction to min(rank(self), rank(other))
        """
        if not rank:
            rank = min((self.k_max, other.k_max))
        res = self + other
        return res.reduce(rank)

    def to_matrix(self):
        """Return full matrix"""
        return self.left_mat * self.right_mat.T

    def reduce(self, new_k):
        """Perform a reduced QR decomposition to rank new_k"""
        q_left, r_left = numpy.linalg.qr(self.left_mat)
        q_right, r_right = numpy.linalg.qr(self.right_mat)
        temp = r_left * r_right.T
        u, s, v = numpy.linalg.svd(temp)
        new_left = q_left * u[:, 0:new_k] * numpy.diag(s[0:new_k])
        new_right = q_right * v[:, 0:new_k]
        # noinspection PyTypeChecker
        return RMat(new_left, new_right, new_k)
