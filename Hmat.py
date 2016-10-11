"""HMat.py"""
import numpy


class Hmat(object):
    """Implement a hierarchical Matrix"""
    def __init__(self):
        self.tl = None
        self.tr = None
        self.bl = None
        self.br = None


class RMat(object):
    """Rank-k matrix"""

    def __init__(self, left_mat, right_mat, max_rank):
        """Build Rank-k matrix

        Arguments:
            left_mat: numpy.matrix n x k
            right_mat: numpy.matrix m x k
            max_rank: integer k_max
        """
        self.k_max = max_rank
        self.left_mat = left_mat
        self.right_mat = right_mat

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
