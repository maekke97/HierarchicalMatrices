"""rmat.py: :class:`RMat`
"""
import numbers

import numpy


class RMat(object):
    """Rank-k matrix

    Implementation of the low rank matrix as described in [HB2015]
    
    .. note:
        There is still an issue with __radd__, when the left operand is a numpy.ndarray object
    """

    def __init__(self, left_mat, right_mat=None, max_rank=None):
        """Build Rank-k matrix

        Arguments:
            left_mat: numpy.matrix n x k
            right_mat: numpy.matrix m x k
            max_rank: integer max_rank
        """
        # check for wrong input
        if right_mat is None and max_rank is None:
            raise ValueError('Not enough input arguments. At least one of right_mat and max_rank has to be specified')
        if max_rank is not None and max_rank <= 0:
            raise ValueError('max_rank must be a positive integer')

        # check for direct conversion
        if right_mat is None:
            self._from_matrix(left_mat, max_rank)
        else:  # default constructor
            left_shape = left_mat.shape
            right_shape = right_mat.shape
            if left_shape[1] != right_shape[1]:
                raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                                 '{0.shape[1]} (dim 1) != {1.shape[1]} (dim 1)'.format(left_mat, right_mat))
            self.max_rank = max_rank
            self.left_mat = left_mat
            self.right_mat = right_mat
            self.shape = (left_shape[0], right_shape[0])
            # check if we have to reduce
            if self.max_rank and self.max_rank < left_shape[1]:
                self._reduce(self.max_rank)

    @staticmethod
    def from_matrix(matrix, max_rank):
        """Convert a full matrix to the rank k format 
        
        :param matrix: Matrix to convert to RMat
        :type matrix: numpy.matrix
        :param max_rank: maximum rank
        :type max_rank: int
        :return: Best approximation of matrix in RMat format
        :rtype: RMat
        """
        u, s, v = numpy.linalg.svd(matrix)
        left_mat = u[:, 0: max_rank] * numpy.diag(s[0: max_rank])
        right_mat = v[0: max_rank, :].T
        return RMat(left_mat, right_mat, max_rank)

    def _from_matrix(self, matrix, max_rank):
        """Convert a full matrix to the rank k format 

        :param matrix: Matrix to convert to RMat
        :type matrix: numpy.matrix
        :param max_rank: maximum rank
        :type max_rank: int
        """
        self.max_rank = max_rank
        u, s, v = numpy.linalg.svd(matrix)
        self.left_mat = u[:, 0: self.max_rank] * numpy.diag(s[0: self.max_rank])
        self.right_mat = v[0: self.max_rank, :].T
        self.shape = matrix.shape

    def __repr__(self):
        left_str = self.left_mat.__repr__().replace('\n', '').replace(' ', '')
        right_str = self.right_mat.__repr__().replace('\n', '').replace(' ', '')
        out_str = '<RMat with left_mat: {0}, right_mat: {1} and max_rank: {2}>'.format(left_str,
                                                                                       right_str,
                                                                                       self.max_rank)
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
        return left_eq and right_eq and self.max_rank == other.max_rank

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        """Addition of self and other"""
        try:
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast together with shapes '
                                 '{0.shape} {1.shape}'.format(self, other))
        except AttributeError:
            if not isinstance(other, numbers.Number):
                raise NotImplementedError('unsupported operand type(s) for +: {0} and {1}'.format(type(self),
                                                                                                  type(other)))
            else:
                return self + numpy.matrix(other)
        if isinstance(other, RMat):
            # if max_rank is defined do exact, else do exact
            if self.max_rank:
                return self.form_add(other, self.max_rank)
            else:
                return self._add_rmat(other)
        elif isinstance(other, numpy.matrix):
            return self._add_matrix(other)
        else:
            raise NotImplementedError('unsupported operand type(s) for +: {0} and {1}'.format(type(self), type(other)))

    def _add_rmat(self, other):
        """Add two Rank-k-matrices

        Resulting matrix will have higher rank
        """
        new_left = numpy.concatenate([self.left_mat, other.left_mat], axis=1)
        new_right = numpy.concatenate([self.right_mat, other.right_mat], axis=1)
        return RMat(new_left, new_right)

    def _add_matrix(self, other):
        """Add full matrix
        
        :type other: numpy.matrix
        """
        # TODO: check if suitable
        addend = self.from_matrix(other, self.max_rank)
        return self + addend

    def __radd__(self, other):
        """Should be commutative so just switch"""
        return self + other

    def __sub__(self, other):
        """Subtract two Rank-k-matrices"""
        return self + (-other)

    def __neg__(self):
        """Unary minus"""
        new_k = self.max_rank
        new_left = numpy.matrix(-self.left_mat)
        new_right = numpy.matrix(self.right_mat)
        return RMat(new_left, new_right, new_k)

    def __abs__(self):
        """Frobenius-norm"""
        return numpy.linalg.norm(self.left_mat * self.right_mat.T)

    def norm(self, order=None):
        """Norm of the matrix

        :param order: order of the norm (see in :func:`numpy.linalg.norm`)
        :return: norm
        :rtype: float
        """
        return numpy.linalg.norm(self.left_mat * self.right_mat.T, ord=order)

    def split(self, start_x, start_y, end_x, end_y):
        """Return the block of self that matches the supplied indices
        
        :param start_x: x start index
        :type start_x: int
        :param start_y: y start index
        :type start_y: int
        :param end_x: x end index
        :type end_x: int
        :param end_y: y end index
        :type end_y: int
        :return: rmat that represents the corresponding block
        :rtype: RMat
        """
        return RMat(left_mat=self.left_mat[start_x: end_x, :],
                    right_mat=self.right_mat[start_y: end_y, :],
                    max_rank=self.max_rank)

    def __mul__(self, other):
        """Multiplication of self and other"""
        if isinstance(other, RMat):
            return self._mul_with_rmat(other)
        elif isinstance(other, numpy.matrix):
            return self._mul_with_mat(other)
        elif isinstance(other, numpy.ndarray):
            return self._mul_with_vector(other)
        elif isinstance(other, numbers.Number):
            return self._mul_with_int(other)
        else:
            raise NotImplementedError('unsupported operand type(s) for *: {0} and {1}'.format(type(self), type(other)))

    def _mul_with_mat(self, other):
        """Multiplication with full matrix"""
        j, r = self.right_mat.shape
        j2, r2 = other.shape
        if j != j2:
            raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                             '{0.shape[1]} (dim 1) != {1.shape[0]} (dim 0)'.format(self, other))
        return RMat(self.left_mat, other.T * self.right_mat, r)

    def _mul_with_rmat(self, other):
        """Multiplication with rmat"""
        i, r1 = self.left_mat.shape
        j, r1 = self.right_mat.shape
        j2, r2 = other.left_mat.shape
        k, r2 = other.right_mat.shape
        out_rank = max((self.max_rank, other.max_rank))
        if j != j2:
            raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                             '{0.shape[1]} (dim 1) != {1.shape[0]} (dim 0)'.format(self, other))
        cost1 = 2 * r1 * r2 * (i + j) - r2 * (i + r1)
        cost2 = 2 * r1 * r2 * (j + k) - r1 * (k + r2)
        if cost2 >= cost1:
            return RMat(self.left_mat * (self.right_mat.T * other.left_mat), other.right_mat, out_rank)
        else:
            return RMat(self.left_mat, other.right_mat * (other.left_mat.T * self.right_mat), out_rank)

    def _mul_with_vector(self, other):
        """Multiplication with vector"""
        return self.left_mat * (self.right_mat.T * other)

    def _mul_with_int(self, other):
        """Multiplication with scalar"""
        return RMat(other * self.left_mat, self.right_mat, self.max_rank)

    def __rmul__(self, other):
        """Multiplication numpy.matrix * RMat"""
        if isinstance(other, numpy.matrix):
            # return RMat(other * self.left_mat, self.right_mat, self.max_rank)
            return other * self.to_matrix()
        elif isinstance(other, numbers.Number):
            return self._mul_with_int(other)
        else:
            raise NotImplementedError('unsupported operand type(s) for *: {0} and {1}'.format(type(self), type(other)))

    def form_add(self, other, rank=None):
        """Formatted addition of self and other, i.e. addition and reduction to rank::

            (self + other).reduce(rank)

        If rank is omitted, reduction to min(rank(self), rank(other))

        :param other: other rank-k matrix
        :type other: RMat
        :param rank: rank after reduction
        :type rank: int
        :return: reduced result
        :rtype: RMat
        """
        if not rank:
            rank = min((self.max_rank, other.max_rank))
        res = self._add_rmat(other)
        res._reduce(rank)
        return res

    def to_matrix(self):
        """Full matrix representation::

            left_mat * right_mat.T

        :return: full matrix
        :rtype: numpy.matrix
        """
        return self.left_mat * self.right_mat.T

    def reduce(self, new_k):
        """Perform a reduced QR decomposition to rank new_k

        :param new_k: rank to reduce to
        :type new_k: int
        :return: reduced matrix
        :rtype: RMat
        """
        q_left, r_left = numpy.linalg.qr(self.left_mat)
        q_right, r_right = numpy.linalg.qr(self.right_mat)
        temp = r_left * r_right.T
        u, s, v = numpy.linalg.svd(temp)
        new_left = q_left * u[:, 0:new_k] * numpy.diag(s[0:new_k])
        new_right = q_right * v[:, 0:new_k]
        # noinspection PyTypeChecker
        return RMat(new_left, new_right, new_k)

    def _reduce(self, new_k):
        """Perform a reduced QR decomposition to rank new_k internal

        :param new_k: rank to reduce to
        :type new_k: int
        """
        q_left, r_left = numpy.linalg.qr(self.left_mat)
        q_right, r_right = numpy.linalg.qr(self.right_mat)
        temp = r_left * r_right.T
        u, s, v = numpy.linalg.svd(temp)
        new_left = q_left * u[:, 0:new_k] * numpy.diag(s[0:new_k])
        new_right = q_right * v[:, 0:new_k]
        self.left_mat = new_left
        self.right_mat = new_right
        self.max_rank = new_k

    def transpose(self):
        """Return transposed copy of self
        
        :rtype: RMat
        """
        out_max_rank = self.max_rank
        out_left = self.right_mat
        out_right = self.left_mat
        return RMat(left_mat=out_left, right_mat=out_right, max_rank=out_max_rank)
