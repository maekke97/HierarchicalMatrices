"""hmat.py: :class:`HMat`
"""
import numpy

from rmat import RMat


class HMat(object):
    """Implement a hierarchical Matrix
    """

    def __init__(self, blocks=(), content=None, shape=(), parent_index=()):
        self.blocks = blocks  # This list contains the lower level HMat
        self.content = content  # If not empty, this is either a full matrix or a RMat
        self.shape = shape  # Tuple of dimensions, i.e. size of index sets
        self.parent_index = parent_index  # Tuple of coordinates for the top-left corner in the parent matrix

    def __repr__(self):
        pass

    def __mul__(self, other):
        if type(other) == numpy.ndarray and other.shape[0] == self.shape[1]:  # Matrix-vector or matrix-matrix product
            try:
                columns = other.shape[1]
            except IndexError:
                columns = 1
            out = numpy.zeros((self.shape[0], columns))  # initialize

    def content_mul(self, other):
        return self.content * other

    def to_matrix(self):
        """Full matrix representation

        :return: full matrix
        :rtype: numpy.matrix
        """
        if self.blocks:  # The matrix has children so fill recursive
            out_mat = numpy.empty(self.shape)
            for block in self.blocks:
                # determine the position of the current block
                vertical_start = block.parent_index[0]
                vertical_end = vertical_start + block.shape[0]
                horizontal_start = block.parent_index[1]
                horizontal_end = horizontal_start + block.shape[1]

                # fill the block with recursive call
                out_mat[vertical_start:vertical_end, horizontal_start:horizontal_end] = block.to_matrix()
            return out_mat
        elif type(self.content) == RMat:  # We have an RMat in content, so return its full representation
            return self.content.to_matrix()
        else:  # We have regular content, so we return it
            return self.content
