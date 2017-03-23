"""hmat.py: :class:`HMat`
"""
import numpy

from rmat import RMat


class HMat(object):
    """Implement a hierarchical Matrix
    """

    def __init__(self, blocks=(), content=None, shape=(), root_index=()):
        self.blocks = blocks  # This list contains the lower level HMat
        self.content = content  # If not empty, this is either a full matrix or a RMat
        self.shape = shape  # Tuple of dimensions, i.e. size of index sets
        self.root_index = root_index  # Tuple of coordinates for the top-left corner in the root matrix

    def __repr__(self):
        return '<HMat with {content}>'.format(content=self.blocks if self.blocks else self.content)

    def __mul__(self, other):
        if type(other) is numpy.ndarray:
            return self._mul_with_vector(other)
        elif type(other) is numpy.matrix:
            return self._mul_with_matrix(other)
        else:
            raise TypeError('unsupported operand type(s) for *: {0} and {1}'.format(type(self), type(other)))

    def _mul_with_vector(self, other):
        """Multiply with a vector

        :param other: vector
        :type other: numpy.array
        :return: result vector
        :rtype: numpy.array
        """
        if self.content is not None:
            # We have content. Multiply and return
            return self.content * other
        else:
            if self.shape[1] != other.shape[0]:
                raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                                 '{0.shape[1]} (dim 1) != {1.shape[0]} (dim 0)'.format(self, other))
            x_start, y_start = self.root_index
            res_length = self.shape[0]
            res = numpy.zeros((res_length, 1))
            for block in self.blocks:
                x_current, y_current = block.root_index
                x_length, y_length = block.shape
                in_index_start = x_current - x_start
                in_index_end = in_index_start + x_length
                res_index_start = y_current - y_start
                res_index_end = res_index_start + y_length
                res[res_index_start: res_index_end] += block * other[in_index_start: in_index_end]
            return res

    def _mul_with_matrix(self, other):
        """Multiply with a matrix

        :param other: matrix
        :type other: numpy.matrix
        :return: result matrix
        :rtype: numpy.matrix
        """
        if self.content is not None:
            return self.content * other
        else:
            if self.shape[1] != other.shape[0]:
                raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                                 '{0.shape[1]} (dim 1) != {1.shape[0]} (dim 0)'.format(self, other))
            res_shape = (self.shape[0], other.shape[1])
            res = numpy.zeros(res_shape)
            row_base, col_base = self.root_index
            for block in self.blocks:
                row_count, col_count = block.shape
                row_current, col_current = block.root_index
                row_start = row_current - row_base
                row_end = row_start + row_count
                res[row_start:row_end, :] += block * other[row_start:row_end, :]
            return res

    def _mul_with_rmat(self, other):
        """Multiplication with an RMat

        :param other: rmatrix.RMat to multiply
        :type other: RMat
        :return:
        """
        if type(self.content) == RMat:
            return self.content * other
        elif self.content is not None:
            return other.__rmul__(self.content)

    def to_matrix(self):
        """Full matrix representation

        :return: full matrix
        :rtype: numpy.matrix
        """
        if self.blocks:  # The matrix has children so fill recursive
            out_mat = numpy.empty(self.shape)
            for block in self.blocks:
                # determine the position of the current block
                vertical_start = block.root_index[0]
                vertical_end = vertical_start + block.shape[0]
                horizontal_start = block.root_index[1]
                horizontal_end = horizontal_start + block.shape[1]

                # fill the block with recursive call
                out_mat[vertical_start:vertical_end, horizontal_start:horizontal_end] = block.to_matrix()
            return out_mat
        elif type(self.content) == RMat:  # We have an RMat in content, so return its full representation
            return self.content.to_matrix()
        else:  # We have regular content, so we return it
            return self.content


def build_hmatrix(block_cluster_tree=None, generate_rmat_function=None, generate_full_matrix_function=None):
    """Factory to build an HMat instance

    :param block_cluster_tree: block cluster tree giving the structure
    :type block_cluster_tree: BlockClusterTree
    :param generate_rmat_function: function taking an admissible block cluster tree and returning a rank-k matrix
    :param generate_full_matrix_function: function taking an inadmissible block cluster tree and returning
        a numpy.matrix
    :return: hmatrix
    :rtype: RMat
    :raises: ValueError if root of BlockClusterTree is admissible
    """
    if block_cluster_tree.admissible:
        raise ValueError("Root of the block cluster tree is admissible, can't generate HMat from that.")
    root = HMat(blocks=[], shape=block_cluster_tree.shape(), root_index=(0, 0))
    recursion_build_hmatrix(root, block_cluster_tree, generate_rmat_function, generate_full_matrix_function)
    return root


def recursion_build_hmatrix(current_hmat, block_cluster_tree, generate_rmat, generate_full_mat):
    """Recursion to :func:`build_hmatrix`
    """
    if block_cluster_tree.admissible:
        # admissible level found, so fill content with rank-k matrix and stop
        current_hmat.content = generate_rmat(block_cluster_tree)
    elif not block_cluster_tree.sons:
        # no sons and not admissible, so fill content with full matrix and stop
        current_hmat.content = generate_full_mat(block_cluster_tree)
    else:
        # recursion: generate new hmatrix for every son in block cluster tree
        for son in block_cluster_tree.sons:
            new_hmat = HMat(blocks=[], shape=son.shape(), root_index=son.plot_info)
            current_hmat.blocks.append(new_hmat)
            recursion_build_hmatrix(new_hmat, son, generate_rmat, generate_full_mat)
