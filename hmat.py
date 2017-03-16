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

    def _mul_with_vector(self, other):
        """Multiply with a vector

        :param other: vector
        :type other: numpy.array
        :return: result vector
        :rtype: numpy.array
        """
        if self.content:
            # We have content. Multiply and return
            return self.content * other
        else:
            x_start, y_start = self.parent_index
            res_length = self.shape[0]
            res = numpy.zeros((res_length, 1))
            for block in self.blocks:
                x_current, y_current = block.parent_index
                x_length, y_length = block.shape
                in_index_start = x_start - x_current
                in_index_end = in_index_start + x_length
                res_index_start = y_start - y_current
                res_index_end = res_index_start + y_length
                res[res_index_start: res_index_end] += block * other[in_index_start: in_index_end]
            return res

    def _mul_with_hmatrix(self, other):
        """Multiply with a matrix

        :param other: matrix
        :type other: HMat
        :return: result matrix
        :rtype: numpy.matrix
        """
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
    root = HMat(blocks=[], shape=block_cluster_tree.shape(), parent_index=(0, 0))
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
            new_hmat = HMat(blocks=[], shape=son.shape(), parent_index=son.plot_info)
            current_hmat.blocks.append(new_hmat)
            recursion_build_hmatrix(new_hmat, son, generate_rmat, generate_full_mat)
