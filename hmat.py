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
    root = HMat(blocks=[], shape=block_cluster_tree.shape())
    pass
