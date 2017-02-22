"""HMat.py"""

import numpy

from Rmat import RMat


class HMat(object):
    """Implement a hierarchical Matrix"""

    def __init__(self, blocks=[], content=None, shape=(), parent_index=()):
        self.blocks = blocks  # This list contains the lower level HMat
        self.content = content  # If not empty, this is either a full matrix or a RMat
        self.shape = shape  # Tuple of dimensions, i.e. size of index sets
        self.parent_index = parent_index  # Tuple of coordinates for the top-left corner in the parent matrix

    def to_matrix(self):
        """Return full matrix"""
        if type(self.content) == RMat:  # We have an RMat in content, so return its full representation
            return self.content.to_matrix()
        elif type(self.content) == numpy.ndarray:  # We have a numpy matrix, so we return it
            return self.content
        else:  # Recursion through all blocks
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
