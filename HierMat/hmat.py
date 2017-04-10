"""hmat.py: :class:`HMat`
"""
import numbers
import operator

import numpy

from HierMat.rmat import RMat


class HMat(object):
    """Implement a hierarchical Matrix
    """

    def __init__(self, blocks=(), content=None, shape=(), root_index=()):
        self.blocks = blocks  # This list contains the lower level HMats
        self.content = content  # If not empty, this is either a full matrix or a RMat
        self.shape = shape  # Tuple of dimensions, i.e. size of index sets
        self.root_index = root_index  # Tuple of coordinates for the top-left corner in the root matrix

    def __getitem__(self, item):
        """Get block at position i, j from root-index"""
        if isinstance(item, int):
            return self.blocks[item]
        else:
            structured_blocks = {block.root_index: block for block in self.blocks}
            return structured_blocks[item]

    def block_structure(self):
        """Find out, what blocks in what alignment we have"""
        # TODO: proper docstring
        structure = {block.root_index: block.shape for block in self.blocks}
        return structure

    def column_sequence(self):
        """Return the sequence of column groups, as in thm. 1.9.4. in :cite:`eves1966elementary`
        
        only for consistent matrices
        
        :return: list of columns in first row
        :rtype: list(int)
        """
        if not self.is_consistent():
            raise StructureWarning('Warning, block structure is not consistent! Results may be wrong')
        pre_sort = sorted(self.block_structure(), key=lambda item: item[1])
        sorted_indices = sorted(pre_sort)
        if not sorted_indices:
            return [self.shape[1]]
        start_col = self.root_index[1]
        current_col = start_col
        max_cols = start_col + self.shape[1]
        col_seq = []
        for index in sorted_indices:
            rows, cols = self.block_structure()[index]
            current_col += cols
            col_seq.append(cols)
            if current_col == max_cols:  # end of column, return list
                return col_seq

    def row_sequence(self):
        """Return the sequence of row groups, as in thm. 1.9.4. in :cite:`eves1966elementary`

        only for consistent matrices

        :return: list of rows in first column
        :rtype: list(int)
        """
        if not self.is_consistent():
            raise StructureWarning('Warning, block structure is not consistent! Results may be wrong')
        pre_sort = sorted(self.block_structure())
        sorted_indices = sorted(pre_sort, key=lambda item: item[1])
        if not sorted_indices:
            return [self.shape[0]]
        start_row = self.root_index[0]
        current_row = start_row
        max_rows = start_row + self.shape[0]
        row_seq = []
        for index in sorted_indices:
            rows, cols = self.block_structure()[index]
            current_row += rows
            row_seq.append(rows)
            if current_row == max_rows:  # end of row, return list
                return row_seq

    def is_consistent(self):
        """Check if the blocks are aligned, i.e. we have consistent rows and columns
        
        :return: True on consistency, false otherwise
        :rtype: bool
        """
        if self.block_structure() == {}:  # if we have no blocks, we just have to check the shape
            return self.content.shape == self.shape
        pre_sort = sorted(self.block_structure(), key=lambda item: item[1])
        sorted_indices = sorted(pre_sort)
        start_row, start_col = self.root_index
        current_row = start_row
        current_col = start_col
        max_rows = start_row + self.shape[0]
        max_cols = start_col + self.shape[1]
        col_rows = 0  # to keep track of the height of each block
        col_seq = []  # sequence of sub-column lengths to compare
        current_col_seq = []
        for index in sorted_indices:
            # iterate over the index list to check each block column by column
            rows, cols = self.block_structure()[index]
            if index != (current_row, current_col):  # starting point of block is not where it should be
                return False
            current_col += cols
            current_col_seq.append(cols)
            if col_rows == 0:  # first block in a column
                col_rows = rows
            if rows != col_rows:  # this block has different height than the others in this column
                return False
            if current_col == max_cols:  # end of column, check against previous and go to next column
                if not col_seq:  # first column, so store for comparison
                    col_seq = current_col_seq
                if col_seq != current_col_seq:  # this column has a different partition than the previous
                    return False
                current_col = start_col
                current_row += col_rows
                col_rows = 0
                current_col_seq = []
        if current_row == max_rows:  # end of row, all fine
            return True
        # if we get to here, the shape of self is exceeded by its blocks in at least one direction
        return False

    def __repr__(self):
        return '<HMat with {content}>'.format(content=self.blocks if self.blocks else self.content)

    def __eq__(self, other):
        """Test for equality
        
        :param other: other HMat
        :type other: HMat
        :return: true on equal
        :rtype: bool
        """
        if not isinstance(self, type(other)):
            return False
        length = len(self.blocks)
        if len(other.blocks) != length:
            return False
        if self.blocks != other.blocks:
            return False
        if not isinstance(self.content, type(other.content)):
            return False
        if isinstance(self.content, RMat) and self.content != other.content:
            return False
        if isinstance(self.content, numpy.matrix) and not numpy.array_equal(self.content, other.content):
            return False
        if self.shape != other.shape:
            return False
        if self.root_index != other.root_index:
            return False
        return True

    def __ne__(self, other):
        """Test for inequality
        
        :param other: other HMat
        :type other: HMat
        :return: true on not equal
        :rtype: bool
        """
        return not self == other

    def __add__(self, other):
        """Addition with several types"""
        try:
            if self.shape != other.shape:
                raise ValueError("operands could not be broadcast together with shapes"
                                 " {0.shape} {1.shape}".format(self, other))
        except AttributeError:
            raise NotImplementedError('unsupported operand type(s) for +: {0} and {1}'.format(type(self), type(other)))
        if isinstance(other, HMat):
            return self._add_hmat(other)
        elif isinstance(other, RMat):
            return self._add_rmat(other)
        elif isinstance(other, numpy.matrix):
            return self._add_matrix(other)
        else:
            raise NotImplementedError('unsupported operand type(s) for +: {0} and {1}'.format(type(self), type(other)))

    def _add_hmat(self, other):
        """Add two hmat objects that have same structure
        
        :param other: HMat to add
        :type other: HMat
        :return: sum
        :rtype: HMat
        """
        # check inputs
        if self.root_index != other.root_index:
            raise ValueError('can not add {0} and {1}. root indices {0.root_index} '
                             'and {1.root_index} not the same'.format(self, other))
        if (self.content is None and other.blocks == ()) or (self.blocks == () and other.content is None):
            raise ValueError('can not add {0} and {1}. block structure is not the same '
                             'for {0} and {1}'.format(self, other))
        if self.content is not None:  # both have content
            return HMat(content=self.content + other.content, shape=self.shape, root_index=self.root_index)
        # if we get here, both have children
        if len(self.blocks) == len(other.blocks):
            blocks = map(operator.add, self.blocks, other.blocks)
            return HMat(blocks=blocks, shape=self.shape, root_index=self.root_index)
        else:
            raise ValueError('can not add {0} and {1}. number of blocks is different'.format(self, other))

    def _add_rmat(self, other):
        # TODO: What here?
        raise NotImplementedError()

    def _add_matrix(self, other):
        """Add full matrix to hmat
        
        :param other: matrix to add
        :type other: numpy.matrix
        :return: sum
        :rtype: HMat
        """
        if self.shape != other.shape:
            raise ValueError('operands could not be broadcast together with shapes'
                             ' {0.shape} {1.shape}'.format(self, other))
        out = HMat(shape=self.shape, root_index=self.root_index)
        if self.blocks != ():
            out.blocks = []
            for block in self.blocks:
                start_x = block.root_index[0] - self.root_index[0]
                start_y = block.root_index[1] - self.root_index[1]
                out.blocks.append(block + other[start_x: start_x + block.shape[0], start_y: start_y + block.shape[1]])
        else:
            out.content = self.content + other
        return out

    def __mul__(self, other):
        if isinstance(other, numpy.matrix):  # order is important here!
            return self._mul_with_matrix(other)
        elif isinstance(other, numpy.ndarray):
            return self._mul_with_vector(other)
        elif isinstance(other, numbers.Number):
            return self._mul_with_scalar(other)
        elif isinstance(other, RMat):
            return self._mul_with_rmat(other)
        elif isinstance(other, HMat):
            return self._mul_with_hmat(other)
        else:
            raise NotImplementedError('unsupported operand type(s) for *: {0} and {1}'.format(type(self), type(other)))

    def __rmul__(self, other):
        if not isinstance(other, numbers.Number):
            raise NotImplementedError('unsupported operand type(s) for *: {0} and {1}'.format(type(self), type(other)))
        return self * other

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
        # TODO: should this be a HMat?
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

        :param other: rmat to multiply
        :type other: RMat
        :return: Hmat containing the product
        :rtype: HMat
        """
        out = HMat(shape=self.shape, root_index=self.root_index)
        if isinstance(self.content, RMat):
            out.content = self.content * other
        elif self.content is not None:
            out.content = other.__rmul__(self.content)
        else:
            # TODO: Check this for correctness
            raise TypeError("Encountered HMat with blocks * RMat! What should I do?!")
        return out

    def _mul_with_hmat(self, other):
        """try to implement block wise
        
        :type other: HMat
        """
        if self.shape[1] != other.shape[0]:
            raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                             '{0.shape[1]} (dim 1) != {1.shape[0]} (dim 0)'.format(self, other))
        out_shape = (self.shape[0], other.shape[1])
        if self.root_index[1] != other.root_index[0]:
            raise ValueError('root indices {0.root_index} and {1.root_index} not aligned: '
                             '{0.root_index[1]} (dim 1) != {1.root_index[0]} (dim 0)'.format(self, other))
        out_root_index = (self.root_index[0], other.root_index[1])
        if self.content is not None and other.content is not None:  # simplest case, both have content
            out_content = self.content * other.content
            return HMat(content=out_content, shape=out_shape, root_index=out_root_index)
        elif self.content is None and other.content is None:  # both have blocks
            return self._mul_with_hmat_blocks(other)

    def _mul_with_hmat_blocks(self, other):
        # TODO: docstring
        out_shape = (self.shape[0], other.shape[1])
        out_root_index = (self.root_index[0], other.root_index[1])
        if self.column_sequence() != other.row_sequence():
            raise ValueError('structures are not aligned. '
                             '{0} != {1}'.format(self.column_sequence(), other.row_sequence()))
        out_blocks = []
        #  formula from Howard Eves: Theorem 1.9.6
        for i in list(set([k[0] for k in self.block_structure()])):
            for j in list(set([k[1] for k in other.block_structure()])):
                out_block = None
                for k in list(set([l[1] for l in self.block_structure()])):
                    if out_block is None:
                        out_block = self[i, k] * other[k, j]
                    else:
                        out_block = out_block + self[i, k] * other[k, j]
                out_blocks.append(out_block)
        return HMat(blocks=out_blocks, shape=out_shape, root_index=out_root_index)

    def _mul_with_scalar(self, other):
        """Multiplication with integer"""
        out = HMat(shape=self.shape, root_index=self.root_index)
        if self.content is not None:
            out.content = self.content * other
            return out
        else:
            out.blocks = [block * other for block in self.blocks]
            return out

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
        elif isinstance(self.content, RMat):  # We have an RMat in content, so return its full representation
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
    :rtype: HMat
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


class StructureWarning(Warning):
    """Special Warning used in this module to indicate bad block structure"""
    pass
