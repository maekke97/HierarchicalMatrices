"""hmat.py: :class:`HMat`, :func:`build_hmatrix`, :func:`recursion_build_hmatrix`, :class:`StructureWarning`
"""
import math
import numbers
import numpy
import copy

from HierMat.rmat import RMat


class HMat(object):
    """Implement a hierarchical Matrix
    
    :param blocks: list of HMat instances (children) (optional)
    :type blocks: list(HMat)
    :param content: the content if the matrix has no children (optional)
    :type content: RMat or numpy.matrix
    :param shape: the shape of the matrix (same as for numpy matrices)
    :type shape: tuple(int, int)
    :param parent_index: the index of this matrix with respect to its containing *parent*-matrix, zero based
    :type parent_index: tuple(int, int)
    
    .. admonition:: Supported operations
        
        * ``+`` (formatted addition)
        
        * ``*`` (formatted multiplication)
        
        * ``-a`` (unary minus)
        
        * ``-`` (formatted subtraction)
        
        * ``==`` (equal)
        
        * ``!=`` (not equal)
        
        * ``abs`` (frobenius norm)
        
        * ``inv`` (inversion)

    """

    def __init__(self, blocks=(), content=None, shape=(), parent_index=()):
        """Implement a hierarchical Matrix
    
        :param blocks: list of HMat instances the children (optional)
        :type blocks: list(HMat)
        :param content: the content if the matrix has no children (optional)
        :type content: RMat or numpy.matrix
        :param shape: the shape of the matrix (same as for numpy matrices)
        :type shape: tuple(int, int)
        :param parent_index: the index of this matrix with respect to its containing *parent*-matrix, zero based
        :type parent_index: tuple(int, int)
        """
        self.blocks = blocks  # This list contains the lower level HMats
        self.content = content  # If not empty, this is either a full matrix or a RMat
        self.shape = shape  # Tuple of dimensions, i.e. size of index sets
        self.parent_index = parent_index  # Tuple of coordinates for the top-left corner in the root matrix

    def __getitem__(self, item):
        """Get block at position i, j from root-index or ith block from blocks"""
        if isinstance(item, int):
            return self.blocks[item]
        else:
            structured_blocks = {block.parent_index: block for block in self.blocks}
            return structured_blocks[item]

    def __setitem__(self, key, value):
        """Set block at index key to value"""
        if isinstance(key, int):
            self.blocks[key] = value
        else:
            index = self.blocks.index(self[key])
            self.blocks[index] = value

    def __repr__(self):
        return '<HMat with {content}>'.format(content=self.blocks if self.blocks else self.content)

    def __eq__(self, other):
        """test for equality
        
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
        eqs = [a == b for a in self.blocks for b in other.blocks]
        if sum(eqs) != length:  # ignore order of blocks
            return False
        if not isinstance(self.content, type(other.content)):
            return False
        if isinstance(self.content, RMat) and self.content != other.content:
            return False
        if isinstance(self.content, numpy.matrix) and not numpy.array_equal(self.content, other.content):
            return False
        if self.shape != other.shape:
            return False
        if self.parent_index != other.parent_index:
            return False
        return True

    def __ne__(self, other):
        """test for inequality
        
        :param other: other HMat
        :type other: HMat
        :return: true on not equal
        :rtype: bool
        """
        return not self == other

    def __abs__(self):
        """Frobenius norm"""
        if isinstance(self.content, RMat):
            return self.content.norm()
        elif isinstance(self.content, numpy.matrix):
            return numpy.linalg.norm(self.content)
        else:
            total = 0
            for block in self.blocks:
                total += abs(block)**2
            return math.sqrt(total)

    def norm(self, order=None):
        """Norm of the matrix

        :param order: order of the norm (see in :func:`numpy.linalg.norm`)
        :return: norm
        :rtype: float
        """
        if order is None or order == 'fro':
            return abs(self)
        else:
            raise NotImplementedError('Only Frobenius implemented so far')

    def __add__(self, other):
        """addition with several types"""
        try:
            if self.shape != other.shape:
                raise ValueError("operands could not be broadcast together with shapes"
                                 " {0.shape} {1.shape}".format(self, other))
        except AttributeError:
            if not isinstance(other, numbers.Number):
                raise NotImplementedError('unsupported operand type(s) for +: {0} and {1}'.format(type(self),
                                                                                                  type(other)))
            else:
                return self._add_matrix(numpy.matrix(other))
        if isinstance(other, HMat):
            return self._add_hmat(other)
        elif isinstance(other, RMat):
            return self._add_rmat(other)
        elif isinstance(other, numpy.matrix):
            return self._add_matrix(other)
        else:
            raise NotImplementedError('unsupported operand type(s) for +: {0} and {1}'.format(type(self), type(other)))

    def _add_hmat(self, other):
        """add two hmat objects
        
        if structures do not match refine to finer structure
        
        :param other: HMat to add
        :type other: HMat
        :return: sum
        :rtype: HMat
        """
        # check inputs
        if self.parent_index != other.parent_index:
            raise ValueError('can not add {0} and {1}. root indices {0.parent_index} '
                             'and {1.parent_index} not the same'.format(self, other))
        if self.content is None and other.blocks == ():
            # only other has content, so restructure other to match structure
            addend = other.restructure(self.block_structure())
            return self + addend
        elif self.blocks == () and other.content is None:
            # only self has content, so restructure self to match structure
            addend = self.restructure(other.block_structure())
            return addend + other
        # both have content
        elif isinstance(self.content, numpy.matrix) and isinstance(other.content, RMat):
            # # take other first to avoid numpy broadcast
            # return HMat(content=other.content + self.content, shape=self.shape, parent_index=self.parent_index)
            # take other to full
            return HMat(content=self.content + other.to_matrix(), shape=self.shape, parent_index=self.parent_index)
        elif self.content is not None:  # both have content, that can be added left to right
            return HMat(content=self.content + other.content, shape=self.shape, parent_index=self.parent_index)
        # if we get here, both have children
        elif len(self.blocks) == len(other.blocks):
            blocks = [self[index] + other[index] for index in self.block_structure()]
            return HMat(blocks=blocks, shape=self.shape, parent_index=self.parent_index)
        else:
            raise ValueError('can not add {0} and {1}. number of blocks is different'.format(self, other))

    def _add_rmat(self, other):
        # TODO: What here?
        raise NotImplementedError()

    def _add_matrix(self, other):
        """add full matrix to hmat
        
        :param other: matrix to add
        :type other: numpy.matrix
        :return: sum
        :rtype: HMat
        """
        if self.shape != other.shape:
            raise ValueError('operands could not be broadcast together with shapes'
                             ' {0.shape} {1.shape}'.format(self, other))
        out = HMat(shape=self.shape, parent_index=self.parent_index)
        if self.blocks != ():
            out.blocks = []
            for block in self.blocks:
                start_x = block.parent_index[0]
                start_y = block.parent_index[1]
                end_x = start_x + block.shape[0]
                end_y = start_y + block.shape[1]
                out.blocks.append(block + other[start_x: end_x, start_y: end_y])
        else:
            out.content = self.content + other
        return out

    def __radd__(self, other):
        """Should be commutative so just switch"""
        return self + other

    def __neg__(self):
        """Unary minus"""
        out = HMat(shape=self.shape, parent_index=self.parent_index)
        if self.content is not None:
            out.content = -self.content
        else:
            out.blocks = []
            for block in self.blocks:
                out.blocks.append(-block)
        return out

    def __sub__(self, other):
        """Subtract other
        
        :type other: HMat
        """
        return self + (-other)

    def __mul__(self, other):
        """multiplication with several types"""
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
        """multiply from the right
        
        only supported for scalars
        enables to write ``a * H``
        """
        if not isinstance(other, numbers.Number):
            raise NotImplementedError('unsupported operand type(s) for *: {0} and {1}'.format(type(self), type(other)))
        return self * other

    def _mul_with_vector(self, other):
        """multiply with a vector

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
            res_length = self.shape[0]
            res = numpy.zeros((res_length, 1))
            for block in self.blocks:
                res_index_start, in_index_start = block.parent_index
                x_length, y_length = block.shape
                in_index_end = in_index_start + y_length
                res_index_end = res_index_start + x_length
                res[res_index_start: res_index_end] += block * other[in_index_start: in_index_end]
            return res

    def _mul_with_matrix(self, other):
        """multiply with a numpy.matrix

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
            for block in self.blocks:
                row_count, col_count = block.shape
                row_start, col_current = block.parent_index
                row_end = row_start + row_count
                res[row_start:row_end, :] += block * other[row_start:row_end, :]
            return res

    def _mul_with_rmat(self, other):
        """multiplication with an rmat

        :param other: rmat to multiply
        :type other: RMat
        :return: Hmat containing the product
        :rtype: HMat
        """
        out = HMat(shape=(self.shape[0], other.shape[1]), parent_index=self.parent_index)
        if isinstance(self.content, RMat):
            out.content = self.content * other
        elif self.content is not None:
            out.content = other.__rmul__(self.content)
        else:
            # TODO: Check this for correctness
            raise TypeError("Encountered HMat with blocks * RMat! What should I do?!")
        return out

    def _mul_with_hmat(self, other):
        """multiplication with other hmat
        
        :type other: HMat
        """
        if self.shape[1] != other.shape[0]:
            raise ValueError('shapes {0.shape} and {1.shape} not aligned: '
                             '{0.shape[1]} (dim 1) != {1.shape[0]} (dim 0)'.format(self, other))
        out_shape = (self.shape[0], other.shape[1])
        if self.parent_index[1] != other.parent_index[0]:
            raise ValueError('root indices {0.parent_index} and {1.parent_index} not aligned: '
                             '{0.parent_index[1]} (dim 1) != {1.parent_index[0]} (dim 0)'.format(self, other))
        out_parent_index = (self.parent_index[0], other.parent_index[1])
        if self.content is not None and other.content is not None:  # simplest case, both have content
            out_content = self.content * other.content
            return HMat(content=out_content, shape=out_shape, parent_index=out_parent_index)
        elif self.content is None and other.content is None:  # both have blocks
            return self._mul_with_hmat_blocks(other, out_shape, out_parent_index)
        elif self.content is not None:  # other has blocks, self has content. Collect other to full matrix
            return HMat(content=self.content * other.to_matrix(), shape=out_shape, parent_index=out_parent_index)
        else:  # other has content, self has blocks. Collect self to full
            return HMat(content=self.to_matrix() * other.content, shape=out_shape, parent_index=out_parent_index)

    def _mul_with_hmat_blocks(self, other, out_shape, out_parent_index):
        """multiplication when self and other have blocks
        
        :type other: HMat
        """
        if self.column_sequence() != other.row_sequence():
            raise ValueError('structures are not aligned. '
                             '{0} != {1}'.format(self.column_sequence(), other.row_sequence()))
        out_blocks = []
        #  formula from Howard Eves: Theorem 1.9.6
        for i in list(set([b[0] for b in self.block_structure()])):
            for j in list(set([b[1] for b in other.block_structure()])):
                out_block = None
                for k in list(set([l[1] for l in self.block_structure()])):
                    if out_block is None:
                        out_block = self[i, k] * other[k, j]
                    else:
                        out_block = out_block + self[i, k] * other[k, j]
                out_blocks.append(out_block)
        return HMat(blocks=out_blocks, shape=out_shape, parent_index=out_parent_index)

    def _mul_with_scalar(self, other):
        """multiplication with integer
        
        :type other: Number.number
        """
        out = HMat(shape=self.shape, parent_index=self.parent_index)
        if self.content is not None:
            out.content = self.content * other
            return out
        else:
            out.blocks = [block * other for block in self.blocks]
            return out

    def block_structure(self):
        """return the block structure of self
        
        produce a dict with parent_index: block paris

        :rtype: dict
        :returns: dict with index: HMatrix pairs
        """
        structure = {block.parent_index: block.shape for block in self.blocks}
        return structure

    def column_sequence(self):
        """Return the sequence of column groups
        as in thm. 1.9.4. in :cite:`eves1966elementary`

        only for consistent matrices

        :return: list of column-sizes in first row
        :rtype: list(int)
        :raises: :class:`StructureWarning` if the matrix is not consistent
        """
        if not self.is_consistent():
            raise StructureWarning('Warning, block structure is not consistent! Results may be wrong')
        pre_sort = sorted(self.block_structure(), key=lambda item: item[1])
        sorted_indices = sorted(pre_sort)
        if not sorted_indices:
            return [self.shape[1]]
        start_col = self.parent_index[1]
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
        """Return the sequence of row groups
        as in thm. 1.9.4. in :cite:`eves1966elementary`

        defined only for consistent matrices

        :return: list of row-sizes in first column
        :rtype: list(int)
        :raises: :class:`StructureWarning` if the matrix is not consistent
        """
        if not self.is_consistent():
            raise StructureWarning('Warning, block structure is not consistent! Results may be wrong')
        pre_sort = sorted(self.block_structure())
        sorted_indices = sorted(pre_sort, key=lambda item: item[1])
        if not sorted_indices:
            return [self.shape[0]]
        start_row = self.parent_index[0]
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
        """check if the blocks are aligned, 
        i.e. we have consistent rows and columns as in def. 1.9.5 in :cite:`eves1966elementary`

        :return: True on consistency, False otherwise
        :rtype: bool
        """
        if self.block_structure() == {}:  # if we have no blocks, we just have to check the shape
            return self.content.shape == self.shape
        pre_sort = sorted(self.block_structure(), key=lambda item: item[1])
        sorted_indices = sorted(pre_sort)
        start_row, start_col = (0, 0)
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

    def restructure(self, structure):
        """Restructure self into blocks to match structure
        
        :param structure: a dict as returned by HMat.block_structure
        :type structure: dict(index: size)
        :returns: HMat that has the specified structure
        :rtype: HMat
        :raises: :class:`NotImplementedError` if self has blocks or if self contains unknown content
        """
        if self.blocks != ():
            raise NotImplementedError('Only implemented for hmat without blocks')
        out_blocks = []
        for index in structure:
            start_x = index[0]
            start_y = index[1]
            end_x = start_x + structure[index][0]
            end_y = start_y + structure[index][1]
            if isinstance(self.content, RMat):
                out_blocks.append(HMat(content=self.split_rmat(start_x, start_y, end_x, end_y),
                                       shape=structure[index], parent_index=index))
            elif isinstance(self.content, numpy.matrix):
                out_blocks.append(HMat(content=self.split_hmat(start_x, start_y, end_x, end_y),
                                       shape=structure[index], parent_index=index))
            else:
                raise NotImplementedError('Illegal structure found in restructure')
        return HMat(blocks=out_blocks, shape=self.shape, parent_index=self.parent_index)

    def split_rmat(self, start_x, start_y, end_x, end_y):
        """Fetch the block specified by indices from a rmat
        
        .. admonition:: Example
            
            .. code:: python
        
                r = RMat(numpy.matrix([[2], [2], [2]]),
                         numpy.matrix([[3], [3], [3]]))
                h = HMat(content=r, shape=(3, 3), parent_index=(0, 0))
                res = h.split_rmat(0, 2, 1, 3)
                res
                <RMat with left_mat: matrix([[2]]), right_mat: matrix([[3]]) and max_rank: None> 
        
        :param start_x: vertical start index
        :type start_x: int
        
        :param start_y: horizontal start index
        :type start_y: int
        :param end_x: vertical end index
        :type end_x: int
        :param end_y: horizontal end index
        :type end_y: int
        :returns: rmat with corresponding block
        :rtype: RMat
        """
        return self.content.split(start_x, start_y, end_x, end_y)

    def split_hmat(self, start_x, start_y, end_x, end_y):
        """Fetch the block specified by indices from a numpy.matrix
        
        :param start_x: vertical start index
        :type start_x: int
        :param start_y: horizontal start index
        :type start_y: int
        :param end_x: vertical end index
        :type end_x: int
        :param end_y: horizontal end index
        :type end_y: int
        :returns: numpy.matrix with corresponding block
        :rtype: numpy.matrix
        """
        return self.content[start_x: end_x, start_y: end_y]

    def to_matrix(self):
        """Full matrix representation

        :return: full matrix
        :rtype: numpy.matrix
        """
        if self.blocks:  # The matrix has children so fill recursive
            out_mat = numpy.matrix(numpy.zeros(self.shape))
            for block in self.blocks:
                # determine the position of the current block
                vertical_start = block.parent_index[0]
                vertical_end = vertical_start + block.shape[0]
                horizontal_start = block.parent_index[1]
                horizontal_end = horizontal_start + block.shape[1]

                # fill the block with recursive call
                out_mat[vertical_start:vertical_end, horizontal_start:horizontal_end] = block.to_matrix()
            return out_mat
        elif isinstance(self.content, RMat):  # We have an RMat in content, so return its full representation
            return self.content.to_matrix()
        else:  # We have regular content, so we return it
            return self.content

    def transpose(self):
        """Return transposed copy of self

        :rtype: RMat
        """
        out_shape = (self.shape[1], self.shape[0])
        if self.blocks == ():
            out_content = self.content.transpose()
            out_blocks = ()
        else:
            out_content = None
            out_blocks = []
            for block in self.blocks:
                out_block = block.transpose()
                out_block.parent_index = (block.parent_index[1], block.parent_index[0])
                out_blocks.append(out_block)
        return HMat(content=out_content, blocks=out_blocks, shape=out_shape, parent_index=self.parent_index)

    def zero(self, make_copy=True):
        """Return a copy of self with zero entries
        
        i.e. an HMat instance with the same structure, but all content is replaced by numpy.zeros
        
        .. note::
        
            If make_copy is True, a deepcopy is made at the beginning. This can be very slow for large matrices.
        
        :param make_copy: make a copy of the matrix to not change the original (Default True)
        :type make_copy: bool
        :return: zero matrix
        :rtype: HMat
        """
        if self.content is not None:
            return HMat(content=numpy.matrix(numpy.zeros(self.shape)), shape=self.shape, parent_index=self.parent_index)
        else:
            if make_copy:
                out = copy.deepcopy(self)
            else:
                out = self
            structure = out.block_structure()
            for index in structure:
                out[index] = out[index].zero()
            return out

    def inv(self, make_copy=True):
        """Invert the matrix according to chapter 7.5.1 in :cite:`hackbusch2015hierarchical`
         
        .. note::
        
            If make_copy is True, a deepcopy is made at the beginning. This can be very slow for large matrices.
        
        :param make_copy: make a copy of the matrix to not change the original (Default True)
        :type make_copy: bool
        :return: approximation of inverse
        :rtype: HMat
        :raises: :class:`numpy.LinAlgError` if matrix is singular
        """
        if isinstance(self.content, numpy.matrix):
            return HMat(content=numpy.linalg.inv(self.content), shape=self.shape, parent_index=self.parent_index)
        if make_copy:
            self_copy = copy.deepcopy(self)
        else:
            self_copy = self
        out_matrix = self.zero()
        row_sequence = self.row_sequence()
        col_sequence = self.column_sequence()
        bls = row_sequence[0]  # block size
        block_check = [bls != c for c in row_sequence]
        if row_sequence != col_sequence or any(block_check):
            raise ValueError('not all squares')
        rows = len(row_sequence)
        for l in xrange(rows):
            out_matrix[l * bls,  l * bls] = self_copy[l * bls, l * bls].inv()
            for j in xrange(l):
                out_matrix[l * bls, j * bls] = out_matrix[l * bls, l * bls] * out_matrix[l * bls, j * bls]
            for j in xrange(l + 1, rows):
                self_copy[l * bls, j * bls] = out_matrix[l * bls, l * bls] * self_copy[l * bls, j * bls]
            for i in xrange(l + 1, rows):
                for j in xrange(l+1):
                    out_matrix[i * bls,
                               j * bls] = out_matrix[i * bls,
                                                     j * bls] - self_copy[i * bls,
                                                                          l * bls] * out_matrix[l * bls,
                                                                                                j * bls]
                for j in xrange(l+1, rows):
                    self_copy[i * bls,
                              j * bls] = self_copy[i * bls,
                                                   j * bls] - self_copy[i * bls,
                                                                        l * bls] * self_copy[l * bls,
                                                                                             j * bls]
        for l in xrange(rows - 1, -1, -1):
            for i in xrange(l - 1, -1, -1):
                for j in xrange(rows):
                    out_matrix[i * bls,
                               j * bls] = out_matrix[i * bls,
                                                     j * bls] - self_copy[i * bls,
                                                                          l * bls] * out_matrix[l * bls,
                                                                                                j * bls]
        return out_matrix

    def solve(self, right_hand_side):
        """Solve the equation

        :param right_hand_side:
        :return:
        """
        pass


def build_hmatrix(block_cluster_tree=None, generate_rmat_function=None, generate_full_matrix_function=None):
    """Factory to build an hierarchical matrix
    
    Takes a block cluster tree and generating functions for full matrices and low rank matrices respectively
    
    The generating functions take a block cluster tree as input and return a RMat or numpy.matrix respectively

    :param block_cluster_tree: block cluster tree giving the structure
    :type block_cluster_tree: BlockClusterTree
    :param generate_rmat_function: function taking an admissible block cluster tree and returning a rank-k matrix
    :param generate_full_matrix_function: function taking an inadmissible block cluster tree and returning
        a numpy.matrix
    :return: hmatrix
    :rtype: HMat
    """
    root = HMat(blocks=[], shape=tuple(block_cluster_tree.shape()), parent_index=(0, 0))
    recursion_build_hmatrix(root, block_cluster_tree, generate_rmat_function, generate_full_matrix_function)
    return root


def recursion_build_hmatrix(current_hmat, block_cluster_tree, generate_rmat, generate_full_mat):
    """Recursion to :func:`build_hmatrix`
    """
    if block_cluster_tree.admissible:
        # admissible level found, so fill content with rank-k matrix and stop
        current_hmat.content = generate_rmat(block_cluster_tree)
        current_hmat.blocks = ()
    elif not block_cluster_tree.sons:
        # no sons and not admissible, so fill content with full matrix and stop
        current_hmat.content = generate_full_mat(block_cluster_tree)
        current_hmat.blocks = ()
    else:
        # recursion: generate new hmatrix for every son in block cluster tree
        x_parent, y_parent = block_cluster_tree.plot_info
        for son in block_cluster_tree.sons:
            x_current, y_current = son.plot_info
            parent_index = (x_current - x_parent, y_current - y_parent)
            new_hmat = HMat(blocks=[], shape=son.shape(), parent_index=parent_index)
            current_hmat.blocks.append(new_hmat)
            recursion_build_hmatrix(new_hmat, son, generate_rmat, generate_full_mat)


class StructureWarning(Warning):
    """Special Warning used in this module to indicate bad block structure"""
    pass
