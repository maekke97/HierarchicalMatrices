#!/usr/bin/env python

"""ClusterTree.py: Implementation of a binary cluster tree for 1D, 2D and 3D lists of indices.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        ClusterTree: Gives a binary cluster tree.
"""


class ClusterTree:
    """Abstract binary cluster tree that is inherited by the 1D, 2D and 3D classes.

    """
    indices = ()
    min_leaf_size = 0
    level = 0
    dimension = 0
    root = None
    children = None
    minimal_rectangle = None
    ax_parallel_rectangle = None

    def __init__(self, indices=(), min_leaf_size=0, _level=0, _root=None,
                 minimal_rectangle=None, ax_parallel_rectangle=None):
        """ClusterTree(indices, min_leaf_size) -> ClusterTree

            mandatory arguments:

            indices:            list of indices,                iterable;
            min_leaf_size:      max size of a leaf,             int;

            optional arguments:

            minimal_rectangle:
                instruction on how to compute the minimal rectangle/cube of a
                subset of indices,                              function;
            ax_parallel_rectangle:
                instruction on how to compute the ax-parallel rectangle/cube of a
                subset of indices,                              function;

            See the examples and README on how to write and use the rectangle functions.
            Defaults are provided for standard Euclidean space.
        """
        length = len(indices)
        self.dimension = len(indices[0])
        self.indices = indices
        self.min_leaf_size = min_leaf_size
        self.level = _level
        self.root = _root
        if minimal_rectangle:
            self.minimal_rectangle = minimal_rectangle
        else:
            # Choose suitable default
            if self.dimension == 3:
                self.minimal_rectangle = self.minimal_rectangle_3d
            elif self.dimension == 2:
                self.minimal_rectangle = self.minimal_rectangle_2d
            else:
                # in 1D, the minimal rectangle of a point is empty
                self.minimal_rectangle = None
        if ax_parallel_rectangle:
            self.ax_parallel_rectangle = ax_parallel_rectangle
        else:
            # Choose suitable default
            if self.dimension == 3:
                self.ax_parallel_rectangle = self.ax_parallel_rectangle_3d
            elif self.dimension == 2:
                self.ax_parallel_rectangle = self.ax_parallel_rectangle_2d
            else:
                self.ax_parallel_rectangle = None
        if length > self.min_leaf_size:
            split = length/2
            if self.dimension == 1:
                self.children = {1: ClusterTree(indices=indices[:split],
                                                min_leaf_size=self.min_leaf_size,
                                                _level=self.level+1,
                                                _root=self,
                                                minimal_rectangle=self.minimal_rectangle,
                                                ax_parallel_rectangle=self.ax_parallel_rectangle
                                                ),
                                 2: ClusterTree(indices=indices[split:],
                                                min_leaf_size=self.min_leaf_size,
                                                _level=self.level+1,
                                                _root=self,
                                                minimal_rectangle=self.minimal_rectangle,
                                                ax_parallel_rectangle=self.ax_parallel_rectangle
                                                )
                                 }
            else:
                indices1, indices2 = self.split_indices(indices)
                self.children = {1: ClusterTree(indices=indices1,
                                                min_leaf_size=self.min_leaf_size,
                                                _level=self.level+1,
                                                _root=self,
                                                minimal_rectangle=self.minimal_rectangle,
                                                ax_parallel_rectangle=self.ax_parallel_rectangle
                                                ),
                                 2: ClusterTree(indices=indices2,
                                                min_leaf_size=self.min_leaf_size,
                                                _level=self.level+1,
                                                _root=self,
                                                minimal_rectangle=self.minimal_rectangle,
                                                ax_parallel_rectangle=self.ax_parallel_rectangle
                                                )
                                 }

    def depth(self):
        """Return the depth of the cluster.

            ClusterTree.depth() -> int
        """
        if not self.children:
            return self.level
        else:
            return max([self.children[1].depth(), self.children[2].depth()])

    def __str__(self):
        out_str = "ClusterTree with properties:\n\tIndices: " + str(self.indices) + ",\n\tDepth: " + str(self.depth()) \
            + ",\n\tLeaf size: " + str(self.min_leaf_size) + "."
        return out_str

    def __repr__(self):
        return str(self)
