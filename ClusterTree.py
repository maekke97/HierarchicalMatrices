#!/usr/bin/env python

"""ClusterTree.py: Implementation of a binary cluster tree for 1D lists of indices.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        ClusterTree: Gives a binary cluster tree.
"""


class ClusterTree:
    """Binary cluster tree for 1D index lists.

    """
    start_index = 0
    list_size = 0
    step_size = 0
    indices = ()
    children = None
    level = 0
    child_id = 0
    leaf_size = 0

    def __init__(self, indices=(), leaf_size=0, step_size=0, start_index=0, level=0, child_id=0, root=None):
        """ClusterTree(indices, leaf_size, step_size) -> ClusterTree

            indices:    list of indices,                iterable;
            leaf_size:  max size of a leaf,             int;
            step_size:  distance between to indices,    float;
        """
        length = len(indices)
        self.indices = indices
        self.leaf_size = leaf_size
        self.start_index = start_index
        self.list_size = length
        self.level = level
        self.child_id = child_id
        self.root = root
        if step_size != 0:
            self.step_size = step_size
        else:
            try:
                self.step_size = float(indices[1]-indices[0])/len(indices)
            except IndexError:
                self.step_size = 0
        if length > self.leaf_size:
            split = length/2
            self.children = {"first": ClusterTree(indices=indices[:split], leaf_size=self.leaf_size, start_index=0,
                                                  level=self.level+1, child_id=1, root=self),
                             "second": ClusterTree(indices=indices[split:], leaf_size=self.leaf_size,
                                                   start_index=split+self.start_index, level=self.level+1,
                                                   child_id=2, root=self),
                             }

    def depth(self):
        """Return the depth of the cluster.

            ClusterTree.depth() -> int
        """
        if not self.children:
            return self.level
        else:
            return max([self.children["first"].depth(), self.children["second"].depth()])

    def __str__(self):
        """Return string representation."""
        out_str = "ClusterTree with properties:\n\tIndices: " + str(self.indices) + ",\n\tDepth: " + str(self.depth()) \
            + ",\n\tLeaf size: " + str(self.leaf_size) + "."
        return out_str

    def __repr__(self):
        """Representation same as __str__."""
        return str(self)
