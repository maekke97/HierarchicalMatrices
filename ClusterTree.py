"""Module_Docstring"""


class ClusterTree:
    """Class_Docstring"""
    start_index = 0
    list_size = 0
    indices = ()
    children = None
    level = 0
    child_id = 0
    leaf_size = 0

    def __init__(self, indices=(), leaf_size=0, start_index=0, level=0, child_id=0, root=None):
        """init_Docstring"""
        length = len(indices)
        self.indices = indices
        self.leaf_size = leaf_size
        self.start_index = start_index
        self.list_size = length
        self.level = level
        self.child_id = child_id
        self.root = root
        if length > self.leaf_size:
            split = length/2
            self.children = {"first": ClusterTree(indices=indices[:split], leaf_size=self.leaf_size, start_index=0,
                                                  level=self.level+1, child_id=1, root=self),
                             "second": ClusterTree(indices=indices[split:], leaf_size=self.leaf_size,
                                                   start_index=split+self.start_index, level=self.level+1,
                                                   child_id=2, root=self),
                             }

    def step_size(self):
        """Return the step_size of the cluster tree."""
        if self.root:
            return self.root.step_size()
        else:
            return 1./len(self.indices)

    def depth(self):
        """Return the depth of the cluster."""
        if not self.children:
            return self.level
        else:
            return max([self.children["first"].depth(), self.children["second"].depth()])

    def __str__(self):
        """String representation."""
        out_str = "ClusterTree with properties:\n\tIndices: " + str(self.indices) + ",\n\tDepth: " + str(self.depth()) \
            + ",\n\tLeaf size: " + str(self.leaf_size) + "."
        return out_str

    def __repr__(self):
        return str(self)

if __name__ == "__main__":
    ct = ClusterTree(indices=range(16), leaf_size=1)
    print ct
