"""cluster_tree.py: :class:`ClusterTree`, :func:`build_cluster_tree`, :func:`recursion_build_cluster_tree`
"""


class ClusterTree(object):
    """Tree structure built according to the :class:`Splitable` in use

    Splits are done until max_leaf_size is reached
    """
    def __init__(self, content, sons=None, max_leaf_size=1, level=0):
        """"""
        self.level = level
        self.content = content
        self.sons = sons if sons else []
        self.max_leaf_size = max_leaf_size

    def __repr__(self):
        optional_string = " with children {0}".format(self.sons) if self.sons else " without children"
        return "<ClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def __len__(self):
        return len(self.content)

    def __str__(self):
        """give str representation of self."""
        cont_str = ",".join([str(p) for p in self.content])
        out_str = "ClusterTree at level {0} with content:\n{1}".format(self.level, cont_str)
        return out_str

    def _plot_str(self):
        """string for plots"""
        cont_str = ''
        for p in self.content:
            cont_str += '['
            cont_str += ",".join(["{0:.2f}".format(i) for i in p])
            cont_str += "] "
        cont_str.rstrip()
        return cont_str

    def __eq__(self, other):
        """Test for equality
        :param other: other cluster tree
        :type other: ClusterTree
        """
        return (self.level == other.level and self.content == other.content
                and self.sons == other.sons and self.max_leaf_size == other.max_leaf_size)

    def __ne__(self, other):
        """Test for inequality
        :param other: other cluster tree
        :type other: ClusterTree
        """
        return not self == other

    def __getitem__(self, item):
        return self.content[item]

    def get_index(self, item):
        """Get index from content

        :param item: index to get
        :type item: int
        :return: index
        :rtype: int
        """
        return self.content.get_index(item)

    def get_grid_item(self, item):
        """Get grid item from content

        :param item: index of item to get
        :type item: int
        """
        return self.content.get_grid_item(item)

    def get_grid_item_support_by_index(self, item):
        """Return supports of i-th item from grid

        :param item: index
        :type item: int
        """
        return self.content.get_grid_item_support_by_index(item)

    def get_grid_item_support(self, item):
        """Return supports ofitem from grid

        :param item: point
        :type item: tuple(float)
        """
        return self.content.get_grid_item_support(item)

    def get_patch_coordinates(self):
        """Return min and max out of indices

        :return: min and max
        :rtype: tuple(int, int)
        """
        return self.content.get_patch_coordinates()

    def to_list(self):
        """Give list representation for export

        Return a list containing the object and a list with its sons

        .. hint::

            **recursive function**
        """
        if self.sons:
            return [self, [son.to_list() for son in self.sons]]
        else:
            return [self, []]

    def to_xml(self):
        """Return a string for xml representation"""
        out_list = self.to_list()
        return self._to_xml(out_list)

    @staticmethod
    def _to_xml(lst, out_string=''):
        if len(lst[1]):
            value_string = str(lst[0])
            display_string = str(len(lst[1]))
        else:
            value_string = str(lst[0])
            display_string = str(lst[0])
        out_string += '<node value="{0}">{1}\n'.format(value_string, display_string)
        if len(lst) > 1 and type(lst[1]) is list:
            for item in lst[1]:
                out_string = ClusterTree._to_xml(item, out_string)
        out_string += "</node>\n"
        return out_string

    def to_dot(self):
        """Return a string for .dot representation"""
        out_list = self.to_list()
        return self._to_dot(out_list)

    @staticmethod
    def _to_dot(lst, out_string=''):
        value_string = str(lst[0])
        label_string = len(lst[0])
        for item in lst[1]:
            item_string = str(item[0])
            out_string += '''"{0}" -- "{1}";
            "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];\n'''.format(
                value_string, item_string, label_string)
            out_string = ClusterTree._to_dot(item, out_string)
        if not lst[1]:
            try:
                content_list = ['{:.2f}'.format(p) for p in lst[0].content]
            except ValueError:

                content_list = [str(['{:.2f}'.format(p) for p in item]) for item in lst[0].content]
            label_string = ', '.join(content_list)
            out_string += '"{0}"[label="{1}",color="#cccccc",style="filled",shape="box"];\n'.format(value_string,
                                                                                                    label_string)
        return out_string

    def depth(self, root_level=None):
        """Get depth of the tree

        :param root_level: internal use.
        :type root_level: int
        :return: depth
        :rtype: int

        .. hint::

            **recursive function**
        """
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def diameter(self):
        """Return the diameter of content

        :return: diameter
        :rtype: float
        """
        return self.content.diameter()

    def distance(self, other):
        """Return distance to other

        :param other: other cluster tree
        :type other: ClusterTree
        :return: distance
        :rtype: float
        """
        return self.content.distance(other.content)


def build_cluster_tree(splitable, max_leaf_size=1, start_level=0):
    """Factory for building a cluster tree

    :param splitable: strategy that decides the structure
    :type splitable: Splitable
    :param max_leaf_size: cluster size at which recursion stops
    :type max_leaf_size: int
    :param start_level: counter to identify levels
    :type start_level: int
    :return: cluster tree
    :rtype: ClusterTree
    """
    root = ClusterTree(splitable, max_leaf_size=max_leaf_size, level=start_level)
    recursion_build_cluster_tree(root)
    return root


def recursion_build_cluster_tree(current_tree):
    """Recursion to :func:`build_cluster_tree`
    """
    if len(current_tree.content) > current_tree.max_leaf_size:
        splits = current_tree.content.split()
        for split in splits:
            new_tree = ClusterTree(content=split,
                                   max_leaf_size=current_tree.max_leaf_size,
                                   level=current_tree.level + 1)
            current_tree.sons.append(new_tree)
            recursion_build_cluster_tree(new_tree)


def admissible(left_clustertree, right_clustertree):
    """Default admissible condition for BlockClusterTree

    True if the smaller diameter of the input is smaller or equal to the distance between the two ClusterTrees

    :param left_clustertree: "Left-side" ClusterTree
    :param right_clustertree: "Right-side" ClusterTree
    :type left_clustertree: ClusterTree
    :type right_clustertree: ClusterTree
    :return: admissible
    :rtype: bool
    """
    diam_min = min(left_clustertree.diameter(), right_clustertree.diameter())
    distance = left_clustertree.distance(right_clustertree)
    return diam_min <= distance
