"""utils.py: Utilities for the HMat class

    Contains all geometry related classes and functions

    Classes:
        BlockClusterTree: Block structure to HMat
        ClusterTree: Structure to BlockClusterTree
        Splitable: Interface for the splitting strategy to ClusterTree
        RegularCuboid: Splitable; Splits with help of regular cuboids
        MinimalCuboid: Splitable; Splits with minimal cuboids
        Balanced: Splitable; Splits evenly
        Cuboid: "Bounding Box" around a cluster. Used by RegularCuboid and MinimalCuboid
        Cluster: Wrapper around discretized grid
        Grid: Discretized grid

    Methods:
        admissible: admissibility condition for BlockClusterTree
        export: Export BlockClusterTree and ClusterTree to various formats
        minimal_cuboid: Build a minimal Cuboid around a Cluster
"""
import numpy

from cuboid import Cuboid


def load(filename):
    """Load a ClusterTree or BlockClusterTree from file."""
    import pickle
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj


def minimal_cuboid(cluster):
    """Build minimal cuboid

    Build minimal cuboid around cluster that is parallel to the axis in Cartesian coordinates

    Args:
        cluster: Cluster instance

    Returns:
        Minimal Cuboid
    """
    points = cluster.grid.points
    low_corner = numpy.array(points[0], float, ndmin=1)
    high_corner = numpy.array(points[0], float, ndmin=1)
    for p in points:
        p = numpy.array(p, float, ndmin=1)
        lower = p >= low_corner
        if not lower.all():
            for i in xrange(len(low_corner)):
                if not lower[i]:
                    low_corner[i] = p[i]
        higher = p <= high_corner
        if not higher.all():
            for i in xrange(len(high_corner)):
                if not higher[i]:
                    high_corner[i] = p[i]
    return Cuboid(low_corner, high_corner)


def admissible(left_clustertree, right_clustertree):
    """Default admissible condition for BlockClusterTree."""
    return max(left_clustertree.diameter(), right_clustertree.diameter()) < left_clustertree.distance(right_clustertree)


class BlockClusterTree(object):
    def __init__(self, left_clustertree, right_clustertree, admissible_function=admissible, level=0):
        self.sons = []
        self.left_clustertree = left_clustertree
        self.right_clustertree = right_clustertree
        self.admissible = admissible_function
        self.level = level
        for left_son in self.left_clustertree.sons:
            for right_son in self.right_clustertree.sons:
                if not self.admissible(left_son, right_son):
                    self.sons.append(BlockClusterTree(left_son, right_son, self.admissible, self.level+1))

    def __repr__(self):
        optional_string = " with children {0!s}".format(self.sons) if self.sons else ""
        return "<BlockClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def __eq__(self, other):
        """Test for equality"""
        return (self.left_clustertree == other.left_clustertree
                and self.right_clustertree == other.right_clustertree
                and self.sons == other.sons
                and self.admissible == other.admissible
                and self.level == other.level
                )

    def depth(self, root_level=None):
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def to_list(self):
        return [self, [son.to_list() for son in self.sons]]

    def export(self, form='xml', out_file='bct_out'):
        """Export obj in specified format.

                implemented: xml, dot, bin
                """

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
                    out_string = _to_xml(item, out_string)
            out_string += "</node>\n"
            return out_string

        def _to_dot(lst, out_string=''):
            if len(lst) > 1 and type(lst[1]) is list:
                for item in lst[1]:
                    if type(item) is list:
                        value_string = str(lst[0])
                        item_string = str(item[0])
                        label_string = len(lst[0])
                        out_string += '''"{0}" -- "{1}";
                                "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];\n'''.format(
                            value_string, item_string, label_string)
                        out_string = _to_dot(item, out_string)
                    else:
                        value_string = str(lst[0])
                        item_string = item
                        label_string = len(eval(value_string.replace('|', ',')))
                        out_string += '''"{0}" -- "{1}";
                                "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];
                                "{1}"[color="#cccccc",style="filled",shape="box"];\n'''.format(value_string,
                                                                                               item_string,
                                                                                               label_string)
            return out_string

        if form == 'xml':
            openstring = 'w'
            export_list = self.to_list()
            head = '<?xml version="1.0" encoding="utf-8"?>\n'
            output = _to_xml(export_list)
            output = head + output
            with open(out_file, "w") as out:
                out.write(output)
        elif form == 'dot':
            openstring = 'w'
            export_list = self.to_list()
            head = 'graph {\n'
            output = _to_dot(export_list)
            tail = '}'
            output = head + output + tail
            with open(out_file, "w") as out:
                out.write(output)
        elif form == 'bin':
            import pickle
            openstring = 'wb'
            file_handle = open(out_file, openstring)
            pickle.dump(self, file_handle, protocol=-1)
            file_handle.close()
        else:
            raise NotImplementedError()


