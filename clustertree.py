"""clustertree.py: Implementation of a cluster tree for lists of indices.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        ClusterTree: Gives a cluster tree.
"""
from cluster import Cluster
from cuboid import minimal_cuboid


def admissible(left_splitable, right_splitable):
    """Default admissible condition for BlockClusterTree."""
    return max(left_splitable.diameter(), right_splitable.diameter()) < left_splitable.distance(right_splitable)


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

    def _export(self):
        return "[{0},{1}]\n".format(self.left_clustertree._export(), self.right_clustertree._export())

    def export(self):
        if self.sons:
            out = [self._export()]
            out.append([son.export() for son in self.sons])
        else:
            out = self._export()
        return out


class ClusterTree(object):
    """
    """
    level = 0
    splitable = None
    sons = []

    def __init__(self, splitable, max_leaf_size=0, level=0):
        """"""
        self.level = level
        self.splitable = splitable
        self.sons = []
        if len(splitable) > max_leaf_size:
            self.splits = splitable.split()
            for split in self.splits:
                self.sons.append(ClusterTree(split, max_leaf_size, self.level + 1))

    def __repr__(self):
        optional_string = " with children {0!s}".format(self.sons) if self.sons else ""
        return "<ClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def _export(self):
        """give str representation of self. (Internal)"""
        return "{0!s}".format(self.splitable.cluster.points)

    def export(self):
        """List representation of tree."""
        if self.sons:
            out = [self._export()]
            out.append([son.export() for son in self.sons])
        else:
            out = self._export()
        return out

    def depth(self):
        if not self.sons:
            return self.level
        else:
            return max([son.depth() for son in self.sons])

    def diameter(self):
        return self.splitable.diameter()

    def distance(self, other):
        return self.splitable.distance(other.splitable)


class Splitable(object):
    """Interface to the different strategies that can be used to split a cluster in two.

    Methods that need to be implemented by subclasses:
        __len__: return the length of the cluster
        split: split the object in two or more, return new instances
        diameter: return the diameter of the object
        distance: return the distance to other object
    """
    def __len__(self):
        # type: () -> float
        raise NotImplementedError()

    def split(self):
        # type: () -> (Splitables)
        raise NotImplementedError()

    def diameter(self):
        # type: () -> float
        raise NotImplementedError()

    def distance(self, other):
        # type: (Splitable) -> float
        raise NotImplementedError()


class RegularCuboid(Splitable):
    """Method of regular cuboids.

    Minimal cuboid is built around initial list of indices. Split is then
    implemented by splitting the surrounding cuboid in half and distributing indices according to the cuboid they
    belong to.

    Gives a binary tree.
    """
    def __init__(self, cluster, cuboid=None):
        """Build a RegularCuboid

        Args:
            cluster: Cluster instance
            cuboid: Cuboid instance surrounding the cluster. (optional)
        """
        self.cluster = cluster
        self.cuboid = cuboid if cuboid else minimal_cuboid(cluster)

    def split(self):
        """Split the cuboid in half and distribute items in cluster according to the cuboid they belong to

        Returns:
            left_RegularCuboid, right_RegularCuboid
        """
        left_cuboid, right_cuboid = self.cuboid.half()
        left_points = []
        right_points = []
        left_links = []
        right_links = []
        for index in xrange(len(self.cluster)):
            if self.cluster.points[index] in left_cuboid:
                left_points.append(self.cluster.points[index])
                left_links.append(self.cluster.links[index])
            else:
                right_points.append(self.cluster.points[index])
                right_links.append(self.cluster.links[index])
        left_cluster = Cluster(left_points, left_links)
        right_cluster = Cluster(right_points, right_links)
        return RegularCuboid(left_cluster, left_cuboid), RegularCuboid(right_cluster, right_cuboid)

    def __len__(self):
        """Return the length of the cluster"""
        return len(self.cluster)

    def diameter(self):
        """Return the diameter of the cuboid"""
        return self.cuboid.diameter()

    def distance(self, other):
        """Return the distance between own cuboid and other cuboid"""
        return self.cuboid.distance(other.cuboid)


class MinimalCuboid(Splitable):
    """
    """
    def __init__(self):
        pass

    def split(self):
        pass

    def __len__(self):
        pass

    def diameter(self):
        pass

    def distance(self, other):
        pass


class Balanced(Splitable):
    """
    """
    def __init__(self):
        pass

    def split(self):
        pass

    def __len__(self):
        pass

    def diameter(self):
        pass

    def distance(self, other):
        pass


def export(obj, form='xml', out_file='./out'):
    """Export obj in specified format.

    implemented: xml, dot
    """
    def _to_xml(lst, out_string=''):
        if type(lst) is list:
            value_string = str(lst[0])
        else:
            value_string = lst
        value_string = value_string.replace('array', '')
        # value_string = value_string.replace(' ', '')
        # value_string = value_string.replace('([', '(')
        # value_string = value_string.replace('])', ')')
        # value_string = value_string.replace('.)', ')')
        # value_string = value_string.replace('.,', ',')
        display_string = len(eval(value_string))
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
                    value_string = value_string.replace('array', '')
                    # value_string = value_string.replace(' ', '')
                    # value_string = value_string.replace('([', '(')
                    # value_string = value_string.replace('])', ')')
                    # value_string = value_string.replace('.)', ')')
                    # value_string = value_string.replace('.,', ',')
                    item_string = str(item[0])
                    item_string = item_string.replace('array', '')
                    # item_string = item_string.replace(' ', '')
                    # item_string = item_string.replace('([', '(')
                    # item_string = item_string.replace('])', ')')
                    # item_string = item_string.replace('.)', ')')
                    # item_string = item_string.replace('.,', ',')
                    label_string = len(eval(value_string))
                    out_string += '''"{0}" -- "{1}";
                    "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];\n'''.format(
                        value_string, item_string, label_string)
                    out_string = _to_dot(item, out_string)
                else:
                    value_string = str(lst[0])
                    value_string = value_string.replace('array', '')
                    # value_string = value_string.replace(' ', '')
                    # value_string = value_string.replace('([', '(')
                    # value_string = value_string.replace('])', ')')
                    # value_string = value_string.replace('.)', ')')
                    # value_string = value_string.replace('.,', ',')
                    item_string = item
                    item_string = item_string.replace('array', '')
                    # item_string = item_string.replace(' ', '')
                    # item_string = item_string.replace('([', '(')
                    # item_string = item_string.replace('])', ')')
                    # item_string = item_string.replace('.)', ')')
                    # item_string = item_string.replace('.,', ',')
                    label_string = len(eval(value_string))
                    out_string += '''"{0}" -- "{1}";
                    "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];
                    "{1}"[color="#cccccc",style="filled",shape="box"];\n'''.format(value_string, item_string, label_string)
        return out_string

    export_list = obj.export()
    if form == 'xml':
        head = '<?xml version="1.0" encoding="utf-8"?>\n'
        output = _to_xml(export_list)
        tail = ''
    elif form == 'dot':
        head = 'graph {\n'
        output = _to_dot(export_list)
        tail = '}'
    else:
        raise NotImplementedError()
    with open(out_file, "w") as out:
        out.write(head + output + tail)
