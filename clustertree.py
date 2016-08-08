"""clustertree.py: Implementation of a binary cluster tree for lists of indices.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        ClusterTree: Gives a cluster tree.
"""
from cluster import Cluster
from cuboid import minimal_cuboid


class BlockClusterTree(object):
    def __init__(self, left_clustertree, right_clustertree, admissible_function=admissible):
        pass


class ClusterTree(object):
    """
    """
    level = 0
    splitable = None
    left_son = None
    right_son = None

    def __init__(self, splitable, max_leaf_size=0, level=0):
        """"""
        self.level = level
        self.splitable = splitable
        if len(splitable) > max_leaf_size:
            left_splitable, right_splitable = splitable.split()
            self.left_son = ClusterTree(left_splitable, max_leaf_size, self.level + 1)
            self.right_son = ClusterTree(right_splitable, max_leaf_size, self.level + 1)

    def __repr__(self):
        optional_string = " with children" if self.left_son and self.right_son else ""
        return "<ClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def export(self, out_string='', outfile='./ct.dot'):
        """.dot representation of tree."""
        if self.level == 0:
            out_string = "graph {\n"
        if self.left_son:
            out_string += "\"{0!s}\" -- \"{1!s}\"\n".format(self.splitable.cluster.points,
                                                        self.left_son.splitable.cluster.points)
            out_string = self.left_son.export(out_string=out_string)
        if self.right_son:
            out_string += "\"{0!s}\" -- \"{1!s}\"\n".format(self.splitable.cluster.points,
                                                        self.right_son.splitable.cluster.points)
            out_string = self.right_son.export(out_string=out_string)
        if not self.level == 0:
            return out_string
        else:
            out_string += "}"
            of_handle = open(outfile, "w")
            of_handle.write(out_string)
            of_handle.close()

    def depth(self):
        if not self.left_son and not self.right_son:
            return self.level
        else:
            return max((self.left_son.depth(), self.right_son.depth()))


class Splitable(object):
    """Interface to the different strategies that can be used to split a cluster in two.

    Methods that need to be implemented by subclasses:
        __len__: return the length of the cluster
        split: split the object in two, return two new instances
        diameter: return the diameter of the object
        distance: return the distance to other object
    """
    def __len__(self):
        # type: () -> float
        raise NotImplementedError()

    def split(self):
        # type: () -> (Splitable, Splitable)
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


def admissible(left_splitable, right_splitable):
    """Default admissible condition for BlockClusterTree."""
    return max(left_splitable.diameter(), right_splitable.diameter()) < left_splitable.distance(right_splitable)
