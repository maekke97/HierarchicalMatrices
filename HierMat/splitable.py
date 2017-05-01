"""splitable.py: :class:`Splitable` (Interface) and iterator and different strategies

- :class:`RegularCuboid`

    Starting from a minimal cuboid that get's split in half on every level, points are distributed according to
    their geometric location

- :class:`MinimalCuboid`

    Same as RegularCuboid, but after every split, the cuboids are reduced to minimal size

- :class:`Balanced`

    Distributes the points evenly on every level, depends solely on the order of the initial list
"""
import numpy

from HierMat.cluster import Cluster
from HierMat.cuboid import Cuboid


class Splitable(object):
    """Interface to the different strategies that can be used to restructure a cluster in two or more.

    .. admonition:: Methods that need to be implemented by subclasses

        - __eq__: check for equality against other Splitable
        - split: restructure the object in two or more, return new instances
        - diameter: return the diameter of the object
        - distance: return the distance to other object

    """
    cluster = None

    def __iter__(self):
        return SplitableIterator(self)

    def __repr__(self):
        return '<Splitable with cluster {0}>'.format(self.cluster)

    def __getitem__(self, item):
        return self.cluster[item]

    def __len__(self):
        """Return the length of the cluster"""
        return len(self.cluster)

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        return not self == other

    def get_index(self, item):
        """Get index from cluster

        :param item: index to get
        :type item: int
        :return: index
        :rtype: int
        """
        return self.cluster.get_index(item)

    def get_grid_item(self, item):
        """Get grid item from cluster

        :param item: index of item to get
        :type item: int
        """
        return self.cluster.get_grid_item(item)

    def get_grid_item_support(self, item):
        """Return supports of item from grid

        :param item: point
        :type item: tuple(float)
        """
        return self.cluster.get_grid_item_support(item)

    def get_grid_item_support_by_index(self, item):
        """Return supports of item from grid

        :param item: index
        :type item: int
        """
        return self.cluster.get_grid_item_support_by_index(item)

    def get_patch_coordinates(self):
        """Return min and max out of cluster indices

        :return: min and max
        :rtype: tuple(int, int)
        """
        return self.cluster.get_patch_coordinates()

    def diameter(self):
        raise NotImplementedError()

    def distance(self, other):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()


class SplitableIterator(object):
    """Iterator to the Splitable implementations."""

    def __init__(self, obj):
        self.obj = obj
        self.counter = 0

    def __iter__(self):
        return self

    def next(self):
        if self.counter >= len(self.obj):
            raise StopIteration
        else:
            self.counter += 1
            return self.obj[self.counter - 1]


class RegularCuboid(Splitable):
    """Method of regular cuboids

    If no cuboid is given, a minimal cuboid is built around initial list of indices. Split is then
    implemented by splitting the surrounding cuboid in half along longest axis and distributing indices according to the
    cuboid they belong to.

    Gives a binary tree
    """

    def __init__(self, cluster, cuboid=None):
        """Build a RegularCuboid

        Args:
            cluster: Cluster instance
            cuboid: Cuboid instance surrounding the cluster. (optional)
        """
        self.cluster = cluster
        self.cuboid = cuboid if cuboid else minimal_cuboid(cluster)

    def __repr__(self):
        return "<RegularCuboid with cluster {0} and cuboid {1}>".format(self.cluster, self.cuboid)

    def __eq__(self, other):
        """:type other: RegularCuboid
        :rtype: bool 
        """
        return self.cluster == other.cluster and self.cuboid == other.cuboid

    def split(self):
        """Split the cuboid and distribute items in cluster according to the cuboid they belong to

        :return: list of smaller regular cuboids 
        :rtype: list(RegularCuboid)
        """
        left_cuboid, right_cuboid = self.cuboid.split()
        left_indices = []
        right_indices = []
        for index in self.cluster.indices:
            if self.cluster.grid.points[index] in left_cuboid:
                left_indices.append(index)
            else:
                right_indices.append(index)
        if len(left_indices) > 0:
            left_cluster = Cluster(self.cluster.grid, left_indices)
            left_rc = RegularCuboid(left_cluster, left_cuboid)
        else:
            left_rc = None
        if len(right_indices) > 0:
            right_cluster = Cluster(self.cluster.grid, right_indices)
            right_rc = RegularCuboid(right_cluster, right_cuboid)
        else:
            right_rc = None
        if not left_rc and right_rc:
            return [right_rc]
        elif not right_rc and left_rc:
            return [left_rc]
        else:
            return [left_rc, right_rc]

    def diameter(self):
        """Return the diameter of the cuboid

        Diameter of the surrounding cuboid

        :return: diameter
        :rtype: float
        """
        return self.cuboid.diameter()

    def distance(self, other):
        """Return the distance between own cuboid and other cuboid

        :param other: other regular cuboid
        :type other: RegularCuboid
        :return: distance
        :rtype: float
        """
        return self.cuboid.distance(other.cuboid)


class MinimalCuboid(RegularCuboid):
    """Method of minimal cuboids

    Split is implemented by splitting the surrounding cuboid in half along longest axis and distributing indices 
    according to the cuboid they belong to. Afterwards all resulting cuboids are shrunk to minimal.

    Gives a binary tree
    """

    def __init__(self, cluster):
        """Build a MinimalCuboid

        :param cluster: cluster to surround
        :type cluster: Cluster
        """
        super(MinimalCuboid, self).__init__(cluster)

    def __repr__(self):
        return "<MinimalCuboid with cluster {0} and cuboid {1}>".format(self.cluster, self.cuboid)

    def split(self):
        """Split the cuboid and distribute items in cluster according to the cuboid they belong to. Reduce every restructure
        to minimal size

        :return: list of minimal cuboids of smaller size
        :rtype: list(MinimalCuboid)
        """
        splits = super(MinimalCuboid, self).split()
        outs = [MinimalCuboid(split.cluster) for split in splits]
        return outs


class Balanced(Splitable):
    """Balanced strategy, where a cluster is always restructure in half only according to its size, without regarding the
    geometry
    
    Gives a binary tree
    """

    def __init__(self, cluster):
        """
        
        :param cluster: a cluster instance
        :type cluster: Cluster
        """
        self.cluster = cluster

    def __eq__(self, other):
        return self.cluster == other.cluster

    def __len__(self):
        return len(self.cluster)

    def __getitem__(self, item):
        return self.cluster[item]

    def __repr__(self):
        return "<Balanced with cluster {0}>".format(self.cluster)

    def get_index(self, item):
        """Get index from cluster

        :param item: index to get
        :type item: int
        :return: index
        :rtype: int
        """
        return self.cluster.get_index(item)

    def get_grid_item(self, item):
        """Get grid item from cluster

        :param item: index of item to get
        :type item: int
        """
        return self.cluster.get_grid_item(item)

    def get_grid_item_support(self, item):
        """Return supports of item from grid

        :param item: point
        :type item: tuple(float)
        """
        return self.cluster.get_grid_item_support(item)

    def get_grid_item_support_by_index(self, item):
        """Return supports of item from grid

        :param item: index
        :type item: int
        """
        return self.cluster.get_grid_item_support_by_index(item)

    def get_patch_coordinates(self):
        """Return min and max out of indices

        :return: min and max
        :rtype: tuple(int, int)
        """
        return self.cluster.get_patch_coordinates()

    def diameter(self):
        """Return the diameter

        Diameter of the contained cluster

        :return: diameter
        :rtype: float
        """
        return self.cluster.diameter()

    def distance(self, other):
        """Distance to other balanced
        
        :param other: other balanced
        :type other: Balanced
        :return: distance
        :rtype: float
        """
        return self.cluster.distance(other.cluster)

    def split(self):
        """Split in two
        
        distribute the first (smaller) half to the left and the rest to the right
        
        return itself if it has only one point
        
        :rtype: list(Balanced)
        """
        split = int(len(self)/2)
        if split >= 1:
            left_cluster = Cluster(self.cluster.grid, self.cluster.indices[:split])
            right_cluster = Cluster(self.cluster.grid, self.cluster.indices[split:])
            return [Balanced(left_cluster), Balanced(right_cluster)]
        else:
            return self


def minimal_cuboid(cluster):
    """Build minimal cuboid

    Build minimal cuboid around cluster that is parallel to the axis in Cartesian coordinates

    :param cluster: cluster to build cuboid around
    :type cluster: Cluster
    :return: minimal cuboid
    :rtype: Cuboid
    """
    points = cluster.grid.points
    low_corner = numpy.array(points[0], float, ndmin=1)
    high_corner = numpy.array(points[0], float, ndmin=1)
    for p in points:
        p = numpy.array(p, float, ndmin=1)
        lowers = (p < low_corner).nonzero()
        for l in lowers:
            low_corner[l] = p[l]
        highers = (p > high_corner).nonzero()
        for h in highers:
            high_corner[h] = p[h]
    return Cuboid(low_corner, high_corner)