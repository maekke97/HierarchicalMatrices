"""splitable.py: :class:`Splitable` (Interface) and iterator and different strategies

- :class:`RegularCuboid`

    Starting from a minimal cuboid that get's split in half on every level, points are distributed according to
    their geometric location

- :class:`MinimalCuboid`

    Same as RegularCuboid, but after every split, the cuboids are reduced to minimal size

- :class:`Balanced`

    Distributes the points evenly on every level, depends solely on the order of the initial list
"""
from cluster import Cluster

from HierMat.utils import minimal_cuboid


class Splitable(object):
    """Interface to the different strategies that can be used to split a cluster in two or more.

    .. note::

        Methods that need to be implemented by subclasses:

        - __len__: return the length of the cluster
        - __iter__: return SplitableIterator(self)
        - __repr__: give a meaningful string representation
        - __getitem__: return item from inner cluster
        - __eq__: check for equality against other Splitable
        - get_index: return index from inner cluster
        - get_grid_item: return grid item from inner cluster
        - get_patch_coordinates: return min and max of index list of cluster
        - split: split the object in two or more, return new instances
        - diameter: return the diameter of the object
        - distance: return the distance to other object

    """
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        return SplitableIterator(self)

    def __repr__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __ne__(self, other):
        return not (self == other)

    def get_index(self, item):
        raise NotImplementedError()

    def get_grid_item(self, item):
        raise NotImplementedError()

    def get_patch_coordinates(self):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()

    def diameter(self):
        raise NotImplementedError()

    def distance(self, other):
        # type: (Splitable) -> float
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

    def __getitem__(self, item):
        return self.cluster[item]

    def __iter__(self):
        return SplitableIterator(self)

    def __len__(self):
        """Return the length of the cluster"""
        return len(self.cluster)

    def __eq__(self, other):
        """Check for equality"""
        return self.cluster == other.cluster and self.cuboid == other.cuboid

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

    def get_patch_coordinates(self):
        """Return min and max out of indices

        :return: min and max
        :rtype: tuple(int, int)
        """
        return self.cluster.get_patch_coordinates()

    def split(self):
        """Split the cuboid and distribute items in cluster according to the cuboid they belong to

        :return: left_RegularCuboid, right_RegularCuboid
        :rtype: RegularCuboid, RegularCuboid
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


class MinimalCuboid(Splitable):
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
        self.cluster = cluster
        self.cuboid = minimal_cuboid(cluster)

    def __eq__(self, other):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def __iter__(self):
        return SplitableIterator(self)

    def split(self):
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

    def __eq__(self, other):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def __iter__(self):
        return SplitableIterator(self)

    def split(self):
        pass

    def diameter(self):
        pass

    def distance(self, other):
        pass
