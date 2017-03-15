from cluster import Cluster
from cuboid import minimal_cuboid


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

    def __iter__(self):
        return SplitableIterator(self)

    def __repr__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def get_index(self, item):
        raise NotImplementedError()

    def get_grid_item(self, item):
        raise NotImplementedError()

    def get_patch_coordinates(self):
        raise NotImplementedError()

    def split(self):
        raise NotImplementedError()

    def diameter(self):
        # type: () -> float
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
    """Method of regular cuboids.

    Minimal cuboid is built around initial list of indices. Split is then
    implemented by splitting the surrounding cuboid in split and distributing indices according to the cuboid they
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
        return self.cluster.get_index(item)

    def get_grid_item(self, item):
        return self.cluster.get_grid_item(item)

    def get_patch_coordinates(self):
        return self.cluster.get_patch_coordinates()

    def split(self):
        """Split the cuboid and distribute items in cluster according to the cuboid they belong to

        Returns:
            left_RegularCuboid, right_RegularCuboid
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
