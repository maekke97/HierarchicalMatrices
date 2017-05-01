"""cluster.py: :class:`Cluster` object and iterator
"""
from HierMat.grid import Grid

import numpy


class Cluster(object):
    """Handles operations on a :class:`Grid` object by manipulating an index list
    """

    def __init__(self, grid, indices=None):
        """Create a cluster

        :param grid: grid to build cluster around
        :type grid: Grid
        :param indices: index list (optional)
        :type indices: list
        """
        self.grid = grid
        self.indices = range(len(grid)) if not indices else indices

    def __getitem__(self, item):
        """Get item

        :param item: index
        :type item: int
        """
        return self.grid[self.indices[item]]

    def __repr__(self):
        """String representation
        :rtype: str
        """
        return "<Cluster object with grid {0} and indices {1}>".format(self.grid, self.indices)

    def __iter__(self):
        """Iterate through Cluster
        """
        return ClusterIterator(self)

    def __len__(self):
        """Number of points
        """
        return len(self.indices)

    def __eq__(self, other):
        """Test for equality
        """
        return self.grid == other.grid and self.indices == other.indices

    def __ne__(self, other):
        """Test for inequality
        """
        return not self == other

    def get_grid_item(self, item):
        """Return item from grid

        :param item: index
        :type item: int
        """
        return self.grid.get_point(item)

    def get_grid_item_support_by_index(self, item):
        """Return supports of i-th item from grid

        :param item: index
        :type item: int
        """
        return self.grid.get_support_by_index(item)

    def get_grid_item_support(self, item):
        """Return supports of item from grid

        :param item: point
        :type item: tuple(float)
        """
        return self.grid.get_support(item)

    def get_index(self, item):
        """Return index at item

        :param item: index
        :type item: int
        :return: item-th index
        :rtype: int
        """
        return self.indices[item]

    def get_patch_coordinates(self):
        """Return min and max out of indices

        :return: min and max
        :rtype: tuple(int, int)
        """
        return min(self.indices), max(self.indices)

    def dim(self):
        """Compute dimension

        :return: dim of Grid
        :rtype: int
        """
        return self.grid.dim()

    def diameter(self):
        """Compute diameter

        Return the maximal Euclidean distance between two points
        For big Clusters this is costly

        :return: diameter
        :rtype: float
        """
        # get all points from grid in indices
        points = [self.grid.points[i] for i in self.indices]
        # compute distance matrix
        dist_mat = [numpy.linalg.norm(numpy.array(x) - numpy.array(y)) for x in points for y in points]
        return max(dist_mat)

    def distance(self, other):
        """Compute distance to other cluster

        Return minimal Euclidean distance between points of self and other

        :param other: another instance of Cluster

        :return: distance
        :rtype: float
        """
        return min([numpy.linalg.norm(x - y) for x in self for y in other])


class ClusterIterator(object):
    """Iterator to Cluster object"""

    def __init__(self, cluster):
        self.cluster = cluster
        self._counter = 0

    def __iter__(self):
        return self

    def next(self):
        if self._counter >= len(self.cluster):
            raise StopIteration
        else:
            self._counter += 1
            return self.cluster[self._counter - 1]
