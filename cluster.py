import numpy as np
from cuboid import Cuboid, minimal_cuboid


class Grid(object):
    """Stores points and links of discretized grid.

    links is a list of lists of points.
    """
    def __init__(self, points, links):
        self.points = points
        self.links = links
        self._current = 0

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return self

    def next(self):
        if self._current > len(self):
            raise StopIteration
        else:
            self._current += 1
            return self.points[self._current - 1], self.links[self._current - 1]


class Cluster(object):
    """Cluster that is a collection of connected points in Cartesian coordinates.

    Attributes:
        indices: list of indices
        grid: Grid object

    Part of master thesis "Hierarchical Matrices".
    """
    def __init__(self, grid, indices=None):
        # type: grid -> Cluster
        """Create a cluster.

        Create a cluster from a list of numpy.arrays.

        Args:
            grid: Grid object

        Raises:
            TypeError: points must be a list of numpy arrays and links must be a list of lists of numpy arrays!
        """
        self.grid = grid
        self.indices = range(len(grid)) if not indices else indices

    def __len__(self):
        return len(self.indices)

    def diameter(self):
        """Compute diameter.

        Return the maximal Euclidean distance between two points.
        For big lists of points this is costly.

        Returns:
            diameter: float.
        """
        # Build distance matrix
        points = [self.grid.points[i] for i in self.indices]
        points += [p for p in self.grid.links[i] for i in self.indices if p not in points]
        dist_mat = [np.linalg.norm(x, y) for x in points for y in points]
        return max(dist_mat)

    def distance(self, other):
        """Compute distance to other cluster.

        Return minimal Euclidean distance between points of self and other

        Args:
            other: another instance of Cluster.

        Returns:
            distance: float.
        """
        return min([np.linalg.norm(x - y) for x in self.points for y in other.points])
