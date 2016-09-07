import numpy


class Grid(object):
    """Discretized grid characterized by points and links

    Attributes:
        points: list of coordinates
        links: list of lists of points
            For every point in points a list of points that it is linked with

    Methods:
        len(): number of points
        dim(): dimension

    """
    def __init__(self, points, links):
        """Create a Grid

        Args:
            points: list of points
            links: list of links for each point. Must have same length as points

        Raises:
            ValueError if points and links are not of same length
        """
        self.points = points
        self.links = links
        if len(self.points) != len(self.links):
            raise ValueError("Points and links must be of same length.")
        self._current = 0

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        """Return point "item" and its links"""
        return self.points[item], self.links[item]

    def __iter__(self):
        """Iterate trough Grid"""
        return self

    def next(self):
        if self._current >= len(self):
            self._current = 0
            raise StopIteration
        else:
            self._current += 1
            return self[self._current - 1]

    def dim(self):
        """Dimension of the Grid"""
        if len(self):
            return len(self[0])
        else:
            return 0


class Cluster(object):
    """Handles operations on a Grid object by manipulating an index list

    Attributes:
        indices: list of indices
        grid: Grid object

    Methods:
        dim: dimension
        diameter: diameter
        distance(other): distance to other Cluster
    """
    def __init__(self, grid, indices=None):
        """Create a cluster

        Args:
            grid: Grid object
        """
        self.grid = grid
        self.indices = range(len(grid)) if not indices else indices
        self._current = 0

    def __getitem__(self, item):
        return self.grid[self.indices[item]]

    def __iter__(self):
        """Iterate through Cluster"""
        return self

    def next(self):
        if self._current >= len(self):
            self._current = 0
            raise StopIteration
        else:
            self._current += 1
            return self[self._current - 1]

    def __len__(self):
        """Number of points"""
        return len(self.indices)

    def dim(self):
        """Compute dimension"""
        return self.grid.dim()

    def diameter(self):
        """Compute diameter

        Return the maximal Euclidean distance between two points
        For big Clusters this is costly

        Returns:
            diameter: float
        """
        # get all points from grid in indices
        points = [self.grid.points[i] for i in self.indices]
        # add relevant links
        points += [p for p in self.grid.links[i] for i in self.indices if p not in points]
        # compute distance matrix
        dist_mat = [numpy.linalg.norm(x, y) for x in points for y in points]
        return max(dist_mat)

    def distance(self, other):
        """Compute distance to other cluster

        Return minimal Euclidean distance between points of self and other

        Args:
            other: another instance of Cluster

        Returns:
            distance: float
        """
        return min([numpy.linalg.norm(x - y) for x in self.points for y in other.points])
