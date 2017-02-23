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
        return self.points[item]

    def __iter__(self):
        """Iterate trough Grid"""
        return GridIterator(self)

    def __eq__(self, other):
        """Test for equality"""
        points_eq = numpy.array_equal(self.points, other.points)
        links_eq = numpy.array_equal(self.links, other.links)
        return points_eq and links_eq

    def dim(self):
        """Dimension of the Grid"""
        return len(self[0])


class GridIterator(object):
    """Iterator to Grid object"""

    def __init__(self, grid):
        self.grid = grid
        self._counter = 0

    def __iter__(self):
        return self

    def next(self):
        if self._counter >= len(self.grid):
            raise StopIteration
        else:
            self._counter += 1
            return self.grid[self._counter - 1]
