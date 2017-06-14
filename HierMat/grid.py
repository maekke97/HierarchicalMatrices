"""grid.py: :class:`Grid` object and iterator
"""
import numpy


class Grid(object):
    """Discretized grid characterized by points and supports
    
    :param points: list of coordinates
    
        The points of the discretized Grid.
            
    :type points: list[tuple(float)]
    :param supports: dictionary mapping points with their supports
    :type supports: dict{point: support}
    :raise ValueError: if points and supports have different length
    """
    def __init__(self, points, supports):
        """Create a Grid"""
        # check input
        if len(points) != len(supports):
            raise ValueError('points and supports must be of same length')
        # fill instance
        self.points = points
        self.supports = supports

    def __len__(self):
        """Return number of points

        :return: len(points)
        :rtype: int
        """
        return len(self.points)

    def __getitem__(self, item):
        """Return point at item

        :param item: index to return
        :type item: int
        """
        return self.points[item]

    def __iter__(self):
        """Iterate trough Grid
        """
        return GridIterator(self)

    def __eq__(self, other):
        """Test for equality

        :param other: other grid
        :type other: Grid
        :return: True on equality
        :rtype: bool
        """
        points_eq = numpy.array_equal(self.points, other.points)
        links_eq = numpy.array_equal(self.supports, other.supports)
        return points_eq and links_eq

    def __ne__(self, other):
        """Test for inequality

        :param other: other grid
        :type other: Grid
        :return: True on inequality
        :rtype: bool
        """
        return not self == other

    def get_point(self, item):
        """Return point at position item

        :param item: index
        :type item: int
        :return: point
        """
        return self.points[item]

    def get_support(self, item):
        """Return support for item

        :param item: point
        :type item: float or tuple(float)
        :return: support
        """
        return self.supports[item]

    def get_support_by_index(self, index):
        """Return support for the i-th item

        :param index: index
        :type index: int
        :return: support
        """
        return self.supports[self.get_point(index)]

    def dim(self):
        """Dimension of the Grid
        
        .. note::
            
            this is the dimension of the first point
             
             if you store points of different dimensions, this will be misleading

        :return: dimension
        :rtype: int
        """
        return len(self[0])


class GridIterator(object):
    """Iterator to Grid object
    """
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
