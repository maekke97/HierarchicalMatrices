import numpy as np


class Cluster(object):
    """Cluster that is a collection of connected points in Cartesian coordinates.

    Attributes:
        points: List of points in Cartesian coordinates.
            List of numpy.arrays
        links: For each point in points a list of points that it is connected with.
            List of lists of numpy.arrays

    Part of master thesis "Hierarchical Matrices".
    """
    points = []
    links = []
    diam = 0

    def __init__(self, points, links):
        """Create a cluster.

        Create a cluster from a list of numpy.arrays.

        Args:
            points: List of points in Cartesian coordinates.
                List of numpy.arrays
            links: For each point in points a list of points that it is connected with.
                List of lists of numpy.arrays

        Raises:
            TypeError: points must be a list of numpy arrays and links must be a list of lists of numpy arrays!
        """
        if isinstance(points[0], np.ndarray) and isinstance(links[0][0], np.ndarray):
            self.points = points
            self.links = links
            self.diam = self._diameter()
        else:
            raise TypeError("points must be list of numpy arrays and links must be list of lists of numpy arrays!")

    def _diameter(self):
        """Compute diameter.

        Return the maximal Euclidean distance between two points.
        For big lists of points this is costly. That's why the diameter is computed at creation and saved to self.diam.

        Returns:
            diameter: float.
        """
        return max([np.linalg.norm(x-y) for x in xrange(len(self.points)) for y in xrange(x+1, len(self.points))])

    def distance(self, other):
        """Compute distance to other cluster.

        Return minimal Euclidean distance between points of self and other

        Args:
            other: another instance of Cluster.

        Returns:
            distance: float.
        """
        return min([np.linalg.norm(x - y) for x in self.points for y in other.points])
