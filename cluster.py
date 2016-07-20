import numpy as np


class Cluster(object):
    """
    Gives a cluster.
    """
    points = []
    links = []
    diam = 0

    def __init__(self, points, links):
        if len(points) == len(links):
            self.points = np.array(points)
            self.links = np.array(links)
            self.diam = self.diameter()
        else:
            raise IOError("points and links must have same length!")

    def diameter(self):
        """
        Return the maximal Euclidean distance between two points.
        :return: diameter (float)
        """
        return max([np.linalg.norm(x-y) for x in xrange(len(self.points)) for y in xrange(x+1, len(self.points))])

    def distance(self, other):
        """
        Return minimal Euclidean distance between points of self and other
        :param other: Cluster
        :return: distance (float)
        """
        return min([np.linalg.norm(x - y) for x in self.points for y in other.points])
