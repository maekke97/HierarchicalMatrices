"""utils.py: Utilities for the :mod:`HMatrix` module
"""
import math
import numpy
from HierMat.cluster import Cluster
from HierMat.cuboid import Cuboid
# TODO: write tests


def load(filename):
    """Load a :class:`ClusterTree` or :class:`BlockClusterTree` from file

    :param filename: file to import
    :type filename: String
    :return: object
    :rtype: BlockClusterTree or ClusterTree

    .. note:: Depends on :mod:`pickle`
    """
    import pickle
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj


def admissible(left_clustertree, right_clustertree):
    """Default admissible condition for BlockClusterTree

    True if the smaller diameter of the input is smaller or equal to the distance between the two ClusterTrees

    :param left_clustertree: "Left-side" ClusterTree
    :param right_clustertree: "Right-side" ClusterTree
    :type left_clustertree: ClusterTree
    :type right_clustertree: ClusterTree
    :return: admissible
    :rtype: bool
    """
    diam_min = min(left_clustertree.diameter(), right_clustertree.diameter())
    distance = left_clustertree.distance(right_clustertree)
    return diam_min <= distance


def divisor_generator(n):
    """Return divisors of n

    :param n: integer to find divisors of
    :type n: int
    :return: divisors
    :rtype: list[int]

    .. warning::
       This is a generator! To get a list with all divisors call::

          list(divisor_generator(n))


    .. note::
       found at
       `StackOverflow
       <http://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number>`_
       on 2017.03.08
    """
    large_divisors = []
    for i in xrange(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


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
