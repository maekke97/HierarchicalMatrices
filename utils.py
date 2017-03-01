"""utils.py: Utilities for the HMat class

    Contains all geometry related classes and functions

    Classes:
        BlockClusterTree: Block structure to HMat
        ClusterTree: Structure to BlockClusterTree
        Splitable: Interface for the splitting strategy to ClusterTree
        RegularCuboid: Splitable; Splits with help of regular cuboids
        MinimalCuboid: Splitable; Splits with minimal cuboids
        Balanced: Splitable; Splits evenly
        Cuboid: "Bounding Box" around a cluster. Used by RegularCuboid and MinimalCuboid
        Cluster: Wrapper around discretized grid
        Grid: Discretized grid

    Methods:
        admissible: admissibility condition for BlockClusterTree
        export: Export BlockClusterTree and ClusterTree to various formats
        minimal_cuboid: Build a minimal Cuboid around a Cluster
"""
import numpy

from cuboid import Cuboid


def load(filename):
    """Load a ClusterTree or BlockClusterTree from file."""
    import pickle
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj


def minimal_cuboid(cluster):
    """Build minimal cuboid

    Build minimal cuboid around cluster that is parallel to the axis in Cartesian coordinates

    Args:
        cluster: Cluster instance

    Returns:
        Minimal Cuboid
    """
    points = cluster.grid.points
    low_corner = numpy.array(points[0], float, ndmin=1)
    high_corner = numpy.array(points[0], float, ndmin=1)
    for p in points:
        p = numpy.array(p, float, ndmin=1)
        lower = p >= low_corner
        if not lower.all():
            for i in xrange(len(low_corner)):
                if not lower[i]:
                    low_corner[i] = p[i]
        higher = p <= high_corner
        if not higher.all():
            for i in xrange(len(high_corner)):
                if not higher[i]:
                    high_corner[i] = p[i]
    return Cuboid(low_corner, high_corner)


def admissible(left_clustertree, right_clustertree):
    """Default admissible condition for BlockClusterTree."""
    return max(left_clustertree.diameter(), right_clustertree.diameter()) < left_clustertree.distance(right_clustertree)


