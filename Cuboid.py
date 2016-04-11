#!/usr/bin/env python

"""Cuboid.py: Implementation of ax-parallel minimal- and splitting-cuboid to provide support for the geometric
    properties of ClusterTree and BlockClusterTree.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        Cuboid: Gives an ax-parallel cuboid.
"""

import numpy as np
import collections


class Cuboid(object):
    """Ax-parallel Cuboid. Only two diagonal corners are stored. NumPy.array used as type.
    """
    low_corner = None
    high_corner = None

    def __init__(self, low_corner, high_corner=None):
        # Check if iterable, cast to np.array (should be safe)
        if isinstance(low_corner, collections.Iterable) and isinstance(high_corner, collections.Iterable):
            self.low_corner = np.array(low_corner, float)
            self.high_corner = np.array(high_corner, float)
        # Only one argument
        elif not high_corner:
            # Copy
            if isinstance(low_corner, Cuboid):
                self.low_corner = low_corner.low_corner
                self.high_corner = low_corner.high_corner
            # Try to build minimal cuboid around input
            else:
                self.__init__(self.make_minimal(low_corner))

    def __eq__(self, other):
        return (self.low_corner == other.low_corner).all() and (self.high_corner == other.high_corner).all()

    def __contains__(self, item):
        return (self.low_corner < item).all() and (self.high_corner > item).all()

    def split(self):
        """Split the cuboid in the largest dimension.
            Return two new Cuboids.
        """
        # determine dimension in which to split
        index = np.argmax(abs(self.low_corner - self.high_corner))
        # determine value at splitting point
        split = (self.high_corner[index] - self.low_corner[index])/2
        low_corner1 = np.array(self.low_corner)
        low_corner2 = np.array(self.low_corner)
        low_corner2[index] = split
        high_corner1 = np.array(self.high_corner)
        high_corner2 = np.array(self.high_corner)
        high_corner1[index] = split
        return Cuboid(low_corner1, high_corner1), Cuboid(low_corner2, high_corner2)

    @staticmethod
    def make_minimal(points):
        """
        :param points:
        :return:
        """
        low_corner = None
        high_corner = None
        return Cuboid(low_corner, high_corner)
