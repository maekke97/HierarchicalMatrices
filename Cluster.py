#!/usr/bin/env python
import numpy as np
"""Cluster: Implementation of a Cluster.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        Cluster: Gives a cluster.
"""


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
        else:
            raise IOError("points and links must have same length!")

    def diameter(self):
        # TODO implement Euclidean default
        return self

    def distance(self, other_cluster):
        # TODO implement Euclidean default
        return self
