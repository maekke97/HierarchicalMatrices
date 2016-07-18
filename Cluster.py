#!/usr/bin/env python

"""Cluster: Implementation of a Cluster.
    Part of master thesis "Hierarchical Matrices".

    Classes:
        Cluster: Gives a cluster.
"""


class Cluster(object):
    """Gives a cluster.

    """
    points = []
    links = []
    diam = 0

    def __init__(self, points, links, diameter=None, distance=None):
        self.points = points
        self.links = links
        if diameter:
            self.diam = diameter(points)
        else:
            self.diam = self.diameter()
        if distance:
            self.distance = distance

    def diameter(self):
        return max(abs(self.points))

    def distance(self, other_cluster):
        return max(abs([self.diam, other_cluster.diam]))
