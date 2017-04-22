"""This is an implementation of the one-dimensional model example from chapter 5.1.2 in :cite:`hackbusch2015hierarchical`.
It shows a basic use-case of hierarchical matrices:
    
.. admonition:: integral equation
    
    we start with the generic one-dimensional integral equation of the form
    
    .. math::
        
        u(x) + \int_0^1 \log |x-y| u(y) dy = g(x)
    
    for :math:`x \in [0,1]`.
    
    

"""
import numpy

import HierMat

import os
import math
import scipy


def model_1d(n=2 ** 5, max_rank=2, gauss_points=1):
    """"""
    h = float(1)/n
    midpoints = [(i + 0.5) * h for i in xrange(n)]
    intervals = [[i * h, (i + 1) * h] for i in xrange(n)]
    supports = {point: lambda x: x == point for point in midpoints}
    grid = HierMat.Grid(points=midpoints, supports=supports)
    cluster = HierMat.Cluster(grid=grid)
    unit_cuboid = HierMat.Cuboid([0], [1])
    strategy = HierMat.RegularCuboid(cluster=cluster, cuboid=unit_cuboid)
    cluster_tree = HierMat.build_cluster_tree(splitable=strategy, max_leaf_size=max_rank)
    HierMat.export(cluster_tree, form='dot', out_file='galerkin_1d_ct.dot')
    os.system('dot -Tpng galerkin_1d_ct.dot > model_1d-ct.png')
    os.system('dot -Tsvg galerkin_1d_ct.dot > model_1d-ct.svg')
    block_cluster_tree = HierMat.build_block_cluster_tree(left_cluster_tree=cluster_tree,
                                                          right_cluster_tree=cluster_tree,
                                                          admissible_function=HierMat.admissible
                                                          )
    HierMat.plot(block_cluster_tree, filename='model_1d-bct.png')


def kernel(x, y):
    """"""
    return math.log(math.fabs(x - y))


def galerkin_1d_rank_k(block_cluster_tree, max_rank):
    """
    
    :param block_cluster_tree: admissible block cluster tree
    :type block_cluster_tree: HierMat.BlockClusterTree
    :param max_rank: separation rank
    :type max_rank: int
    :return: 
    """
    x_length, y_length = block_cluster_tree.shape()
    left_matrix = numpy.matrix(numpy.zeros((x_length, max_rank)))
    right_matrix = numpy.matrix(numpy.zeros((y_length, max_rank)))

    return HierMat.RMat()


def collocation_1d_rank_k(block_cluster_tree):

    return HierMat.RMat()


if __name__ == '__main__':
    model_1d()
