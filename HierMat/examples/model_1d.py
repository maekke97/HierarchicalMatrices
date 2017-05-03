"""This is an implementation of the one-dimensional model example from :cite:`borm2003hierarchical`.
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
import scipy.integrate as integrate


def model_1d(n=2 ** 5, max_rank=1, n_min=1):
    """"""
    midpoints = [((i + 0.5)/n,) for i in xrange(n)]
    intervals = {p: [p[0] - 0.5/n, p[0] + 0.5/n] for p in midpoints}
    grid = HierMat.Grid(points=midpoints, supports=intervals)
    cluster = HierMat.Cluster(grid=grid)
    unit_cuboid = HierMat.Cuboid([0], [1])
    strategy = HierMat.RegularCuboid(cluster=cluster, cuboid=unit_cuboid)
    cluster_tree = HierMat.build_cluster_tree(splitable=strategy, max_leaf_size=n_min)
    HierMat.export(cluster_tree, form='dot', out_file='galerkin_1d_ct.dot')
    os.system('dot -Tpng galerkin_1d_ct.dot > model_1d-ct.png')
    os.system('dot -Tsvg galerkin_1d_ct.dot > model_1d-ct.svg')
    block_cluster_tree = HierMat.build_block_cluster_tree(left_cluster_tree=cluster_tree,
                                                          right_cluster_tree=cluster_tree,
                                                          admissible_function=HierMat.admissible
                                                          )
    HierMat.plot(block_cluster_tree, filename='model_1d-bct.png')
    hmat = HierMat.build_hmatrix(block_cluster_tree=block_cluster_tree,
                                 generate_rmat_function=lambda bct: galerkin_1d_rank_k(bct, max_rank),
                                 generate_full_matrix_function=galerkin_1d_full
                                 )
    hmat_full = hmat.to_matrix()
    x = numpy.ones((n, 1))
    for i in xrange(1, n, 2):
        x[i] = 2
    galerkin_full = galerkin_1d_full(block_cluster_tree)
    HierMat.export(hmat, form='bin', out_file='hmat.bin')
    numpy.savetxt('hmat_full.txt', hmat_full)
    numpy.savetxt('gallmat_full.txt', galerkin_full)
    rmat = galerkin_1d_rank_k(block_cluster_tree=block_cluster_tree, max_rank=max_rank)
    return True


def kernel(x, y):
    """"""
    out = numpy.log(numpy.linalg.norm(x - y))
    if out in [numpy.inf, -numpy.inf]:
        return 0
    else:
        return out


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
    # determine the interval
    first_point_x = block_cluster_tree.left_clustertree[0]
    last_point_x = block_cluster_tree.left_clustertree[-1]
    lower_boundary_x = block_cluster_tree.left_clustertree.get_grid_item_support(first_point_x)[0]
    upper_boundary_x = block_cluster_tree.left_clustertree.get_grid_item_support(last_point_x)[-1]
    taylor_midpoint = float(lower_boundary_x + upper_boundary_x)/2
    # x_interpolation
    for k in xrange(max_rank):
        def integral_function_x(x):
            return (x - taylor_midpoint)**k

        for i in xrange(x_length):
            midpoint = block_cluster_tree.left_clustertree[i]
            lower, upper = block_cluster_tree.left_clustertree.get_grid_item_support(midpoint)
            left_matrix[i, k] = integrate.fixed_quad(integral_function_x, lower, upper)[0]

        for j in xrange(y_length):
            midpoint = block_cluster_tree.right_clustertree[j]
            lower, upper = block_cluster_tree.right_clustertree.get_grid_item_support(midpoint)
            if k == 0:
                def integral_function_y(y):
                    return kernel(taylor_midpoint, y)

                right_matrix[j, k] = integrate.fixed_quad(integral_function_y, lower, upper)[0]
            else:
                def integral_function_y(y):
                    return (taylor_midpoint - y) ** (-k)

                integral = integrate.fixed_quad(integral_function_y, lower, upper)[0]
                right_matrix[j, k] = (-float(1))**(k + 1)/k * integral
    return HierMat.RMat(left_mat=left_matrix, right_mat=right_matrix, max_rank=max_rank)


def galerkin_1d_full(block_cluster_tree):
    """
    
    :param block_cluster_tree:
    :type block_cluster_tree: HierMat.BlockClusterTree
    :return:
    :rtype: numpy.matrix
    """
    x_length, y_length = block_cluster_tree.shape()
    out_matrix = numpy.matrix(numpy.zeros((x_length, y_length)))
    for i in xrange(x_length):
        midpoint_x = block_cluster_tree.left_clustertree[i]
        x_lower, x_upper = block_cluster_tree.left_clustertree.get_grid_item_support(midpoint_x)
        for j in xrange(y_length):
            midpoint_y = block_cluster_tree.right_clustertree[j]
            y_lower, y_upper = block_cluster_tree.right_clustertree.get_grid_item_support(midpoint_y)
            inner_integral = lambda x: integrate.fixed_quad(lambda y: kernel(x, y), y_lower, y_upper)[0]
            out_matrix[i, j] = integrate.fixed_quad(inner_integral, x_lower, x_upper)[0]
    return out_matrix


if __name__ == '__main__':
    model_1d(n=2**5, max_rank=2, n_min=1)
