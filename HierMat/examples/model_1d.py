"""This is an implementation of the one-dimensional model example from :cite:`borm2003hierarchical`.
It shows a basic use-case of hierarchical matrices:
    
.. admonition:: integral equation
    
    we start with the one-dimensional integral equation of the form
    
    .. math::
        
        u(x) + \int_0^1 \log |x-y| u(y) dy = g(x)
    
    for :math:`x \in [0,1]`.
    
    

"""
import numpy

import HierMat

import os
import math


def model_1d(n=2 ** 3, max_rank=1, n_min=1, b=numpy.random.rand(2**5, 1)):
    """"""
    midpoints = [((i + 0.5)/n,) for i in xrange(n)]
    intervals = {p: (p[0] - 0.5/n, p[0] + 0.5/n) for p in midpoints}
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
                                 generate_full_matrix_function=lambda bct: galerkin_1d_full(bct)
                                 )
    hmat_full = hmat.to_matrix()
    galerkin_full = galerkin_1d_full(block_cluster_tree)
    HierMat.export(hmat, form='bin', out_file='hmat.bin')
    numpy.savetxt('hmat_full.txt', hmat_full)
    numpy.savetxt('gallmat_full.txt', galerkin_full)
    return numpy.linalg.norm(hmat_full-galerkin_full)


def kernel(x):
    """"""
    out = x**2 * (numpy.log(abs(x))-0.5)
    if math.isnan(out):
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

    def ker(x, y):
        """"""
        return numpy.log(numpy.abs(x - y))

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
        for i in xrange(x_length):
            midpoint = block_cluster_tree.left_clustertree[i]
            lower, upper = block_cluster_tree.left_clustertree.get_grid_item_support(midpoint)
            left_matrix[i, k] = ((upper - taylor_midpoint)**(k+1) - (lower - taylor_midpoint)**(k+1))/(k+1)
        for j in xrange(y_length):
            midpoint = block_cluster_tree.right_clustertree[j]
            lower, upper = block_cluster_tree.right_clustertree.get_grid_item_support(midpoint)
            if k == 0:
                right_matrix[j, k] = 0
            else:
                right_matrix[j, k] = (-float(1))**(k + 1)/(k * (k+1)) * ((taylor_midpoint - upper)**(1-k)
                                                                         - (taylor_midpoint - lower)**(1-k))
    return HierMat.RMat(left_mat=left_matrix, right_mat=right_matrix, max_rank=max_rank)


def galerkin_1d_full(block_cluster_tree):
    """Exact calculation of the integral

    .. math::

        A_{i,j}=A_{\\tau,t}^{Gal}=\int_t\int_\\tau\log\Vert x-y\Vert \;dydx
    
    :param block_cluster_tree:
    :type block_cluster_tree: HierMat.BlockClusterTree
    :return: matrix with same shape as block_cluster_tree.shape()
    :rtype: numpy.matrix
    """
    x_length, y_length = block_cluster_tree.shape()
    out_matrix = numpy.matrix(numpy.zeros((x_length, y_length)))
    for i in xrange(x_length):
        for j in xrange(y_length):
            c, d = block_cluster_tree.left_clustertree.get_grid_item_support_by_index(i)
            a, b = block_cluster_tree.right_clustertree.get_grid_item_support_by_index(j)
            out_matrix[i, j] = (-kernel(d-b)+kernel(c-b)+kernel(d-a)-kernel(c-a))/2+(a-b)*(d-c)
    return out_matrix


def gauss_legendre_interpolation(a=-1, b=1):
    """compute abscissas and weights for Gauss-Legendre formulas, I=(-1,1)
    abscissas are the zeroes of the Legendre polynomials
    weights are computed by integrating the associated Lagrange polynomials
    fixed at 5 gauss points
    """
    d = float(a+b)/2
    e = float(b-a)/2

    # tabular values
    out = dict()
    out[0] = (d, 2*e)
    out[1] = ([d-1/math.sqrt(3)*e, d+1/math.sqrt(3)*e], [e, e])
    out[2] = ([d-math.sqrt(15)/5*e, d, d+math.sqrt(15)/5*e], [5 * e/9, 8 * e/9, 5 * e/9])

    xa = math.sqrt(525-70*math.sqrt(30))/35*e
    xb = math.sqrt(525+70*math.sqrt(30))/35*e
    wa = (18+math.sqrt(30))/36*e
    wb = (18-math.sqrt(30))/36*e

    out[3] = ([d-xb, d-xa, d+xa, d+xb], [wb, wa, wa, wb])

    xd = math.sqrt(245-14*math.sqrt(70))/21*e
    xe = math.sqrt(245+14*math.sqrt(70))/21*e
    wd = (322+13*math.sqrt(70))/900*e
    we = (322-13*math.sqrt(70))/900*e

    out[4] = ([d-xe, d-xd, d, d+xd, d+xe], [we, wd, 128/225*e, wd, we])

    return out


def get_interpol_points(k):
    return [math.cos((2*v+1)*math.pi/(2*k)) for v in xrange(k)].sort()


if __name__ == '__main__':
    model_1d(b=numpy.random.rand(2**3), n=2**3, max_rank=1, n_min=1)
