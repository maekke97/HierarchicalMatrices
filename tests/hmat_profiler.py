import argparse
from timeit import default_timer as timer
from memory_profiler import profile
import numpy as np

from HierMat import *


def build_rank(bct):
    n = len(bct.left_clustertree)
    m = len(bct.right_clustertree)
    left = np.matrix(np.random.rand(n, 1))
    right = np.matrix(np.random.rand(m, 1))
    return RMat(left_mat=left, right_mat=right, max_rank=1)


def build_full(bct):
    n = len(bct.left_clustertree)
    m = len(bct.right_clustertree)
    return np.matrix(np.random.rand(n, m))


def main(limit):
    file_str = '/compute/nem/bct_{0}'.format(limit)
    bct = load(file_str)
    hmat = build_hmatrix(block_cluster_tree=bct,
                         generate_rmat_function=build_rank,
                         generate_full_matrix_function=build_full)
    start = timer()
    add_res = hmat + hmat
    end = timer()
    print "Addition with n={0} and k={1} took {2} seconds.".format(limit, rank, end - start)
    start = timer()
    add_res = hmat - hmat
    end = timer()
    print "Subtraction with n={0} and k={1} took {2} seconds.".format(limit, rank, end - start)
    start = timer()
    mul_res = hmat * hmat
    end = timer()
    print "Multiplication with n={0} and k={1} took {2} seconds.".format(limit, rank, end - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("HMatrix Profiler")
    parser.add_argument("limit", type=int)
    try:
        args = parser.parse_args()
        print '\n' + '*' * 74 + '\n'
        print 'Starting run with n={0} and k={1}'.format(args.limit, args.rank) + '\n'
        main(args.limit)
        print '\n' + 'Finished run with n={0}'.format(args.limit)
        print '\n' + '*' * 74 + '\n'
    except SystemExit:
        main(20)
