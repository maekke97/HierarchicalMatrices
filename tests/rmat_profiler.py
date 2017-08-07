import argparse
from timeit import default_timer as timer
from memory_profiler import profile
import numpy as np

from HierMat import *


@profile
def main(limit, rank):
    left1 = np.matrix(np.random.rand(limit, rank))
    right1 = np.matrix(np.random.rand(limit, rank))
    rmat1 = RMat(left_mat=left1, right_mat=right1, max_rank=rank)
    left2 = np.matrix(np.random.rand(limit, rank))
    right2 = np.matrix(np.random.rand(limit, rank))
    rmat2 = RMat(left_mat=left2, right_mat=right2, max_rank=rank)
    start = timer()
    add_res = rmat1 + rmat2
    end = timer()
    print "Addition with n={0} and k={1} took {2} seconds.".format(limit, rank, end - start)
    start = timer()
    add_res = rmat1 - rmat2
    end = timer()
    print "Subtraction with n={0} and k={1} took {2} seconds.".format(limit, rank, end - start)
    start = timer()
    mul_res = rmat1 * rmat2
    end = timer()
    print "Multiplication with n={0} and k={1} took {2} seconds.".format(limit, rank, end - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("RMatrix Profiler")
    parser.add_argument("limit", type=int)
    parser.add_argument("rank", type=int)
    try:
        args = parser.parse_args()
        print '\n' + '*' * 74 + '\n'
        print 'Starting run with n={0} and k={1}'.format(args.limit, args.rank) + '\n'
        main(args.limit, args.rank)
        print '\n' + 'Finished run with n={0}'.format(args.limit)
        print '\n' + '*' * 74 + '\n'
    except SystemExit:
        main(20, 2)
