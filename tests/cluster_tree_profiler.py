import argparse
import math
from timeit import default_timer as timer
import numpy as np
from memory_profiler import profile

from HierMat import *


@profile
def main(limit, dimension):
    lim = limit
    if dimension == 1:
        points = [((i + 0.5) / lim,) for i in xrange(lim)]
        supports = {points[i]: (points[i][0] - 0.5/lim, points[i][0] + 0.5/lim) for i in xrange(lim)}
    elif dimension == 2:
        points = [(math.sin(float(i)/lim), math.cos(float(j)/lim)) for i in xrange(lim) for j in xrange(lim)]
        supports = {points[k]: (points[k-1], points[k+1]) for k in xrange(lim * lim - 1)}
        supports[points[lim*lim-1]] = (points[lim*lim-2], points[0])
    else:
        points = [(float(i) / lim, float(j) / lim, float(k) / lim) for i in xrange(lim) for j in
                  xrange(lim) for k in xrange(lim)]
        supports = {points[k]: (points[k - 1], points[k + 1]) for k in xrange(lim * lim * lim - 1)}
        supports[points[lim * lim * lim - 1]] = (points[lim * lim * lim - 2], points[0])
    grid = Grid(points, supports)
    cluster = Cluster(grid)
    rc = RegularCuboid(cluster)
    start = timer()
    ct_rc = build_cluster_tree(rc, 1)
    end = timer()
    print "ClusterTree build-up with RegularCuboid took {0} seconds.".format(end - start)
    start = timer()
    bct_rc_rc = build_block_cluster_tree(ct_rc, ct_rc)
    end = timer()
    print "BlockClusterTree build-up with RegularCuboid took {0} seconds." .format(end - start)
    file_str = '/compute/nem/bct_{0}'.format(limit)
    export(bct_rc_rc, form='bin', out_file=file_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("HMatrix Profiler")
    parser.add_argument("dimension", type=int, choices=[1, 2, 3])
    parser.add_argument("limit", type=int)
    try:
        args = parser.parse_args()
        print '\n' + '*' * 74 + '\n'
        print 'Starting run with n={0}'.format(args.limit) + '\n'
        main(args.limit, args.dimension)
        print '\n' + 'Finished run with n={0}'.format(args.limit)
        print '\n' + '*' * 74 + '\n'
    except SystemExit:
        main(20, 2)
