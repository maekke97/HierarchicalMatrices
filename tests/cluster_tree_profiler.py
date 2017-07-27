import argparse
import random
from timeit import default_timer as timer
import numpy as np
from memory_profiler import profile

from HierMat import *


@profile
def main(args):
    lim = args.limit
    if args.dimension == 1:
        points = [((i + 0.5) / lim,) for i in xrange(lim)]
        supports = {points[i]: (points[i][0] - 0.5/lim, points[i][0] + 0.5/lim) for i in xrange(lim)}
    elif args.dimension == 2:
        points = [(float(i) / lim, float(j) / lim) for i in xrange(lim) for j in xrange(lim)]
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
    mc = MinimalCuboid(cluster)
    bc = Balanced(cluster)
    start = timer()
    ct_rc = build_cluster_tree(rc, 1)
    end = timer()
    print "ClusterTree build-up with RegularCuboid took {0} seconds.".format(end - start)
    export(ct_rc, 'bin', 'ct_rc.bin')
    # start = timer()
    # ct_mc = build_cluster_tree(mc, 1)
    # end = timer()
    # print "ClusterTree build-up with MinimalCuboid took {0} seconds.".format(end - start)
    # export(ct_mc, 'bin', 'ct_mc.bin')
    start = timer()
    ct_bc = build_cluster_tree(bc, 1)
    end = timer()
    print "ClusterTree build-up with Balanced took {0} seconds.".format(end - start)
    export(ct_bc, 'bin', 'ct_bc.bin')
    start = timer()
    bct_rc_rc = build_block_cluster_tree(ct_rc, ct_rc)
    end = timer()
    print "BlockClusterTree build-up with RegularCuboid took {0} seconds." .format(end - start)
    export(bct_rc_rc, 'bin', 'bct_rc.bin')
    # start = timer()
    # bct_mc_mc = build_block_cluster_tree(ct_mc, ct_mc)
    # end = timer()
    # print "BlockClusterTree build-up with MinimalCuboid took {0} seconds.".format(end - start)
    # export(bct_mc_mc, 'bin', 'bct_mc.bin')
    start = timer()
    bct_bc_bc = build_block_cluster_tree(ct_bc, ct_bc)
    end = timer()
    print "BlockClusterTree build-up with Balanced took {0} seconds.".format(end - start)
    export(bct_bc_bc, 'bin', 'bct_bc.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("HMatrix Profiler")
    parser.add_argument("dimension", type=int, choices=[1, 2, 3])
    parser.add_argument("limit", type=int)
    args = parser.parse_args()
    print '\n' + '*' * 74 + '\n'
    main(args)
    print '\n' + '*' * 74 + '\n'
