import argparse
import random
from timeit import default_timer as timer
import numpy as np
from profilehooks import profile

from HierMat import *


@profile
def main(args):
    lim = args.limit
    link_num = 2
    if args.dimension == 1:
        points = [((i + 0.5) / lim,) for i in xrange(lim)]
        supports = {points[i]: (points[i][0] - 0.5/lim, points[i][0] + 0.5/lim) for i in xrange(lim)}
    elif args.dimension == 2:
        points = [(float(i) / lim, float(j) / lim) for i in xrange(lim) for j in xrange(lim)]
        supports = {points[k]: (points[k-1], points[k+1]) for k in xrange(lim*lim)}
    else:
        points = [(float(i) / lim, float(j) / lim, float(k) / lim) for i in xrange(lim) for j in
                  xrange(lim) for k in xrange(lim)]
        supports = {points[k]: (points[k - 1], points[k + 1]) for k in xrange(lim * lim * lim)}
    grid = Grid(points, supports)
    cluster = Cluster(grid)
    rc = RegularCuboid(cluster)
    mc = MinimalCuboid(cluster)
    bc = Balanced(cluster)
    start = timer()
    ct_rc = ClusterTree(rc, 1)
    end = timer()
    print "ClusterTree build-up with RegularCuboid took " + str(end - start)
    start = timer()
    ct_mc = ClusterTree(mc, 1)
    end = timer()
    print "ClusterTree build-up with MinimalCuboid took " + str(end - start)
    start = timer()
    ct_bc = ClusterTree(bc, 1)
    end = timer()
    print "ClusterTree build-up with Balanced took " + str(end - start)
    start = timer()
    bct_rc_rc = BlockClusterTree(ct_rc, ct_rc)
    end = timer()
    print "BlockClusterTree build-up with RegularCuboid took " + str(end - start)
    start = timer()
    bct_mc_mc = BlockClusterTree(ct_mc, ct_mc)
    end = timer()
    print "BlockClusterTree build-up with MinimalCuboid took " + str(end - start)
    start = timer()
    bct_bc_bc = BlockClusterTree(ct_bc, ct_bc)
    end = timer()
    print "BlockClusterTree build-up with Balanced took " + str(end - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("HMatrix Profiler")
    parser.add_argument("dimension", type=int, choices=[1, 2, 3])
    parser.add_argument("limit per dimension", type=int)
    args = parser.parse_args()
    main(args)