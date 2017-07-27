from HierMat import *
import numpy as np

n = 2**5
points = [((i+0.5)/n,) for i in xrange(n)]
supports = {point: (point[0] - 0.5/n, point[0] + 0.5/n) for point in points}
grid = Grid(points=points, supports=supports)
cluster_1 = Cluster(grid=grid, indices=[0, 3, 7])
print len(cluster_1)
print cluster_1.diameter()
cluster_2 = Cluster(grid=grid)
print cluster_2.diameter()


class NewCluster(Cluster):
    def diameter(self):
        points = [self.grid.points[i] for i in self.indices]
        supps = [self.grid.supports[p][0] for p in points]
        supps += [self.grid.supports[p][1] for p in points]
        points += supps
        dist_mat = [np.linalg.norm(np.array(x) - np.array(y)) for x in points for y in points]
        return max(dist_mat)

new_cluster_1 = NewCluster(grid=grid, indices=[0, 3, 7])
new_cluster_2 = NewCluster(grid=grid)
print new_cluster_1.diameter()
print new_cluster_2.diameter()

strategy_1 = RegularCuboid(cluster=cluster_2)
unit_cuboid = Cuboid(0, 1)
strategy_2 = RegularCuboid(cluster=cluster_2, cuboid=unit_cuboid)
print strategy_1.diameter()
print strategy_2.diameter()

clustertree = build_cluster_tree(strategy_1)
export(clustertree, form='dot', out_file='clustertree.dot')


def admissible(left_clustertree, right_clustertree):
    diam_min = min(left_clustertree.diameter(), right_clustertree.diameter())
    distance = left_clustertree.distance(right_clustertree)
    return diam_min <= distance

blockclustertree = build_block_cluster_tree(
    left_cluster_tree=clustertree,
    right_cluster_tree=clustertree,
    admissible_function=admissible)
plot(blockclustertree, filename='blockclustertree.png', ticks=True)


def admissible_2(left_clustertree, right_clustertree):
    diam_min = min(left_clustertree.diameter(), right_clustertree.diameter())
    distance = left_clustertree.distance(right_clustertree)
    return diam_min < distance

blockclustertree_2 = build_block_cluster_tree(
    left_cluster_tree=clustertree,
    right_cluster_tree=clustertree,
    admissible_function=admissible_2)
plot(blockclustertree_2, filename='blockclustertree_2.png', ticks=True)

A = np.matrix([[1, 0], [2, 1], [1, 1]])
B = np.matrix([[0, 5], [1, 1], [1, 0]])
M_1 = RMat(left_mat=A, right_mat=B)
print M_1

C = np.matrix([[1, 0, 5], [2, 1, 1], [3, 1, 0]])
M_2 = RMat(left_mat=C, max_rank=2)
print M_2
