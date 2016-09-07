"""utils.py: Utilities for the HMat class

    Contains all geometry related classes and functions

    Classes:
        BlockClusterTree: Block structure to HMat
        ClusterTree: Structure to BlockClusterTree
        Splitable: Interface for the splitting strategy to ClusterTree
        RegularCuboid: Splitable; Splits with help of regular cuboids
        MinimalCuboid: Splitable; Splits with minimal cuboids
        Balanced: Splitable; Splits evenly
        Cuboid: "Bounding Box" around a cluster. Used by RegularCuboid and MinimalCuboid
        Cluster: Wrapper around discretized grid
        Grid: Discretized grid

    Methods:
        admissible: admissibility condition for BlockClusterTree
        export: Export BlockClusterTree and ClusterTree to various formats
        minimal_cuboid: Build a minimal Cuboid around a Cluster
"""
import numpy
from numpy import array


class BlockClusterTree(object):
    def __init__(self, left_clustertree, right_clustertree, admissible_function=admissible, level=0):
        self.sons = []
        self.left_clustertree = left_clustertree
        self.right_clustertree = right_clustertree
        self.admissible = admissible_function
        self.level = level
        for left_son in self.left_clustertree.sons:
            for right_son in self.right_clustertree.sons:
                if not self.admissible(left_son, right_son):
                    self.sons.append(BlockClusterTree(left_son, right_son, self.admissible, self.level+1))

    def __repr__(self):
        optional_string = " with children {0!s}".format(self.sons) if self.sons else ""
        return "<BlockClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def depth(self, root_level=None):
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def _export(self):
        return "[{0}|{1}]\n".format(self.left_clustertree._export(), self.right_clustertree._export())

    def export(self):
        if self.sons:
            out = [self._export()]
            out.append([son.export() for son in self.sons])
        else:
            out = self._export()
        return out


class ClusterTree(object):
    """
    """
    level = 0
    splitable = None
    sons = []

    def __init__(self, splitable, max_leaf_size=0, level=0):
        """"""
        self.level = level
        self.splitable = splitable
        self.sons = []
        if len(splitable) > max_leaf_size:
            splits = splitable.split()
            for split in splits:
                self.sons.append(ClusterTree(split, max_leaf_size, self.level + 1))

    def __repr__(self):
        optional_string = " with children {0!s}".format(self.sons) if self.sons else ""
        return "<ClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def _export(self):
        """give str representation of self. (Internal)"""
        return ",".join([str(p) for p in self.splitable.cluster.indices])

    def export(self):
        """List representation of tree."""
        out = [self._export()]
        out.append([son.export() for son in self.sons])
        return out

    def depth(self, root_level=None):
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def diameter(self):
        return self.splitable.diameter()

    def distance(self, other):
        return self.splitable.distance(other.splitable)


class Splitable(object):
    """Interface to the different strategies that can be used to split a cluster in two.

    Methods that need to be implemented by subclasses:
        __len__: return the length of the cluster
        split: split the object in two or more, return new instances
        diameter: return the diameter of the object
        distance: return the distance to other object
    """
    def __len__(self):
        # type: () -> float
        raise NotImplementedError()

    def split(self):
        # type: () -> (Splitables)
        raise NotImplementedError()

    def diameter(self):
        # type: () -> float
        raise NotImplementedError()

    def distance(self, other):
        # type: (Splitable) -> float
        raise NotImplementedError()


class RegularCuboid(Splitable):
    """Method of regular cuboids.

    Minimal cuboid is built around initial list of indices. Split is then
    implemented by splitting the surrounding cuboid in half and distributing indices according to the cuboid they
    belong to.

    Gives a binary tree.
    """
    def __init__(self, cluster, cuboid=None):
        """Build a RegularCuboid

        Args:
            cluster: Cluster instance
            cuboid: Cuboid instance surrounding the cluster. (optional)
        """
        self.cluster = cluster
        self.cuboid = cuboid if cuboid else minimal_cuboid(cluster)

    def split(self):
        """Split the cuboid in half and distribute items in cluster according to the cuboid they belong to

        Returns:
            left_RegularCuboid, right_RegularCuboid
        """
        left_cuboid, right_cuboid = self.cuboid.half()
        left_points = []
        right_points = []
        for index in self.cluster.indices:
            if self.cluster.grid.points[index] in left_cuboid:
                left_points.append(index)
            else:
                right_points.append(index)
        left_cluster = Cluster(self.cluster.grid, left_points)
        right_cluster = Cluster(self.cluster.grid, right_points)
        return RegularCuboid(left_cluster, left_cuboid), RegularCuboid(right_cluster, right_cuboid)

    def __len__(self):
        """Return the length of the cluster"""
        return len(self.cluster)

    def diameter(self):
        """Return the diameter of the cuboid"""
        return self.cuboid.diameter()

    def distance(self, other):
        """Return the distance between own cuboid and other cuboid"""
        return self.cuboid.distance(other.cuboid)


class MinimalCuboid(Splitable):
    """
    """
    def __init__(self):
        pass

    def split(self):
        pass

    def __len__(self):
        pass

    def diameter(self):
        pass

    def distance(self, other):
        pass


class Balanced(Splitable):
    """
    """
    def __init__(self):
        pass

    def split(self):
        pass

    def __len__(self):
        pass

    def diameter(self):
        pass

    def distance(self, other):
        pass


class Cuboid(object):
    """Cuboid that is parallel to the axis in Cartesian coordinates.

    Characterized by two diagonal corners.

    Attributes:
        low_corner: numpy.array with minimal values in each dimension
        high_corner: numpy.array with maximal values in each dimension

    Part of master thesis "Hierarchical Matrices".
    """
    low_corner = None
    high_corner = None

    def __init__(self, low_corner, high_corner):
        """Build a cuboid.

        Args:
            low_corner, high_corner: numpy.array of same length.
        """
        self.low_corner = array(low_corner, float)
        self.high_corner = array(high_corner, float)

    def __eq__(self, other):
        """Check if self is equal to other.

        Checks for equality in both low_corners and high_corners.

        Args:
            other: another instance of Cuboid.
        """
        low_eq = self.low_corner == other.low_corner
        high_eq = self.high_corner == other.high_corner
        return low_eq.all() and high_eq.all()

    def __contains__(self, point):
        """Check if point is inside the cuboid

        True if point is between low_corner and high_corner

        Args:
            point: numpy.array of correct dimension

        Returns:
            contained: boolean
        """
        lower = self.low_corner <= point
        higher = point <= self.high_corner
        return lower.all() and higher.all()

    def __repr__(self):
        """Representation of Cuboid"""
        return "Cuboid({0},{1})".format(str(self.low_corner), str(self.high_corner))

    def __str__(self):
        """Return string representation"""
        return "Cuboid with:\n\tlow corner: {0},\n\thigh corner{1}.".format(str(self.low_corner),
                                                                               str(self.high_corner))

    def half(self, axis=None):
        """Split the cuboid in half

        If axis is specified, the cuboid is split along the given axis, else the maximal axis is chosen.

        Optional args:
            axis: integer specifying the axis to choose.

        Returns:
            cuboid1, cuboid2: Cuboids
        """
        if axis:
            index = axis
        else:
            # determine dimension in which to half
            index = numpy.argmax(abs(self.high_corner - self.low_corner))
        # determine value at splitting point
        split = (self.high_corner[index] + self.low_corner[index])/2
        low_corner1 = array(self.low_corner)
        low_corner2 = array(self.low_corner)
        low_corner2[index] = split
        high_corner1 = array(self.high_corner)
        high_corner2 = array(self.high_corner)
        high_corner1[index] = split
        return Cuboid(low_corner1, high_corner1), Cuboid(low_corner2, high_corner2)

    def diameter(self):
        """Return the diameter of the cuboid.

        Diameter is the Euclidean distance between low_corner and high_corner.

        Returns:
            diameter: float
        """
        return numpy.linalg.norm(self.high_corner-self.low_corner)

    def distance(self, other):
        """Return distance to other Cuboid.

        Distance is the minimal Euclidean distance between points in self and points in other.

        Args:
            other: another instance of Cuboid.

        Returns:
            distance: float.
        """
        dimension = len(self.low_corner)
        distance1 = self.low_corner - other.low_corner
        distance2 = self.low_corner - other.high_corner
        distance3 = self.high_corner - other.low_corner
        distance4 = self.high_corner - other.high_corner
        distance_matrix = array((distance1, distance2, distance3, distance4))
        checks = abs(numpy.sum(numpy.sign(distance_matrix), 0)) == 4*numpy.ones(dimension)
        distance_vector = array(checks, dtype=float)
        min_vector = numpy.amin(abs(distance_matrix), axis=0)
        return numpy.linalg.norm(min_vector * distance_vector)


class Cluster(object):
    """Handles operations on a Grid object by manipulating an index list

    Attributes:
        indices: list of indices
        grid: Grid object

    Methods:
        dim: dimension
        diameter: diameter
        distance(other): distance to other Cluster
    """
    def __init__(self, grid, indices=None):
        """Create a cluster

        Args:
            grid: Grid object
        """
        self.grid = grid
        self.indices = range(len(grid)) if not indices else indices
        self._current = 0

    def __getitem__(self, item):
        return self.grid[self.indices[item]]

    def __iter__(self):
        """Iterate through Cluster"""
        return self

    def next(self):
        if self._current >= len(self):
            self._current = 0
            raise StopIteration
        else:
            self._current += 1
            return self[self._current - 1]

    def __len__(self):
        """Number of points"""
        return len(self.indices)

    def dim(self):
        """Compute dimension"""
        return self.grid.dim()

    def diameter(self):
        """Compute diameter

        Return the maximal Euclidean distance between two points
        For big Clusters this is costly

        Returns:
            diameter: float
        """
        # get all points from grid in indices
        points = [self.grid.points[i] for i in self.indices]
        # add relevant links
        points += [p for p in self.grid.links[i] for i in self.indices if p not in points]
        # compute distance matrix
        dist_mat = [numpy.linalg.norm(x, y) for x in points for y in points]
        return max(dist_mat)

    def distance(self, other):
        """Compute distance to other cluster

        Return minimal Euclidean distance between points of self and other

        Args:
            other: another instance of Cluster

        Returns:
            distance: float
        """
        return min([numpy.linalg.norm(x - y) for x in self.points for y in other.points])


class Grid(object):
    """Discretized grid characterized by points and links

    Attributes:
        points: list of coordinates
        links: list of lists of points
            For every point in points a list of points that it is linked with

    Methods:
        len(): number of points
        dim(): dimension

    """
    def __init__(self, points, links):
        """Create a Grid

        Args:
            points: list of points
            links: list of links for each point. Must have same length as points

        Raises:
            ValueError if points and links are not of same length
        """
        self.points = points
        self.links = links
        if len(self.points) != len(self.links):
            raise ValueError("Points and links must be of same length.")
        self._current = 0

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        """Return point "item" and its links"""
        return self.points[item], self.links[item]

    def __iter__(self):
        """Iterate trough Grid"""
        return self

    def next(self):
        if self._current >= len(self):
            self._current = 0
            raise StopIteration
        else:
            self._current += 1
            return self[self._current - 1]

    def dim(self):
        """Dimension of the Grid"""
        if len(self):
            return len(self[0])
        else:
            return 0


def admissible(left_clustertree, right_clustertree):
    """Default admissible condition for BlockClusterTree."""
    return max(left_clustertree.diameter(), right_clustertree.diameter()) < left_clustertree.distance(right_clustertree)


def export(obj, form='xml', out_file='./out'):
    """Export obj in specified format.

    implemented: xml, dot, bin
    """
    def _to_xml(lst, out_string=''):
        if len(lst[1]):
            value_string = str(lst[0])
            display_string = str(len(lst[1]))
        else:
            value_string = str(lst[0])
            display_string = str(lst[0])
        out_string += '<node value="{0}">{1}\n'.format(value_string, display_string)
        if len(lst) > 1 and type(lst[1]) is list:
            for item in lst[1]:
                out_string = _to_xml(item, out_string)
        out_string += "</node>\n"
        return out_string

    def _to_dot(lst, out_string=''):
        if len(lst) > 1 and type(lst[1]) is list:
            for item in lst[1]:
                if type(item) is list:
                    value_string = str(lst[0])
                    item_string = str(item[0])
                    label_string = len(eval(value_string.replace('|', ',')))
                    out_string += '''"{0}" -- "{1}";
                    "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];\n'''.format(
                        value_string, item_string, label_string)
                    out_string = _to_dot(item, out_string)
                else:
                    value_string = str(lst[0])
                    item_string = item
                    label_string = len(eval(value_string.replace('|', ',')))
                    out_string += '''"{0}" -- "{1}";
                    "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];
                    "{1}"[color="#cccccc",style="filled",shape="box"];\n'''.format(value_string, item_string,
                                                                                   label_string)
        return out_string

    if form == 'xml':
        openstring = 'w'
        export_list = obj.export()
        head = '<?xml version="1.0" encoding="utf-8"?>\n'
        output = _to_xml(export_list)
        output = head + output
        with open(out_file, "w") as out:
            out.write(output)
    elif form == 'dot':
        openstring = 'w'
        export_list = obj.export()
        head = 'graph {\n'
        output = _to_dot(export_list)
        tail = '}'
        output = head + output + tail
        with open(out_file, "w") as out:
            out.write(output)
    elif form == 'bin':
        import pickle
        openstring = 'wb'
        file_handle = open(out_file, openstring)
        pickle.dump(obj, file_handle, protocol=-1)
        file_handle.close()
    else:
        raise NotImplementedError()


def minimal_cuboid(cluster):
    """Build minimal cuboid

    Build minimal cuboid around cluster that is parallel to the axis in Cartesian coordinates

    Args:
        cluster: Cluster instance

    Returns:
        Minimal Cuboid
    """
    points = cluster.grid.points
    low_corner = numpy.array(points[0], float, ndmin=1)
    high_corner = numpy.array(points[0], float, ndmin=1)
    for p in points:
        p = numpy.array(p, float, ndmin=1)
        lower = p >= low_corner
        if not lower.all():
            for i in xrange(len(low_corner)):
                if not lower[i]:
                    low_corner[i] = p[i]
        higher = p <= high_corner
        if not higher.all():
            for i in xrange(len(high_corner)):
                if not higher[i]:
                    high_corner[i] = p[i]
    return Cuboid(low_corner, high_corner)
