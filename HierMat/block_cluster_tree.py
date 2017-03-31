"""block_cluster_tree.py: :class:`BlockClusterTree`,
:func:`build_block_cluster_tree`,
:func:`recursion_build_block_cluster_tree`
"""
from utils import admissible, divisor_generator


class BlockClusterTree(object):
    """Compares two cluster trees level wise with respect to an admissibility condition and builds a tree
    """
    def __init__(self, left_clustertree, right_clustertree, sons=None, level=0, is_admissible=False, plot_info=None):
        self.sons = sons if sons else []
        self.left_clustertree = left_clustertree
        self.right_clustertree = right_clustertree
        self.admissible = is_admissible
        self.level = level
        self.plot_info = plot_info

    def __repr__(self):
        optional_string = " with children {0!s}".format(self.sons) if self.sons else ""
        return "<BlockClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def __len__(self):
        left_len = len(self.left_clustertree)
        right_len = len(self.right_clustertree)
        return left_len * right_len

    def __eq__(self, other):
        """Test for equality
        :type other: BlockClusterTree
        """
        return (self.left_clustertree == other.left_clustertree
                and self.right_clustertree == other.right_clustertree
                and self.sons == other.sons
                and self.admissible == other.admissible
                and self.level == other.level
                and self.plot_info == other.plot_info
                )

    def __ne__(self, other):
        return not self == other

    def depth(self, root_level=None):
        """Get depth of the tree

        :param root_level: internal use.
        :type root_level: int
        :return: depth
        :rtype: int

        .. hint::

            **recursive function**
        """
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def to_list(self):
        """Give list representation for export

        Return a list containing the object and a list with its sons

        .. hint::

            **recursive function**
        """
        return [self, [son.to_list() for son in self.sons]]

    def shape(self):
        """Return length of left and right cluster tree

        :return: x and y dimension
        :rtype: tuple(int, int)
        """
        return len(self.left_clustertree), len(self.right_clustertree)

    def draw(self, axes, admissible_color='#1e26bc', inadmissible_color='#bc1d38'):
        """Draw a patch into given axes

        :param axes: axes instance to draw in
        :type axes: matplotlib.pyplot.axes
        :param admissible_color: color for admissible patch (see matplotlib for color specs)
        :type admissible_color: str
        :param inadmissible_color: color for inadmissible patch
        :type inadmissible_color: str
        """
        # set x coordinates for patch
        x_min, y_min = self.plot_info
        x_max = x_min + len(self.left_clustertree)
        y_max = y_min + len(self.right_clustertree)
        x = [x_min, x_min, x_max, x_max]
        # set y coordinates for patch
        y = [y_min, y_max, y_max, y_min]
        color = admissible_color if self.admissible else inadmissible_color
        axes.fill(x, y, color, ec='k', lw=0.1)

    def plot(self, filename=None, ticks=False, face_color='#133f52',
             admissible_color='#76f7a8', inadmissible_color='#ff234b'):
        """Plot the block cluster tree

        :param filename: filename to save the plot. if omitted, the plot will be displayed
        :type filename: str
        :param ticks: show ticks in the plot
        :type ticks: bool
        :param face_color: background color (see matplotlib for color specs)
        :param admissible_color: color for admissible patch
        :type admissible_color: str
        :param inadmissible_color: color for inadmissible patch
        :type inadmissible_color: str

        .. note::

            depends on :mod:`matplotlib.pyplot`

        """
        import matplotlib.pyplot as plt

        plt.rc('axes', linewidth=0.5, labelsize=4)
        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)
        fig = plt.figure(figsize=(3, 3), dpi=400)
        fig.patch.set_facecolor(face_color)
        # get max of the ticks
        x_min, x_max = self.left_clustertree.get_patch_coordinates()
        y_min, y_max = self.right_clustertree.get_patch_coordinates()
        axes = plt.axes()
        axes.set_xlim(x_min, x_max + 1)
        axes.set_ylim(y_min, y_max + 1)
        if ticks:
            x_divisors = list(divisor_generator(x_max + 1))
            y_divisors = list(divisor_generator(y_max + 1))
            if len(x_divisors) > 4:
                x_ticks = x_divisors[-4]
            else:
                x_ticks = x_divisors[-1]
            if len(y_divisors) > 4:
                y_ticks = y_divisors[-4]
            else:
                y_ticks = y_divisors[-1]
            axes.set_xticks(range(x_min, x_max + 2, x_ticks))
            axes.set_yticks(range(y_min, y_max + 2, y_ticks))
        else:
            axes.set_xticks([])
            axes.set_yticks([])
        axes.tick_params(length=2, width=0.5)
        axes.xaxis.tick_top()
        axes.invert_yaxis()
        self.plot_recursion(axes, admissible_color=admissible_color, inadmissible_color=inadmissible_color)
        fig.add_axes(axes)
        if not filename:
            return fig
        else:
            # remove whitespace around the plot
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(filename, format='png', facecolor=fig.get_facecolor(), edgecolor=None)

    def plot_recursion(self, axes, admissible_color='#1e26bc', inadmissible_color='#bc1d38'):
        if self.sons:
            for son in self.sons:
                son.plot_recursion(axes, admissible_color=admissible_color, inadmissible_color=inadmissible_color)
        else:
            self.draw(axes, admissible_color=admissible_color, inadmissible_color=inadmissible_color)

    def export(self, form='xml', out_file='bct_out'):
        """Export obj in specified format.

        :param form: format specifier
        :type form: str
        :param out_file: path to output file
        :type out_file: str
        :raises NotImplementedError: if form is not supported

        .. note::

            implemented so far:

            - xml
            - dot
            - bin
        """
        if form == 'xml':
            export_list = self.to_list()
            head = '<?xml version="1.0" encoding="utf-8"?>\n'
            output = self._to_xml(export_list)
            output = head + output
            with open(out_file, "w") as out:
                out.write(output)
        elif form == 'dot':
            export_list = self.to_list()
            head = 'graph {\n'
            output = self._to_dot(export_list)
            tail = '}'
            output = head + output + tail
            with open(out_file, "w") as out:
                out.write(output)
        elif form == 'bin':
            import pickle
            file_handle = open(out_file, "wb")
            pickle.dump(self, file_handle, protocol=-1)
            file_handle.close()
        else:
            raise NotImplementedError()

    @staticmethod
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
                out_string = BlockClusterTree._to_xml(item, out_string)
        out_string += "</node>\n"
        return out_string

    @staticmethod
    def _to_dot(lst, out_string=''):
        if len(lst) > 1:
            for item in lst[1]:
                if type(item) is list:
                    value_string = str(lst[0])
                    item_string = str(item[0])
                    label_string = len(lst[0])
                    out_string += '''"{0}" -- "{1}";
                            "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];\n'''.format(
                        value_string, item_string, label_string)
                    out_string = BlockClusterTree._to_dot(item, out_string)
        return out_string


def build_block_cluster_tree(left_cluster_tree, right_cluster_tree=None, start_level=0, admissible_function=admissible):
    """Factory for building a block cluster tree

    :param left_cluster_tree: "left side" cluster tree
    :type left_cluster_tree: ClusterTree
    :param right_cluster_tree: "right side" cluster tree
    :type right_cluster_tree: ClusterTree
    :param start_level: counter that keeps track of the level
    :type start_level: int
    :param admissible_function: function that determines whether two cluster trees are admissible or not. This is
        crucial for the structure of the block cluster tree. (See :mod:`utils.admissible` for an example)
    :type admissible_function: function that returns a bool
    :return: block cluster tree
    :rtype: BlockClusterTree
    """
    if not right_cluster_tree:
        right_cluster_tree = left_cluster_tree
    is_admissible = admissible_function(left_cluster_tree, right_cluster_tree)
    x_min, x_max = left_cluster_tree.get_patch_coordinates()
    y_min, y_max = right_cluster_tree.get_patch_coordinates()
    plot_info = [x_min, y_min]
    root = BlockClusterTree(left_cluster_tree, right_cluster_tree,
                            level=start_level, is_admissible=is_admissible, plot_info=plot_info)
    recursion_build_block_cluster_tree(root, admissible_function)
    return root


def recursion_build_block_cluster_tree(current_tree, admissible_function):
    """Recursion to :func:`build_block_cluster_tree`
    """
    if not admissible_function(current_tree.left_clustertree, current_tree.right_clustertree):
        # get top left corner of current block
        x_min, y_min = current_tree.plot_info
        x_current = x_min
        y_current = y_min
        for left_son in current_tree.left_clustertree.sons:
            for right_son in current_tree.right_clustertree.sons:
                new_tree = BlockClusterTree(left_son, right_son,
                                            level=current_tree.level + 1,
                                            is_admissible=False,
                                            plot_info=[x_current, y_current]
                                            )
                current_tree.sons.append(new_tree)
                recursion_build_block_cluster_tree(new_tree, admissible_function)
                y_current += len(right_son)
            y_current = y_min
            x_current += len(left_son)
    else:
        current_tree.admissible = True
