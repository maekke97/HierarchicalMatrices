from utils import admissible, divisor_generator


class BlockClusterTree(object):
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
        """Test for equality"""
        return (self.left_clustertree == other.left_clustertree
                and self.right_clustertree == other.right_clustertree
                and self.sons == other.sons
                # and self.admissible == other.admissible
                and self.level == other.level
                )

    def depth(self, root_level=None):
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def to_list(self):
        return [self, [son.to_list() for son in self.sons]]

    def draw(self, axes, admissible_color='#1e26bc', inadmissible_color='#bc1d38'):
        # set x coordinates for patch
        ## x_min, x_max = self.left_clustertree.get_patch_coordinates()
        x_min, y_min = self.plot_info
        x_max = x_min + len(self.right_clustertree)
        y_max = y_min + len(self.left_clustertree)
        x = [x_min, x_min, x_max, x_max]
        # set y coordinates for patch
        ## y_min, y_max = self.right_clustertree.get_patch_coordinates()
        y = [y_min, y_max, y_max, y_min]
        color = admissible_color if self.admissible else inadmissible_color
        axes.fill(x, y, color, ec='k', lw=0.1)

    def plot(self, filename=None, face_color='#96acd1', admissible_color='#1e26bc', inadmissible_color='#bc1d38'):
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
        axes.tick_params(length=2, width=0.5)
        axes.xaxis.tick_top()
        axes.invert_yaxis()
        self._plot(axes)
        fig.add_axes(axes)
        if not filename:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename, format='png', facecolor=fig.get_facecolor(), edgecolor=None)

    def _plot(self, axes, admissible_color='#1e26bc', inadmissible_color='#bc1d38'):
        if self.sons:
            for son in self.sons:
                son._plot(axes, admissible_color=admissible_color, inadmissible_color=inadmissible_color)
        else:
            self.draw(axes, admissible_color=admissible_color, inadmissible_color=inadmissible_color)

    def export(self, form='xml', out_file='bct_out'):
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
                        label_string = len(lst[0])
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
                                "{1}"[color="#cccccc",style="filled",shape="box"];\n'''.format(value_string,
                                                                                               item_string,
                                                                                               label_string)
            return out_string

        if form == 'xml':
            export_list = self.to_list()
            head = '<?xml version="1.0" encoding="utf-8"?>\n'
            output = _to_xml(export_list)
            output = head + output
            with open(out_file, "w") as out:
                out.write(output)
        elif form == 'dot':
            export_list = self.to_list()
            head = 'graph {\n'
            output = _to_dot(export_list)
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


def build_block_cluster_tree(left_cluster_tree, right_cluster_tree=None, start_level=0, admissible_function=admissible):
    if not right_cluster_tree:
        right_cluster_tree = left_cluster_tree
    is_admissible = admissible_function(left_cluster_tree, right_cluster_tree)
    x_min, x_max = left_cluster_tree.get_patch_coordinates()
    y_min, y_max = left_cluster_tree.get_patch_coordinates()
    plot_info = [x_min, y_min]
    root = BlockClusterTree(left_cluster_tree, right_cluster_tree,
                            level=start_level, is_admissible=is_admissible, plot_info=plot_info)
    recursion_build_block_cluster_tree(root, admissible_function)
    return root


def recursion_build_block_cluster_tree(current_tree, admissible_function):
    if not admissible_function(current_tree.left_clustertree, current_tree.right_clustertree):
        # check for sons on both sides
        # if not current_tree.left_clustertree.sons and current_tree.right_clustertree.sons:
        #     # only right cluster has sons
        #     left_cluster_sons = [current_tree.left_clustertree]
        #     right_cluster_sons = current_tree.right_clustertree.sons
        # elif current_tree.left_clustertree.sons and not current_tree.right_clustertree.sons:
        #     # only left cluster has sons
        #     left_cluster_sons = current_tree.left_clustertree.sons
        #     right_cluster_sons = [current_tree.right_clustertree]
        # elif current_tree.left_clustertree.sons and current_tree.right_clustertree.sons:
        #     # both have sons
        #     left_cluster_sons = current_tree.left_clustertree.sons
        #     right_cluster_sons = current_tree.right_clustertree.sons
        # else:
        #     # no sons on both sides, so stop recursion
        #     return None
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
                x_current += len(right_son)
            x_current = x_min
            y_current += len(left_son)
    else:
        current_tree.admissible = True
