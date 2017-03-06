class ClusterTree(object):
    """
    """

    def __init__(self, content, sons=None, max_leaf_size=1, level=0):
        """"""
        self.level = level
        self.content = content
        self.sons = sons if sons else []
        self.max_leaf_size = max_leaf_size

    def __repr__(self):
        optional_string = " with children {0}".format(self.sons) if self.sons else " without children"
        return "<ClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def __len__(self):
        return len(self.content)

    def __str__(self):
        """give str representation of self."""
        cont_str = ",".join([str(p) for p in self.content])
        out_str = "ClusterTree at level {0} with content:\n{1}".format(self.level, cont_str)
        return out_str

    def __eq__(self, other):
        """Test for equality"""
        return self.level == other.level and self.content == other.content and self.sons == other.sons

    def __getitem__(self, item):
        return self.content[item]

    def get_index(self, item):
        return self.content.get_index(item)

    def get_grid_item(self, item):
        return self.content.get_grid_item(item)

    def to_list(self):
        """Give list representation for export"""
        if self.sons:
            return [self, [son.to_list() for son in self.sons]]
        else:
            return [self, []]

    def export(self, form='xml', out_file='out'):
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
                    value_string = str(lst[0])
                    item_string = str(item[0])
                    label_string = len(lst[0])
                    out_string += '''"{0}" -- "{1}";
                    "{0}"[label="{2}",color="#cccccc",style="filled",shape="box"];\n'''.format(
                        value_string, item_string, label_string)
                    out_string = _to_dot(item, out_string)
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

    def depth(self, root_level=None):
        if root_level is None:
            root_level = self.level
        if not self.sons:
            return self.level - root_level
        else:
            return max([son.depth(root_level) for son in self.sons])

    def diameter(self):
        return self.content.diameter()

    def distance(self, other):
        return self.content.distance(other.content)


def build_cluster_tree(splitable, max_leaf_size=1, start_level=0):
    root = ClusterTree(splitable, max_leaf_size=max_leaf_size, level=start_level)
    recursion_build_cluster_tree(root)
    return root


def recursion_build_cluster_tree(current_tree):
    if len(current_tree.content) > current_tree.max_leaf_size:
        splits = current_tree.content.split()
        for split in splits:
            new_tree = ClusterTree(content=split,
                                   max_leaf_size=current_tree.max_leaf_size,
                                   level=current_tree.level + 1)
            current_tree.sons.append(new_tree)
            recursion_build_cluster_tree(new_tree)
