class ClusterTree(object):
    """
    """

    def __init__(self, splitable, max_leaf_size=1, level=0):
        """"""
        self.level = level
        self.splitable = splitable
        self.sons = []
        if len(splitable) > max_leaf_size:
            splits = splitable.split()
            for split in splits:
                self.sons.append(ClusterTree(split, max_leaf_size, self.level + 1))

    def __repr__(self):
        optional_string = " with children {0}".format(self.sons) if self.sons else ""
        return "<ClusterTree at level {0}{1}>".format(str(self.level), optional_string)

    def __len__(self):
        return len(self.splitable)

    def __str__(self):
        """give str representation of self."""
        return ",".join([str(p) for p in self.splitable])

    def __eq__(self, other):
        """Test for equality"""
        return self.level == other.level and self.splitable == other.splitable and self.sons == other.sons

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
                        "{1}"[color="#cccccc",style="filled",shape="box"];\n'''.format(value_string, item_string,
                                                                                       label_string)
            return out_string

        if form == 'xml':
            openstring = 'w'
            export_list = self.to_list()
            head = '<?xml version="1.0" encoding="utf-8"?>\n'
            output = _to_xml(export_list)
            output = head + output
            with open(out_file, "w") as out:
                out.write(output)
        elif form == 'dot':
            openstring = 'w'
            export_list = self.to_list()
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
        return self.splitable.diameter()

    def distance(self, other):
        return self.splitable.distance(other.splitable)
