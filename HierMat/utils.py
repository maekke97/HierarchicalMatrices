"""utils.py: Utilities for the :mod:`HMatrix` module
"""
import math

from HierMat.block_cluster_tree import BlockClusterTree
from HierMat.cluster_tree import ClusterTree
from HierMat.grid import Grid


def export(obj, form='xml', out_file='out'):
    """Export obj in specified format.
    
    :param obj: object to export
    :type obj: BlockClusterTree or ClusterTree or HMat or RMat
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
        head = '<?xml version="1.0" encoding="utf-8"?>\n'
        output = obj.to_xml()
        output = head + output
        with open(out_file, "w") as out:
            out.write(output)
    elif form == 'dot':
        head = 'graph {\nnodesep=0.1;\nranksep=1.5;\n'
        output = obj.to_dot()
        tail = '}'
        output = head + output + tail
        with open(out_file, "w") as out:
            out.write(output)
    elif form == 'bin':
        import pickle
        file_handle = open(out_file, "wb")
        pickle.dump(obj, file_handle, protocol=-1)
        file_handle.close()
    else:
        raise NotImplementedError()


def plot(obj, filename=None, **kwargs):
    """plot an object
    
    :param obj: object to plot
    :type obj: BlockClusterTree or Grid
    :param filename: filename to save the plot to (if omitted, the plot will be displayed)
    :type filename: str
    :param kwargs: optional arguments to specific plot commands
        see the respective documentations
    """
    if isinstance(obj, BlockClusterTree):
        return block_cluster_tree_plot(obj, filename, **kwargs)
    else:
        raise NotImplementedError('object can not be plotted')


def block_cluster_tree_plot(obj, filename=None, ticks=False, face_color='#133f52',
                            admissible_color='#76f7a8', inadmissible_color='#ff234b'):
    """Plot the block cluster tree
    
    :param obj: block cluster tree to plot
    :type obj: BlockClusterTree
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
    x_min, x_max = obj.left_clustertree.get_patch_coordinates()
    y_min, y_max = obj.right_clustertree.get_patch_coordinates()
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
    obj.plot_recursion(axes, admissible_color=admissible_color, inadmissible_color=inadmissible_color)
    fig.add_axes(axes)
    if not filename:
        return fig
    else:
        # remove whitespace around the plot
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(filename, format='png', facecolor=fig.get_facecolor(), edgecolor=None)


def load(filename):
    """Load a :class:`ClusterTree` or :class:`BlockClusterTree` from file

    :param filename: file to import
    :type filename: String
    :return: object
    :rtype: BlockClusterTree or ClusterTree

    .. note:: Depends on :mod:`pickle`
    """
    import pickle
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj


def divisor_generator(n):
    """Return divisors of n

    :param n: integer to find divisors of
    :type n: int
    :return: divisors
    :rtype: list[int]

    .. warning::
       This is a generator! To get a list with all divisors call::

          list(divisor_generator(n))


    .. note::
       found at
       `StackOverflow
       <http://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number>`_
       on 2017.03.08
    """
    large_divisors = []
    for i in xrange(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor
