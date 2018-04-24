# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:26:10 2017

@author: Raluca Sandu
"""

"""graphing.py -- helper functions for graphing with matplotlib/pyplot
This software is licensed under the terms of the MIT License as
follows:
Copyright (c) 2013 Jessica B. Hamrick

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def save(path, fignum=None, close=True, width=None, height=None,
         ext=None, verbose=False):
    """Save a figure from pyplot.
    Parameters:
    path [string] : The path (and filename, without the extension) to
    save the figure to.
    fignum [integer] : The id of the figure to save. If None, saves the
    current figure.
    close [boolean] : (default=True) Whether to close the figure after
    saving.  If you want to save the figure multiple times (e.g., to
    multiple formats), you should NOT close it in between saves or you
    will have to re-plot it.
    width [number] : The width that the figure should be saved with. If
    None, the current width is used.
    height [number] : The height that the figure should be saved with. If
    None, the current height is used.
    ext [string or list of strings] : (default='png') The file
    extension. This must be supported by the active matplotlib backend
    (see matplotlib.backends module).  Most backends support 'png',
    'pdf', 'ps', 'eps', and 'svg'.
    verbose [boolean] : (default=True) Whether to print information
    about when and where the image has been saved.
    """

    # get the figure
    if fignum is not None:
        fig = plt.figure(fignum)
    else:
        fig = plt.gcf()

    # set its dimenions
    if width:
        fig.set_figwidth(width)
    if height:
        fig.set_figheight(height)

    # make sure we have a list of extensions
    if ext is not None and not hasattr(ext, '__iter__'):
        ext = [ext]

    # Extract the directory and filename from the given path
    directory, basename = os.path.split(path)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # infer the extension if ext is None
    if ext is None:
        basename, ex = os.path.splitext(basename)
        ext = [ex[1:]]

    for ex in ext:
        # The final path to save to
        filename = "%s.%s" % (basename, ex)
        savepath = os.path.join(directory, filename)

        if verbose:
            sys.stdout.write("Saving figure to '%s'..." % savepath)

        # Actually save the figure
#        plt.savefig(savepath)
        plt.savefig(savepath,pad_inches=0)

    # Close it
    if close:
        plt.close()

    if verbose:
        sys.stdout.write("Done\n")


def plot_to_array(fig=None):
    """Convert a matplotlib figure `fig` to RGB pixels.
    Note that this DOES include things like ticks, labels, title, etc.
    Parameters
    ----------
    fig : int or matplotlib.figure.Figure (default=None)
        A matplotlib figure or figure identifier. If None, the current
        figure is used.
    Returns
    -------
    out : numpy.ndarray
        (width, height, 4) array of RGB values
    """

    # get the figure object
    if fig is None:
        fig = plt.gcf()
    try:
        fig = int(fig)
    except TypeError:
        pass
    else:
        fig = plt.figure(fig)

    # render the figure
    fig.canvas.draw()

    # convert the figure to a rgb string, and read that buffer into a
    # numpy array
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.renderer.tostring_rgb()
    arr = np.fromstring(buf, dtype=np.uint8)
    arr.resize(h, w, 3)

    # convert values from 0-255 to 0-1
    farr = arr / 255.

    return farr


def clear_top(ax=None):
    """Remove the top edge of the axis bounding box.
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')


def clear_bottom(ax=None):
    """Remove the bottom edge of the axis bounding box.
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks_position('top')


def clear_top_bottom(ax=None):
    """Remove the top and bottom edges of the axis bounding box.
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks([])


def clear_left(ax=None):
    """Remove the left edge of the axis bounding box.
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.yaxis.set_ticks_position('right')


def clear_right(ax=None):
    """Remove the right edge of the axis bounding box.
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')


def clear_left_right(ax=None):
    """Remove the left and right edges of the axis bounding box.
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks([])


def outward_ticks(ax=None, axis='both'):
    """Make axis ticks face outwards rather than inwards (which is the
    default).
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    axis : string (default='both')
        The axis (either 'x', 'y', or 'both') for which to set the tick
        direction.
    """

    if ax is None:
        ax = plt.gca()
    if axis == 'both':
        ax.tick_params(direction='out')
    else:
        ax.tick_params(axis=axis, direction='out')


def set_xlabel_coords(y, x=0.5, ax=None):
    """Set the y-coordinate (and optionally the x-coordinate) of the x-axis
    label.
    Parameters
    ----------
    y : float
        y-coordinate for the label
    x : float (default=0.5)
        x-coordinate for the label
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots
    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_label_coords(x, y)


def set_ylabel_coords(x, y=0.5, ax=None):
    """Set the x-coordinate (and optionally the y-coordinate) of the y-axis
    label.
    Parameters
    ----------
    x : float
        x-coordinate for the label
    y : float (default=0.5)
        y-coordinate for the label
    ax : axis object (default=pyplot.gca())
    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots
    """
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_label_coords(x, y)


def align_ylabels(xcoord, *axes):
    """Align the y-axis labels of multiple axes.
    Parameters
    ----------
    xcoord : float
        x-coordinate of the y-axis labels
    *axes : axis objects
        The matplotlib axis objects to format
    """
    for ax in axes:
        set_ylabel_coords(xcoord, ax=ax)


def align_xlabels(ycoord, *axes):
    """Align the x-axis labels of multiple axes
    Parameters
    ----------
    ycoord : float
        y-coordinate of the x-axis labels
    *axes : axis objects
        The matplotlib axis objects to format
    """
    for ax in axes:
        set_xlabel_coords(ycoord, ax=ax)


def no_xticklabels(ax=None):
    """Remove the tick labels on the x-axis (but leave the tick marks).
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xticklabels([])


def no_yticklabels(ax=None):
    """Remove the tick labels on the y-axis (but leave the tick marks).
    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    """
    if ax is None:
        ax = plt.gca()
    ax.set_yticklabels([])


def set_figsize(width, height, fig=None):
    """Set the figure width and height.
    Parameters
    ----------
    width : float
        Figure width
    height : float
        Figure height
    fig : figure object (default=pyplot.gcf())
    """

    if fig is None:
        fig = plt.gcf()
    fig.set_figwidth(width)
    fig.set_figheight(height)


def set_scientific(low, high, axis=None, ax=None):
    """Set the axes or axis specified by `axis` to use scientific notation for
    ticklabels, if the value is <10**low or >10**high.
    Parameters
    ----------
    low : int
        Lower exponent bound for non-scientific notation
    high : int
        Upper exponent bound for non-scientific notation
    axis : str (default=None)
        Which axis to format ('x', 'y', or None for both)
    ax : axis object (default=pyplot.gca())
        The matplotlib axis object to use
    """
    # get the axis
    if ax is None:
        ax = plt.gca()
    # create the tick label formatter
    fmt = plt.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((low, high))
    # format the axis/axes
    if axis is None or axis == 'x':
        ax.get_yaxis().set_major_formatter(fmt)
    if axis is None or axis == 'y':
        ax.get_yaxis().set_major_formatter(fmt)


def sync_ylims(*axes):
    """Synchronize the y-axis data limits for multiple axes. Uses the maximum
    upper limit and minimum lower limit across all given axes.
    Parameters
    ----------
    *axes : axis objects
        List of matplotlib axis objects to format
    Returns
    -------
    out : ymin, ymax
        The computed bounds
    """
    ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes])
    ymin = min(ymins)
    ymax = max(ymaxs)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    return ymin, ymax


def sync_xlims(*axes):
    """Synchronize the x-axis data limits for multiple axes. Uses the maximum
    upper limit and minimum lower limit across all given axes.
    Parameters
    ----------
    *axes : axis objects
        List of matplotlib axis objects to format
    Returns
    -------
    out : yxin, xmax
        The computed bounds
    """
    xmins, xmaxs = zip(*[ax.get_xlim() for ax in axes])
    xmin = min(xmins)
    xmax = max(xmaxs)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        return xmin, xmax