"""
Contains methods for making various kinds of pretty graphs
with matplotlib. Plotly graphs are usually in their own
jupyter notebook.
"""
from scipy.signal import savgol_filter
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#===============================================================================

def plot_rmsd_adaptive(rmsds, maxgen, genlen, btimes=None,
                       colfg=None, colbg=None, cutoff=7):
    """
    Plots an adaptive graph, all pretty

    Args:
        rmsds (Pandas dataframe): RMSD to bound state, index in ps
        maxgen (int): Maximum generation to plot
        genlen (int): Nanoseconds per generation
        btimes (list): Times at which a binding event occurs, in ps. Will
            be shown as an arrow.
        colfg (3 tuple): Foreground graph color
        colbg (3 tuple): Background graph color
        cutoff (float): Maximum Angstroms to show

    Returns:
        (matplotlib figure): The graph
    """

    data = rmsds[rmsds < (cutoff+3.)].dropna(how="all")
    xmax = genlen*maxgen

    fig = plt.figure(figsize=(20, 5), dpi=300)
    ax = fig.gca()
    for gen in range(1, maxgen):
        for k in data[[_ for _ in data if "G%d_" % gen in _]]:
            if not len(data[k]): continue
            ax.plot(data[k].index/1000.,
                    savgol_filter(data[k], window_length=29,
                                  polyorder=5, mode="constant"),
                    alpha=1, c=colfg, zorder=1)
            ax.plot(data[k].index/1000., data[k],
                    alpha=0.6, c=colbg, zorder=0)

    # Show binding times
    if btimes is not None:
        for b in btimes:
            ax.arrow(x=b/1000., y=cutoff, dx=0, dy=-1, color="black", width=1.5,
                     length_includes_head=True, head_length=0.25, zorder=2)

    # Show vertical lines each resampling event, but just a tic at the top
    ax.grid(which="minor", axis="x")
    ax.set_xticks(np.arange(1, xmax, genlen), minor=True)

    ax.set_xlim([0, xmax])
    ax.set_ylim([0, cutoff])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD to bound pose (A)")

    return fig

#===============================================================================

def plot_rmsd_traditional(rmsds, maxtime, btimes=None,
                          colfg=None, colbg=None, cutoff=7):
    """
    Plots a traditional MD graph, all pretty

    Args:
        rmsds (Pandas dataframe): RMSD to bound state, index in ps
        maxtime (int): Maximum time to plot, in ns
        btimes (list): Times at which a binding event occurs, in ps. Will
            be shown as an arrow.
        colfg (3 tuple): Foreground graph color
        colbg (3 tuple): Background graph color
        cutoff (float): Maximum angstroms to show on y axis

    Returns:
        (matplotlib Figure): the graph
    """
    data = rmsds[rmsds < (cutoff+3.)].dropna(how="all")

    fig = plt.figure(figsize=(20, 5), dpi=300)
    ax = fig.gca()
    for k in data:
        if not len(data[k]): continue
        ax.plot(data[k].index/1000.,
                savgol_filter(data[k], window_length=29,
                              polyorder=5, mode="constant"),
                alpha=1, c=colfg, zorder=1)
        ax.plot(data[k].index/1000., data[k],
                alpha=0.6, c=colbg, zorder=0)

    if btimes is not None:
        for b in btimes:
            ax.arrow(x=b/1000., y=cutoff, dx=0, dy=-1, color="black", width=1.5,
                     length_includes_head=True, head_length=0.25, zorder=2)

    ax.set_xlim([0, maxtime])
    ax.set_ylim([0, cutoff])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD to bound pose (A)")

    return fig

#===============================================================================
