"""
Contains methods for generally visualizing MSMs that are
loaded into DensitySamplers, usually
"""
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import msmexplorer as msme
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
try:
    from vmd import molecule, molrep, vmdnumpy, display, atomsel
    atomsel = atomsel.atomsel
except:
    import vmd
    import molecule
    import molrep
    import vmdnumpy
    import display
    from atomsel import atomsel

#==============================================================================

def add_frames(samp, cluster, num=10):
    """
    Adds representative frames of the given molecule to the visualizer
    session.

    Args:
        samp (DensitySampler): Sampler to add frames to
        cluster (int): Cluster label to visualize
        num (int): Number of frames to load

    """
    for _ in range(num):
        samp._load_frame(cluster)

#==============================================================================

def show_msm(samp, nodelist=None):
    """
    Visualises the MSM using a msmexplorer-like visualization.
    This is a combination of networkx and msmexplorer to have pickable
    nodes that tell you what they are when clicked.

    Args:
        samp (DensitySampler): Sampler to visualize
        nodelist (list of int): Nodes to visualize, or None for all
    """

    # Hide all clusters
    for c in samp.molids.keys():
        samp.show_cluster(c, shown=False)

    msm = samp.msm
    graph = nx.Graph(msm.transmat_)
    node_size = 0.08 / np.log([i/sum(msm.populations_) for i in msm.populations_])
    pos = nx.shell_layout(graph)

    plt.ion()
    fig = plt.figure()
    ax = plt.gca()

    if nodelist is None:
        nodelist = graph.nodes()
    xy = np.asarray([pos[v] for v in nodelist])
    labels = msm.inverse_transform(list(msm.mapping_))[0]

    drawn = []
    for i in range(len(nodelist)):
        ax.add_artist(plt.Circle(xy[i],
                                 radius=node_size[i],
                                 label=labels[i],
                                 edgecolor="black",
                                 facecolor="blue",
                                 zorder=2,
                                 picker=5.))

    edgelist = [(n1, n2) for n1 in graph.nodes() for n2 in graph.nodes()]
    widths = [10.*msm.transmat_[n1][n2] for n1 in graph.nodes() for n2 in graph.nodes()]
    nx.draw_networkx_edges(graph, pos,
                           arrows=True,
                           width=widths,
                           ax=ax,
                           edgelist=edgelist)


    #def show_graph():
    #    while (1):
    #        plt.pause(0.05)
        #plt.draw()


    #fig.canvas.mpl_connect("pick_event", pick_handler)

    fig.canvas.mpl_connect("pick_event", samp._graph_callback)
    plt.show()
    #p = mp.Process(target=show_graph)

    #fig.canvas.mpl_connect("key_press_event", close_handler)
   # p.start()

    print("returning")


#==============================================================================

