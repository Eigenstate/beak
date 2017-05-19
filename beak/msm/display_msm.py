#!/usr/bin/env python
"""
Standalone threaded script that displays a MSM.
Pipe stdout from here to have VMD update
"""

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import colors

from msmbuilder.tpt import net_fluxes, top_path
from beak.msm import utils
import networkx as nx
import numpy as np
import sys

class Grapher(object):

    def __init__(self, msmfile, scores=None, **kwargs):
        # Construct graph
        plt.ion()
        self.msm = utils.load(msmfile)
        #self.scores = utils.load(scores) if scores is not None else None
        self.solvent = np.argmax(self.msm.populations_)
        self.labels = list(self.msm.mapping_)
        self.nodes = {}

        # Show graph
        self.fig = plt.figure()
        self.ax = plt.gca()
        self.show_msm()

        while  True :
            plt.pause(0.10)

    #==========================================================================

    def graph_callback(self, event):
        """
        Graph callback function for matplotlib integration
        """
        artist = event.artist
        if artist.get_edgecolor() == colors.to_rgba("black"):
            artist.set_linewidth(5)
            self.color_neighbors(artist.get_label(), reset=False)
            artist.set_edgecolor("red")
            print(artist.get_label())
            sys.stdout.flush()
        else:
            artist.set_edgecolor("black")
            artist.set_linewidth(1)
            self.color_neighbors(artist.get_label(), reset=True)
            print(artist.get_label())
            sys.stdout.flush()

    #==========================================================================

    def get_color(self, num, indices):
        colors = plt.cm.RdYlGn(indices)

        i=0
        while i < len(indices)-1:
            if indices[i] <= num and \
               indices[i+1] > num: break
            i += 1
        return colors[i]

    #==========================================================================

    def color_neighbors(self, label, reset=False):
        """
        Colors all neighbors in the graph according to their
        transition probability. Makes it easy to pick things.
        """
        label = int(label)
        idx = self.msm.transform([label])[0][0]

        flux = net_fluxes(sources=[self.solvent], sinks=[idx], msm=self.msm)
        path = list(top_path(sources=[self.solvent], sinks=[idx], net_flux=flux)[0])
        colors = plt.cm.winter(np.linspace(0, len(path), len(path)))

        for i, p in enumerate(path):
            if p == self.solvent: continue
            node = self.nodes[self.labels[p]]
            if node.get_linewidth() == 5:
                node.set_edgecolor("green")
            if reset:
                node.set_facecolor("grey")
            else:
                node.set_facecolor(colors[i])
        return

        for i in range(len(self.msm.transmat_)):
            freq = self.msm.transmat_[idx][i]
            node = self.nodes[self.labels[i]]
            if node.get_linewidth() == 5:
                node.set_edgecolor("pink")
            if reset:
                node.set_facecolor("grey")
            else:
                node.set_facecolor(self.get_color(freq, indices))

    #==========================================================================

    def show_msm(self, nodelist=None):
        """
        Visualises the MSM using a msmexplorer-like visualization.
        This is a combination of networkx and msmexplorer to have pickable
        nodes that tell you what they are when clicked.

        Args:
            nodelist (list of int): Nodes to visualize, or None for all
        """

        graph = nx.Graph(self.msm.transmat_)
        node_size = np.abs(1./np.log([i/sum(self.msm.populations_) \
                                   for i in self.msm.populations_]))
        # Decrease size of solvent cluster
        #node_size[self.solvent] = sorted(node_size)[-2]

        #pos = nx.fruchterman_reingold_layout(graph, scale=np.sum(node_size), iterations=0, k=5.)
        #pos = nx.spring_layout(graph,scale=np.sum(node_size),iterations=100,
                #pos={x:(0.,0.,0.) for x in range(len(self.msm.transmat_))}, fixed=[solvent])
        pos = nx.circular_layout(graph, scale=np.sum(node_size))

        if nodelist is None:
            nodelist = graph.nodes()
        xy = np.asarray([pos[v] for v in nodelist])

        for i in range(len(nodelist)):
            if i == self.solvent:
                continue
            n = plt.Circle(xy[i],
                           radius=node_size[i],
                           label=self.labels[i],
                           edgecolor="black",
                           facecolor="grey",
                           zorder=2,
                           picker=5.)
            self.ax.text(xy[i][0], xy[i][1], self.labels[i],
                         label=self.labels[i],
                         fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.5))
            self.nodes[self.labels[i]] = n
            self.ax.add_artist(n)

        edgelist = [(n1, n2) for n1 in graph.nodes() for n2 in graph.nodes()
                    if n1 != self.solvent and n2 != self.solvent]
        widths = [25.*self.msm.transmat_[n1][n2] for n1 in graph.nodes()
                  for n2 in graph.nodes() if n1 != self.solvent and n2 != self.solvent]
        nx.draw_networkx_edges(graph, pos,
                               arrows=True,
                               width=widths,
                               ax=self.ax,
                               edgelist=edgelist)
        self.fig.canvas.mpl_connect("pick_event", self.graph_callback)

#==============================================================================

if __name__ == "__main__":
    msm = sys.argv[1]
    scores = None
    if len(sys.argv) > 2:
        scores = sys.argv[2]
    g = Grapher(sys.argv[1], scores=scores)

#==============================================================================
