#!/usr/bin/env python
"""
Visualizers for convergence, interactively
"""
import os
import sys
import numpy as np
import matplotlib
import pickle
import time

from beak.msm import comparator
from configparser import ConfigParser
from subprocess import PIPE, Popen
from threading import Thread
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import colors
from vmd import display, molecule, molrep, trans

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class GraphProcess(object):
    """
    A graph guy i guess
    """

    def __init__(self, data):
        self.data = data
        self.traces = []

        # Construct graph
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.gca()

        # Get minimum and max generations
        self.mingen = min(p.generation for t in self.data for p in t)
        self.maxgen = max(p.generation for t in self.data for p in t)
        self.ax.set_xlim([self.mingen, self.maxgen])
        self.ax.set_ylim([0, 0.010])

        # Show graph and loop forever
        self.show_graph()

        while True:
            plt.pause(0.10)

    #===========================================================================

    def graph_callback(self, event):
        """
        Graph callback function for matplotlib integration
        """
        artist = event.artist
        if artist.get_facecolor() == colors.to_rgba("grey"):
            artist.set_facecolor("red")
        else:
            artist.set_facecolor("grey")

        print(artist.get_label())

    #===========================================================================

    def show_graph(self):
        """
        Actually shows the graph
        """
        colormap = plt.cm.winter(np.linspace(0, 10.0, 100.0))
        width = 0.2
        height = 0.01*(width/(self.maxgen-self.mingen-width))

        for trace in data:
            self.ax.plot([x.generation for x in trace],
                         [y.population for y in trace],
                         "-",
                         color="black"
                        )

            # Have to do points individually for callback
            for point in trace:
                n = Ellipse([point.generation, point.population],
                            width=width,
                            height=height,
                               label=point.get_label(),
                               edgecolor="black",
                               facecolor="grey",
                               #radius=0.05,
                               #facecolor=colormap[np.mean(point.rmsd_to_bound],
                               zorder=2,
                               picker=2.)
                self.ax.add_artist(n)
        self.fig.canvas.mpl_connect("pick_event", self.graph_callback)
        plt.show()

    #===========================================================================

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ConvergencePlotter(object):
    """
    Object that handles all visualization, keeps track of loaded
    molecules etc.
    """

    def __init__(self, configfile, mingen, maxgen):
        self.config = ConfigParser(interpolation=None)
        self.config.read(configfile)

        self.rootdir = self.config.get("system", "rootdir")
        self.reference = self.config.get("system", "reference")
        os.chdir(self.rootdir)

        self.plotdata = comparator.get_cluster_progress(configfile,
                                                        mingen,
                                                        maxgen)
        self.molids = {}

    #===========================================================================

    def show_cluster(self, generation, cluster, visstate=None):
        """
        Shows or hides a cluster. If not loaded, loads it
        """
        tag = "G%sc%s" % (generation, cluster)
        if self.molids.get(tag) is None:
            self.load_cluster(generation, cluster)
        else:
            if visstate is None:
                visstate = not molecule.get_visible(self.molids[tag])
            molecule.set_visible(self.molids[tag], visstate)

    #===========================================================================

    def load_cluster(self, generation, cluster):
        """
        Loads the corresponding cluster
        """
        # Remember view to reset later
        #display.update_off()
        #center = None
        #if len(self.molids):
        #    center = trans.get_center(self.molids.values()[0])
        #    scale = trans.get_scale(self.molids.values()[0])
        #    transl = trans.get_trans(self.molids.values()[0])
        #    rotatl = trans.get_rotation(self.molids.values()[0])

        if "prmtop" in self.reference:
            mid = molecule.load("parm7", self.reference, "crdbox",
                                self.reference.replace(".prmtop", ".inpcrd"))
        elif "psf" in self.reference:
            mid = molecule.load("psf", self.reference, "pdb",
                                self.reference.replace(".psf", ".pdb"))

        cfile = os.path.join(self.rootdir, "clusters", str(generation),
                             "%s.dx" % cluster)
        molecule.read(mid, "dx", cfile, waitfor=-1)
        molecule.rename(mid, "G%sc%s" % (generation, cluster))

        molrep.delrep(mid, 0)
        molrep.addrep(mid, style="NewRibbons 0.1 12.0 12.0",
                      selection="protein or resname ACE NMA",
                      color="Molecule", material="Opaque")
        molrep.addrep(mid, style="Isosurface 0.05 0 0 1 1",
                      color="Molecule", material="Opaque")
        self.molids["G%sc%s" % (generation, cluster)] = mid

        # Reset view
        #if center:
        #    for m in self.molids.values():
        #        trans.set_center(m, center)
        #        trans.set_scale(m, scale)
        #        trans.set_trans(m, transl)
        #        trans.set_rotation(m, rotatl)
        #display.update_on()

    #===========================================================================

    def _enqueue_output(self, process):
        while process.poll() is None:
            try:
                for line in iter(process.stdout.readline, b''):
                    data = line.decode("utf-8").strip().split(" ")
                    self.show_cluster(*data)
            except: pass
            time.sleep(0.05)

    #===========================================================================

    def graph(self):
        """
        Displays the graph in another process
        """
        ON_POSIX = 'posix' in sys.builtin_module_names

        # Start the graph process and send it the data
        p = Popen([sys.executable, "-u", __file__], stdin=PIPE, stdout=PIPE,
                  bufsize=1, close_fds=ON_POSIX)
        p.stdin.write(pickle.dumps(self.plotdata, protocol=-1))
        p.stdin.flush()
        p.stdin.close()

        # Start the listener thread and attach it to stdout
        tq = Thread(target=self._enqueue_output, args=(p,))
        tq.daemon = True
        tq.start()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":

    data = pickle.loads(sys.stdin.buffer.read())
    g = GraphProcess(data)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
