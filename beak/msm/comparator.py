"""
Contains classes for comparing clusters across multiple generations.
"""
import os
import numpy as np

from beak.msm import utils
from configparser import ConfigParser
from glob import glob
from matplotlib import pyplot as plt

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Cluster comparators                              #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ClusterEquivalent(object):
    """
    Represents two equivalent clusters, and their associated stats.

    Attributs:
        gen1 (int): Generation of first cluster
        gen2 (int): Generation of second cluster
        label1 (int): Label of first clustesr
        label2 (int): Label of second cluster
        mean_rmsd (float): RMSD between mean coordinates of both clusters
    """
    def __init__(self, gen1, gen2, label1, label2, rmsd):
        self.gen1 = gen1
        self.gen2 = gen2
        self.label1 = label1
        self.label2 = label2
        self.rmsd = rmsd

    def __str__(self):
        return "%d <- %d RMSD=%f" % (self.label1, self.label2, np.mean(self.rmsd))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def compare_clusters(gen, prevgen, rootdir,
                     varthreshold=5.0, threshold=10.0):
    """
    Finds clusters that match up in terms of mean position.

    Args:
        gen (int): Matches up clusters from this generation...
        prevgen (int): ... to clusters in this previous generation
        rootdir (str): Root simulation directory
        varthreshold (float): Variance thereshold factor (TODO)
        threshold (float): RMSD threshold factor (TODO)

    Returns:
        (list of ClusterEquivalent): Matching clusters, inorder
    """
    means = utils.load(os.path.join(rootdir, "clusters", str(gen), "means.pkl"))
    varxs = utils.load(os.path.join(rootdir, "clusters", str(gen),
                                    "variance.pkl"))
    oldmeans = utils.load(os.path.join(rootdir, "clusters", str(prevgen),
                                       "means.pkl"))
    oldvarxs = utils.load(os.path.join(rootdir, "clusters", str(prevgen),
                                       "variance.pkl"))
    validmeans = [m for m in means if np.mean(varxs[m]) < varthreshold]
    validoldmeans = [m for m in oldmeans if np.mean(oldvarxs[m]) < varthreshold]

    connecteds = []
    for m in validmeans:
        bestrmsd = threshold * np.ones(means[0].shape)
        candidate = ()
        for mm in validoldmeans:
            rmsd = np.sqrt(np.sum((means[m] - oldmeans[mm])**2, axis=0))
            if np.all(rmsd < bestrmsd):
                candidate = (mm, m)
                bestrmsd = rmsd
        if candidate:
            connecteds.append(candidate)

    return connecteds

#==============================================================================

def plot_cluster_progress(config, mingen, maxgen):
    """
    Plots the cluster progress, using all available data

    Args:
        config (str): Path to configuration file for run

    Returns:
        (fig) Plottable matplotlib figure
    """
    assert(os.path.isfile(config))
    cfg = ConfigParser(interpolation=None)
    cfg.read(config)
    rootdir = cfg.get("system", "rootdir")

    # Accumulate generations with a variance pickle file
    # TODO: Actually calculate variance for older generations where
    #       it wasn't calculated at the time
#    gen = cfg.getint("production", "generation")
#    ggen = (g for g in range(1, gen+1)
#            if os.path.isfile(os.path.join(rootdir, "clusters", str(gen),
#                                           "variance.pkl")))

    # Obtain a list of clusters across generations
    prevmsm = utils.load(os.path.join(rootdir, "production", str(mingen),
                                      "mmsm_G%d.pkl" % mingen))
    msm = None
    plot_data = { k : [] for k in range(mingen, maxgen) }
    counter = 0 # DEBUG

    for g in range(mingen, maxgen):
        msm = utils.load(os.path.join(rootdir, "production", str(g+1),
                                      "mmsm_G%d.pkl" % (g+1)))
        result = compare_clusters(gen=g+1, prevgen=g, rootdir=rootdir)
        counter += len(result)
        #print("Gen: %d result %s" % (g, result))

        # Aggregate and get populations
        for r in result:
            l1 = prevmsm.inverse_transform([r[0]])[0][0]
            l2 = msm.inverse_transform([r[1]])[0][0]

#            if g == mingen: # First run, init lists of lists
#                plot_data[g].append([prevmsm.populations_[l1],
#                                     msm.populations_[l2]])
#                continue

            for gstart in plot_data.values():
                for p in gstart:
                    if p[-1] == prevmsm.populations_[l1]:
                        p.append(msm.populations_[l2])
                        break
                else:
                    plot_data[g].append([prevmsm.populations_[l1],
                                         msm.populations_[l2]])

        prevmsm = msm

    return plot_data
    print("should be %d is %d" % (counter, sum(len(p) for p in plot_data.values())))

    # Generate plot
    for mg, data in plot_data.items():
        for d in data:
            plt.plot(x=range(mg, mg+len(d)), y=d)

#==============================================================================
