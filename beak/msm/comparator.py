"""
Contains classes for comparing clusters across multiple generations.
"""
import os
import numpy as np

from beak.msm import utils
from configparser import ConfigParser
from vmd import molecule, vmdnumpy

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

def get_bound_state(rootdir):
    """
    Returns bound coordinates as numpy array. Only b2ar supported
    """
    if "B2AR" in rootdir and "50ns" in rootdir:
        mid = molecule.load("mae", os.path.join(rootdir, "..", "bound_5.mae"))
        mask = vmdnumpy.atomselect(mid, 0, "noh and resname DALP")
        coords = np.compress(mask, vmdnumpy.timestep(mid, 0), axis=0)
        molecule.delete(mid)
        return coords

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

    if len(varxs.keys()) != 50:
        print("MISSING KEYS: gen %d" % gen)

    validmeans = [m for m in means if np.mean(varxs[m]) < varthreshold]
    validoldmeans = [m for m in oldmeans if np.mean(oldvarxs[m]) < varthreshold]

    # Get + load bound state
    boundcoords = get_bound_state(rootdir)

    connecteds = []
    for m in validmeans:
        bestrmsd = threshold * np.ones(means[0].shape)
        candidate = ()
        for mm in validoldmeans:
            rmsd = np.sqrt(np.sum((means[m] - oldmeans[mm])**2, axis=0))
            if np.all(rmsd < bestrmsd):
                brms = np.mean(np.sqrt(np.sum((means[m]-boundcoords)**2,
                                              axis=0)))
                candidate = (mm, m, brms)
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
    plot_data = { k : [] for k in range(mingen+1, maxgen+1) }
    counter = 0 # DEBUG

    # Initialize array with first found cluster set
    msm = None
    prevmsm = utils.load(os.path.join(rootdir, "production", str(mingen),
                                      "mmsm_G%d.pkl" % mingen))
    #plot_data[mingen] = [[pop] for pop in prevmsm.populations_]

    #for g in range(mingen+1, maxgen+1):
    for g in range(mingen+1, maxgen+1):
        msm = utils.load(os.path.join(rootdir, "production", str(g),
                                      "mmsm_G%d.pkl" % (g)))
        result = compare_clusters(gen=g, prevgen=g-1, rootdir=rootdir)

        # Aggregate and get populations
        appended = []
        for r in result:
            l1 = prevmsm.inverse_transform([r[0]])[0][0]
            l2 = msm.inverse_transform([r[1]])[0][0]

            for biglist in plot_data.values():
                for entry in biglist:
                    if entry[-1][0] == prevmsm.populations_[l1]:
                        entry.append((msm.populations_[l2], r[2]))
                        appended.append(l2)
                        counter += 1

        # Handle entries that have no correspondence with previous generation
        for c in [_ for _  in msm.mapping_.values() if _ not in appended]:
            plot_data[g].append([ (msm.populations_[c], 100.0) ])
            counter += 1
        prevmsm = msm

    # Trim out entries with no sequence (TODO)
    print("should be %d is %d ideal %d" % (counter, sum(len(_) for p in plot_data.values() for _ in p),
                                           (maxgen-mingen)*50))
    return plot_data

#==============================================================================
