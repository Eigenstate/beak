"""
Contains classes for comparing clusters across multiple generations.
"""
#pylint: disable=invalid-name, wrong-import-order
import os
import numpy as np

from beak.msm import utils
from configparser import ConfigParser
from vmd import molecule, vmdnumpy

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Cluster comparators                              #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Cluster(object):
    """
    Represents a single cluster, with associated data relevant for plotting

    Attributes:
        label (int): Cluster label
        generation (int): Generation cluster came from
        rmsd_to_bound (float): RMSD to bound state
        hub_score (float): Hub score
        population (float): Predicted equilibrium population
    """
    def __init__(self, **kwargs):
        self.label = kwargs.get("label")
        self.generation = kwargs.get("generation")
        self.rmsd_to_bound = kwargs.get("rmsd_to_bound")
        self.hub_score = kwargs.get("hub_score")
        self.population = kwargs.get("population")

    #===========================================================================

    def __repr__(self):
        return "G%sc%s" % (self.generation, self.label)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_bound_state(rootdir):
    """
    Returns bound coordinates as numpy array. Only b2ar supported
    """
    mid = None
    if "B2AR" in rootdir and "50ns" in rootdir:
        mid = molecule.load("mae", os.path.join(rootdir, "..", "bound_5.mae"))
        mask = vmdnumpy.atomselect(mid, 0, "noh and resname DALP")
    elif "dipp" in rootdir:
        mid = molecule.load("mae", os.path.join(rootdir, "..", "pep_bound",
                                                "prep", "4RWD_prepped_bound.mae"))
        mask = vmdnumpy.atomselect(mid, 0,
                                   "noh and same fragment as resname DI7 DI78")

    if mid:
        coords = np.compress(mask, vmdnumpy.timestep(mid, 0), axis=0)
        molecule.delete(mid)
        return coords
    else:
        return None

#==============================================================================

def rmsd_to(coords, boundcoords):
    """
    Gets the RMSD, coordinate wise, between 2 things
    """
    return np.sqrt(np.sum((coords-boundcoords)**2, axis=0))

#==============================================================================

def compare_clusters(gen, prevgen, rootdir,
                     varthreshold=5.0, threshold=5.0, **kwargs):
    """
    Finds clusters that match up in terms of mean position.

    Args:
        gen (int): Matches up clusters from this generation...
        prevgen (int): ... to clusters in this previous generation
        rootdir (str): Root simulation directory
        varthreshold (float): Variance thereshold factor (TODO)
        threshold (float): RMSD threshold factor (TODO)

    Returns:
        (list of tuple of Clusters): Matching clusters, inorder
    """
    d = os.path.join(rootdir, "clusters", str(gen))
    means = kwargs.get("means", utils.load(os.path.join(d, "means.pkl")))
    varxs = utils.load(os.path.join(d, "variance.pkl"))

    d = os.path.join(rootdir, "production", str(gen))
    msm = kwargs.get("msm",
                     utils.load(os.path.join(d, "mmsm_G%d.pkl" % gen)))
    scores = kwargs.get("scores",
                        utils.load(os.path.join(d, "mmsm_scores.pkl")))

    d = os.path.join(rootdir, "clusters", str(prevgen))
    oldmeans = utils.load(os.path.join(d, "means.pkl"))
    oldvarxs = utils.load(os.path.join(d, "variance.pkl"))

    d = os.path.join(rootdir, "production", str(prevgen))
    oldmsm = utils.load(os.path.join(d, "mmsm_G%d.pkl" % prevgen))
    oldscores = utils.load(os.path.join(d, "mmsm_scores.pkl"))

    if len(varxs.keys()) != 50:
        print("MISSING KEYS: gen %d" % gen)

    validmeans = [m for m in means if np.mean(varxs[m]) < varthreshold]
    validoldmeans = [m for m in oldmeans if np.mean(oldvarxs[m]) < varthreshold]

    # Get + load bound state, solvent population
    boundcoords = get_bound_state(rootdir)

    connecteds = []
    for m in validmeans:
        bestrmsd = threshold * np.ones(list(means.values())[0].shape)
        candidate = ()
        for mm in validoldmeans:
            rmsd = rmsd_to(means[m], oldmeans[mm])
            if np.all(rmsd < bestrmsd):

                l1 = oldmsm.transform([mm])[0][0]
                l2 = msm.transform([m])[0][0]
                c1 = Cluster(label=mm,
                             generation=prevgen,
                             population=oldmsm.populations_[l1],
                             hub_score=oldscores[l1],
                             rmsd_to_bound=np.mean(rmsd_to(oldmeans[mm],
                                                           boundcoords))
                                           if np.any(boundcoords)
                                           else None)
                c2 = Cluster(label=m,
                             generation=gen,
                             population=msm.populations_[l2],
                             hub_score=scores[l2],
                             rmsd_to_bound=np.mean(rmsd_to(means[m],
                                                           boundcoords))
                                           if np.any(boundcoords)
                                           else None)

                candidate = (c1, c2)
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
        (list of lists of Clusters) : Equivalent clusters
    """
    assert os.path.isfile(config)
    cfg = ConfigParser(interpolation=None)
    cfg.read(config)
    rootdir = cfg.get("system", "rootdir")
    boundcoords = get_bound_state(rootdir)

    # Obtain a list of clusters across generations
    plot_data = []
    counter = 0 # DEBUG

    for g in range(mingen+1, maxgen+1):
        mdir = os.path.join(rootdir, "production", str(g))
        msm = utils.load(os.path.join(mdir, "mmsm_G%d.pkl" % g))
        scores = utils.load(os.path.join(mdir, "mmsm_scores.pkl"))
        means = utils.load(os.path.join(rootdir, "clusters",
                                        str(g), "means.pkl"))
        appended = []

        # Aggregate and get populations
        result = compare_clusters(gen=g, prevgen=g-1, rootdir=rootdir,
                                  msm=msm, scores=scores, means=means)
        for r in result:
            for sequence in plot_data:
                if sequence[-1].label == r[0].label and \
                   sequence[-1].generation == g-1:
                    sequence.append(r[1])
                    appended.append(r[1].label)
                    counter += 1
                    # Do NOT break here as it's not necessarily 1:1

        # Handle entries that have no correspondence with previous generation
        for c, l in [ x for x in msm.mapping_.items() if x[0] not in appended]:
            plot_data.append([ Cluster(label=c,
                                     generation=g,
                                     rmsd_to_bound=np.mean(rmsd_to(boundcoords,
                                                                   means[c]))
                                                   if np.any(boundcoords)
                                                   else None,
                                     hub_score=scores[l],
                                     population=msm.populations_[l])
                             ])
            counter += 1

    # Trim out entries with no sequence
    plot_data = [p for p in plot_data if len(p) > 1]
    print("should be %d is %d ideal %d" % (counter,
                                           sum(len(_) for _ in plot_data),
                                           (maxgen-mingen)*50))
    return plot_data

#==============================================================================
