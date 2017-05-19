"""
Contains methods for dealing with condensed cluster
MSM representations
"""
import os
import numpy as np
from msmbuilder.tpt import hub_scores, net_fluxes, top_path
from vmd import molecule, molrep, vmdnumpy, atomsel

#==============================================================================

def closest_to_bound(samp, msm, truesel, trueid):
    """
    Visualizes and returns the closest cluster to the bound pose.

    Args:
        samp (ClusterSampler): MSM trajectory object
        clust (cluster): Which clusters to consider
        msm (MarkovStateModel): Markov model
        truesel (str): Atom selection for true bound pose
        trueid (int): VMD molecule ID of molecule with bound pose

    Returns:
        (list of float/int): Cluster labels, in order, of closest to bound
    """
    # Get the atom selection mask for the known bound ligand
    tmask = vmdnumpy.atomselect(trueid, 0, "noh and (%s)" % truesel)
    bound = np.compress(tmask, vmdnumpy.timestep(trueid, 0), axis=0)

    # Find the closest frame to that one
    rmsds = {}
    for m in samp.molids:
        mask = vmdnumpy.atomselect(m, 0, "noh and resname %s" \
                                   % " ".join(samp.ligands))
        clustmean = np.compress(mask, vmdnumpy.timestep(m, 0), axis=0)
        r = np.sum(np.sqrt(np.sum((bound-clustmean)**2, axis=1)))
        rmsds[r] = molecule.name(m).split("_")[1]

    return [rmsds[x] for x in np.sort(rmsds.keys())]

#==============================================================================

def show_binding_pathway(samp, bound, clust, msm, scores=None):
    """
    Visualizes the clusters along the binding pathway compared to
    a known bound pose. Pathway is the most probable one to bulk solvent.

    Args:
        samp (ClusterSampler): MSM trajectory object
        bound (cluster): Which cluster corresponds to the bound pose
        clust (clusters): Which clusters to consider
        msm (MarkovStateModel): Which MSM to consider
        scores (list of float): Hub scores, or will be computed if none

    Returns:
        (list of int): Clusters along the binding pathway, including
            the closest one to the known bound pose, not including bulk solvent.
    """
    # Compute scores if undefined
    if scores is None:
        scores = hub_scores(msm)

    # Identify the solvent cluster and the bound cluster
    solvent = np.argmax(scores)

    # Handle single cluster
    if type(bound) != type(list()):
        bound = [bound]

    # Get the top pathway and remove solvent from it
    # Do this for each specified cluster since we want path to each cluster
    pathway = []
    for b in bound:
        flux = net_fluxes(sources=[solvent], sinks=[b], msm=msm)
        path = list(top_path(sources=[solvent], sinks=[b], net_flux=flux)[0])
        pathway.append(path)
        print(path)

    show_clusters(samp, path)

    return pathway

#==============================================================================

def show_clusters(samp, clusters):
    """
    Visualizes the indicated clusters

    Args:
        samp (ClusterSampler): MSM trajectory thing
        clusters (list of int): Clusters to visualize
    """

    for m in samp.molids:
        molecule.set_visible(m, False)
        if any(molecule.name(m) == "%d_%s" \
                % (samp.generation, c) for c in clusters):
            molecule.set_visible(m, True)

#==============================================================================

def trajs_with_clust(samp, cluster):
    """
    Returns the trajectories where a cluster is represented

    Args:
        samp (ClusterSampler): MSM cluster object
        clusts (int): Cluster label to look for

    Returns:
        dict str->list of int): Filename, frames where cluster
            frame is present
    """
    frames = {k:v for k, v in {i:np.ravel(np.where(c == cluster)) \
              for i, c in enumerate(samp.mclust)}.iteritems() \
              if len(v)}

    files = {}
    for clustidx, frames in frames.items():
        pos = samp.prodfiles[clustidx/samp.num_ligands]
        if files.get(pos) is not None:
            files[pos].extend(frames)
        else:
            files[pos] = list(frames)

    return files

#==============================================================================

def load_traj(samp, filename):
    """
    Loads a trajectory and aligns it, sets representations

    Args:
        filename (str): Production filename to load

    Returns:
        (int): Loaded molid
    """
    # Load data
    gen = filename.split('/')[-3]
    rep = filename.split('/')[-2]

    id = molecule.load("psf", os.path.join(samp.dir, "systems", gen, "%s.psf" % rep))
    molecule.read(id, "netcdf", filename, skip=5, waitfor=-1)
    molecule.rename(id, "traj-G%s-r%s" % (gen, rep))

    # Set representation
    molrep.delrep(id, 0)
    molrep.addrep(id, style="NewRibbons", material="Opaque",
                  selection="protein and not same fragment as "
                            "(resname %s)" % " ".join(samp.ligands),
                            color="Type")
    molrep.addrep(id, style="Licorice", material="Opaque",
                  selection="noh and same fragment as "
                            "(resname %s)" % " ".join(samp.ligands),
                            color="Type")

    # Align to other molids
    refsel = atomsel("protein and not same fragment as "
                     "(resname %s)" % " ".join(samp.ligands),
                     molid=samp.molids[0], frame=0)

    for frame in range(molecule.numframes(id)):
        psel = atomsel("protein and not same fragment as "
                       "(resname %s)" % " ".join(samp.ligands),
                       molid=id, frame=frame)
        tomove = psel.fit(refsel)
        atomsel("all", molid=id, frame=frame).move(tomove)

    return id

#==============================================================================
