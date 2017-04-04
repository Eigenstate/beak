"""
Contains useful utilities for running MSMs
"""
from __future__ import print_function
import os
import h5py
import numpy as np
from vmd import atomsel, molecule, vmdnumpy
atomsel = atomsel.atomsel

#==============================================================================

def save_features_h5(dataset, filename):
    """
    Saves the given feature set as an hdf5 file

    Args:
        dataset (list of ndarray): Data to save
        filename (str): Filename to save as

    Returns:
        True on sucess
    """

    h5f = h5py.File(filename, 'w-') # w- means fail on existence
    for i, fset in enumerate(dataset):
        h5f.create_dataset(name=str(i),
                           shape=fset.shape,
                           dtype=fset.dtype,
                           data=fset,
                           compression="gzip")
    h5f.close()
    return True

#==============================================================================

def load_features_h5(filename):
    """
    Loads a feature set from an hdf5 file

    Args:
        filename (str): Filename to load

    Returns:
        (list of ndarray) Loaded features
    """

    h5f = h5py.File(filename, 'r')
    feated = []
    for i in sorted(h5f.keys(), key=int):
        feated.append(h5f[i][:])
    h5f.close()

    return feated

#==============================================================================

def get_psf_from_traj(filename, rootdir):
    """
    Returns the psf file corresponding to a given trajectory file,
    using my directory layout. No fallback for older psfs.

    Args:
        filename (str): File name of trajectory
        rootdir (str): Root directory of simulation

    Returns:
        (str): File name of corresponding psf
    """
    rep = filename.split('/')[-2]
    gen = filename.split('/')[-3]
    psf = os.path.join(rootdir, "systems", gen, "%s.psf" % rep)

    return psf

#==============================================================================

def get_cluster_center(prodfiles, clusts, cluster, ligands, topology):
    """
    Obtains a representative structure of the given cluster.
    Only reads in one trajectory at a time, so may not be completely
    the best structure but works when all trajs can't be loaded into
    memory.

    Args:
        prodfiles (list of str): Production files, ordered the same as
            for cluster obtaining
        clusts (list of ndarray): Cluster data
        cluster (int): Which cluster to get center of
        ligands (list of str): Ligand resnames
        topology (str): Path to psf, or rootdir for autopsf

    Returns:
        (int) VMD molid of representative frame
        (float) RMSD of representatitive frame to mean
    """
    mean = None      # Current mean (ligheavyatoms, 3)
    count = None     # number of frames with this cluster
    bestidx = None   # fileindex, frameindex, ligindex of best
    bestpos = None   # Current best (ligheavyatoms, 3)

    for trajidx, trajfile in enumerate(prodfiles):
        # Read in a new molecule
        molid = molecule.load("psf", topology if os.path.isfile(topology)
                              else get_psf_from_traj(trajfile, topology))
        molecule.read(molid, "netcdf", trajfile, waitfor=-1)

        # Get residue number for each ligand
        ligids = sorted(set(atomsel("resname %s" % " ".join(ligands),
                                    molid=molid).get("residue")))

        # Initialize arrays if uninitialized according to ligand size
        if mean is None:
            ligheavyatoms = len(atomsel("noh and same fragment as residue %d"
                                        % ligids[0], molid=molid))
            mean = np.zeros((ligheavyatoms, 3))
            count = 0

        # Now handle the ligands independently
        for i, lig in enumerate(ligids):
            # Get atom selection mask for the ligand
            mask = vmdnumpy.atomselect(molid, 0,
                                       "noh and same fragment as residue %d"
                                       % lig)
            # Get frames that contain this cluster
            cidx = trajidx*len(ligids) + i
            if cluster == "nan":
                frames = [_ for _, d in enumerate(clusts[cidx]) if np.isnan(d)]
            else:
                frames = [_ for _, d in enumerate(clusts[cidx]) if d == cluster]

            # Add frames to running total
            count += len(frames)
            dev = []
            for f in frames:
                mean += np.compress(mask, vmdnumpy.timestep(molid, f), axis=0)
                dev.append(np.sqrt(1./ligheavyatoms *
                                   np.sum((np.compress(mask, vmdnumpy.timestep(molid, f),
                                                       axis=0) - (mean/float(count)))**2,
                                          axis=1)))

            # Find if there's a more representative frame in this structure
            if bestpos is not None:
                olddev = np.sqrt(1./ligheavyatoms * np.sum((bestpos-(mean/float(count)))**2,
                                                           axis=1))
            if min(dev) < olddev or bestpos is None:
                bestpos = np.compress(mask,
                                      vmdnumpy.timestep(molid, frames[np.argmin(dev)]),
                                      axis=0)
                bestidx = [(trajidx, molid, lig), min(dev)]
                print("Found a better mean: %f" % min(dev))
            else:
                bestidx[1] = olddev # Update score in case it's better now

        # Clean up
        molecule.delete(molid)

    return bestidx

#==============================================================================

def save_cluster_centers(prodfiles, clust, msm, ligands, outdir, topology):
    """
    Saves cluster centers to a file

    Args:
        prodfiles (list of str): Production filenames, same as clust order
        clust (list of ndarray): Cluster data
        msm (MarkovStateModel): MSM with cluster labels
        ligands (list of str): Ligand resnames in system
        outdir (str): Output directory for cluster centers
        topology (str): Either root directory for system or psf file
    """

    protsel = "(protein or resname ACE NMA) and not same fragment as " \
              "resname %s" % " ".join(ligands)
    fn = open(os.path.join(outdir, "rmsds"), 'w', 0) # Unbuffered

    for cl in msm.mapping_.values():
        print("On cluster: %s" % cl)
        lg, rms = get_cluster_center(prodfiles, clust, cl, ligands, topology)
        m, f, l = lg
        print("  Molid: %d\tFrame: %d\tLigand: %d" % (m, f, l))
        fn.write("%s\t%f\n" % (cl, rms))
        atomsel("(%s) or (same fragment as residue %d)" % (protsel, l),
                molid=m, frame=f).write("mae",
                                        os.path.join(outdir, "%s.mae" % cl))
    fn.close()

#==============================================================================

