"""
Contains useful utilities for running MSMs
"""
from __future__ import print_function
import os
from glob import glob
import pickle
import h5py
import numpy as np
from msmbuilder.msm import MarkovStateModel
from vmd import atomsel, molecule
atomsel = atomsel.atomsel

#==============================================================================

def save_features_h5(dataset, filename, num_ligands=0, trajfiles=None):
    """
    Saves the given feature set as an hdf5 file

    Args:
        dataset (list of ndarray): Data to save
        filename (str): Filename to save as
        num_ligands (int): Number of ligands in the system
        trajfiles (list of str): Filenames of trajectories. This lets the
            "filename" attribute be populated

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
        if num_ligands:
            h5f[str(i)].attrs["filename"] = trajfiles[i/num_ligands]

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

def get_topology(filename, rootdir):
    """
    Returns the psf file corresponding to a given trajectory file,
    using my directory layout. No fallback for older psfs.

    Args:
        filename (str): File name of trajectory
        rootdir (str): Root directory of simulation

    Returns:
        (str): File name of corresponding topology. Could be psf or prmtop.
    """
    rep = filename.split('/')[-2]
    gen = filename.split('/')[-3]

    # Handle special psf case for stripped trajectories
    if "strip" in filename:
        topo = os.path.join(rootdir, "systems", gen, "%s_stripped.prmtop" % rep)
    else:
        topo = os.path.join(rootdir, "systems", gen, "%s.psf" % rep)

    return topo

#==============================================================================

def get_trajectory_format(filename):
    """
    Gets the filetype for VMD loading for the given filename

    Returns:
        (str): VMD format string for loading the given file
    """
    if ".dcd" in filename:
        return "dcd"
    elif ".nc" in filename:
        return "netcdf"
    else:
        raise ValueError("No known format for file %s" % filename)

#==============================================================================

def load_trajectory(filename, rootdir, **kwargs):
    """
    Loads a trajectory with the correct topology and format based on
    the filename. Also aligns to the reference structure using the
    appropriate reference selection.

    Args:
        filename (str): The reimaged trajectory file to load
        rootdir (str): Root directory of the sampling run
        aselref (atomsel): Reference atom selection to align to
        psfref (str): Atom selection strings for PSFs / normal resids
        prmref (str): Atom selection strings for prmtops / altered resids
        frame (int): Single frame to load, or None for all frames, or (beg,end)
        topology (str): Manual topology choice to use, or None for autodetect
        molid (int): Molecule ID to load trajectory into, or None for new molid
        skip (int): Stride to load, defaults to 1

    Returns:
        (int): VMD molecule ID of loaded and aligned trajectory
    """
    # Load the topology
    topology = kwargs.get("topology", None)
    if topology is None:
        topology = get_topology(filename, rootdir)

    if kwargs.get("molid") is None:
        mid = molecule.load("psf" if "psf" in topology else "parm7", topology)
    else:
        mid = int(kwargs.get("molid"))

    aselref = kwargs.get("aselref", None)
    psfref = kwargs.get("psfref", None)
    prmref = kwargs.get("prmref", None)
    frame = kwargs.get("frame", None)
    skip = kwargs.get("skip", 1)

    # Load the trajectory in
    fmt = get_trajectory_format(filename)
    if frame is None:
        molecule.read(mid, fmt, filename, waitfor=-1, skip=skip)
    elif isinstance(frame, int):
        molecule.read(mid, fmt, filename, beg=frame, end=frame, waitfor=-1)
    elif len(frame) == 2:
        molecule.read(mid, fmt, filename, beg=frame[0], end=frame[1],
                      waitfor=-1, skip=skip)
    else:
        raise ValueError("I don't understand loading frames: %s" % frame)

    # Align, if desired
    if aselref is None:
        return mid
    framsel = atomsel(psfref if "psf" in topology else prmref, molid=mid)
    for frame in range(molecule.numframes(mid)):
        molecule.set_frame(mid, frame)
        framsel.update()
        atomsel("all", molid=mid, frame=frame).move(framsel.fit(aselref))
    return mid

#==============================================================================

def align(molid, refid, refsel):
    """
    Aligns all frames found in the trajectory specified by molid to
    a reference structure.

    Args:
        molid (int): VMD molecule ID for loaded trajectory to align
        refid (int): VMD molecule ID for loaded reference molecule
        refsel (str): Atom selection string for atoms to align.
    """
    ref = atomsel(refsel, molid=refid)
    for frame in range(molecule.numframes(molid)):
        psel = atomsel(refsel, molid=molid, frame=frame)
        tomove = psel.fit(ref)
        atomsel("all", molid=molid, frame=frame).move(tomove)

#==============================================================================

def get_prodfiles(generation, rootdir, new=False):
    """
    Gets the sorted list of all production files for a given adaptive
    sampling run. Supports legacy non-stripped and non-equilibrated
    reimaged trajectories.

    Args:
        generation (int): Latest generation to load
        rootdir (str): Root directory for sampling run
        new (bool): Only return production files from this generation, if True

    Returns:
        (list of str): Production files, inorder
    """
    prodfiles = []
    for gen in range(generation if new else 1, generation+1):
        rpath = os.path.join(rootdir, "production", str(gen), "*")
        pfs = glob(os.path.join(rpath, "Reimaged_strip_Eq1*.nc"))

        # Fallpack to previous non-stripped reimaging
        # Fallback again to even older non-equilibration reimaging
        if not len(pfs):
            pfs = glob(os.path.join(rpath, "Reimaged_strip_Eq6*.nc"))
        if not len(pfs):
            pfs = glob(os.path.join(rpath, "Reimaged_Eq1*.nc"))
        if not len(pfs):
            pfs = glob(os.path.join(rpath, "Reimaged_Eq6*.nc"))

        prodfiles.extend(sorted(pfs, key=lambda x: int(x.split('/')[-2])))

    return prodfiles

#==============================================================================

def load(filename):
    """
    Python 2/3 compatible load function
    """
    return pickle.load(open(filename, 'rb'), encoding="latin-1")

#==============================================================================

def get_equivalent_clusters(label, clust1, clust2):
    """
    Returns equivalent cluster(s) between MSMs.

    Args:
        label (int): Which cluster to look at, in clust1
        clust1 (list of ndarray): First set of clusters
        clust2 (list of ndarray): Second set of clusters

    Returns:
        (list of int): Label(s) in clust2 that match input label
    """
    found = set()
    frames = {k:v for k, v in {i:np.ravel(np.where(c == label))
                               for i, c in enumerate(clust1)
                              }.items() if len(v)}

    for trajidx, cidx in frames.items():
        found.update([clust2[trajidx][c] for c in cidx \
                      if c < len(clust2[trajidx])])

    return found

#==============================================================================

def generate_truelabeled_msm(truelabels, length, lag):
    """
    Generates a MSM of given length with the labelled data

    Args:
        truelabels (list of ndarray): Cluster labels
        length (int): Length for labels to use
        lag (int): Lag time, in frames, for MSM

    Returns:
        (MarkovStateModel): the model
    """
    msm = MarkovStateModel(lag_time=lag,
                           reversible_type="transpose",
                           ergodic_cutoff="off",
                           prior_counts=0.000001)

    msm.fit([t[:length] for t in truelabels])
    return msm

#==============================================================================
