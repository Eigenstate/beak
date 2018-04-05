"""
Contains useful utilities for running MSMs
"""
from __future__ import print_function
import os
import stat
from glob import glob
import pickle
import h5py
import random
import numpy as np
from configparser import ConfigParser
from vmd import atomsel, molecule

#==============================================================================

def save_tica_h5(tica, filename, overwrite=False):
    """
    Dumps a tica object to a h5 file that can be loaded with load_tica_h5.
    This is because pickling it results in crazy huge files with lots
    of memory usage

    Args:
        tica (tICA): tICA object to dump
        filename (str): Filename to save as
        overwrite (bool): Will allow overwriting of an existing file
    """

    h5f = h5py.File(filename, 'w' if overwrite else 'w-', libver="latest")
    for attribute in ["n_features", "n_components",
                      "n_observations_", "n_sequences_",
                      "_outer_0_to_T_lagged", "_sum_0_to_TminusTau",
                      "_sum_tau_to_T", "_sum_0_to_T",
                      "_outer_0_to_TminusTau", "_outer_offset_to_T"]:
        data = getattr(tica, attribute)
        h5f.create_dataset(name=attribute,
                           shape=(1,) if np.isscalar(data) else data.shape,
                           dtype=int if np.isscalar(data) else data.dtype,
                           compression=None if np.isscalar(data) else "gzip",
                           data=data)

    h5f.close()

    # Change permissions to be read only
    if not overwrite:
        os.chmod(filename, stat.S_IRUSR)

#==============================================================================

def load_tica_h5(filename):
    """
    Loads a tica dataset from the compact h5 format

    Args:
        filename (str): File to load

    Returns:
        (tICA object): The tICA object
    """
    from msmbuilder.decomposition import tICA

    # Extract data from the h5 file
    # Handle array types and int types differently
    h5f = h5py.File(filename, 'r')
    tica = tICA()
    for attribute, data in h5f.items():
        if len(data) == 1:
            setattr(tica, attribute, data[0])
        else:
            newdata = np.empty(data.shape, dtype=data.dtype)
            setattr(tica, attribute, newdata)
            data.read_direct(newdata)

    # Set some attributes ourselves
    tica._initialized = True
    tica._is_dirty = True

    return tica

#==============================================================================

def save_features_h5(dataset, filename, num_ligands=0, trajfiles=None,
                     overwrite=False):
    """
    Saves the given feature set as an hdf5 file

    Args:
        dataset (list of ndarray): Data to save
        filename (str): Filename to save as
        num_ligands (int): Number of ligands in the system
        trajfiles (list of str): Filenames of trajectories. This lets the
            "filename" attribute be populated
        overwrite (bool): Allow a file with the same name to be clobbered

    Returns:
        True on sucess
    """

    h5f = h5py.File(filename, 'w' if overwrite else 'w-',
                    libver="latest") # w- means fail on existence
    for i, fset in enumerate(dataset):
        h5f.create_dataset(name=str(i),
                           shape=fset.shape,
                           dtype=fset.dtype,
                           data=fset,
                           compression="gzip")
        if num_ligands:
            h5f[str(i)].attrs["filename"] = trajfiles[i/num_ligands]

    h5f.close()

    # Change permissions to be read only
    if not overwrite:
        os.chmod(filename, stat.S_IRUSR)

    return True

#==============================================================================

def load_features_h5(filename, maxframes=None):
    """
    Loads a feature set from an hdf5 file

    Args:
        filename (str): Filename to load
        maxframes (int): Maximum frames to load. If None, will load all frames.

    Returns:
        (list of ndarray) Loaded features
    """

    h5f = h5py.File(filename, 'r')
    feated = []
    for i in sorted(h5f.keys(), key=int):
        if maxframes is not None:
            feated.append(h5f[i][:maxframes])
        else:
            feated.append(h5f[i][:])
    h5f.close()

    return feated

#==============================================================================

def get_topology(filename, rootdir=None):
    """
    Returns the psf file corresponding to a given trajectory file,
    using my directory layout. No fallback for older psfs.

    Args:
        filename (str): File name of trajectory
        rootdir (str): Root directory of simulation, or None
            to guess from the fullpath

    Returns:
        (str): File name of corresponding topology. Could be psf or prmtop.
    """
    rep = filename.split('/')[-2]
    gen = filename.split('/')[-3]

    if rootdir is None:
        rootdir = "/%s" % os.path.join(*os.path.abspath(filename).split('/')[:-4])

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

def load_trajectory(filename, **kwargs):
    """
    Loads a trajectory with the correct topology and format based on
    the filename. Also aligns to the reference structure using the
    appropriate reference selection.

    Args:
        filename (str): The reimaged trajectory file to load
        molid (int): Molecule ID to load trajectory into, or None for new molid
        stride (int): Stride to load, defaults to 1
        config (str): Path to configfile to parse for following args

        For manually specifying many options:
        rootdir (str): Root directory of the sampling run
        aselref (atomsel): Reference atom selection to align to
        psfref (str): Atom selection strings for PSFs / normal resids
        prmref (str): Atom selection strings for prmtops / altered resids
        frame (int): Single frame to load, or None for all frames, or (beg,end)
        topology (str): Manual topology choice to use, or None for autodetect

    Returns:
        (int): VMD molecule ID of loaded and aligned trajectory
    """
    # Process options that wouldn't be in a config file
    topology = kwargs.get("topology", None)
    frame = kwargs.get("frame", None)
    stride = kwargs.get("stride", 1)
    # Set up alignment stuff. Prefer to use config file
    if kwargs.get("config"):
        if isinstance(kwargs.get("config"), str):
            config = ConfigParser(interpolation=None)
            config.read(kwargs.get("config"))
        elif isinstance(kwargs.get("config"), ConfigParser):
            config = kwargs.get("config")
        else:
            raise ValueError("%s is not a string or ConfigParser"
                             % kwargs.get("config"))
        if topology is None:
            topology = get_topology(filename,
                                    rootdir=config["system"]["rootdir"])

        # Load reference topology and get relevant atom selection
        ref = config["system"]["reference"]
        psfref = config["system"]["canonical_sel"]
        prmref = config["system"]["refsel"]

        if "psf" in ref:
            refid = molecule.load("psf", ref,
                                  "pdb", ref.replace("psf", "pdb"))
            aselref = atomsel(psfref, molid=refid)
        elif "prmtop" in ref:
            refid = molecule.load("parm7", ref,
                                  "crdbox", ref.replace("prmtop", "inpcrd"))
            aselref = atomsel(prmref, molid=refid)
        else:
            raise ValueError("Unknown type of reference '%s'" % ref)

    else: # Legacy... delete TODO
        if topology is None:
            topology = get_topology(filename, kwargs.get("rootdir"))

        aselref = kwargs.get("aselref", None)
        psfref = kwargs.get("psfref", None)
        prmref = kwargs.get("prmref", None)

    if kwargs.get("molid") is None:
        mid = molecule.load("psf" if "psf" in topology else "parm7", topology)
    else:
        mid = int(kwargs.get("molid"))

    # Load the trajectory in
    fmt = get_trajectory_format(filename)
    if frame is None:
        molecule.read(mid, fmt, filename, waitfor=-1, skip=stride)
    elif isinstance(frame, int):
        molecule.read(mid, fmt, filename, beg=frame, end=frame, waitfor=-1)
    elif len(frame) == 2:
        molecule.read(mid, fmt, filename, beg=frame[0], end=frame[1],
                      waitfor=-1, skip=stride)
    else:
        raise ValueError("I don't understand loading frames: %s" % frame)

    # Align, if desired
    if aselref is None:
        return mid
    framsel = atomsel(psfref if "psf" in topology else prmref, molid=mid)

    for frame in range(molecule.numframes(mid)):
        molecule.set_frame(mid, frame)
        framsel.update()
        try:
            atomsel("all", molid=mid, frame=frame).move(framsel.fit(aselref))
        except ValueError:
            print("Can't align the following molecules:\n%s\n\n%s"
                  % (", ".join(molecule.get_filenames(mid)),
                     ", ".join(molecule.get_filenames(refid))))
            quit(1)

    if kwargs.get("config"): # Clean up reference molecule
        molecule.delete(refid)
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

def get_prodfiles(generation, rootdir, new=False, equilibration=False):
    """
    Gets the sorted list of all production files for a given adaptive
    sampling run. Supports legacy non-stripped and non-equilibrated
    reimaged trajectories.

    Args:
        generation (int): Latest generation to load
        rootdir (str): Root directory for sampling run
        new (bool): Only return production files from this generation, if True
        equilibration (bool): If equilibrated trajectory should be included

    Returns:
        (list of str): Production files, inorder
    """
    prefix = "Eq1" if equilibration else "Eq6"

    prodfiles = []
    for gen in range(generation if new else 1, generation+1):
        rpath = os.path.join(rootdir, "production", str(gen), "*")
        pfs = glob(os.path.join(rpath, "Reimaged_strip_%s_*.nc" % prefix))

        # Fallpack to previous non-stripped reimaging
        if not len(pfs):
            pfs = glob(os.path.join(rpath, "Reimaged_%s_*.nc" % prefix))
        prodfiles.extend(sorted(pfs, key=lambda x: int(x.split('/')[-2])))

    return prodfiles

#==============================================================================

def load(filename):
    """
    Python 2/3 compatible load function
    """
    return pickle.load(open(filename, 'rb'), encoding="latin-1")

#==============================================================================

def dump(thing, filename, overwrite=False):
    """
    Provide load and dump together for simplicity
    """
    pickle.dump(thing, open(filename, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    # Change permissions to be read only
    if not overwrite:
        os.chmod(filename, stat.S_IRUSR)

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
    Don't forget to truncate the amount of label data to match
    the actual number of production files!

    Args:
        truelabels (list of ndarray): Cluster labels
        length (int): Length for labels to use
        lag (int): Lag time, in frames, for MSM

    Returns:
        (MarkovStateModel): the model
    """
    from msmbuilder.msm import MarkovStateModel
    msm = MarkovStateModel(lag_time=lag,
                           reversible_type="transpose",
                           ergodic_cutoff="off",
                           prior_counts=0.000001)

    msm.fit([t[:length] for t in truelabels])
    return msm

#==============================================================================

def get_rmsd_to(meansfile, coords):
    """
    Returns the RMSD from each cluster in meansfile to the given
    atom selection.

    Args:
        meansfile (str): Path to pickled cluster means object
        coords (ndarray natoms x 3): Coordinates to get RMSD to

    Returns:
        (dict str->float): Cluster label and RMSD to it
    """
    rmsds = {}
    means = load(meansfile)
    for cl, dat in means.items():
        if len(dat) != len(coords):
            raise ValueError("Different natoms between cluster mean and sel!")
        rmsds[cl] = np.sqrt(np.mean(np.square(dat-coords)))
    return rmsds

#==============================================================================

def get_mfpt_from_solvent(msm, sinks):
    """
    Gets the mean first passage time from the identified solvent cluster (based
    on population) to the sinks. Returns whichever passage time is fastest to
    the sinks for each cluster in msm. Uses 10ns for lag time in MSM.

    Args:
        msm (MarkovStateModel): The MSM to compute on
        sinks (list of int): Cluster labels for bound states

    Returns:
        Fastest time to get from solvent to any cluster in sinks
    """
    from msmbuilder.tpt import mfpts
    data = mfpts(msm, sinks=None)[:][sinks]
    return np.min(data, axis=0)

#==============================================================================

def get_frame(cluster, dataset, nligs):
    """
    Picks a frame corresponding to the given cluster.

    Args:
        cluster (int): The cluster to sample
        dataset (list of ndarray): Clustered dataset
        nligs (int): Number of ligands in the system

    Returns:
        (str, int, int): Path to the trajectory file and frame index
            within that file corresponding to the given cluster, and
            the index of the ligand belonging to the cluster
    """
    # Randomly choose a frame where a ligand is in this cluster
    frames = {k:v for k, v in {i:np.ravel(np.where(c == cluster)) \
              for i, c in enumerate(dataset)}.items() \
              if len(v)}
    clustindex = int(random.choice(list(frames.keys())))
    frameindex = int(random.choice(frames[clustindex]))
    fileindex = int(clustindex / nligs)

    # Get the index of the ligand in the cluster
    ligidx = clustindex % nligs

    return (fileindex, frameindex, ligidx)

#==========================================================================
