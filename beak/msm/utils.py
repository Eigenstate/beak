"""
Contains useful utilities for running MSMs
"""
from __future__ import print_function
import os
import h5py
import numpy as np
from glob import glob
from Dabble import VmdSilencer
try:
    from vmd import atomsel, molecule, vmdnumpy
    atomsel = atomsel.atomsel
except:
    import vmd
    import molecule, vmdnumpy
    from atomsel import atomsel

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

