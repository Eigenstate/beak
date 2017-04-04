"""
Contains useful utilities for running MSMs
"""

import os
import h5py
import numpy as np

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

    h5f = h5py.File(output, 'w-') # w- means fail on existence
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

