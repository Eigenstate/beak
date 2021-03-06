"""
Contains methods for counting binding events
"""

import pandas as pd
import numpy as np

#===============================================================================

def get_time_bound(rmsds, threshold, dt, maxtime):
    """
    Obtains the total time in the bound state over time.

    Args:
        rmsds (Pandas DataFrame): RMSD values, with index as time. Must
            have integer index as otherwise things are impossible.
        threshold (float): RMSD value below which ligand is considered bound
        dt (int): Time chunks
        maxtime (int): Maximum time to return

    Returns:
        (Pandas Series): Number of frames with bound, over time, with time
            chunked into dt sized chunks.
    """
    allbound = rmsds[rmsds <= threshold]
    counts = []
    for i in range(0, maxtime, dt):
        counts.append(allbound.loc[i:dt+i].count().sum())
    return pd.Series(counts, index=range(0, maxtime, dt))

#===============================================================================

def get_binding_times_thresholded(rmsds, dhigh, dlow):
    """
    Obtains the times at which binding events occur.
    We define a binding event as going from RMSD > dhigh to
    RMSD < dlow in some time interval dt.

    Args:
        rmsd (Pandas DataFrame): RMSD values, with index as time. Must
            have integer index because otherwise things are impossible.
        dhigh (float): RMSD value above which ligand is considered unbound
        dlow (float): RMSD value below which ligand is considered bound

    Returns:
        (numpy ndarray): List of times at which a binding event occurred

    """
    # Check input data is actually pandas series
    if not isinstance(rmsds, pd.DataFrame):
        raise ValueError("rmsds must be a pandas DataFrame. Got %s instead"
                         % type(rmsds))
    if not isinstance(rmsds.index, pd.Int64Index):
        raise ValueError("rmsds must have integer index. Got %s instead"
                         % type(rmsds.index))

    # Remove data points that are out of range, for simplicity
    alltlow = rmsds[rmsds <= dlow]
    allthigh = rmsds[rmsds >= dhigh]
    btimes = []

    for key in rmsds.keys():

        tlow = alltlow[key].dropna()
        thigh = allthigh[key].dropna()

        # Require starting unbound.
        # This misses binding across adaptive resampling borders.
        # TODO: is it a problem? Superimpose on graphs and check.
        t = thigh.index.min()

        ## Always count the first binding event I find
        #t = 0

        ## But, don't count if we start in the bound pose
        #if tlow.index().min() == rmsds[key].dropna().index().min():
        #    t = thigh.index.min()

        # We travel strictly forward in time, gathering binding events
        # as we go.
        while t < tlow.index.max():
            # Get closest bound time to this unbind, using next value
            bidx = tlow.index.get_loc(t, method="backfill")
            t = tlow.index[bidx]
            btimes.append(t)

            try:
                ubidx = thigh.index.get_loc(t, method="backfill")
                t = thigh.index[ubidx]
            except KeyError: # It doesn't unbind again before the end
                break

    return np.asarray(btimes)

#===============================================================================
