"""
Methods to analyze the amount of sampling in a simulation
"""
import numpy as np
from vmd import atomsel

#===============================================================================

def round_to_half(n):
    """
    Rounds a number to the nearest grid spacing, which is on the 0.5
    """
    correction = 0.5 if n >=0 else -0.5
    return int(n + correction) + correction

#===============================================================================

def get_box(residue, molid):
    """
    Gets a box around a given residue. Box is sized for atom
    center + 1/2 largest VDW radius.

    Args:
        residue (int): Residue number
        molid (int): VMD molecule ID
        precision (float): Round to nearest THIS

    Returns:
        [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    """
    sel = atomsel("residue %d" % residue, molid)
    radius = max(sel.get("radius"))
    result = []
    for coordinate in ["x", "y", "z"]:
        minval = min(sel.get(coordinate)) + radius
        maxval = max(sel.get(coordinate)) + radius
        result.append([round_to_half(minval),
                       round_to_half(maxval)])

    return result

#===============================================================================

def integrate(residue, molid, grid):
    """
    Sums the values of the grid covering the given residue
    """

    indices = []
    for aid, axis in enumerate(get_box(residue, molid)):
        minval = np.min(np.where(grid.edges[aid] > axis[0])) - 1
        maxval = np.max(np.where(grid.edges[aid] < axis[1])) + 1
        indices.append([minval, maxval])

    return indices
#===============================================================================
