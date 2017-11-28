"""
Methods to analyze the amount of sampling in a simulation
"""
import numpy as np
from gridData import Grid
from vmd import atomsel

#===============================================================================

def get_box(residue, molid):
    """
    Gets a box around a given residue, excluding hydrogens.
    Box is sized for atom center + VDW radius, on a per-atom basis.
    There's no need to round the box to the nearest grid spacing
    as the integration function finds this.

    Args:
        residue (int): Residue number
        molid (int): VMD molecule ID

    Returns:
        (list of float): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    """
    sel = atomsel("noh and residue %d" % residue, molid)
    radius = np.asarray(sel.get("radius"))
    result = []
    for coordinate in ["x", "y", "z"]:
        minval = np.min(np.asarray(sel.get(coordinate)) - radius)
        maxval = np.max(np.asarray(sel.get(coordinate)) + radius)
        result.append([minval, maxval])

    return result

#===============================================================================

def get_indices(residue, molid, density):
    """
    Gets indices into grid that match coordinates of given residue.

    Args:
        residue (int): Residue number
        molid (int): VMD molecule ID
        density (Grid): Density map to index into

    Returns:
        (list of int): [[xmin, xmax], [ymin, ymax], [zmin,zmax]]
    """
    indices = []
    for aid, axis in enumerate(get_box(residue, molid)):
        minval = np.min(np.where(density.edges[aid] > axis[0])) - 1
        maxval = np.max(np.where(density.edges[aid] < axis[1])) + 1
        indices.append([minval, maxval])
    return indices

#===============================================================================

def integrate(residue, molid, density):
    """
    Sums the values of the grid covering the given residue.
    First finds the grid indices corresponding to the residue box,
    erring on the side of a larger box (minimum box enclosing residue).

    Args:
        residue (int): Residue number
        molid (int): VMD molecule ID
        density (Grid): gridData Grid containing ligand density to integrate

    Returns:
        (float): Sum of grid squares residsue covers
    """

    indices = get_indices(residue, molid, density)
    # Use numpy fancy indexing to get a view into the grid
    total = np.sum(density.grid[indices[0][0]:indices[0][1],
                                indices[1][0]:indices[1][1],
                                indices[2][0]:indices[2][1]])

    return total

#===============================================================================

def get_subgrid(residue, molid, density):
    """
    Returns a subgrid centered around the given residue. No interpolation
    is performed

    Args:
        residue (int): Residue number
        molid (int):  VMD molecule ID
        density (Grid): Grid to subsample

    Returns:
        (Grid): Subgrid covered by residue
    """
    indices = get_indices(residue, molid, density)
    # Use numpy fancy indexing to get a view into the grid
    subgrid = density.grid[indices[0][0]:indices[0][1],
                           indices[1][0]:indices[1][1],
                           indices[2][0]:indices[2][1]]
    edges = [density.edges[i][e[0]:e[1]] for i, e in enumerate(indices)]

    return Grid(grid=subgrid, edges=edges)

#===============================================================================
