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
            or None if the residue is out of bounds on the grid
    """
    indices = []

    for aid, axis in enumerate(get_box(residue, molid)):
        # Sanity check it's in the box at all
        if axis[0] > np.max(density.edges[aid]) or \
           axis[1] < np.min(density.edges[aid]):
            return None

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
        residue (int, or 3 tuple): Residue number, or list of indices
            [[xmin, xmax], [ymin, ymax], [zmin,zmax]]
        molid (int): VMD molecule ID
        density (Grid): gridData Grid containing ligand density to integrate

    Returns:
        (float): Sum of grid squares residue covers
    """

    if len(residue) > 1:
        indices = residue
    else:
        indices = get_indices(residue, molid, density)

    # If it's out of bounds, just return 0
    if indices is None:
        return 0.

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

    Raises:
        ValueError: if residue is not covered by the grid
    """
    indices = get_indices(residue, molid, density)

    if indices is None:
        raise ValueError("Residue is not covered by the grid!")

    # Use numpy fancy indexing to get a view into the grid
    subgrid = density.grid[indices[0][0]:indices[0][1],
                           indices[1][0]:indices[1][1],
                           indices[2][0]:indices[2][1]]
    edges = [density.edges[i][e[0]:e[1]] for i, e in enumerate(indices)]

    return Grid(grid=subgrid, edges=edges)

#===============================================================================

def integrate_no_doublecounting(selection, molid, density):
    """
    Sums the values of the grid covering the given selection. Grid cells
    are not double counted even if multiple molecules in the selection
    cover them.

    Use a MaskedArray view onto the original grid to simplify summation.

    Args:
        selection (str): Atom selection to integrate
        molid (int): VMD molecule ID
        density (Grid): Grid to integrate

    Returns:
        (float): Sum of grid squares covered by atom selection
    """
    sumgrid = np.ma.MaskedArray(data=density.grid, copy=False, fillvalue=0.0,
                                mask=np.ones(density.grid.shape))
    mask = np.ma.getmask(sumgrid)
    residues = set(atomsel(selection, molid=molid).get("residue"))

    for res in residues:
        indices = get_indices(res, molid, density)
        if indices is not None:
            mask[indices[0][0]:indices[0][1],
                 indices[1][0]:indices[1][1],
                 indices[2][0]:indices[2][1]] = False

    return np.sum(sumgrid)

#===============================================================================

def integrate_each_residue(molid, density):
    return [integrate_no_doublecounting("residue %d" % r,
                                        molid=molid,
                                        density=density)
            for r in sorted(set(atomsel("all", molid).get("residue")))]

#===============================================================================

def get_control_value(molid, density):
    """
    Returns the value of an integral assuming ligands are distributed
    completely evenly on the grid

    Args:
        selection (str): Atom selection to integrate
        molid (int): VMD molecule ID
        density (Grid): Grid of correct shape. Will not be altered

    Returns:
        (float): Sum of grid squares covered by atom selection, assuming
            constant ligand density
    """

    # Assume ligands distributed completely evenly
    g = density.grid
    g2 = np.copy(density.grid)
    g2.fill(1./len(np.ravel(g2)))

    density.grid = g2
    #result = integrate_no_doublecounting(selection, molid, density)
    result = integrate_each_residue(molid, density)
    density.grid = g
    return result

#===============================================================================
