"""
Metrics

Author: Robin Betz
Copyright (C) 2015 Robin Betz

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330
Boston, MA 02111-1307, USA.
"""
from scipy.spatial import distance
from vmd import atomsel, vmdnumpy
import numpy as np

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_distance(x, y):
    """
    Gets the distance between two coordinates in Cartesian space

    Args:
        x (array of 3 floats): First coordinate
        y (array of 3 floats): Second coordinate


    Returns:
        float: The Euclidean distance between the points

    Raises:
        ValueError if the arrays are incorrectly sized
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != 3 or len(y) != 3:
        raise ValueError("These aren't valid coordinates!")

    return np.sqrt(np.sum( (x-y)**2 ))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_min_distance(sel1, sel2, molid, frame):
    """
    Returns the minimum distance between two atom selections.

    Args:
        sel1 (atomsel): First atom selection
        sel2 (atomsel): Second atom selection

    Returns:
        float: The minimum distance between the selections
    """

    if isinstance(sel1, str):
        sel1 = atomsel(sel1, molid=molid, frame=frame)
    if isinstance(sel2, str):
        sel2 = atomsel(sel2, molid=molid, frame=frame)

    xyz = vmdnumpy.timestep(molid, frame)
    a1 = xyz[sel1.get("index")]
    a2 = xyz[sel2.get("index")]

    return np.min(distance.cdist(a1, a2))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

