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
from math import sqrt

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
    if len(x) != 3 or len(y) != 3:
        raise ValueError("These aren't valid coordinates!")

    d = (x[0]-y[0])*(x[0]-y[0]) + \
        (x[1]-y[1])*(x[1]-y[1]) + \
        (x[2]-y[2])*(x[2]-y[2])
    return sqrt(d)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_min_distance(sel1, sel2):
    """
    Returns the minimum distance between two atom selections.

    Args:
        sel1 (atomsel): First atom selection
        sel2 (atomsel): Second atom selection

    Returns:
        float: The minimum distance between the selections
    """
    sel1x = sel1.get('x')
    sel1y = sel1.get('y')
    sel1z = sel1.get('z')
    sel2x = sel2.get('x')
    sel2y = sel2.get('y')
    sel2z = sel2.get('z')

    return min([get_distance((sel1x[i],sel1y[i],sel1z[i]), \
                             (sel2x[j],sel2y[j],sel2z[j])) \
                              for i in range(len(sel1)) \
                              for j in range(len(sel2))])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
