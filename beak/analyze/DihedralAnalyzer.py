"""
DihedralAnalyzer

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

from __future__ import print_function
from . import Analyzer

import numpy as np
import math
from vmd import atomsel

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class DihedralAnalyzer(Analyzer):

    #==========================================================================

    def __init__(self, data=None, a1=None, a2=None, a3=None, a4=None):

        super(DihedralAnalyzer, self).__init__(data)

        # Prompt for selection
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

        if not (self.a1 or self.a2 or self.a3 or self.a4):
            self.a1 = input("What's the first atom selection? > ")
            self.a2 = input("What's the second atom selection? > ")
            self.a3 = input("What's the third atom selection? > ")
            self.a4 = input("What's the fourth atom selection? > ")

    #==========================================================================

    def _analyze_one(self, trajset):
        """
        Analyzes a single trajectory set

        Args:
            trajset (TrajectorySet): One trajectory set to analyze

        Returns:
            list of 1D numpy arrays: A list of a n x 1 numpy arrays, with one
                replicate in each array. This accounts for varying number of
                frames per replicate.
            float: Dihedral angle for the requested selection in this traj

        Raises:
            ValueError if trajset isn't a TrajectorySet
            ValueError if a selection is empty
        """

        if not isinstance(trajset, TrajectorySet):
            raise ValueError("Not a trajectory set")

        data = []
        for rep in trajset.trajectories:
            sel1 = atomsel(self.a1, molid=int(rep))
            sel2 = atomsel(self.a2, molid=int(rep))
            sel3 = atomsel(self.a3, molid=int(rep))
            sel4 = atomsel(self.a4, molid=int(rep))
            array = np.empty((rep.numFrames()))
            # Loop through all frames
            for i in range(rep.numFrames()):
                rep.setFrame(i)
                sel1.update()
                sel2.update()
                sel3.update()
                sel4.update()

                # Check for empty selection
                if len(sel1) != 1:
                    raise ValueError("Wrong selection: %s" % self.a1)
                if len(sel2) != 1:
                    print(sel2.get('name'))
                    raise ValueError("Wrong selection: %s" % self.a2)
                if len(sel3) != 1:
                    raise ValueError("Wrong selection: %s" % self.a3)
                if len(sel4) != 1:
                    raise ValueError("Wrong selection: %s" % self.a4)

                array[i] = calculate_dihedral_angle(sel1, sel2, sel3, sel4)
            data.append(array)

        # Calculate reference value
        sel1 = atomsel(self.a1, molid=int(trajset.reference))
        sel2 = atomsel(self.a2, molid=int(trajset.reference))
        sel3 = atomsel(self.a3, molid=int(trajset.reference))
        sel4 = atomsel(self.a4, molid=int(trajset.reference))

        # Check for empty selection
        if len(sel1) != 1:
            raise ValueError("Wrong selection: %s" % self.a1)
        if len(sel2) != 1:
            raise ValueError("Wrong selection: %s" % self.a2)
        if len(sel3) != 1:
            raise ValueError("Wrong selection: %s" % self.a3)
        if len(sel4) != 1:
            raise ValueError("Wrong selection: %s" % self.a4)
        reference = calculate_dihedral_angle(sel1, sel2, sel3, sel4)

        return (data, reference)

    #==========================================================================

    def plot(self, title=None, xlabel="Time (ns)", ylabel=None, smoothing=5):
        """
        Creates a nice plot of the data returned from analysis

        Args:
            title (str): Title of the plot. If not provided, will prompt
            xlabel (str): x axis label. If not provided, will prompt
            ylabel (str): y axis label. If not provided, will prompt

        Returns:
            matplotlib figure: Graph that can be displayed or saved

        Raises:
            ValueError if a data array has more than 2 columns
        """
        
        return super(DihedralAnalyzer, self).plot(title, xlabel, ylabel, smoothing)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def calculate_dihedral_angle(a1, a2, a3, a4):
    """
    Calculates a dihedral angle from 4 atoms, in order.

    Args:
        a1-4 (atomsel): Atom selection for each diheral
    Returns:
        (float) The dihedral angle
    Raises:
        ValueError if more than one atom is in a selection
    """
    # Mine gives different results. Use evaltcl
    #from VMD import evaltcl
    #command = "measure dihed {%d %d %d %d}" % (a1.get('index')[0],
    #                                           a2.get('index')[0],
    #                                           a3.get('index')[0],
    #                                           a4.get('index')[0])
    #return float(evaltcl(command))

    # Calculate the dihedral angle
    a1x = a1.get('x')[0]
    a1y = a1.get('y')[0]
    a1z = a1.get('z')[0]
    a2x = a2.get('x')[0]
    a2y = a2.get('y')[0]
    a2z = a2.get('z')[0]
    a3x = a3.get('x')[0]
    a3y = a3.get('y')[0]
    a3z = a3.get('z')[0]
    a4x = a4.get('x')[0]
    a4y = a4.get('y')[0]
    a4z = a4.get('z')[0]

    v1x = a2x-a1x
    v1y = a2y-a1y
    v1z = a2z-a1z
    v2x = a2x-a3x
    v2y = a2y-a3y
    v2z = a2z-a3z
    v3x = a3x-a4x
    v3y = a3y-a4y
    v3z = a3z-a4z

    v1Xv2xv2Xv3 = (v1y * v2z - v1z * v2y) * (v2y * v3z - v2z * v3y) + \
                  (v2x * v1z - v1x * v2z) * (v3x * v2z - v2x * v3z) + \
                  (v1x * v2y - v2x * v1y) * (v2x * v3y - v3x * v2y)

    v1Xv2 = (v1y * v2z - v1z * v2y) * (v1y * v2z - v1z * v2y) + \
            (v2x * v1z - v1x * v2z) * (v2x * v1z - v1x * v2z) + \
            (v1x * v2y - v2x * v1y) * (v1x * v2y - v2x * v1y)

    v2Xv3 = (v2y * v3z - v2z * v3y) * (v2y * v3z - v2z * v3y ) + \
            (v3x * v2z - v2x * v3z) * (v3x * v2z - v2x * v3z ) + \
            (v2x * v3y - v3x * v2y) * (v2x * v3y - v3x * v2y )

    cosdihe = v1Xv2xv2Xv3 / math.sqrt(v1Xv2 * v2Xv3)
    if cosdihe > 1.0:  cosdihe = 1.0
    if cosdihe < -1.0: cosdihe = -1.0

    sign = v1x * (v2y * v3z - v3y * v2z) - v1y * (v2x * v3z - v3x * v2z) \
         + v1z * (v2x * v3y - v3x * v2y);

    dihedral = math.acos(cosdihe)
    #if sign > 0 and dihedral < 0: dihedral = dihedral*-1.0
    #if sign < 0 and dihedral > 0: dihedral = dihedral*-1.0
    return math.degrees(dihedral)
