"""
RMSDAnalyzer

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
from beak.TrajectorySet import TrajectorySet

import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

import vmd
from atomsel import atomsel
import Molecule

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class MinDistanceAnalyzer(Analyzer):
    """
    Conducts analysis of the minimum distance between any atom in the
    two selections

    Attributes:
        data (array of TrajectorySet): data
        calc (dict str->numpy 2D array): Analyzed data for all stuff,
            one data column per replicate in the array
        sel1 (str): First selection string
        sel2 (str): Second selection string
        times (dict str->numpy 1D array): Time in ps per frame for each
            trajectory, keyed by name
        colors (dict str->str): Color to make each trajectory set when
            plotted, keyed by name
    """

    #==========================================================================

    def __init__(self, data=None, selection1=None, selection2=None):

        super(MinDistanceAnalyzer, self).__init__(data)

        # Prompt for selection
        self.sel1 = selection1
        self.sel2 = selection2
        if (not self.sel1) or (not self.sel2):
            self.sel1 = raw_input("What's the first atom selection? > ")
            self.sel2 = raw_input("What's the second atom selection? > ")

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
            float: Minimum distance between selection in the reference
                structure for this trajectory

        Raises:
            ValueError if trajset isn't a TrajectorySet
            ValueError if a selection is empty
       """

        if not isinstance(trajset, TrajectorySet):
            raise ValueError("Not a trajectory set")

        data = []
        for rep in trajset.trajectories:
            sel1 = atomsel(self.sel1, molid=int(rep))
            sel2 = atomsel(self.sel2, molid=int(rep))
            array = np.empty((rep.numFrames()))
            # Loop through all frames
            for i in range(rep.numFrames()):
                rep.setFrame(i)
                sel1.update()
                sel2.update()

                # Check for empty selection
                if not len(sel1):
                    raise ValueError("Empty selection: %s" % self.sel1)
                if not len(sel2):
                    raise ValueError("Empty selection: %s" % self.sel2)

                array[i] = get_min_distance(sel1, sel2)
            data.append(array)

        # Calculate reference value
        sel1 = atomsel(self.sel1, molid=int(trajset.reference))
        sel2 = atomsel(self.sel2, molid=int(trajset.reference))
        # Check for empty selection
        if not len(sel1):
            raise ValueError("Empty selection: %s" % self.sel1)
        if not len(sel2):
            raise ValueError("Empty selection: %s" % self.sel2)
        reference = get_min_distance(sel1, sel2)

        return (data, reference)

    #==========================================================================

    def plot(self, title=None, xlabel="Time (ns)", ylabel=None):
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
        
        return super(MinDistanceAnalyzer, self).plot(title, xlabel, ylabel)

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


