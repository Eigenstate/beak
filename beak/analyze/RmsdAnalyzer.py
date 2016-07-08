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

import vmd
from atomsel import atomsel
import Molecule

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class RmsdAnalyzer(Analyzer):
    """
    Conducts analysis of the RMSD between two or more atom selections

    Attributes:
        data (array of TrajectorySet): data
        sel (str): Selection string to get RMSD of
        calc (dict str->numpy 2D array): Analyzed data for all stuff,
            one data column per replicate in the array
        times (dict str->numpy 1D array): Time in ps per frame for each
            trajectory, keyed by name
        colors (dict str->str): Color for each trajectory when plotting,
            keyed by name
    """
    #==========================================================================

    def __init__(self, data=None, selection=None, refmol=None):
        super(RmsdAnalyzer, self).__init__(data)

        # Prompt for RMSD selection
        self.sel = selection
        if not self.sel:
            self.sel = raw_input("What's the selection to get RMSD of? > ")
        self.refmol = refmol

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
            float: RMSD of reference structure. In this case it's always 0

        Raises:
            ValueError if trajset isn't a TrajectorySet
       """

        if not isinstance(trajset, TrajectorySet):
            raise ValueError("Not a trajectory set")

        data = []
        if not self.refmol:
            refsel = atomsel(self.sel, molid=int(trajset.reference))
        else:
            refsel = atomsel(self.sel, molid=self.refmol)
        print(refsel.get('mass'))
        for rep in trajset.trajectories:
            framesel = atomsel(self.sel, molid=int(rep))
            print(framesel.get('mass'))
            array = np.empty((rep.numFrames()))
            for i in range(rep.numFrames()):
                rep.setFrame(i)
                framesel.update()
                array[i] = framesel.rmsd(refsel)
            data.append(array)

        return (data, 0.0)

    #==========================================================================

    def plot(self, title=None, xlabel=None, ylabel="Time (ns)"):
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

        return super(RmsdAnalyzer, self).plot(title, xlabel, ylabel)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


