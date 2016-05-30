"""
TransferRateAnalyzer

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
import math
import matplotlib.pyplot as plt
import numpy

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class TransferRateAnalyzer(object):
    """
    Analyzes a trajectory for number of molecules of a specified type that
    go in or out of a region. Provides both the number of molecules in
    the region over time as well as a transfer rate in and out.

    Attributes:
        data (list of TrajectorySet): Raw trajectories to analyze
        _calc (dict str->numpy 2D array): Analyzed data for all stuff,
            one data column per replicate in the array
        times (dict str->numpy 1D array): Time in ps per frame for each
            trajectory, keyed by name
        colors (dict str->str): Color for each trajectory when plotting,
            keyed by name
        _ref (dict str->float): Reference structure calculation for each
            trajectory, keyed by name
    """

    #==========================================================================

    def __init__(self, data=None, xmin=None, ymin=None, zmin=None,
                 xmax=None, ymax=None, zmax=None):
        self.data = data

        # Prompt for any number of trajectories
        uinput = ""
        self.data = []
        try:
            self.data.extend(data)
        except:
            self.data = [ data ]

        self.times = {}
        self.colors = {}
        self._calc = {}
        self._refs = {}
        if not self.data:
            self.data = []
            while uinput != "EOF":
                uinput = raw_input("Type EOF to stop prompting for trajectories > ")
                self.data.append(TrajectorySet())

    #==========================================================================

    def analyze(self):
        """
        Conducts the bulk of the analysis. By calling analyze on each
        individual trajectory.

        Returns:
            dict str -> numpy array: A dictionary of data to plot, where a
                descriptive string links to a n x 2 numpy array containing
                x and y values to plot
        """
        for dat in self.data:
            (self._calc[dat.name], self._refs[dat.name]) = self._analyze_one(dat)
            self.times[dat.name] = dat.times
            self.colors[dat.name] = dat.color
        return self._calc

    #==========================================================================

    def _analyze_one(self, trajset):
        """
        Analyzes a single trajectory set to populate one entry in the
        _calc array. Should be overruled in each child Analyzer to do
        whatever specific analysis is needed.

        Args:
            trajset (TrajectorySet): Individual trajectory to analyze
        Returns:
            numpy array: Calculated data for this trajectory set
            float: Reference value for this trajectory set
        """
        pass

    #==========================================================================

    def plot(self, title=None, xlabel=None, ylabel=None, smoothing=5):
        """
        Creates a nice plot of the data returned from analysis. If 
        the _calc array is not populated yet, will do the calculation

        Args:
            title (str): Title of the plot. If not provided, will prompt
            xlabel (str): x axis label. If not provided, will prompt
            ylabel (str): y axis label. If not provided, will prompt
            smoothing (int): Window size for smoothing

        Returns:
            matplotlib figure: Graph that can be displayed or saved

        Raises:
            ValueError if a data array has more than 2 columns
        """
        if not self._calc:
            print("Performing calculation...")
            self.analyze()

        # Prompt for plot info if not set
        if title is None:
            title = raw_input("What should the plot title be? > ")
        if xlabel is None:
            xlabel = raw_input("What should the plot xlabel be? > ")
        if ylabel is None:
            ylabel = raw_input("What should the plot ylabel be? > ")

        (figure, axes) = plt.subplots()

        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)

        for label in self._calc:
            # Plot each replicate
            data = self._calc[label]
            i = 0
            for dat in data:
                if i==0: l=label
                else: l=""
                axes.plot(self.times[label][:len(dat)],
                          sliding_mean(dat, window=smoothing),
                          color=self.colors[label],
                          linewidth=1.0, label=l)
                i = 1
            # Plot reference value
            axes.plot([0,self.times[label][-1]],
                      [self._refs[label], self._refs[label]],
                       linestyle='--', linewidth=1.0,
                       color=self.colors[label])
            print("REFERENCE = %f" % self._refs[label])
        axes.legend()
        #axes.set_xlim([0, self.times[

        return figure

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def sliding_mean(data_array, window=5):
    """
    Smooths an array of data with the given window size

    Args:
        data_array (numpy array): The 1D data array to smooth
        window (int): Size of the smoothing window

    Returns:
        numpy array or list: The smoothed data set

    Raises:
        ValueError if the data array isn't a numpy array or list
        ValueError if the data array is not one dimensional
    """

    if not isinstance(data_array, (numpy.ndarray, list)):
        raise ValueError("data_array must be numpy array or list")
    if data_array.ndim != 1:
        raise ValueError("data_array must be one dimensional!")

    new_list = numpy.empty((len(data_array)))
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list[i] = avg

    return new_list

#==============================================================================

def get_threshold_percent(array, threshold):
    """
    Returns the percentage of an array that is below a given threshold

    Args:
        array (list or numpy array): The array to evaluate
        threshold (float or int): The value to check if array elements
            are strictly less than

    Returns:
        float: The percentage of the array (range 0->100) that is below
            the threshold

    Raises:
        ValueError if the data array isn't a numpy array or list
        ValueError if the data array isn't one-dimensional
    """
    if not isinstance(array, (numpy.ndarray, list)):
        raise ValueError("data_array must be numpy array or list")
    if data.array.ndim != 1:
        raise ValueError("data_array must be one dimensional!")

    a = array[array < threshold]
    return float(a)/float(len(array))*100.0

#==============================================================================

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
    return math.sqrt(d)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

