"""
Analyzer

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

class Analyzer(object):
    """
    Analyzes a trajectory. All analyzer objects should inherit from this.
    Provides an analyze() method that does the bulk of the analysis, and
    has its own TrajectorySet. Includes some useful functions that are
    commonly used in analysis.

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

    def __init__(self, data=None):
        self.data = data
        self.tableau20 = [(31, 119, 180), (174, 199, 232), (214, 39, 40),
                          (255, 152, 150), (44, 160, 44), (255, 127, 14),
                          (255, 187, 120), (152, 223, 138), (148, 103, 189),
                          (197, 176, 213), (140, 79, 75), (196, 156, 148),
                          (227, 119, 194), (247, 182, 210), (127, 127, 127),
                          (199, 199, 199), (188, 189, 34), (219, 219, 141),
                          (23, 190, 207), (158, 218, 229)]
        for i in range(len(self.tableau20)):
            r, g, b = self.tableau20[i]
            self.tableau20[i] = (r / 255., g / 255., b / 255.)


        # Prompt for any number of trajectories
        uinput = ""
        self.data = data
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

        pass

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
        if not title:
            title = raw_input("What should the plot title be? > ")
        if not xlabel:
            xlabel = raw_input("What should the plot xlabel be? > ")
        if not ylabel:
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
            axes.plot((0,len(self.times)),
                      (self._refs[label], self._refs[label]),
                      '--', linewidth=1.5, color=self.colors[label])
            print("REFERENCE = %f" % self._refs[label])
        axes.legend()

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

