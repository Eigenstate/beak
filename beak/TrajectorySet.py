"""
Trajectory loader

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
import vmd
import os
import readline
from glob import glob
from Molecule import Molecule
from atomsel import atomsel

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class TrajectorySet(object):
    """
    A collection of molecular dynamics trajectories.
    Includes a translated psf file, a reference structure,
    and any number of replicates.

    Attributes:
        name (str): A short name describing the set
        psf (str): Path to psf file with the topology
        directory (str): Path to the directory with the trajectories
        ligand (str): Residue name of the ligand
        trajectories (list of Molecule): Molecule for loaded replicates
        reference (Molecule): Molecule for reference structure
        save_frequency (float): Frequency file was written to, in ps
        times (list of floats): Time in ps when each loaded frame was written
        stride (int): Stride for reading from data file
        color (str): Color to graph this replicate set in
        reimaged (bool): Whether to load the reimaged single glob trajectory
            instead of individual Prod_[0-9]*.nc files
    """

    #==========================================================================

    def __init__(self, name=None, psf=None, directory=None, ligand=None,
                 stride=None, save_frequency=None, color=None, reimaged=True):
        self.name = name
        self.psf = psf
        self.directory = directory
        self.ligand = ligand
        self.color = color
        self._stride = stride
        self._save_frequency = save_frequency
        self.reimaged = reimaged

        self.trajectories = []
        self.reference = Molecule()
        self.times = []

        # Ask user what to load, then load
        self._prompt_for_paths()
        self._load_production_data()
        self._load_reference_data()
        #self._align_trajectories()

    #==========================================================================

    def _load_reference_data(self):
        '''
        Loads reference structure matching name of input psf
        '''
        self.reference.load(filename=self.psf, filetype='psf')

        #if os.path.isfile(self.psf.replace("_trans.psf", ".pdb")):
        #    print("LOADING TRANS")
        #    self.reference.load(filename=self.psf.replace("_trans.psf", ".pdb"),
        #                        filetype='pdb')
        #else:
        self.reference.load(filename=self.psf.replace(".psf", ".pdb"),
                            filetype='pdb')
        self.reference.rename(self.name + "_ref")

    #==========================================================================

    def _load_production_data(self):
        '''
        Loads all replicates found in a top-level directory
        matching the specified psf file.

        Raises:
            IOError: if a reimaged file isn't found and we are looking for it
        '''
       
        thedir = os.path.abspath(self.directory)
        dirs = [name for name in os.listdir(thedir) if \
                os.path.isdir(os.path.join(thedir,name))]
        for replicate in dirs:
            sim = Molecule()
            sim.load(os.path.abspath(self.psf), filetype='psf')
            sim.rename("%s_%d" % (self.name, int(sim)))

            if self.reimaged:
                p = glob(os.path.join(self.directory, replicate, "Reimaged*.nc"))[0]
                #p = os.path.join(self.directory, replicate, "Prod_all_reimaged.nc")
                if not p or not os.path.isfile(p):
                    raise IOError("No reimaged file %s" % p)
                sim.load(p, filetype='netcdf', step=self._stride, waitfor=-1)
            else:
                # Read in production data, reimaged
                prods = glob(os.path.join(self.directory, replicate,
                             "Prod_[0-9]*.nc"))
                prods.sort()
                for p in prods:
                    if "reimaged" in p: continue
                    sim.load(p, filetype='netcdf', step=self._stride, waitfor=-1)
            self.trajectories.append(sim)

        # Calculate times
        time_per_frame = self._stride * self._save_frequency
        max_len = max([ r.numFrames() for r in self.trajectories])
        self.times = [i*time_per_frame/1000. for i in range(max_len)]

    #==========================================================================

    def _prompt_for_paths(self):
        '''
        Prompts user for info about the molecule to load
        '''
        # Autocomplete directories at prompt
        def complete(text, state):
            return (glob(text+'*')+[None])[state]

        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(' \t\n;')
        readline.set_completer(complete)

        # Prompt for stuff
        if not self.psf:
            self.psf = raw_input("Where is the psf file? > ").strip()
        if not self.directory:
            self.directory = raw_input("Where is the production directory? > ").strip()
        if not self.name:
            self.name = raw_input("What is the name of this trajectory set? > ").strip()
        if not self.ligand:
            self.ligand = raw_input("What is the resname of the ligand? > ").strip()
        if not self.color:
            self.color = raw_input("What color should this set be when graphed? > ").strip()
        if not self._save_frequency:
            self._save_frequency = float(raw_input("What is the save interval, in ps? > "))
        if not self._stride:
            self._stride = int(raw_input("What should the stride for loading be? > "))

    #==========================================================================

    def _align_trajectories(self):
        """
        Aligns all loaded trajectories to the reference
        structure. Modifies them in-place.
        """
        rsel = atomsel('protein', molid=int(self.reference))
        for s in self.trajectories:
            # Align frames to reference structure
            sel = atomsel('protein', molid=int(s))
            for frame in range(s.numFrames()):
                s.setFrame(frame)
                sel.update()
                t = sel.fit(rsel)
                atomsel('all', molid=int(s)).move(t)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
