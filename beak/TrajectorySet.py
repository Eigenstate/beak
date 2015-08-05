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
from Molecule import *
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
    """

    #==========================================================================

    def __init__(self):
        self.name = ""
        self.psf = ""
        self.directory = ""
        self.ligand = ""
        self.trajectories = []
        self.reference = Molecule()

        # Ask user what to load, then load
        self._prompt_for_paths()
        self._load_production_data()
        self._load_reference_data()
        self._align_trajectories()

    #==========================================================================

    def _load_reference_data(self):
        '''
        Loads reference structure matching name of input psf
        '''
        self.reference.load(filename=self.psf, filetype='psf')
        self.reference.load(filename=self.psf.replace("_trans.psf", ".pdb"),
                            filetype='pdb')
        self.reference.rename(self.name + "_ref")

    #==========================================================================

    def _load_production_data(self):
        '''
        Loads all replicates found in a top-level directory
        matching the specified psf file.
        '''

        for replicate in os.listdir(os.path.abspath(directory)):
            sim = Molecule()
            sim.load(os.path.abspath(self.psf))
            sim.rename("%s_%d" % (self.name, int(sim)))

            # Read in production data, reimaged
            prods = glob(os.path.join(self.directory, replicate,
                         "Prod_[0-9]*.nc"))
            prods.sort()
            for p in prods:
                if "reimaged" in p: continue
                sim.load(p, filetype='netcdf', step=10, waitfor=-1)
            self.trajectories.append(sim)

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
        print("Where is the psf file?")
        self.psf = raw_input()
        print("Where is the production directory?")
        self.directory = raw_input()
        print("What is the name of this trajectory set?")
        self.name = raw_input()
        print("What is the resname of the ligand?")
        self.ligand = raw_input()

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
