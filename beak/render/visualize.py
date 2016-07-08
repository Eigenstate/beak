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
import axes, display
from Molecule import MoleculeRep
from beak.TrajectorySet import TrajectorySet 

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def go(reimaged=False):
    """
    Visualizes a trajectory set with standard representations
    """
    # Load data
    data = TrajectorySet(reimaged=reimaged)

    # Turn off display updating while we make changes
    axes.set_location("OFF")
    display.update_off()

    # Set representations for the reference
    proteinref = MoleculeRep(style='NewCartoon', color='ColorID 6',
                             material='Transparent',
                             selection='protein or resname ACE NMA')
    ligandref = MoleculeRep(style='Licorice', color='ColorID 6',
                            material='Transparent',
                            selection='noh and resname %s' % data.ligand)
    data.reference.clearReps()
    data.reference.addRep(proteinref)
    data.reference.addRep(ligandref) 

    # Representations for replicates
    protein = MoleculeRep(style='NewCartoon', color='ColorID 1',
                          selection='protein or resname ACE NMA')
    ligand = MoleculeRep(style='Licorice', color='Type',
                         selection='noh and resname %s' % data.ligand)
    hbonds = MoleculeRep(style='HBonds', color='ColorID 4',
                         selection='within 5 of resname %s' % data.ligand)

    # Set replicate representations
    for r in data.trajectories:
        r.clearReps()
        r.addRep(protein)
        r.addRep(ligand)
        r.addRep(hbonds)

    # Turn the display back on
    display.update_on()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
