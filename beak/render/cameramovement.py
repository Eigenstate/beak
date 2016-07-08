"""
CameraMovement

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
from beak.rendertools.quaternion import Quaternion
from numpy import *
import vmd, trans, display

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class CameraMovement(object):
    """
    Moves the camera smoothly from one position and rotation to another.
    Uses spherical linear interpolation to move slowly at the ends

    Attributes:
        render (function handle): The function to call to render. Takes index
            as a parameter
        index (int): Frame index being rendered
        nsteps (int): Number of steps to take
        startpos (3 vector): Starting camera position
        endpos (3 vector): Ending camera position
        startrot (3x3 matrix): Starting camera rotation
        endrot (3x3 matrix): Ending camera rotation
        molids (list of int): Molecule IDs to apply movement to
    """

    #==========================================================================

    def __init__(self, render, index, nsteps, startpos, endpos,
                 startrot, endrot, startzoom, endzoom, molids):
        self.render = render
        self.index = index
        self.nsteps = nsteps
        self.startpos = startpos
        self.endpos = endpos
        self.startrot = Quaternion(startrot)
        self.endrot = Quaternion(endrot)
        self.startzoom = startzoom
        self.endzoom = endzoom
        self.molids = molids

        self._step = 0
        self._pos = startpos
        self._rot = Quaternion(startrot)
        self._zoom = startzoom
        self.omega = arccos(sum([startpos[i]*endpos[i] for i in range(3)])/ \
                         ( sum([startpos[i]*startpos[i] for i in range(3)]) * \
                          sum([endpos[i]*endpos[i] for i in range(3)]) ))
        self.om2 = arccos( 1./(startzoom*endzoom) )

    #==========================================================================
    
    def go(self):
        """
        Does the interpolation and calls render each time.
        Sets the orientation of the camera for all molids

        Returns the next index to render
        """
        
        for i in range(self.nsteps):
            self._take_step()
            print(self._zoom)
            for m in self.molids:
                trans.set_rotation(m, self._rot.tr().reshape(1,16).tolist()[0])
                trans.set_center(m, self._pos)
                trans.set_scale(m, self._zoom)
            display.update()
            self.render(self.index)
            self.index += 1

        return self.index

    #==========================================================================

    def _take_step(self):
        """
        Takes a step in the interpolation.
        Updates _pos and _rot internally.
        """
        t = float(self._step)/float(self.nsteps)

        # Do the position here
        p0 = ( sin((1-t)*self.omega) )/( sin(self.omega) )
        p1 = ( sin(t*self.omega) )/( sin(self.omega) )

        self._pos = [p0*self.startpos[i] + p1*self.endpos[i] for i in range(3)]

        # Now the scale
        z0 = ( sin((1-t)*self.om2) )/( sin(self.om2) )
        z1 = ( sin(t*self.om2) )/( sin(self.om2) )
        self._zoom = z0 * self.startzoom + z1 * self.endzoom

        # Use quaternion library to handle the rotation
        self._rot = Quaternion.interp(self.startrot, self.endrot, t)

        self._step += 1

    #==========================================================================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
