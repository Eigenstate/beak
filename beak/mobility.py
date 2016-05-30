#!/usr/bin/env python

import vmd
import numpy as np
import molecule
from atomsel import atomsel

def align(molid):
    m = Molecule(id=molid)
    rsel = atomsel('protein', molid=molid)
    sel = atomsel('protein', molid=molid)
    for i in range(m.numFrames()):
        m.setFrame(i)
        sel.update()
        t = sel.fit(rsel)
        atomsel('all', molid=molid).move(t)

def get_rmsd(molid, outfile):

    refsel = atomsel('protein', molid=molid, frame=0)
    framesel = atomsel('protein', molid=molid)
    array = np.empty((molecule.numframes(molid)))

    for i in range(molecule.numframes(molid)):
        molecule.set_frame(molid, i)
        framesel.update()
        array[i] = framesel.rmsd(refsel)

    np.savetxt(outfile, array)
    #return array

