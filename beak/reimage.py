#!/usr/bin/env python

from glob import glob
import os
import subprocess
import sys
import vmd, molecule
from atomsel import atomsel

def get_protein_residues(psf):
    """
    Gets the prmtop ids of the resids of the first chain
    of the protein.

    Args:
        psf (str): Path to the psf file
    Returns: (str) String of residues for anchor, in prmtop numbers
    """
    molid = molecule.load('psf', psf)
    resids = set(atomsel('pfrag 1', molid=molid).get('residue'))
    return ",".join([str(x+1) for x in resids])

def reimage(prmtop, psf, thedir):
    """
    Reimages all Prod_[0-9]*.nc files in a given directory
    into one Prod_all_reimaged.nc file, with the protein at
    the center

    Args:
        prmtop (str): Path to the topology file
        psf (str): Path to the psf file, used to identify protein
        thedir (str): Path to directory containing all replicates

    Raises:
        IOError if the topology file is not present
        IOError if the replicate directory is not present or empty
        ValueError if the cpptraj call fails
    """
    # Error checking
    if not os.path.isfile(prmtop):
        raise IOError("%s not a valid file" % prmtop)
    if not os.path.isdir(thedir):
        raise IOError("%s not a valid directory" % thedir)

    # Go into the folder n collect filenames
    dirs = [name for name in os.listdir(thedir) if \
            os.path.isdir(os.path.join(thedir,name))]
    if not dirs:
        raise IOError("No replicates found in directory %s" % thedir)

    for replicate in dirs:
        tempfile = open('tempfile', 'w')

        # Read in production data, reimaged
        prods = glob(os.path.join(thedir, replicate,
                     "Prod_[0-9]*.nc"))
        prods.sort()
        for p in prods:
            if "reimaged" in p: continue
            tempfile.write("trajin %s offset 10\n" % p)

        protein_residues = get_protein_residues(psf)

        tempfile.write("autoimage anchor :%s\n" % protein_residues)
        tempfile.write("trajout %s/Prod_all_reimaged.nc\n" % os.path.join(thedir, replicate))
        tempfile.write("go\n")

        os.system("%s/bin/cpptraj -p %s < tempfile" % (os.environ['AMBERHOME'], prmtop))
        #if subprocess.check_call("%s/bin/cpptraj -p %s < tempfile" % (os.environ['AMBERHOME'], prmtop)):
        #    raise ValueError("Error converting replicate %s" % replicate)

if __name__ == "__main__":
    # Process args
    if len(sys.argv) != 4:
        print("Usage: %s <prmtop> <psf> <folder>" % sys.argv[0])
        quit(1)

    prmtop = os.path.abspath(sys.argv[1])
    psf = os.path.abspath(sys.argv[2])
    thedir = os.path.abspath(sys.argv[3])
    reimage(prmtop, psf, thedir)

