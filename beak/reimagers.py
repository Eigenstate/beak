#!/usr/bin/env python
"""
Does reimaging, either as a standalone python script or
an importable function
"""
from __future__ import print_function
from glob import glob
import os
import re
import subprocess
import sys
import itertools
try:
    from vmd import molecule, atomsel
    atomsel = atomsel.atomsel
except ImportError:
    import vmd
    import molecule
    from atomsel import atomsel

#==============================================================================

# pylint: disable=missing-docstring,invalid-name
def group_output(inputset):
    """
    Groups the integers in input set into ranges
    in a string parseable by parseSelection
    """
    # Find ranges using itertools
    def ranges(i):
        for _, b in itertools.groupby(enumerate(i),
                                      lambda x_y: x_y[1]-x_y[0]):
            b = list(b)
            yield b[0][1], b[-1][1]
    l = list(ranges(inputset))

    # Put tuples together into a passable list
    result = ""
    for i in l:
        if i[0] == i[1]:
            result += "%d," % i[0]
        else:
            result += "%d-%d," % (i[0]-1, i[1]+1)
    return result[:-1]
# pylint: enable=missing-docstring,invalid-name

#==============================================================================

def check_empty(filename):
    """
    Checks if the netcdf file contains frames.

    Args:
        filename (str): Path to the netcdf file to check
    Returns:
        (bool): True if there were no frames
    """
    if not os.path.isfile(filename):
        return False
    checker = subprocess.Popen(["ncdump", "-h", filename],
                               stdout=subprocess.PIPE)
    output = checker.stdout.read()
    r = re.compile(r"\(\d+ currently\)")
    r2 = re.compile(r"\d+")
    match = r.search(output)
    if match is None:return True
    m2 = r2.search(match.group())
    if m2 is None: return True
    return not bool(int(m2.group()))

#==============================================================================

def get_protein_residues(psf):
    """
    Gets the prmtop ids of the resids of the first chain
    of the protein.

    Args:
        psf (str): Path to the psf file
    Returns: (str) String of residues for anchor
    """
    molid = molecule.load('psf', psf)
    ace = min(atomsel('resname ACE').get('residue'))
    nma = min(atomsel('resname NMA').get('residue'))
    return "%d-%d" % (ace, nma)
    fragment = set(atomsel('pfrag 1').get('fragment')).pop()
    print(set(atomsel('fragment %d' % fragment).get('resname')))
    residues = [x for x in set(atomsel('fragment %d' % fragment).get('residue'))]
    #residues = [ x+1 for x in set(atomsel('protein or resname ACE NMA').get('residue')) ]
    residues.sort()
    molecule.delete(molid)
    return group_output(residues)

#==============================================================================

def reimage(psf, revision, skip, alleq, align, stripmask=None):
    """
    Reimages all Prod_[0-9]*.nc files in a given directory
    into one Prod_all_reimaged.nc file, with the protein at
    the center

    Args:
        psf (str): Path to the psf file, used to identify protein
        revision (str): The revision to reimage
        skip (int): Offset
        alleq (bool): Whether or not to include all equilibration setps
        align (bool): Whether or not to also align the trajectory
        stripmask (str): AMBER selection to remove from reimaged trajectory

    Raises:
        IOError if the topology file is not present
        IOError if the replicate directory is not present or empty
        ValueError if the cpptraj call fails
    """
    revision = str(revision)
    # Error checking
    if not os.path.isfile(psf):
        raise IOError("%s not a valid file" % psf)
    if not os.path.isdir("production/" + revision):
        raise IOError("production/%s not a valid directory" % revision)
    if not os.path.isdir("equilibration/" + revision):
        raise IOError("%s not a valid directory" % revision)

    # Go into the folder n collect filenames
    dirs = [name for name in os.listdir("production/" + revision) if \
            os.path.isdir(os.path.join("production", revision, name))]
    if not dirs:
        raise IOError("No replicates found in directory %s"
                      % os.path.join("production", revision))

    for replicate in dirs:
        reimage_single_dir(psf, replicate, revision, skip, alleq, align,
                           stripmask)

#==============================================================================

def reimage_single_dir(psf, replicate, revision, skip, alleq, align,
                       stripmask=None):

    # Make em strings
    replicate = str(replicate)
    revision = str(revision)

    # Enumerate production files
    proddir = os.path.join("production", revision, replicate)
    prods = sorted(glob(os.path.join(proddir, "Prod_[0-9]*.nc")),
                   key=lambda x: int(x.replace(os.path.join(proddir, "Prod_"),
                                               "").replace(".nc", "")))
    if not len(prods):
        print("NO production simulation in Rev %s Rep %s" % (revision, replicate))
        return

    # Get number of last production file
    lastnum = int(prods[-1].replace(os.path.join(proddir, "Prod_"), "").replace(".nc", ""))
    # If output file already exists, continue
    if alleq:
        ofile = os.path.join(proddir, "Reimaged_Eq1_to_%d_skip_%s.nc" % (lastnum, skip))
        ofile2 = os.path.join(proddir, "Reimaged_strip_Eq1_to_%d_skip_%s.nc" % (lastnum, skip))
    else:
        ofile = os.path.join(proddir, "Reimaged_Eq6_to_%d_skip_%s.nc" % (lastnum, skip))
        ofile2 = os.path.join(proddir, "Reimaged_strip_Eq6_to_%d_skip_%s.nc" % (lastnum, skip))
    if os.path.isfile(ofile) and os.path.isfile(ofile2):
        print("EXISTS reimaged file for Rev %s Rep %s" % (revision, replicate))
        return
    else:
        if alleq:
            if stripmask:
                rems = glob(os.path.join("production", revision, replicate,
                                         "Reimaged_strip_Eq1_to_*_skip_%s.nc" % skip))
            else:
                rems = glob(os.path.join("production", revision, replicate,
                                         "Reimaged_Eq1_to_*_skip_%s.nc" % skip))
        else:
            rems = glob(os.path.join("production", revision, replicate,
                                     "Reimaged_Eq6_to_*_skip_%s.nc" % skip))
        # Delete reimaged files that are older than last modified production trajectory
        for r in rems:
            if os.path.getmtime(r) > os.path.getmtime(prods[-1]):
                print("Removing: %s" % r)
                sys.stdout.flush()
                os.remove(r)

    # Now write cpptraj input
    tempfile = open(os.path.join(proddir, "tempfile"), 'w')

    if align:
        tempfile.write("reference %s parm %s [ref]\n" % (psf.replace("psf", "inpcrd"), psf))

    # equilibration written 8x more frequently so downsample
    if alleq:
        # Find if we're MSMing with separate directory or not
        eqdir = os.path.join("equilibration", revision)
        if os.path.isdir(os.path.join(eqdir, replicate)):
            eqdir = os.path.join(eqdir, replicate)
        eqs = sorted(glob(os.path.join(eqdir, "Eq_[0-5]*.nc")),
                     key=lambda x: int(x.replace(os.path.join(eqdir, "Eq_"),
                                                 "").replace(".nc", "")))
        for e in eqs:
            tempfile.write("trajin %s 1 last %d\n" % (e, int(skip)*8))

    # Last equilibration in
    if not check_empty(os.path.join(proddir, "Eq_6.nc")):
        tempfile.write("trajin %s 1 last %d\n" % (os.path.join(proddir, "Eq_6.nc"), int(skip)*8))
    else:
        prods = [] # Don't write out production files if no Eq_6 present
        ofile = ofile.replace("_to_%d_skip" % lastnum, "_to_Eq5_skip")

    # Read in production data, reimaged
    for p in prods:
        if "Reimaged" in p:
            continue
        if not check_empty(p):
            tempfile.write("trajin %s\n" % p)

    protein_residues = get_protein_residues(psf)

    tempfile.write("center origin (:%s)\n" % protein_residues)
    tempfile.write("image origin center\n")

    if align:
        tempfile.write("rms toRef ref [ref] @CA\n")

    if stripmask is not None:
        ofile = ofile.replace("Reimaged_", "Reimaged_strip_")
        tempfile.write("strip (%s) parmout %s\n"
                       % (stripmask, psf.replace(".psf", "_stripped.prmtop")))
    tempfile.write("trajout %s offset %s\n" % (ofile, skip))
    tempfile.write("go\n")
    tempfile.close()

    return subprocess.call("%s/bin/cpptraj -p %s -i %s/tempfile" %
                           (os.environ['AMBERHOME'], psf, proddir),
                           shell=True)

#==============================================================================

