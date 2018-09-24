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
from multiprocessing import Pool
from vmd import molecule, atomsel

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
        (bool): True if there were no frames or file is missing
    """
    # Say the file is empty if it doesn't exist
    if not os.path.isfile(filename):
        return True

    checker = subprocess.Popen(["ncdump", "-h", filename],
                               stdout=subprocess.PIPE)
    output = checker.stdout.read()
    r = re.compile(r"\(\d+ currently\)")
    r2 = re.compile(r"\d+")
    match = r.search(output.decode("utf-8"))
    if match is None: return True
    m2 = r2.search(match.group())
    if m2 is None: return True
    return not bool(int(m2.group()))

#==============================================================================

def get_protein_residues(topology):
    """
    Gets the prmtop ids of the resids of the first chain
    of the protein.

    Args:
        topology(str): Path to the input topology file
    Returns: (str) String of residues for anchor
    """
    if topology.split(".")[-1] == "prmtop":
        molid = molecule.load('parm7', topology)
        field = "resid"
    else:
        molid = molecule.load('psf', topology)
        field = "residue"

    ace = min(atomsel('resname ACE', molid).get(field))
    nma = min(atomsel('resname NMA NME', molid).get(field))

    molecule.delete(molid)
    return "%d-%d" % (ace, nma)

    # LEGACY
    #fragment = set(atomsel('pfrag 1').get('fragment')).pop()
    #print(set(atomsel('fragment %d' % fragment).get('resname')))
    #residues = [x for x in set(atomsel('fragment %d' % fragment).get('residue'))]
    #residues = [ x+1 for x in set(atomsel('protein or resname ACE NMA').get('residue')) ]
    #residues.sort()
    #molecule.delete(molid)
    #return group_output(residues)

#==============================================================================

def reimage(basedir, psf, revision, skip, alleq, align, stripmask=None):
    """
    Reimages all Prod_[0-9]*.nc files in a given directory
    into one Prod_all_reimaged.nc file, with the protein at
    the center

    Args:
        basedir (str): Base directory to find all replicates
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
    os.chdir(basedir)
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

    p = Pool(int(os.environ.get("SLURM_NTASKS", "1")))
    p.starmap(reimage_single_dir, [(psf, replicate, revision, skip, alleq,
                                    align, stripmask) for replicate in dirs])
    p.close()

#==============================================================================

def reimage_single_dir(topology, replicate, revision, skip, alleq, align,
                       stripmask=None):

    # Make em strings
    replicate = str(replicate)
    revision = str(revision)

    # Enumerate production files
    proddir = os.path.join("production", revision, replicate)
    prods = sorted(glob(os.path.join(proddir, "Prod_[0-9]*.nc")),
                   key=lambda x: int(x.replace(os.path.join(proddir, "Prod_"),
                                               "").replace(".nc", "")))
    prods = [x for x in prods if "Reimaged" not in x]

    if check_empty(os.path.join(proddir, "Eq_6.nc")) and \
       check_empty(os.path.join(proddir, "Eq_unrestrained.nc")):
        print("NO production simulation in Rev %s Rep %s" % (revision, replicate))
        return None

    # Get number of last production file
    if len(prods):
        lastnum = prods[-1].replace(os.path.join(proddir,
                                                 "Prod_"), "").replace(".nc", "")
    else:
        lastnum = "EqU"

    # Set output file names
    rprefix = "Reimaged_strip" if stripmask else "Reimaged"
    ofile = os.path.join(proddir, "%s_Eq%s_to_%s_skip_%s.nc"
                         % (rprefix, "1" if alleq else "U",
                            lastnum, skip))

    # If reimaged output exists, only continue if latest production
    # file has been updated since then. Delete all matching reimaged
    # files that are older so directories aren't cluttered up.
    if alleq:
        rems = glob(os.path.join(proddir, "%s_Eq1_to_*_skip_%s.nc"
                                 % (rprefix, skip)))
    # Handle both old and new Eq6_ vs EqU_
    else:
        rems = glob(os.path.join(proddir, "%s_Eq6_to_*_skip_%s.nc"
                                 % (rprefix, skip)))
        rems.extend(glob(os.path.join(proddir, "%s_EqU_to_*_skip_%s.nc"
                                 % (rprefix, skip))))
    for r in rems:
        if os.path.getmtime(r) > os.path.getmtime(prods[-1]):
            print("Removing: %s" % r)
            sys.stdout.flush()
            os.remove(r)

    # Now write cpptraj input
    tempfile = open(os.path.join(proddir, "tempfile"), 'w')

    topoprefix = ".".join(topology.split(".")[:-1])
    if align:
        tempfile.write("reference %s.inpcrd parm %s [ref]\n" % (topoprefix, topology))

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

    # Last equilibration in, handle legacy Eq_6 for now
    if os.path.isfile(os.path.join(proddir, "Eq_6.nc")):
        fn = "Eq_6.nc"
    else:
        fn = "Eq_unrestrained.nc"

    tempfile.write("trajin %s 1 last %d\n" % (os.path.join(proddir, fn),
                                              int(skip)*8))

    # Read in production data, reimaged
    for p in [x for x in prods if not check_empty(x)]:
        tempfile.write("trajin %s\n" % p)

    protein_residues = get_protein_residues(topology)

    tempfile.write("center origin (:%s)\n" % protein_residues)
    tempfile.write("image origin center\n")

    if align:
        tempfile.write("rms toRef ref [ref] @CA\n")

    if stripmask is not None:
        tempfile.write("strip (%s) parmout %s_stripped.prmtop\n"
                       % (stripmask, topoprefix))
    tempfile.write("trajout %s offset %s\n" % (ofile, skip))
    tempfile.write("go\n")
    tempfile.close()

    # Returns 0 on success
    return subprocess.call("%s/bin/cpptraj -p %s -i %s/tempfile" %
                           (os.environ['AMBERHOME'], topology, proddir),
                           shell=True)

#==============================================================================

def reimage_single_mdstep(topology, basedir, skip, alleq,
                          align, stripmask=None):
    """
    Reimages a trajectory that is contained entirely in one directory
    with the "system.psf" naming convention associated with mdstep runs.

    Args:
        topology (str): Topology filename
        basedir (str): Directory containing all files
        skip (int): Stride for outputting frames
        alleq (bool): True to include all the equilibration, too
        align (bool): True to align all frames to first one
        stripmask (str): Amber-style atom selection of atoms to omit
            from the reimaged trajectory

    Returns:
        0 on success
    """
    # Enumerate production files
    prods = sorted(glob(os.path.join(basedir, "Prod_[0-9]*.nc")),
                   key=lambda x: int(x.replace(os.path.join(basedir, "Prod_"),
                                               "").replace(".nc", "")))
    prods = [x for x in prods if "Reimaged" not in x]

    if check_empty(os.path.join(basedir, "Eq_6.nc")) and \
       check_empty(os.path.join(basedir, "Eq_unrestrained.nc")):
        print("No production simulation in %s" % basedir)
        return None

    # Get number of last production file
    if len(prods):
        lastnum = prods[-1].replace(os.path.join(basedir,
                                                 "Prod_"), "").replace(".nc", "")
    else:
        lastnum = "EqU"

    # Set output file names
    rprefix = "Reimaged_strip" if stripmask else "Reimaged_"
    ofile = os.path.join(basedir, "%s_Eq%s_to_%s_skip_%s.nc"
                         % (rprefix, "1" if alleq else "U",
                            lastnum, skip))

    # If reimaged output exists, only continue if latest production
    # file has been updated since then. Delete all matching reimaged
    # files that are older so directories aren't cluttered up.
    if alleq:
        rems = glob(os.path.join(basedir, "%s_Eq1_to_*_skip_%s.nc"
                                 % (rprefix, skip)))
    # Handle both old and new Eq6_ vs EqU_
    else:
        rems = glob(os.path.join(basedir, "%s_Eq6_to_*_skip_%s.nc"
                                 % (rprefix, skip)))
        rems.extend(glob(os.path.join(basedir, "%s_EqU_to_*_skip_%s.nc"
                                 % (rprefix, skip))))

    for r in rems:
        if os.path.getmtime(r) < os.path.getmtime(prods[-1]):
            print("Removing: %s" % r)
            sys.stdout.flush()
            os.remove(r)

    # Now write cpptraj input
    tempfile = open(os.path.join(basedir, "tempfile"), 'w')

    topoprefix = ".".join(topology.split(".")[:-1])
    if align:
        tempfile.write("reference %s.inpcrd parm %s [ref]\n"
                       % (topoprefix, topology))

    # equilibration written 8x more frequently so downsample
    if alleq:
        eqs = sorted(glob(os.path.join(basedir, "Eq_[0-5]*.nc")),
                     key=lambda x: int(x.replace(os.path.join(basedir, "Eq_"),
                                                 "").replace(".nc", "")))
        for e in eqs:
            tempfile.write("trajin %s 1 last %d\n" % (e, int(skip)*8))

    # Last equilibration in
    if os.path.isfile(os.path.join(basedir, "Eq_6.nc")):
        fn = "Eq_6.nc"
    else:
        fn = "Eq_unrestrained.nc"
    tempfile.write("trajin %s 1 last %d\n" % (os.path.join(basedir, fn),
                                              int(skip)*8))

    # Read in production data, reimaged
    for p in [x for x in prods if not check_empty(x)]:
        tempfile.write("trajin %s\n" % p)

    protein_residues = get_protein_residues(topology)

    tempfile.write("center origin (:%s)\n" % protein_residues)
    tempfile.write("image origin center\n")

    if align:
        tempfile.write("rms toRef ref [ref] @CA\n")

    if stripmask is not None:
        tempfile.write("strip (%s) parmout %s_stripped.prmtop\n"
                       % (stripmask, topoprefix))
    tempfile.write("trajout %s offset %s\n" % (ofile, skip))
    tempfile.write("go\n")
    tempfile.close()

    # Returns 0 on success
    return subprocess.call("%s/bin/cpptraj -p %s -i %s/tempfile" %
                           (os.environ['AMBERHOME'], topology, basedir),
                           shell=True)

#==============================================================================

def reimage_mdstep(basedir, skip, alleq, align, stripmask=None):
    """
    Reimages all subdirectories of this one, with mdstep conventions
    for file naming.

    Args:
        basedir (str): Base directory to collect replicates from
        skip (int): Offset
        alleq (bool): Whether or not to include all equilibration setps
        align (bool): Whether or not to also align the trajectory
        stripmask (str): AMBER selection to remove from reimaged trajectory

    Raises:
        ValueError if no valid replicate directories with system.psf are found
        ValueError if the cpptraj call fails
    """
    replicate_dirs = [d for d in os.listdir(basedir) \
                      if os.path.isdir(d) and \
                      os.path.isfile(os.path.join(d, "system.psf"))]

    # Error checking
    if not replicate_dirs:
        raise ValueError("No replicate directories in '%s' "
                         "Each replicate directory needs a system.psf to be "
                         "present to be correctly identified" % basedir)

    p = Pool(int(os.environ.get("SLURM_NTASKS", "1")))
    p.starmap(reimage_single_mdstep, [(os.path.join(d, "system.psf"), d,
                                       skip, alleq, align, stripmask)
                                      for d in replicate_dirs])
    p.close()

#==============================================================================
