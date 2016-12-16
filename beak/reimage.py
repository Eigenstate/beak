#!/usr/bin/env python

from glob import glob
import os
import re
import subprocess
import sys
import itertools
import vmd, molecule
from atomsel import atomsel

#====
def groupOutput(inputset):
    """
    Groups the integers in input set into ranges
    in a string parseable by parseSelection
    """
    # Find ranges using itertools
    def ranges(i):
        for a,b in itertools.groupby(enumerate(i),
                                     lambda (x,y): y-x):
            b = list(b)
            yield b[0][1], b[-1][1]
    l = list(ranges(inputset))

    # Put tuples together into a passable list
    result = ""
    for i in l:
        if i[0] == i[1]: result += "%d," % i[0]
        else: result += "%d-%d," % (i[0]-1,i[1]+1)
    return result[:-1]

#==============================================================================

def check_empty(filename):
    """
    Checks if the netcdf file contains frames.

    Args:
        filename (str): Path to the netcdf file to check
    Returns:
        (bool): True if there were no frames
    """

    checker = subprocess.Popen(["ncdump", "-h", filename],
                               stdout=subprocess.PIPE)
    output = checker.stdout.read()
    r = re.compile("\(\d+ currently\)")
    r2 = re.compile("\d+")
    match  = r.search(output)
    if match is None: return True
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
    print set(atomsel('fragment %d' % fragment).get('resname'))
    residues = [ x for x in set(atomsel('fragment %d' % fragment).get('residue')) ]
    #residues = [ x+1 for x in set(atomsel('protein or resname ACE NMA').get('residue')) ]
    residues.sort()
    return groupOutput(residues)

def reimage(psf, revision, skip, alleq, align):
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
                      % os.path.join("production", revision, name))

    for replicate in dirs:
        # Enumerate production files
        prods = [ x.replace("%s/Prod_"% os.path.join("production", revision, replicate),"").replace(".nc","") for x in 
                  glob( "%s/Prod_[0-9]*.nc" % os.path.join("production", revision, replicate) ) if "imaged" not in x ]
        prods.sort(key=int)
        if not len(prods):
            print("NO production simulation in Rev %s Rep %s" % (revision, replicate))
            continue

        # If output file already exists, continue
        if alleq:
            ofile = os.path.join("production", revision, replicate, "Reimaged_Eq1_to_%s_skip_%s.nc" % (prods[-1], skip))
        else:
            ofile = os.path.join("production", revision, replicate, "Reimaged_Eq6_to_%s_skip_%s.nc" % (prods[-1], skip))
        if os.path.isfile(ofile):
            print("EXISTS reimaged file for Rev %s Rep %s" % (revision, replicate))
            continue
        else:
            if alleq:
                rems = glob(os.path.join("production", revision, replicate, "Reimaged_Eq1_to_*_skip_%s.nc" % skip))
            else:
                rems = glob(os.path.join("production", revision, replicate, "Reimaged_Eq6_to_*_skip_%s.nc" % skip))
            for r in rems:
                num = r.split('_')[3]
                if num <= prods[-1]:
                    print("Removing: %s" % r)
                    os.remove(r)

        # Now write cpptraj input
        tempfile = open('production/%s/tempfile' % os.path.join(revision, replicate), 'w')

        if align:
            tempfile.write("reference %s parm %s [ref]\n" % (psf.replace("psf","inpcrd"), psf))

        # equilibration written 8x more frequently so downsample
        if alleq:
            eqs = [ x.replace("equilibration/%s/Eq_"%revision,"").replace(".nc","") for x in
                    glob("equilibration/%s/Eq_[0-5]*.nc" % revision) if "imaged" not in x ]
            eqs.sort(key=int)
            for e in eqs:
                tempfile.write("trajin equilibration/%s/Eq_%s.nc 1 last %d\n" % (revision, e, int(skip)*8))

        # Last equilibration in 
        tempfile.write("trajin production/%s/%s/Eq_6.nc 1 last %d\n" % (revision, replicate, int(skip)*8))

        # Read in production data, reimaged
        for p in prods:
            if "Reimaged" in p: continue
            pfile = "%s/Prod_%s.nc" % (os.path.join("production", revision, replicate), p)
            if not check_empty(pfile):
                tempfile.write("trajin %s\n" % pfile)

        protein_residues = get_protein_residues(psf)
        
        tempfile.write("center origin (:%s)\n" % protein_residues)
        tempfile.write("image origin center\n")
        
        if align:
            tempfile.write("rms toRef ref [ref] @CA\n") 

        tempfile.write("trajout %s offset %s\n" % (ofile, skip))
        #tempfile.write("trajout %s/Reimaged_200_to_%s_skip_%s.nc start 1000 offset %s\n" % (os.path.join("production", revision, replicate),prods[-1],skip,skip))
        tempfile.write("go\n")
        tempfile.close()

        subprocess.call("%s/bin/cpptraj -p %s -i %s/tempfile" % (os.environ['AMBERHOME'], psf, os.path.join("production", revision, replicate)), shell=True)



if __name__ == "__main__":
    # Process args
    if len(sys.argv) != 6:
        print("Usage: %s <psf> <revision> <skip> <alleq> <align>" % sys.argv[0])
        quit(1)

    psf = os.path.abspath(sys.argv[1])
    revision = sys.argv[2]
    skip = sys.argv[3]
    alleq = (sys.argv[4] == "True")
    align = (sys.argv[5] == "True")
    reimage(psf, revision, skip, alleq, align)

