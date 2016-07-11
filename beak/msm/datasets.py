"""
Contains functionality for loading project-specific datasets
"""
from glob import glob
from socket import gethostname

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        PROJECT-SPECIFIC METHODS                            +
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def b2ar_alprenolol():
    """
    Returns topology, filenames, and identifiers for B2AR-alprenolol
    """
    filenames = []
    identifiers = []

    if gethostname() == "platyrhynchos":
        prefix = "/home/robin/Work/Projects/thesis/sherlock/"
    else:
        prefix = "/scratch/PI/rondror/"
    topology = prefix + "rbetz/C.psf"
    skelly = prefix + "rbetz/pnasc/DESRES-Trajectory_pnas2011a-C-%d-no-water-no-lipid/pnas2011a-C-%d-no-water-no-lipid/pnas2011a-C-%d-no-water-no-lipid-*.dcd"
    for i in range(10):
        for x,g in enumerate(sorted(glob(skelly % (i,i,i)))):
            filenames.append(g)
            identifiers.append("C%d-%d" % (i,x))

    return filenames, identifiers, topology

#==============================================================================
