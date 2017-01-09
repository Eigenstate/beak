"""
Contains functionality for loading project-specific datasets
"""
import os
import pickle
from . import vmdfunctions as bk
from msmbuilder.lumping import PCCAPlus
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

def b2ar_10dalps(generation):
    filenames = []
    identifiers = []

    if gethostname() == "platyrhynchos":
        prefix = "/home/robin/Work/Projects/thesis/sherlock/rbetz/b2ar_TEN_dalps/"
    else:
        prefix = "/scratch/PI/rondror/rbetz/b2ar_TEN_dalps/"
    tf = prefix + "prep/inp01_b2ar_10dalps_hmr.psf"

    for gen in range(1, generation+1):
        news = sorted(glob(os.path.join(prefix, "production", str(gen),
                                        "*", "Reimaged_*.nc")),
                      key=lambda x: int(x.split('/')[-2]))
        filenames.extend(news)
        identifiers.extend("10dalpG%d-%d" % (gen, i) for i in range(1,len(news)+1))

    return filenames, identifiers, tf, prefix

#==============================================================================

def dor(generation):
    filenames = []
    identifiers = []

    if gethostname() == "platyrhynchos":
        prefix = "/home/robin/Work/Projects/thesis/sherlock/rbetz/DOR/3ligs/"
    else:
        prefix = "/scratch/PI/rondror/rbetz/DOR/3ligs/"
    topology = prefix + "prep/inp01_4rwd_3ligs.psf"
    skelly = prefix + "production/%d/%d/Reimaged_Eq6_*_skip_1.nc"

    for gen in range(1, generation+1):
        news = sorted(glob(os.path.join(prefix, "production", str(gen),
                                        "*", "Reimaged_*.nc")),
                      key=lambda x: int(x.split('/')[-2]))
        filenames.extend(news)
        identifiers.extend("dorG%d-%d" % (gen, i) for i in range(1,len(news)+1))

    return filenames, identifiers, topology, prefix

#==============================================================================

def b2test(rep, generation):
    filenames = []
    identifiers = []

    if gethostname() == "platyrhynchos":
        prefix = "/home/robin/Work/Projects/thesis/sherlock/rbetz/tests_b2ar/50ns_%d/" % rep
        tf = "/home/robin/Work/Projects/thesis/sherlock/rbetz/b2ar10_rs/inp01_b2ar_10dalps_hmr.psf"
    else:
        prefix = "/scratch/PI/rondror/rbetz/tests_b2ar/50ns_%d/" % rep
        tf = "/scratch/PI/rondror/rbetz/b2ar10_rs/inp01_b2ar_10dalps_hmr.psf"
    skelly = prefix + "production/%d/%d/Reimaged_Eq6_*_skip_1.nc"

    for gen in range(1, generation+1):
        news = sorted(glob(os.path.join(prefix, "production", str(gen),
                                        "*", "Reimaged_*.nc")),
                      key=lambda x: int(x.split('/')[-2]))
        filenames.extend(news)
        identifiers.extend("b2G_%d_%d-%d" % (rep, gen, i) for i in range(1,len(news)+1))

    return filenames, identifiers, tf, prefix

#==============================================================================

def b2ar10_rs(generation):
    filenames = []
    identifiers = []

    if gethostname() == "platyrhynchos":
        prefix = "/home/robin/Work/Projects/thesis/sherlock/rbetz/b2ar10_rs/"
    else:
        prefix = "/scratch/PI/rondror/rbetz/b2ar10_rs/"
    tf = prefix + "inp01_b2ar_10dalps_hmr.psf"
    skelly = prefix + "production/%d/%d/Reimaged_Eq6_*_skip_1.nc"

    for gen in range(1, generation+1):
        news = sorted(glob(os.path.join(prefix, "production", str(gen),
                                        "*", "Reimaged_*.nc")),
                      key=lambda x: int(x.split('/')[-2]))
        filenames.extend(news)
        identifiers.extend("b2G%d-%d" % (gen, i) for i in range(1,len(news)+1))

    return filenames, identifiers, tf, prefix

#==============================================================================

def dor10_rs(generation):
    filenames = []
    identifiers = []

    if gethostname() == "platyrhynchos":
        prefix = "/home/robin/Work/Projects/thesis/sherlock/rbetz/DOR10_rs/"
    else:
        prefix = "/scratch/PI/rondror/rbetz/DOR10_rs/"
    tf = prefix + "inp02_4rwd_10ligs.psf"
    skelly = prefix + "production/%d/%d/Reimaged_Eq6_*_skip_1.nc"

    for gen in range(1, generation+1):
        news = sorted(glob(os.path.join(prefix, "production", str(gen),
                                        "*", "Reimaged_*.nc")),
                      key=lambda x: int(x.split('/')[-2]))
        filenames.extend(news)
        identifiers.extend("dorG%d-%d" % (gen, i) for i in range(1,len(news)+1))

    return filenames, identifiers, tf, prefix
#==============================================================================

def load_dataset(dataset, generation, stride, ligands):

    # Load data, topology, get ligand residues
    fn, ids, tf, prefix = dataset(generation)
    topo = bk.load_topology(fn[0], tf)
    molids = bk.load_data(fn, ids, tf, stride)
    molids = sorted(molids.values())
    ligs = bk.get_ligand_residues(molids[0], topo, ligands)

    # Find and load the msm
    msm = pickle.load(open(os.path.join(prefix, "production", str(generation), "mmsm_G%d.pkl" % generation)))
    tica = pickle.load(open(os.path.join(prefix, "production", str(generation), "testing.tica.pkl")))
    clust = pickle.load(open(os.path.join(prefix, "production", str(generation), "testing.cluster.pkl")))

    # Stride it correctly
    tica = [t[::stride] for t in tica]
    clust = [c[::stride] for c in clust]

    # Show the clusters for next generation
    mins, clustl = bk.get_msm_clusters(molids, msm, clust, ligs)
    bk.show_clusters(mins[:10], clustl)

    return mins, msm, clust, tica, ligs, topo

#==============================================================================
