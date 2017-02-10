"""
Contains functionality for loading project-specific datasets
"""
import vmd
import molecule
import molrep
import numpy as np
import os
import trans
import vmdnumpy
from atomsel import atomsel
from configparser import RawConfigParser
from glob import glob
from msmbuilder.lumping import PCCAPlus
from msmbuilder.msm import MarkovStateModel
from msmbuilder.tpt import hub_scores
from msmbuilder.utils import load
from socket import gethostname
from VMD import evaltcl, graphics

#==============================================================================

class Sampler(object):

    def __init__(self, configfile, generation, stride):
        """
        Creates a sampler explorer object.

        Args:
            configfile (str): Configuration file to read generation and
                file info from
            generation (int): Last generation to read in
            stride (int): Amount to stride read in trajectories
        """
        self.config = RawConfigParser()
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.num_ligands = int(self.config["system"]["num_ligands"])
        self.generation = generation
        assert(generation <= self.config["production"]["generation"])
        self.nreps = int(self.config["model"]["samplers"])
        self.dir = self.config["system"]["rootdir"]
        if gethostname() == "platyrhynchos":
            print("DIR: %s" % self.dir)
            self.dir = self.dir.replace("/scratch/PI/rondror/rbetz/",
                                        "/home/robin/Work/Projects/thesis/sherlock/")
            print("DIR: %s" % self.dir)
        self.name = self.config["system"]["jobname"]
        self.stride = stride

        # Actually load all the data
        self.molids, self.filenames = self.load_trajectories()
        self.align_trajectories()
        self.features = [] # Only load features if requested

        # Find and load the msm, tica, and clusters
        substr = os.path.join(self.dir, "production", str(self.generation))
        self.mmsm = load(os.path.join(substr, "mmsm_G%d.pkl" % self.generation))
        self.msm = load(os.path.join(substr, "msm_G%d.pkl" % self.generation))
        tica = load(os.path.join(substr, "testing.tica.pkl"))
        clust = load(os.path.join(substr,"testing.cluster.pkl"))
        if os.path.isfile(os.path.join(substr, "mmsm_scores.pkl")):
            self.scores = load(os.path.join(substr, "mmsm_scores.pkl"))
        else:
            self.scores = []


        # Handle saved pre-msm clusters, as naming scheme changed...
        if os.path.isfile(os.path.join(substr, "testing.mcluster.pkl")):
            mclust = load(os.path.join(substr, "testing.mcluster.pkl"))
            self.mclust = [c[::self.stride] for c in mclust]
            self.clust = [c[::self.stride] for c in clust]
        else:
            self.mclust = [c[::self.stride] for c in clust]
            self.clust = None

        # Stride it correctly
        self.tica = [t[::self.stride] for t in tica]

        # Show the clusters for next generation
        #mins, clustl = self.get_msm_clusters(molids, msm, clust, ligs)
        #bk.show_clusters(mins[:10], clustl)
        # Load the features?

    #==========================================================================

    def load_features(self):
        """
        Loads features on demand
        """
        for gen in range(1, self.generation+1):
            self.features.extend(self.config["system"]["featurized"] % gen)

    #==========================================================================

    def align_trajectories(self):
        """
        Aligns all loaded trajectories
        """
        refsel = atomsel("protein and not same fragment as resname %s"
                         % " ".join(self.ligands),
                         molid=self.molids[0], frame=0)

        for m in self.molids:
            for frame in range(molecule.numframes(m)):
                psel = atomsel("protein and not same fragment as resname %s"
                               % " ".join(self.ligands),
                               molid=m, frame=frame)
                tomove = refsel.fit(psel)
                atomsel("all", molid=m, frame=frame).move(tomove)

    #==========================================================================

    def load_trajectories(self):
        """
        Loads all of the trajectories
        """
        molids = []
        filenames = []
        for gen in range(1, self.generation+1):
            for rep in range(1, self.nreps+1):
                psfname = os.path.join(self.dir, "systems",
                                       str(gen), "%d.psf" % rep)
                if os.path.isfile(psfname) or os.path.islink(psfname):
                    a = molecule.load("psf", psfname)
                else:
                    print("NOT A PSF: %s" % psfname)
                    a = molecule.load("psf",
                                      os.path.abspath(self.config["system"]["topologypsf"]))
                g = glob(os.path.join(self.dir, "production", str(gen),
                                      str(rep), "Reimaged_*.nc"))
                assert(len(g)==1 or len(g)==0)
                if len(g) == 0:
                    print("Missing reimaged file for %d, skipping" % rep)
                    continue

                filenames.append(g[0])
                molecule.read(a, "netcdf", g[0], skip=self.stride, waitfor=-1)
                molecule.rename(a, "%s-G%d-r%d" % (self.name, gen, rep))
                molids.append(a)
        return molids, filenames

    #==========================================================================

    def get_ligand_residues(self, molid):
        """
        Gets the VMD resids corresponding to the mdtraj ligands, in order.
        This order is the same as featurization

        Args:
            molid (int): VMD molecule ID to colour
              up ligands
        Returns:
            (list of int): Residue numbers of each ligands, in order
        """
        ligids = sorted(set(atomsel("resname %s" % " ".join(self.ligands)),
                                    molid=molid).get("residue"))
        return ligids

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

