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

#==============================================================================

class Sampler(object):

    def __init__(self, configfile, generation, stride, firstgen=1,
                 sampstride=1):
        """
        Creates a sampler explorer object.

        Args:
            configfile (str): Configuration file to read generation and
                file info from
            generation (int): Last generation to read in
            stride (int): Amount to stride read in trajectories
            firstgen (int): First generation to read in (1)
            sampstride (int): Only read in every Nth sampler
        """
        self.config = RawConfigParser()
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.num_ligands = int(self.config["system"]["num_ligands"])
        self.generation = generation
        self.firstgen = firstgen

        assert(generation <= int(self.config["production"]["generation"]))
        self.nreps = int(self.config["model"]["samplers"])
        self.dir = self.config["system"]["rootdir"]
        if gethostname() == "platyrhynchos":
            print("DIR: %s" % self.dir)
            self.dir = self.dir.replace("/scratch/PI/rondror/rbetz/",
                                        "/home/robin/Work/Projects/thesis/sherlock/")
            print("DIR: %s" % self.dir)
        self.name = self.config["system"]["jobname"]
        self.stride = stride
        self.sampstride = sampstride

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
        self.scores = []
        if os.path.isfile(os.path.join(substr, "mmsm_scores.pkl")):
            print("Loading scores")
            self.scores = load(os.path.join(substr, "mmsm_scores.pkl"))
        else:
            print("Couldn't find scores: %s"
                  % os.path.join(substr, "mmsm_scores.pkl"))

        # Handle saved pre-msm clusters, as naming scheme changed...
        if os.path.isfile(os.path.join(substr, "testing.mcluster.pkl")):
            mclust = load(os.path.join(substr, "testing.mcluster.pkl"))
            mclust = [c[::self.stride] for c in mclust]
            clust = [c[::self.stride] for c in clust]
        else:
            mclust = [c[::self.stride] for c in clust]
            clust = None

        # Stride it correctly
        self.tica = [t[::self.stride] for t in tica]

        # Based on first generation read in, trim others
        offset = sum(len(glob(os.path.join(self.dir, "production",
                                            str(gen), "*",
                                            "Reimaged_*.nc"))) \
                     for gen in range(1, self.firstgen)) * self.num_ligands

        print("OFFSET: %d" % offset)

        tica = tica[offset:]
        mclust = mclust[offset:]
        if clust is not None:
            clust = clust[offset:]
        assert len(tica) == len(mclust)

        # Handle sample striding. Can't use array slicing since need a few in a row
        self.tica = []
        self.mclust = []
        self.clust = []
        for i in [_ for _ in range(len(tica)/self.num_ligands) if _ % self.sampstride==0]:
            self.tica.extend(tica[i*self.num_ligands:(i+1)*self.num_ligands])
            self.mclust.extend(mclust[i*self.num_ligands:(i+1)*self.num_ligands])
            if self.clust is not None:
                self.clust.extend(clust[i*self.num_ligands:(i+1)*self.num_ligands])

    #==========================================================================

    def __del__(self):
        for m in self.molids:
            molecule.delete(m)
        del self.tica
        del self.mclust
        if self.clust is not None:
            del self.clust

    #==========================================================================

    def load_features(self):
        """
        Loads features on demand
        """
        for gen in range(self.firstgen, self.generation+1):
            self.features.extend(self.config["system"]["featurized"] % gen)

    #==========================================================================

    def align_trajectories(self):
        """
        Aligns all loaded trajectories
        """
        print("Aligining")
        refsel = atomsel("protein and not same fragment as resname %s"
                         % " ".join(self.ligands),
                         molid=self.molids[0], frame=0)

        for m in self.molids:
            for frame in range(molecule.numframes(m)):
                psel = atomsel("protein and not same fragment as resname %s"
                               % " ".join(self.ligands),
                               molid=m, frame=frame)
                tomove = psel.fit(refsel)
                atomsel("all", molid=m, frame=frame).move(tomove)

    #==========================================================================

    def load_trajectories(self):
        """
        Loads all of the trajectories
        """
        molids = []
        filenames = []

        counter = -1
        for gen in range(self.firstgen, self.generation+1):
            for rep in range(1, self.nreps+1):
                # Stride trajectories regardless of generation
                counter += 1
                if counter % self.sampstride != 0:
                    continue

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
                    counter -= 1
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

