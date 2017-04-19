"""
Contains functionality for loading project-specific datasets
"""
from __future__ import print_function
import os
import numpy as np
import random
from configparser import RawConfigParser
from glob import glob
from msmbuilder.utils import load
from socket import gethostname
from . import utils
#pylint: disable=import-error,no-name-in-module
try:
    from VMD import evaltcl
    import vmd
    import molecule
    import molrep
    import vmdnumpy
    from atomsel import atomsel
except ImportError:
    from vmd import evaltcl, molecule, molrep, vmdnumpy, atomsel
    atomsel = atomsel.atomsel
#pylint: enable=import-error,no-name-in-module

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Sampler(object):
    """
    Sampler that loads in all of the trajectory files
    """

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
        # Set visualization stuff
        evaltcl("color scale method BGR")
        evaltcl("display projection Orthographic")

        self.config = RawConfigParser()
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.num_ligands = int(self.config["system"]["num_ligands"])
        self.generation = generation
        self.firstgen = firstgen

        assert generation <= int(self.config["production"]["generation"])
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
        clust = load(os.path.join(substr, "testing.cluster.pkl"))
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
            print("Striding: %d" % self.stride)
            self.mclust = [c[::self.stride] for c in mclust]
            self.clust = [c[::self.stride] for c in clust]
        else:
            self.mclust = [c[::self.stride] for c in clust]
            self.clust = None

        # Stride it correctly
        self.tica = [t[::self.stride] for t in tica]

        # Based on first generation read in, trim others
        offset = -1*self.num_ligands*len(self.molids)
        self.tica = tica[offset:]
        self.mclust = mclust[offset:]
        if clust is not None:
            self.clust = clust[offset:]

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
# TODO: use utils
        pass

    #==========================================================================

    def align_trajectories(self):
        """
        Aligns all loaded trajectories
        """
        refmol = molecule.load("psf", self.config["system"]["reference"],
                               "pdb", self.config["system"]["reference"].replace("psf", "pdb"))
        print("Aligning")
        refsel = atomsel("(%s) and not same fragment as resname %s"
                         % (self.config["system"]["refsel"],
                            " ".join(self.ligands)),
                         molid=refmol)

        for m in self.molids:
            for frame in range(molecule.numframes(m)):
                psel = atomsel("(%s) and not same fragment as resname %s"
                               % (self.config["system"]["refsel"],
                                  " ".join(self.ligands)),
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
                                      str(rep), "Reimaged_strip_Eq1*.nc"))
                assert len(g) == 1 or len(g) == 0
                if len(g) == 0:
                    g = glob(os.path.join(self.dir, "production", str(gen),
                                          str(rep), "Reimaged_Eq1*.nc"))
                if len(g) == 0:
                    g = glob(os.path.join(self.dir, "production", str(gen),
                                          str(rep), "Reimaged_Eq6*.nc"))
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
        ligids = sorted(set(atomsel("resname %s" % " ".join(self.ligands),
                                    molid=molid).get("residue")))
        return ligids

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ClusterSampler(object):
    """
    Loads and visualizes a representative frame and density for each cluster
    """

    def __init__(self, configfile, generation):
        """
        Creates a sampler explorer object.

        Args:
            configfile (str): Configuration file to read generation and
                file info from
            generation (int): Last generation to read in
        """
        # Set visualization stuff
        evaltcl("color scale method BGR")
        evaltcl("display projection Orthographic")

        self.config = RawConfigParser()
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.num_ligands = int(self.config["system"]["num_ligands"])
        self.generation = generation

        assert generation <= int(self.config["production"]["generation"])
        self.nreps = int(self.config["model"]["samplers"])
        self.dir = self.config["system"]["rootdir"]
        if gethostname() == "platyrhynchos":
            self.dir = self.dir.replace("/scratch/PI/rondror/rbetz/",
                                        "/home/robin/Work/Projects/thesis/sherlock/")
        self.name = self.config["system"]["jobname"]

        # List all of the filenames so we can look them up later
        self.prodfiles = []
        for gen in range(1, self.generation+1):
            # Yields gen/0/Reim, gen/1/Reim, gen/2/Reim, ...
            self.prodfiles.extend(
                sorted(glob(os.path.join(self.dir, "production",
                                         str(gen), "*",
                                         "Reimaged_*.nc")),
                       key=lambda x: int(x.split('/')[-2])))

        # Actually load all the data
        self.molids = []
        self.rmsds = None
        self.load_clusters()

        # Find and load the msm, and clusters
        substr = os.path.join(self.dir, "production", str(self.generation))
        self.mmsm = load(os.path.join(substr, "mmsm_G%d.pkl" % self.generation))
        self.msm = load(os.path.join(substr, "msm_G%d.pkl" % self.generation))
        self.clust = load(os.path.join(substr, "testing.cluster.pkl"))
        self.scores = []
        if os.path.isfile(os.path.join(substr, "mmsm_scores.pkl")):
            print("Loading scores")
            self.scores = load(os.path.join(substr, "mmsm_scores.pkl"))
        else:
            print("Couldn't find scores: %s"
                  % os.path.join(substr, "mmsm_scores.pkl"))

        # Handle saved pre-msm clusters, as naming scheme changed...
        if os.path.isfile(os.path.join(substr, "testing.mcluster.pkl")):
            self.mclust = load(os.path.join(substr, "testing.mcluster.pkl"))
        else:
            self.mclust = self.clust
            self.clust = None

        for gen in range(1, self.generation+1):
            # Yields gen/0/Reim, gen/1/Reim, gen/2/Reim, ...
            self.prodfiles.extend(
                sorted(glob(os.path.join(self.dir, "production",
                                         str(gen), "*",
                                         "Reimaged_*.nc")),
                       key=lambda x: int(x.split('/')[-2])))

    #==============================================================================

    def __del__(self):
        for m in self.molids:
            molecule.delete(m)
        del self.mclust
        if self.clust is not None:
            del self.clust

    #==============================================================================

    def load_clusters(self):
        """
        Loads all of the clusters associated with a generation

        Args:
            dir (str): Path to root directory
            generation (int): Which generation to load

        Returns:
            (list of int): Loaded molids
            (dict int -> float): RMSD of each molid to cluster mean
        """

        with open(os.path.join(self.dir, "clusters", str(self.generation), "rmsds")) as fn:
            lines = fn.readlines()

        rmsds = {int(x.split()[0].strip()): float(x.split()[1].strip()) for x in lines}

        for molname in sorted(glob(os.path.join(self.dir, "clusters",
                                                str(self.generation), "*.mae")),
                              key=lambda x: int(x.split('/')[-1].replace(".mae", ""))):
            nam = molname.split('/')[-1].replace(".mae", "")
            a = molecule.load("mae", molname)
            molecule.rename(a, "%d_%s" % (self.generation, nam))

            # Load the density map
            molecule.read(a, "dx", molname.replace("mae", "dx"), waitfor=-1)
            #evaltcl("mol addrep %d Isosurface 0.1 0 1 1 1" % a)
            molrep.addrep(a, style="Isosurface 0.05 0 0 1 1",
                          color="ColorID 2")
            molrep.set_scaleminmax(a, molrep.num(a)-1, 0, 2.)

            # Get per-atom rmsd
            if self.rmsds is None:
                self.rmsds = rmsds

            atomsel("all", molid=a).set("user", self.rmsds[int(nam)])

            molrep.delrep(a, 0)
            molrep.addrep(a, style="NewRibbons", material="Opaque",
                          color="User", selection="protein and not "
                          "same fragment as (resname %s)"
                          % " ".join(self.ligands))
            if len(self.molids) > 1:
                molrep.set_visible(a, molrep.num(a)-1, False)
            molrep.set_scaleminmax(a, molrep.num(a)-1, 0, 2.)
            molrep.addrep(a, style="Licorice", material="Opaque",
                          selection="noh and same fragment as "
                                    "(resname %s)" % " ".join(self.ligands),
                          color="User")
            molrep.set_scaleminmax(a, molrep.num(a)-1, 0, 2.)
            self.molids.append(a)

        # Align
# These files should all already be aligned to the reference structure when
# they are created.
#        refmol = molecule.load("psf", self.config["system"]["reference"],
#                               "pdb", self.config["system"]["reference"].replace("psf","pdb"))
#        refsel = atomsel("(%s) and not same fragment as resname %s"
#                         % (self.config["system"]["refsel"],
#                            " ".join(self.ligands)),
#                         molid=refmol)
#
#        for m in self.molids:
#            for frame in range(molecule.numframes(m)):
#                psel = atomsel("(%s) and not same fragment as resname %s"
#                               % (self.config["system"]["refsel"],
#                                  " ".join(self.ligands)),
#                               molid=m, frame=frame)
#                tomove = psel.fit(refsel)
#                atomsel("all", molid=m, frame=frame).move(tomove)

    #==========================================================================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class DensitySampler(object):
    """
    Loads up a bunch of densities along with random frames representing
    each cluster
    """

    def __init__(self, configfile, **kwargs):
        """
        Creates a sampler explorer object.

        Args:
            configfile (str): Configuration file to read generation and
                file info from
            filenames (list of str): Production files
            clustdir (str): Directory with cluster dx files
            clusters (str): Cluster data pickle path
            msm (str): MSM pickle path
        """
        # Set visualization stuff
        evaltcl("color scale method BGR")
        evaltcl("display projection Orthographic")

        self.config = RawConfigParser()
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.nligs = int(self.config["system"]["num_ligands"])

        self.generation = kwargs.get("generation")
        self.clustdir = kwargs.get("clustdir")
        self.prodfiles = kwargs.get("prodfiles")

        # Load reference structure for later alignment
        self.reference = self.config["system"]["reference"]
        if "prmtop" in self.reference:
            self.refid = molecule.load("parm7", self.reference,
                    "crdbox", self.reference.replace("prmtop", "inpcrd"))
        elif "psf" in self.reference:
            self.refid = molecule.load("psf", self.reference,
                    "pdb", self.reference.replace("psf", "pdb"))
        self.refsel = self.config["system"]["refsel"]
        self.aselref = atomsel(self.refsel, molid=self.refid)
        self.psfsel = self.config["system"]["canonical_sel"]

        # List all of the filenames so we can look them up later
        self.molids = {}
        self._load_densities()

        # Find and load the msm, and clusters
        self.msm = load(kwargs.get("msm"))
        self.clusters = load(kwargs.get("clusters"))

    #==========================================================================

    def __del__(self):
        for mid in self.molids.values():
            molecule.delete(mid)
        del self.clusters
        del self.msm

    #==========================================================================

    def _load_densities(self):
        """
        Loads all clusters and density map
        """
        for cfile in sorted(glob(os.path.join(self.clustdir, "*.dx")),
                key=lambda x:int(x.split("/")[-1].replace(".dx",""))):
            cid = int(cfile.split("/")[-1].replace(".dx", ""))
            if "prmtop" in self.reference:
                self.molids[cid] = molecule.load("parm7", self.reference, "crdbox",
                                                 self.reference.replace("prmtop", "inpcrd"))
            else:
                self.molids[cid] = molecule.load("psf", self.reference, "pdb",
                                                 self.reference.replace("psf", "pdb"))

            molecule.rename(self.molids[cid], str(cid))
            molecule.read(self.molids[cid], "dx", cfile, waitfor=-1)
            molrep.delrep(self.molids[cid], 0)
            molrep.addrep(self.molids[cid], style="NewRibbons 0.1 12.0 12.0",
                          selection="protein or resname ACE NMA",
                          color="ColorID %d" % (cid%33), material="Opaque")
            evaltcl("mol drawframes %d %d %d"
                    % (self.molids[cid], molrep.num(self.molids[cid])-1, 0))
            molrep.addrep(self.molids[cid], style="Isosurface 0.05 0 0 1 1",
                          color="ColorID %d" % (cid%33))

    #==========================================================================

    def _load_frame(self, cluster):
        """
        Picks a frame corresponding to the given cluster and loads it,
        setting representation to the correct ligand
        Args:
            cluster(int): The cluster to sample
        """
        # Pick a frame at random containing this cluster
        frames = {k:v for k, v in {i:np.ravel(np.where(c == cluster)) \
                  for i, c in enumerate(self.clusters)}.items() \
                  if len(v)}
        clustindex = int(random.choice(frames.keys()))
        frameindex = int(random.choice(frames[clustindex]))
        filename = self.prodfiles[clustindex / self.nligs]

        # Load this frame
        molecule.read(self.molids[cluster],
                      utils.get_trajectory_format(filename), filename,
                      beg=frameindex, end=frameindex, waitfor=-1)

        # Set representations so this frame appears
        ligidx = clustindex % self.nligs
        molrep.addrep(self.molids[cluster], style="Licorice 0.3 12.0 12.0",
                      selection="noh and same fragment as residue %d" % ligidx,
                      color="ColorID %d" % (cluster%33), material="Opaque")
        evaltcl("mol drawframes %d %d %d"
                % (self.molids[cluster], molrep.num(self.molids[cluster])-1,
                   molecule.numframes(self.molids[cluster])))

    #==========================================================================

    def add_frames(self, cluster, num=10):
        """
        Adds num random frames depicting the given cluster to
        the given molecule
        """
        for _ in range(num):
            self._load_frame(cluster)

