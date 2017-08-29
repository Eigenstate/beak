"""
Contains functionality for loading project-specific datasets
"""
from __future__ import print_function
import os
import numpy as np
import random
import sys
import time
from configparser import ConfigParser
from glob import glob
from socket import gethostname
from beak.msm import utils

from beak.msm.utils import load
from subprocess import PIPE, Popen
from threading import Thread
from queue import Queue, Empty # Python 3 queue
from vmd import evaltcl, molecule, molrep, atomsel

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

        self.config = ConfigParser(interpolation=None)
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.num_ligands = self.config.getint("system", "num_ligands")
        self.generation = generation
        self.firstgen = firstgen

        assert generation <= self.config.getint("production", "generation")
        self.nreps = self.config.getint("model", "samplers")
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
        self.mmsm = utils.load(os.path.join(substr, "mmsm_G%d.pkl" % self.generation))
        self.msm = utils.load(os.path.join(substr, "msm_G%d.pkl" % self.generation))
        tica = utils.load(os.path.join(substr, "testing.tica.pkl"))
        clust = utils.load(os.path.join(substr, "testing.cluster.pkl"))
        self.scores = []
        if os.path.isfile(os.path.join(substr, "mmsm_scores.pkl")):
            print("Loading scores")
            self.scores = utils.load(os.path.join(substr, "mmsm_scores.pkl"))
        else:
            print("Couldn't find scores: %s"
                  % os.path.join(substr, "mmsm_scores.pkl"))

        # Handle saved pre-msm clusters, as naming scheme changed...
        if os.path.isfile(os.path.join(substr, "testing.mcluster.pkl")):
            mclust = utils.load(os.path.join(substr, "testing.mcluster.pkl"))
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

        self.config = ConfigParser(interpolation=None)
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.num_ligands = self.config.getint("system", "num_ligands")
        self.generation = generation

        assert generation <= self.config.getint("production", "generation")
        self.nreps = self.config.getint("model", "samplers")
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

        #with open(os.path.join(self.dir, "clusters", str(self.generation), "rmsds")) as fn:
        #    lines = fn.readlines()

        #rmsds = {int(x.split()[0].strip()): float(x.split()[1].strip()) for x in lines}

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
            #if self.rmsds is None:
            #    self.rmsds = rmsds

            #atomsel("all", molid=a).set("user", self.rmsds[int(nam)])

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
            topology (str): One topology for all input files, for DESRES
            load_clusters (bool): Whether clusters should be loaded
        """
        # Set visualization stuff
        evaltcl("color scale method BGR")
        evaltcl("display projection Orthographic")

        self.config = ConfigParser(interpolation=None)
        self.config.read(configfile)

        self.ligands = self.config["system"]["ligands"].split(',')
        self.nligs = self.config.getint("system", "num_ligands")
        self.dir = self.config["system"]["rootdir"]

        self.generation = kwargs.get("generation")
        self.clustdir = kwargs.get("clustdir")
        self.prodfiles = kwargs.get("prodfiles", None)
        if self.prodfiles is None:
            self.prodfiles = utils.get_prodfiles(self.generation, self.dir,
                                                 new=False,
                                                 equilibration=self.config.getboolean("model",
                                                                                      "include_equilibration"))

        # Load reference structure for later alignment
        # Hide it
        self.reference = self.config["system"]["reference"]
        if "prmtop" in self.reference:
            self.refid = molecule.load("parm7", self.reference,
                    "crdbox", self.reference.replace("prmtop", "inpcrd"))
        elif "psf" in self.reference:
            self.refid = molecule.load("psf", self.reference,
                    "pdb", self.reference.replace("psf", "pdb"))
        molrep.delrep(self.refid, 0)
        molecule.rename(self.refid, "Reference")

        self.prmsel = self.config["system"]["refsel"]
        self.aselref = atomsel(self.prmsel, molid=self.refid)
        self.psfsel = self.config["system"]["canonical_sel"]

        # List all of the filenames so we can look them up later
        self.molids = {}

        if kwargs.get("load_clusters", True):
            self._load_densities()
            self._load_means()

        # Find and load the msm, and clusters, and tics
        self.msmname = kwargs.get("msm")
        if os.path.isfile(kwargs.get("scores", "")):
            self.scores = load(kwargs.get("scores"))
        else:
            self.scores = kwargs.get("scores")
        self.msm = utils.load(self.msmname)
        self.clusters = utils.load(kwargs.get("clusters"))
        self.tica = None
        if kwargs.get("tica") is not None:
            self.tica = utils.load(kwargs.get("tica"))
        self.topology = kwargs.get("topology")

        # Multithreaded updates
        self._queue = Queue()
        self._threads = []

    #==========================================================================

    def __del__(self):
        if self.refid in molecule.listall():
            molecule.delete(self.refid)
        for mid in self.molids.values():
            for _ in mid:
                molecule.delete(_)
        for t in self._threads:
            if t.isAlive(): t.terminate()
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
                molid = molecule.load("parm7", self.reference, "crdbox",
                                       self.reference.replace("prmtop", "inpcrd"))
            else:
                molid = molecule.load("psf", self.reference, "pdb",
                                      self.reference.replace("psf", "pdb"))
            self.molids[cid] = [molid]

            molecule.rename(molid, str(cid))
            molecule.read(molid, "dx", cfile, waitfor=-1)
            molrep.delrep(molid, 0)
            color = 9 if (cid%33 == 8) else cid % 33
            molrep.addrep(molid, style="NewRibbons 0.1 12.0 12.0",
                          selection="protein or resname ACE NMA",
                          color="ColorID %d" % color, material="Opaque")
            evaltcl("mol drawframes %d %d %d"
                    % (molid, molrep.num(molid)-1, 0))
            molrep.addrep(molid, style="Isosurface 0.05 0 0 1 1",
                          color="ColorID %d" % color)

    #==========================================================================

    def _load_means(self):
        """
        Loads means file with center coordinate of each cluster
        """
        if os.path.isfile(os.path.join(self.clustdir, "means.pkl")):
            self.means = utils.load(os.path.join(self.clustdir, "means.pkl"))

    #==========================================================================

    def load_frame(self, cluster):
        """
        Picks a frame corresponding to the given cluster and loads it,
        setting representation to the correct ligand. User fields 1-5 will
        be populated with the first 5 tics.
        Args:
            cluster (int): The cluster to sample
        """
        # Pick a frame at random containing this cluster
        frames = {k:v for k, v in {i:np.ravel(np.where(c == cluster)) \
                  for i, c in enumerate(self.clusters)}.items() \
                  if len(v)}
        clustindex = int(random.choice(list(frames)))
        frameindex = int(random.choice(frames[clustindex]))
        filename = self.prodfiles[int(clustindex / self.nligs)]
        ligidx = clustindex % self.nligs

        # Figure out if we need an alternate molid for this frame
        # ie if it came from a non stripped trajectory we need to load
        # the non stripped topology as an alternate molid for the cluster
        if self.topology is None:
            topology = utils.get_topology(filename, self.dir)
        else:
            topology = self.topology

        molid = None
        for mol in self.molids[cluster]:
            if topology in molecule.get_filenames(mol):
                molid = mol
                break

        # Load and align this frame
        m2 = utils.load_trajectory(filename,
                                   config=self.config,
                                   #aselref=self.aselref,
                                   #psfref=self.psfsel,
                                   #prmref=self.prmsel,
                                   frame=frameindex,
                                   topology=topology,
                                   molid=molid)
        molecule.set_visible(m2, molecule.get_visible(self.molids[cluster][0]))
        if m2 not in self.molids[cluster]:
            self.molids[cluster].append(m2)

        # Set representations so this frame appears along with parent cluster
        molecule.rename(m2, "frame_%d" % cluster)
        molrep.delrep(m2, 0)
        ligands = sorted(set(atomsel("resname %s" % " ".join(self.ligands),
                                     molid=m2).get("residue")))
        color = 9 if (cluster%33 == 8) else cluster % 33
        molrep.addrep(m2, style="Licorice 0.3 12.0 12.0",
                      selection="noh and same fragment as residue %d"
                                % ligands[ligidx],
                      color="User", material="Opaque")
        molrep.set_colorupdate(m2, molrep.num(m2)-1, True)
        molrep.set_scaleminmax(m2, molrep.num(m2)-1, 0., 1.)
        evaltcl("mol drawframes %d %d 0:%d" % (m2, molrep.num(m2)-1,
                                               molecule.numframes(m2)))

        if self.tica is None:
            molrep.modrep(m2, molrep.num(m2)-1, color="ColorID %d" % color)
                      #color="ColorID %d" % color, material="Opaque")

        # Set ligand user field by first 5 tics
        lsel = atomsel("noh and same fragment as residue %d"
                       % ligands[ligidx], molid=m2)

        if self.tica is not None:
            for featidx in range(1,5):
                field = "user" if featidx == 1 else "user%d" % featidx
                minl = min(min(d[:,featidx]) for d in self.tica)
                rge = max(max(d[:,featidx]) for d in self.tica) - minl
                dat = (self.tica[clustindex][frameindex, featidx] - minl)/rge
                lsel.set(field, dat)


    #==========================================================================

    def show_cluster(self, cluster, shown=None):
        """
        Shows all frames and density for the specified cluster

        Args:
            cluster (int): Cluster label to show
            shown (bool): If I should show or hide. None for toggle
        """
        visstate = shown
        for molid in self.molids[cluster]:
            if visstate is None:
                visstate = not molecule.get_visible(molid)
            molecule.set_visible(molid, visstate)

    #==========================================================================

    def _enqueue_output(self, graphproc):
        while graphproc.poll() is None: # while it's running
            try:
                for line in iter(graphproc.stdout.readline, b''):
                    #queue.put(line)
                    data = int(line.decode("utf-8"))
                    self.show_cluster(data)
                #graphproc.stdout.close()
            except: pass
            time.sleep(0.05)

    #==========================================================================

    def _dequeue_output(self, queue, enquer):
        while enquer.isAlive():
            #time.sleep(0.05)
            #try: line = queue.get_nowait()
            try:
                sys.stdout.flush()
                line = queue.get(timeout=0.05)
                sys.stdout.flush()
            except Empty:
                #time.sleep(0.05)
                continue
            data = int(line.decode("utf-8"))
            self.show_cluster(data)

    #==========================================================================

    def graph_msm(self):
        # Hide all clusters to start
        for cluster in self.molids:
            self.show_cluster(cluster, shown=False)
        # Start the graph in a new process
        # It should have a new socket to X rather than sharing this one's...
        ON_POSIX = 'posix' in sys.builtin_module_names
        p = Popen(["/home/robin/Work/Code/beak/beak/msm/display_msm.py",
                   self.msmname
                   ],
                   stdout=PIPE,
                  bufsize=1, close_fds=ON_POSIX)
        #q = Queue()
        t1 = Thread(target=self._enqueue_output, args=(p, ))
        t1.daemon = True
        t1.start()
        self._threads.append(t1)

        #t2 = Thread(target=self._dequeue_output, args=(q, t1))
        #t2.daemon = True
        #t2.start()

    #==========================================================================

    def closest_to(self, coords):
        """
        Returns the clusters sorted in order of closeness to the
        given coordinates
        """
        return sorted(self.means,
                      key=lambda k: np.sqrt(np.sum((coords-self.means[k])**2))
                     )

    #==========================================================================

