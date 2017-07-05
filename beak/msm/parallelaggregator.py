"""
Contains classes for aggregating clusters into useful files so that
all trajectory doesn't have to be loaded.
"""
from __future__ import print_function
import os
import sys
import numpy as np

from beak.msm import utils
from Dabble import VmdSilencer
from gridData import Grid
from multiprocessing import Pool
from threading import Thread, Lock
from queue import Queue
from vmd import atomsel, molecule, vmdnumpy

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#              Cluster means and most representative structure                #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ClusterCenter(object):
    """
    A cluster center.

    Attributes:
        mean (ndarray): Sum of positions so far, not divided by count
        count (int): Number of this cluster represented in mean
        reptraj (str): Filename of trajectory where ligand can be found
        repframe (int): Frame where ligand can be found
        repligid (int): Residue number of representative ligand
        rms (float): RMS of best ligand to mean
        bestpos (ndarray): Coordinates of most representative ligand
    """

    #==========================================================================

    def __init__(self, natoms):
        self.mean = np.zeros((natoms, 3))
        self.bestpos = None
        self.count = 0
        self.reptraj = None
        self.repframe = None
        self.repligid = None
        self.rms = None

    #==========================================================================

    def update(self, molid, trajfile, frame, ligid, **kwargs):
        """
        Updates the mean and the most representative ligand, if found.

        Args:
            molid (int): VMD molecule ID
            trajfile (str): Loaded trajectory filename
            frame (int): Frame to check
            ligid (int): Ligand residue ID to check
            mask (ndarray): Atom selection mask for ligand.
                Faster if precomputed

        Returns:
            (bool) if the most representative frame was updated
        """
        mask = kwargs.get("mask", None)
        coords = np.compress(mask, vmdnumpy.timestep(molid, frame), axis=0)
        dev = self._do_update(coords)

        if self.bestpos is not None:
            olddev = np.sqrt(1./len(self.bestpos) *
                             np.sum((self.bestpos - self.mean/float(self.count))**2))

        if self.bestpos is None or dev < olddev:
            self.bestpos = coords
            self.reptraj = trajfile
            self.repframe = frame
            self.repligid = ligid
            self.rms = dev
            return True
        else:
            self.rms = olddev

    #==========================================================================

    def _do_update(self, coords):
        """
        Gets the deviation according to whatever whacko metric I'm using
        for the given coordinates. Updates the mean using these coordinates

        Args:
            coords (ndarray): Coordinates to check
        Returns:
            (bool) If those coordinates were updated
        """
        # Update mean
        self.mean += coords
        self.count += 1
        dev = np.sqrt(1./len(coords) * np.sum((coords - self.mean/float(self.count))**2))

        return dev

    #==========================================================================

    def save_mae(self, filename, topofile, protsel):
        """
        Saves the cluster center as a mae file

        Args:
            filename (str): Filename to write
            topofile (str): Either a psf file or a root directory in
                            which to find topologies
            protsel (str): Protein selection string
        """
        print("  Trajectory: %s\nFrame: %d\tLigand: %d"
              % (self.reptraj, self.repframe, self.repligid))

        # This is deprecated so I'm not bothering translating it to utils
        # function call here
        if os.path.isfile(topofile):
            topo = topofile
        else:
            topo = utils.get_topology(self.reptraj, topofile)
        molid = molecule.load("psf" if "psf" in topo else "parm7", topo)
        molecule.read(molid, "dcd" if ".dcd" in self.reptraj else "netcdf",
                      self.reptraj, beg=self.repframe, end=self.repframe,
                      waitfor=-1)

        # Frame number differs here since we only loaded one frame
        atomsel("(%s) or (same fragment as residue %d)"
                % (protsel, self.repligid),
                molid=molid, frame=0).write("mae", filename)
        molecule.delete(molid)

#==============================================================================

def get_cluster_centers(prodfiles, clusts, clusters, ligands, topology):
    """
    Obtains a representative structure of the given cluster.
    Only reads in one trajectory at a time, so may not be completely
    the best structure but works when all trajs can't be loaded into
    memory.

    Args:
        prodfiles (list of str): Production files, ordered the same as
            for cluster obtaining
        clusts (list of ndarray): Cluster data
        clusters (list of int): Which clusters to get center of
        ligands (list of str): Ligand resnames
        topology (str): Path to psf, or rootdir for autopsf

    Returns:
        (dict int->ClusterCenter): Cluster centers with labels
    """

    bests = {} # Will be a bunch of ClusterIndice

    for trajidx, trajfile in enumerate(prodfiles):
        # Read in a new molecule
        if os.path.isfile(topology):
            topofile = topology
        else:
            topofile = utils.get_topology(trajfile, topology)

        # Deprecated but ok
        molid = molecule.load("psf" if "psf" in topofile else "parm7", topofile)
        molecule.read(molid, "dcd" if ".dcd" in trajfile else "netcdf",
                      trajfile, waitfor=-1)

        # Get residue number for each ligand
        ligids = sorted(set(atomsel("resname %s" % " ".join(ligands),
                                    molid=molid).get("residue")))
        if not len(bests):
            ligheavyatoms = len(atomsel("noh and same fragment as residue %d"
                                        % ligids[0], molid=molid))


        # Now handle the ligands independently
        for i, lig in enumerate(ligids):
            for cls in clusters:
                # Initialize clusters
                if not bests.get(cls):
                    bests[cls] = ClusterCenter(ligheavyatoms)

                # Get atom selection mask for the ligand
                mask = vmdnumpy.atomselect(molid, 0,
                                           "noh and same fragment as residue %d"
                                           % lig)

                # Get frames that contain this cluster
                cidx = trajidx*len(ligids) + i
                if cls == "nan":
                    frames = [_ for _, d in enumerate(clusts[cidx]) if np.isnan(d)]
                else:
                    frames = [_ for _, d in enumerate(clusts[cidx]) if d == cls]
                if not len(frames):
                    continue # No frames represented
                if len(clusts[cidx]) != molecule.numframes(molid):
                    raise ValueError("Frames mismatch between trajidx %d clustidx %d" \
                                     "I have %d, %d\nFilename was %s"
                                     % (trajidx, cidx, len(clusts[cidx]),
                                        molecule.numframes(molid), trajfile))

                # Add frames to running total
                for frame in frames:
                    bests[cls].update(molid, trajfile, frame, lig, mask=mask)

        # Clean up
        molecule.delete(molid)

    return bests

#==============================================================================

def save_cluster_centers(prodfiles, clust, msm, ligands, outdir, topology):
    """
    Saves cluster centers to a file

    Args:
        prodfiles (list of str): Production filenames, same as clust order
        clust (list of ndarray): Cluster data
        msm (MarkovStateModel): MSM with cluster labels
        ligands (list of str): Ligand resnames in system
        outdir (str): Output directory for cluster centers
        topology (str): Either root directory for system or psf/prmtop file
    """

    protsel = "(protein or resname ACE NMA) and not same fragment as " \
              "resname %s" % " ".join(ligands)

    with VmdSilencer(output=os.path.join(outdir, "vmd.log")):
        centers = get_cluster_centers(prodfiles, clust,
                                      clusters=msm.mapping_.values(),
                                      ligands=ligands,
                                      topology=topology)

        rfn = open(os.path.join(outdir, "rmsds"), 'w', 0) # Unbuffered
        for clust, center in centers.items():
            center.save_mae(os.path.join(outdir, "%s.mae" % clust),
                            topology, protsel)
            rfn.write("%d %f\n" % (clust, center.rms))
        rfn.close()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Cluster densities                                #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class DensityWorker(Thread):
    """
    Actually does the work for cluster density. A gather operation pulls
    a bunch of these together.

    Attributes:
        grids (dict int->Grid): Densities for each cluster, indexed by label
        counts (dict int->int): Number of times each cluster was observed
        means (dict int->ndarray): Mean coordinate of each cluster
    """

    #==========================================================================

    def __init__(self, thread_id, inqueue,
                 means_q, densities_q, counts_q, **kwargs):
        super(DensityWorker, self).__init__()
        self.id = thread_id
        self.grids = {}
        self.counts = {}
        self.means = {}
        self.clusters = kwargs.get("clusters")

        self.kwargs = kwargs
        self.inqueue = inqueue
        self.means_q = means_q
        self.densities_q = densities_q
        self.counts_q = counts_q
        self.lock = kwargs.get("atomsel_lock")

    #==========================================================================

    def run(self):
        while not self.inqueue.empty():
            trajidx, trajfile = self.inqueue.get()
            print("%d: Got trajidx %d" % (self.id, trajidx))
            sys.stdout.flush()
            self._process_traj(trajidx, trajfile)
            self.inqueue.task_done()

        self.means_q.put(self.means)
        self.densities_q.put(self.grids)
        self.counts_q.put(self.counts)

    #==========================================================================

    def _update_grid(self, label, data):
        """
        Updates the grid with a given label and data
        """
        binned, edges = np.histogramdd(data,
                                       bins=self.kwargs.get("dimensions"),
                                       range=self.kwargs.get("ranges"),
                                       normed=False)


        if self.grids.get(label):
            self.grids[label][0] += binned
            self.means[label] += data
            self.counts[label] += 1
        else:
            self.grids[label] = [binned, edges]
            self.means[label] = data
            self.counts[label] = 1

    #==========================================================================

    def _process_traj(self, trajidx, trajfile):
        """
        Updates the densities so far from the given trajectory and cluster
        assignments.

        Args:
            trajfile (str): Trajectory file to process
        """
        # Load the trajectory
        if self.kwargs.get("maxframes"):
            maxframe = self.kwargs.get("maxframes")[trajidx]-1
        else:
            maxframe = -1

        molid = utils.load_trajectory(trajfile, self.kwargs.get("rootdir"),
                                      aselref=self.kwargs.get("aselref"),
                                      psfref=self.kwargs.get("psfsel"),
                                      prmref=self.kwargs.get("refsel"),
                                      frame=(0, maxframe),
                                      topology=self.kwargs.get("topology"),
                                      lock=self.lock)

        # Get residue number for each ligand
        self.lock.acquire()
        ligids = sorted(set(atomsel("resname %s" % " ".join(self.kwargs.get("lignames")),
                                    molid=molid).get("residue")))
        masks = [vmdnumpy.atomselect(molid, 0,
                                     "noh and same fragment as residue %d" % l)
                 for l in ligids]
        self.lock.release()

        # DEBUG print out nframes
        if len(self.clusters[trajidx*len(ligids)]) != molecule.numframes(molid):
            print("Mismatch trajectory: %s" % trajfile)
            print("%d clusters but %d frames"
                  % (len(self.clusters[trajidx*len(ligids)]),
                     molecule.numframes(molid)))
            sys.stdout.flush()

        # Go through each frame just once
        # Update density for each ligand
        for frame in range(molecule.numframes(molid)):
            for i in range(len(ligids)):
                coords = np.compress(masks[i],
                                     vmdnumpy.timestep(molid, frame),
                                     axis=0)
                cidx = trajidx*len(ligids) + i
                self._update_grid(self.clusters[cidx][frame], coords)

        molecule.delete(molid)

#==========================================================================

class ClusterDensity(object): #pylint: disable=too-many-instance-attributes
    """
    Obtains cluster locations as a density map. Aims to be memory efficient
    and as fast as possible by interating through trajectories only once,
    loading only one at a time.
    """
    #==========================================================================

    def __init__(self, prodfiles, clusters, config, **kwargs):
        """
        Args:
            trajfiles (list of str): Trajectory files
            clusters (list of ndarray): Cluster data
            config (ConfigParser): Config file with other system information
            maxframes (list of int): Number of frames to read from each file
            topology (str): Single topology to use for all frames
        """
        # Sanity check there are actually production files to process
        self.prodfiles = prodfiles
        if not len(self.prodfiles):
            raise ValueError("No trajectory files to process")

        self.means = None
        self.counts = None
        self.densities = None
        self.outdir = "unset"

        # Handle optional arguments
        self.kwargs = kwargs

        # Precalculate box edges
        self.kwargs["clusters"] = clusters
        self.kwargs["dimensions"] = [float(d) for d in config["dabble"]["dimensions"].split(',')]
        self.kwargs["ranges"] = [[-r/2., r/2.] for r in self.kwargs["dimensions"]]

        self.kwargs["lignames"] = config["system"]["ligands"].split(',')
        self.kwargs["rootdir"] = config["system"]["rootdir"]

        # Load reference structure
        refname = config["system"]["reference"]
        self.kwargs["refsel"] = config["system"]["refsel"]
        self.kwargs["psfsel"] = config["system"]["canonical_sel"]

        # Each thread needs its own reference structure
        if "prmtop" in refname:
            self.kwargs["refid"] = molecule.load("parm7", refname,
                                                 "crdbox", refname.replace("prmtop",
                                                                           "inpcrd"))
        elif "psf" in refname:
            self.kwargs["refid"] = molecule.load("psf", refname,
                                                 "pdb", refname.replace("psf", "pdb"))
        else:
            raise ValueError("Unknown format of reference file %s" % refname)
        self.kwargs["aselref"] = atomsel(self.kwargs["refsel"],
                                         molid=self.kwargs.get("refid"))

        # Shared lock for atom selection
        self.kwargs["atomsel_lock"] = Lock()

    #==========================================================================

    def sum_queue(self, q_sum): #pylint: disable=no-self-use
        """
        Sums the values in a queue, as the data is a dictionary
        """
        thedata = {}
        while not q_sum.empty():
            data = q_sum.get()
            for k, v in data.items():
                if thedata.get(k) is not None:
                    thedata[k] += v
                else:
                    thedata[k] = v
        return thedata

    #==========================================================================

    def save_single_file(self, label):
        hist = self.densities[label]
        den = Grid(hist[0], edges=hist[1], origin=[0., 0., 0.])
        den /= float(self.counts[label])
        den.export(os.path.join(self.outdir, "%s.dx" % label),
                   file_format="dx")
        self.means[label] /= float(self.counts[label])

    #==========================================================================

    def save_densities(self, outdir):
        """
        Calculates all density maps for observed ligands. Then,
        normalizes all of the density maps according to observed count.
        That way, a density of 1.0 in some location means a ligand atom
        was there 100% of the time.

        Args:
            outdir (str): Output directory in which to put dx map files
        """

        self.outdir = outdir
        with VmdSilencer(output=os.path.join(self.outdir, "vmd.log")):

            # Add all production files to the queue
            todos = Queue(maxsize=len(self.prodfiles))
            for i, f in enumerate(self.prodfiles): todos.put_nowait((i, f))
            print("Evaluating %d prodfiles" % len(self.prodfiles))
            sys.stdout.flush()

            means_q = Queue()
            density_q = Queue()
            counts_q = Queue()

            # Create the appropriate number of workers
            for _ in range(int(os.environ.get("SLURM_NTASKS", 4))):
                w = DensityWorker(_, todos, means_q, density_q, counts_q,
                                  **self.kwargs)
                w.daemon = True
                w.start()

            # Wait until entire queue is processed
            todos.join()

            # Gather up all data
            self.means = self.sum_queue(means_q)
            self.densities = self.sum_queue(density_q)
            self.counts = self.sum_queue(counts_q)

        # Delete things that aren't pickled
        del self.kwargs

        pool = Pool(int(os.environ.get("SLURM_NTASKS", 4)))
        pool.map(self.save_single_file, self.densities.keys())
        utils.dump(self.means, os.path.join(self.outdir, "means.pkl"))

    #==========================================================================

#==============================================================================
