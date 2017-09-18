"""
Contains classes for aggregating clusters into useful files so that
all trajectory doesn't have to be loaded.
"""
from __future__ import print_function
import os
import sys
import numpy as np

from beak.msm import utils
from configparser import ConfigParser
from Dabble import VmdSilencer
from multiprocessing import Process, Queue
from gridData import Grid
from vmd import atomsel, molecule, vmdnumpy

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Cluster densities                                #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ClusterDensity(object): #pylint: disable=too-many-instance-attributes
    """
    Obtains cluster locations as a density map. Aims to be memory efficient
    and as fast as possible by interating through trajectories only once,
    loading only one at a time.

    Attributes:
        grids (dict int->Grid): Densities for each cluster, indexed by label
        counts (dict int->int): Number of times each cluster was observed
    """
    #==========================================================================

    def __init__(self, prodfiles, clusters, **kwargs):
        """
        Args:
            trajfiles (list of str): Trajectory files
            clusters (list of ndarray): Cluster data
            config (ConfigParser): Config file with other system information, will
                read info from this if possible
            maxframes (list of int): Number of frames to read from each file
            topology (str): Single topology to use for all frames
            dimensions (list of 3 floats): System box size
            ligands (list of str): Ligand residue names
            reference (str): Path to reference structure for alignment
            rootdir (str): Root directory containing files etc
            alignsel (str): String for alignment on reference structure
        """
        self.prodfiles = prodfiles
        self.clusters = clusters

        if kwargs.get("config") is not None:
            if isinstance(kwargs.get("config"), str):
                config = ConfigParser(interpolation=None)
                config.read(kwargs.get("config"))
            else:
                config = kwargs.get("config")
            self.dimensions = [float(d) for d in config["dabble"]["dimensions"].split(',')]
            self.lignames = config["system"]["ligands"].split(',')
            self.rootdir = config["system"]["rootdir"]
            refname = config["system"]["reference"]
            self.refsel = config["system"]["refsel"]
            # For backwards / psf compatibility
            self.psfsel = config["system"]["canonical_sel"]
        else:
            self.dimensions = kwargs.get("dimensions")
            self.lignames = kwargs.get("ligands")
            self.rootdir = kwargs.get("rootdir")
            refname = kwargs.get("reference")
            self.refsel = kwargs.get("alignsel")
            self.psfsel = kwargs.get("alignsel")


        # Precalculate box edges
        self.ranges = [[-r/2., r/2.] for r in self.dimensions]
        self.grids = {}

        # Accumulators for cluster statistics
        self.counts = {}
        self.means = {}
        self.variances = {}
        self.accumulate_only = kwargs.get("accumulate_only", False) # Omit last divide

        # Handle optional arguments
        self.maxframes = kwargs.get("maxframes", None)
        self.topology = kwargs.get("topology", None)

        # Load reference structure
        if "prmtop" in refname:
            self.refid = molecule.load("parm7", refname,
                                       "crdbox", refname.replace("prmtop", "inpcrd"))
        elif "psf" in refname:
            self.refid = molecule.load("psf", refname,
                                       "pdb", refname.replace("psf", "pdb"))
        else:
            raise ValueError("Unknown format of reference file %s" % refname)

        # Sanity check that saved structure could actually be read
        if not molecule.exists(self.refid):
            raise ValueError("Couldn't load reference structure %s"
                             % refname)

        # Sanity check there are actually production files to process
        if not self.prodfiles:
            raise ValueError("No trajectory files to process")

        self.aselref = atomsel(self.refsel, molid=self.refid)

    #==========================================================================

    def _update_grid(self, label, data):
        """
        Updates the grid with a given label and data.
        Mean and variance are updated in-place.

        Args:
            label (int): Which cluster to update data
            data (ndarray, natoms x 3): Coordinates of ligand to update with
        """
        binned, edges = np.histogramdd(data,
                                       bins=self.dimensions,
                                       range=self.ranges,
                                       normed=False)

        # Initialize if necessary
        if self.grids.get(label) is None:
            # Edges is the same, so just initialize it here
            self.grids[label] = [np.zeros(binned.shape), edges]
            self.counts[label] = 0
            self.means[label] = np.zeros(data.shape)
            self.variances[label] = np.zeros(data.shape)

        # Update, using Welford's method to update mean and variance in place
        self.counts[label] += 1
        delta = data - self.means[label]
        self.means[label] += delta/self.counts[label]
        delta2 = data - self.means[label]
        self.variances[label] += np.multiply(delta, delta2)

        # Update grid / histogram
        self.grids[label][0] += binned

    #==========================================================================

    def _process_traj(self, trajfile):
        """
        Updates the densities so far from the given trajectory and cluster
        assignments.

        Args:
            trajfile (str): Trajectory file to process
        """
        # Load the trajectory
        assert trajfile in self.prodfiles
        trajidx = self.prodfiles.index(trajfile)
        maxframe = -1 if self.maxframes is None else self.maxframes[trajidx]-1

        molid = utils.load_trajectory(trajfile,
                                      rootdir=self.rootdir,
                                      aselref=self.aselref,
                                      psfref=self.psfsel,
                                      prmref=self.refsel,
                                      frame=(0, maxframe),
                                      topology=self.topology)

        # Get residue number for each ligand
        ligids = sorted(set(atomsel("resname %s" % " ".join(self.lignames),
                                    molid=molid).get("residue")))
        masks = [vmdnumpy.atomselect(molid, 0,
                                     "noh and same fragment as residue %d" % l)
                 for l in ligids]

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

    def process_all_trajectories(self):
        for i, traj in enumerate(self.prodfiles):
            print("  On trajfile %d of %d" % (i, len(self.prodfiles)))
            sys.stdout.flush()
            self._process_traj(traj)

        # Check if we would NaN doing the reduce step
        if self.accumulate_only:
            return

        # Last step of Welford's algorithm
        for label in self.variances:
            print("Counts of %s is %d" % (label, self.counts[label]))
            self.variances[label] /= float(self.counts[label] - 1)

    #==========================================================================

    def save(self, outdir):
        """
        Calculates all density maps for observed ligands. Then,
        normalizes all of the density maps according to observed count.
        That way, a density of 1.0 in some location means a ligand atom
        was there 100% of the time.

        Args:
            outdir (str): Output directory in which to put dx map files
        """
        with VmdSilencer(output=os.path.join(outdir, "vmd.log")):
            self.process_all_trajectories()
            self.save_densities(outdir)

    #==========================================================================

    def save_densities(self, outdir):
        """
        Saves the densities, means, and variances to output directory
        """
        for label, hist in self.grids.items():
            den = Grid(hist[0], edges=hist[1], origin=[0., 0., 0.])
            den /= float(self.counts[label])
            den.export(os.path.join(outdir, "%s.dx" % label),
                       file_format="dx")

        utils.dump(self.means, os.path.join(outdir, "means.pkl"))
        utils.dump(self.variances, os.path.join(outdir, "variance.pkl"))

    #==========================================================================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                     Parallelized cluster densities                          #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_cluster_worker(resultqueue, prodfiles, clusters, **kwargs):
    """
    Runs a single ClusterDensity worker and returns it
    """
    densitor = ClusterDensity(prodfiles, clusters, **kwargs)
    densitor.process_all_trajectories()
    resultqueue.put(densitor)

#==============================================================================

class ParallelClusterDensity(object): #pylint: disable=too-many-instance-attributes
    """
    Uses many worker tasks to compute cluster densities in parallel.
    Obtains cluster locations as a density map. Aims to be memory efficient
    and as fast as possible by interating through trajectories only once,
    loading only one at a time.


    Attributes:
        grids (dict int->Grid): Densities for each cluster, indexed by label
        counts (dict int->int): Number of times each cluster was observed
    """
    #==========================================================================

    def __init__(self, prodfiles, clusters, **kwargs):
        """
        Args:
            trajfiles (list of str): Trajectory files
            clusters (list of ndarray): Cluster data
            config (ConfigParser): Config file with other system information, will
                read info from this if possible
            maxframes (list of int): Number of frames to read from each file
            topology (str): Single topology to use for all frames
            dimensions (list of 3 floats): System box size
            ligands (list of str): Ligand residue names
            reference (str): Path to reference structure for alignment
            rootdir (str): Root directory containing files etc
            alignsel (str): String for alignment on reference structure
        """
        self.prodfiles = prodfiles
        self.clusters = clusters
        self.kwargs = kwargs

        # Precalculate box edges
        self.grids = {}
        self.counts = {}
        self.means = {}
        self.variances = {}

        # Sanity check there are actually production files to process
        if not self.prodfiles:
            raise ValueError("No trajectory files to process")

    #==========================================================================

    def _start_workers(self):
        """
        Starts parallel calculation jobs. Does work with a pool of cluster
        density workers, with trajectories divided up evenly between them.
        """
        nproc = int(os.environ.get("SLURM_NTASKS", "8"))
        chunksize = len(self.prodfiles) // nproc
        nligs = len(self.clusters) // len(self.prodfiles)

        results = Queue()
        workers = []
        for i in range(nproc):
            idx = i * chunksize
            jobber = Process(target=run_cluster_worker,
                             args=(results,
                                   self.prodfiles[idx:idx+chunksize],
                                   self.clusters[idx*nligs:(idx+chunksize)*nligs]
                                  ),
                             kwargs=dict(self.kwargs, accumulate_only=True),
                             daemon=True)
            jobber.start()
            workers.append(jobber)

        return results, workers

    #==========================================================================

    def _aggregate_results(self, dataqueue, workers):
        """
        Aggregates all the data from the multiple ClusterDensity objects
        contained in the data Queue
        """
        # Now queue full so terminate workers
        for w in workers:
            w.join()

        data = []
        while not dataqueue.empty():
            print("Got a guy from the queue")
            data.append(dataqueue.get())

        for label in data[0].grids:
            for worker in data:
                # Initialize if necessary
                if self.grids.get(label) is None:
                    self.grids[label] = worker.grids[label]
                    self.counts[label] = worker.counts[label]
                    self.means[label] = worker.means[label]
                    self.variances[label] = worker.variances[label]
                else:
                    # Parallel algorithm from Chan et. al.
                    delta = self.means[label] - worker.means[label]
                    na = worker.counts[label]
                    nb = self.counts[label]

                    self.means[label] += delta * nb / (na + nb)
                    self.variances[label] += worker.variances[label] \
                                          +  delta**2 * (na * nb)/(na + nb)

                    # Now do easy update of grids and counts
                    self.grids[label][0] += worker.grids[label][0]
                    self.counts[label] += worker.counts[label]

            # Do final step of welford now it's been accumulated
            self.variances[label] /= float(self.counts[label] - 1)

    #==========================================================================

    def save(self, outdir):
        """
        Calculates all density maps for observed ligands. Then,
        normalizes all of the density maps according to observed count.
        That way, a density of 1.0 in some location means a ligand atom
        was there 100% of the time.

        Args:
            outdir (str): Output directory in which to put dx map files
        """
        data, workers = self._start_workers()
        self._aggregate_results(data, workers)
        self.save_densities(outdir)

    #==========================================================================

    def save_densities(self, outdir):
        """
        Save density maps, means, and variances, into a output directory
        """
        for label, hist in self.grids.items():
            den = Grid(hist[0], edges=hist[1], origin=[0., 0., 0.])
            den /= float(self.counts[label])
            den.export(os.path.join(outdir, "%s.dx" % label),
                       file_format="dx")

        utils.dump(self.means, os.path.join(outdir, "means.pkl"))
        utils.dump(self.variances, os.path.join(outdir, "variance.pkl"))

    #==========================================================================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
