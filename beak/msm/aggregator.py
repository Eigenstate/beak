"""
Contains classes for aggregating clusters into useful files so that
all trajectory doesn't have to be loaded.
"""
from __future__ import print_function
import os
import sys
import numpy as np

from beak.msm import utils
from configparser import RawConfigParser
from Dabble import VmdSilencer
from gridData import Grid
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
                config = RawConfigParser()
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
        self.counts = {}
        self.means = {}
        self.variances = {}

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
        if not len(self.prodfiles):
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
            self.means[label] = [np.zeros(data.shape)]

        # Update, using Welford's method to update mean and variance in place
        self.grids[label][0] += binned

        self.counts[label] += 1
        delta = data - self.means[label]
        self.means[label] += delta/self.counts[label]
        delta2 = data - self.means[label]
        self.variances[label] += np.multiply(delta, delta2)

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

    def save(self, outdir):
        """
        Calculates all density maps for observed ligands. Then,
        normalizes all of the density maps according to observed count.
        That way, a density of 1.0 in some location means a ligand atom
        was there 100% of the time.

        Args:
            outdir (str): Output directory in which to put dx map files
        """
        self.save_densities(outdir)

    #==========================================================================

    def save_densities(self, outdir):
        """
        Old style naming for save
        """
        with VmdSilencer(output=os.path.join(outdir, "vmd.log")):
            for i, traj in enumerate(self.prodfiles):
                print("  On trajfile %d of %d" % (i, len(self.prodfiles)))
                sys.stdout.flush()
                self._process_traj(traj)

        for label, hist in self.grids.items():
            den = Grid(hist[0], edges=hist[1], origin=[0., 0., 0.])
            den /= float(self.counts[label])
            den.export(os.path.join(outdir, "%s.dx" % label),
                       file_format="dx")

            # Last step of Welford's algorithm
            self.variances[label] /= float(self.counts[label] - 1)
            # Means is already divided
            # self.means[label] /= float(self.counts[label])

        utils.dump(self.means, os.path.join(outdir, "means.pkl"))
        utils.dump(self.variances, os.path.join(outdir, "variance.pkl"))

    #==========================================================================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Cluster means only                               #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ClusterMeans(object): #pylint: disable=too-many-instance-attributes
    """
    Obtains mean position of each ligand atom in clusters.
    Aims to be memory efficient and as fast as possible by interating through
    trajectories only once, loading only one at a time.

    Attributes:
        means (dict int->ndarray): Means for each cluster, indexed by label
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
                config = RawConfigParser()
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


        self.counts = {}
        self.means = {}

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

    def _update_mean(self, label, data):
        """
        Updates the grid with a given label and data
        """
        if self.means.get(label) is not None:
            self.means[label] += data
            self.counts[label] += 1
        else:
            self.means[label] = data
            self.counts[label] = 1

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
                self._update_mean(self.clusters[cidx][frame], coords)

        molecule.delete(molid)

    #==========================================================================

    def save(self, outname):
        """
        Calculates all density maps for observed ligands. Then,
        normalizes all of the density maps according to observed count.
        That way, a density of 1.0 in some location means a ligand atom
        was there 100% of the time.

        Args:
            outname (str): Output file name for means pickle
        """
        with VmdSilencer(output="vmd.log"):
            for i, traj in enumerate(self.prodfiles):
                print("  On trajfile %d of %d" % (i, len(self.prodfiles)))
                sys.stdout.flush()
                self._process_traj(traj)

        for label in self.means:
            self.means[label] /= float(self.counts[label])

        utils.dump(self.means, outname)

    #==========================================================================

#==============================================================================
