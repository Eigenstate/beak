"""
Contains useful utilities for running MSMs
"""
from __future__ import print_function
import os
import h5py
import numpy as np
from Dabble import VmdSilencer
try:
    from vmd import atomsel, molecule, vmdnumpy
    atomsel = atomsel.atomsel
except:
    import vmd
    import molecule, vmdnumpy
    from atomsel import atomsel

#==============================================================================

def save_features_h5(dataset, filename, num_ligands=0, trajfiles=None):
    """
    Saves the given feature set as an hdf5 file

    Args:
        dataset (list of ndarray): Data to save
        filename (str): Filename to save as
        num_ligands (int): Number of ligands in the system
        trajfiles (list of str): Filenames of trajectories. This lets the
            "filename" attribute be populated

    Returns:
        True on sucess
    """

    h5f = h5py.File(filename, 'w-') # w- means fail on existence
    for i, fset in enumerate(dataset):
        h5f.create_dataset(name=str(i),
                           shape=fset.shape,
                           dtype=fset.dtype,
                           data=fset,
                           compression="gzip")
        if num_ligands:
            h5f[str(i)].attrs["filename"] = trajfiles[i/num_ligands]

    h5f.close()
    return True

#==============================================================================

def load_features_h5(filename):
    """
    Loads a feature set from an hdf5 file

    Args:
        filename (str): Filename to load

    Returns:
        (list of ndarray) Loaded features
    """

    h5f = h5py.File(filename, 'r')
    feated = []
    for i in sorted(h5f.keys(), key=int):
        feated.append(h5f[i][:])
    h5f.close()

    return feated

#==============================================================================

def get_topology(filename, rootdir):
    """
    Returns the psf file corresponding to a given trajectory file,
    using my directory layout. No fallback for older psfs.

    Args:
        filename (str): File name of trajectory
        rootdir (str): Root directory of simulation

    Returns:
        (str): File name of corresponding topology. Could be psf or prmtop.
    """
    rep = filename.split('/')[-2]
    gen = filename.split('/')[-3]

    # Handle special psf case for stripped trajectories
    if "strip" in filename:
        topo = os.path.join(rootdir, "systems", gen, "%s_stripped.prmtop" % rep)
    else:
        topo = os.path.join(rootdir, "systems", gen, "%s.psf" % rep)

    return topo

#==============================================================================

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

    def __init__(self, natoms):
        self.mean = np.zeros((natoms, 3))
        self.bestpos = None
        self.count = 0
        self.reptraj = None
        self.repframe = None
        self.repligid = None
        self.rms = None


    def update(self, molid, trajfile, frame, ligid, mask=None):
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

        if os.path.isfile(topofile):
            topo = topofile
        else:
            topo = get_topology(self.reptraj, topofile)
        m = molecule.load("psf" if "psf" in topo else "parm7", topo)
        molecule.read(m, "dcd" if ".dcd" in self.reptraj else "netcdf",
                      self.reptraj, beg=self.repframe, end=self.repframe,
                      waitfor=-1)

        # Frame number differs here since we only loaded one frame
        atomsel("(%s) or (same fragment as residue %d)"
                % (protsel, self.repligid),
                molid=m, frame=0).write("mae", filename)
        molecule.delete(m)

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
            topofile = get_topology(trajfile, topology)

        molid = molecule.load("psf" if "psf" in topo else "parm7", topofile)
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
            for cl in clusters:
                # Initialize clusters
                if not bests.get(cl):
                    bests[cl] = ClusterCenter(ligheavyatoms)

                # Get atom selection mask for the ligand
                mask = vmdnumpy.atomselect(molid, 0,
                                           "noh and same fragment as residue %d"
                                           % lig)

                # Get frames that contain this cluster
                cidx = trajidx*len(ligids) + i
                if cl == "nan":
                    frames = [_ for _, d in enumerate(clusts[cidx]) if np.isnan(d)]
                else:
                    frames = [_ for _, d in enumerate(clusts[cidx]) if d == cl]
                if not len(frames): continue # No frames represented
                if len(clusts[cidx]) != molecule.numframes(molid):
                    raise ValueError("Frames mismatch between trajidx %d clustidx %d" \
                                     "I have %d, %d\nFilename was %s"
                                     % (trajidx, cidx, len(clusts[cidx]),
                                        molecule.numframes(molid), trajfile))

                # Add frames to running total
                for f in frames:
                    bests[cl].update(molid, trajfile, f, lig, mask)

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
        topology (str): Either root directory for system or psf file
    """

    protsel = "(protein or resname ACE NMA) and not same fragment as " \
              "resname %s" % " ".join(ligands)

    with VmdSilencer(output=os.path.join(outdir, "vmd.log")):
        centers = get_cluster_centers(prodfiles, clust,
                                      clusters=msm.mapping_.values(),
                                      ligands=ligands,
                                      topology=topology)

        fn = open(os.path.join(outdir, "rmsds"), 'w', 0) # Unbuffered
        for cl, center in centers.items():
            center.save_mae(os.path.join(outdir, "%s.mae" % cl),
                            topology, protsel)
            fn.write("%d %f\n" % (cl, center.rms))
        fn.close()

#==============================================================================

