import mdtraj as md
import molecule
import molrep
import numpy as np
import trans
from atomsel import atomsel
from glob import glob
from msmbuilder.msm import MarkovStateModel
from VMD import evaltcl

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        ALL PURPOSE METHODS                                 +
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_ligand_residues(molid, topology, ligands):
    """
    Gets the VMD resids corresponding to the mdtraj ligands, in order.
    This is necessary in order to match up features for each ligand to
    visualization.

    Args:
        molid (int): VMD molecule ID to colour
        topology (mdtraj Topology): topology for the molecule, for matching
          up ligands
        ligands (list of str): Ligand resnames to consider
    Returns:
        (list of int): Residue numbers of each ligands, in order
    """
    mligs = [r.index for r in topology.residues if
             any(r.name == l for l in ligands)]

    # Some sanity checking
    lsel = atomsel("resname %s" % " or resname ".join(ligands), molid)
    vligs = atomsel("residue %s" % ' '.join(str(i) for i in mligs), molid)
    assert len(mligs) == len(set(lsel.get("residue")))
    assert len(set(vligs.get("residue"))) == len(set(lsel.get("residue")))

    return sorted(mligs)

#==============================================================================

def color_ligands(data, molids, topology, ligands, featidx):
    """
    Sets the user field of each alprenolol molecule according to the
    values in data, on a per frame basis.

    Args:
        data (list of 1D array): Should be a list same length as the
          number of ligand molecules, with array length same as trajectory.
    """
    assert len(data) == len(ligands)*len(molids)

    # Normalize the feature across all trajectories
    minl = min(min(d[:,featidx]) for d in data)
    rge = max(max(d[:,featidx]) for d in data) - minl

    for mx,m in enumerate(sorted(molids)):
        sels = [ atomsel("residue %d" % l, m) for l in ligands ]
        for i,s in enumerate(sels):
            assert len(data[mx*len(ligands)+i]) == molecule.numframes(m)

            dat = (data[mx*len(ligands)+i][:,featidx] - minl)/rge
            assert len(dat) == molecule.numframes(m)

            for f in range(molecule.numframes(m)):
                molecule.set_frame(m, f)
                s.update()
                #s.set("user", data[len(ligands)*mx+i][f][featidx])
                s.set("user", dat[f])

#==============================================================================

def clear_representations(molid):
    """
    Deletes all representations associated with a molecule
    """
    while molrep.num(molid):
       molrep.delrep(molid, 0)

#==============================================================================

def generate_cluster_representations(molids, data, ligands, clustvis):
    """
    Generates sets of representations for each cluster. The resulting
    data from these sets can be used to visualized.

    Args:
        molids (list of int): Molecule IDs to generate clusters for
        data (numpy array): Cluster ID data from msmbuilder
        ligands (list of int): Ligand IDs, in order
        clustvis (list of int): Which clusters to display

    Returns:
        (dict int -> list of tuple): Cluster dictionary, connecting cluster
            name to list of representations, which are each a tuple consisting
            first of molid and then unique representation name. There is
            also an entry "-1" for the basic protein representation
    """

    assert len(data) == len(ligands)*len(molids)

    clusters = {}
    clusters[-1] = []
    for c in clustvis:
        clusters[c] = []

    for mx, m in enumerate(molids):
        clear_representations(m)
        # Create the basic set of representations
        molrep.addrep(m, style="NewRibbons", selection="protein or resname ACE NMA",
                      color="ColorID 6")
        repname = molrep.get_repname(m, molrep.num(m)-1)
        clusters[-1].append((m, repname))
        # Basic ligand representation to always show
        #molrep.addrep(m, style="CPK",
        #              selection="residue %s" % " ".join(str(_) for _ in ligands),
        #              color="Type")
        #repname = molrep.get_repname(m, molrep.num(m)-1)
        #clusters[-1].append((m, repname))
        
        # Now ones for each cluster
        for c in clustvis:
            for i, l in enumerate(ligands):

                # Get the frames corresponding to this cluster
                assert len(data[mx*len(ligands)+i]) == molecule.numframes(m)
                frames = [_ for _, d in enumerate(data[mx*len(ligands)+i]) if d==c]
                if not len(frames): continue

                # Add a representation with all these frames
                molrep.addrep(m, style="Licorice",
                              selection="residue %d and noh" % l,
                              color="Type")
                molrep.set_visible(m, molrep.num(m)-1, False)
                repname = molrep.get_repname(m, molrep.num(m)-1)
                evaltcl("mol drawframes %d %d %s"
                        % (m, molrep.num(m)-1, ','.join(str(_) for _ in frames)))

                # Put this pair on the stack of cluster datas
                clusters[c].append((m,repname))

    return clusters

#==============================================================================

def show_clusters(clusters, clustertable):
    """
    Sets the representations for each molecule to show representations
    for the specified cluster

    Args:
        cluster (list of int): The cluster(s) to display
        clustertable: Output from generate_cluster_representations
    """
   
    # Hide all representations
    assert clustertable.get(-1) is not None
    for m,_ in clustertable.get(-1):
        for r in range(molrep.num(m)):
            molrep.set_visible(m, r, False)

    # Show basic representations
    for m, nam in clustertable.get(-1):
        molrep.set_visible(m, molrep.repindex(m, nam), True)

    # Now show representations for the given cluster in the right color
    for i,c in enumerate(clusters):
        assert clustertable.get(c) is not None
        for m, nam in clustertable.get(c):
            rep = molrep.repindex(m, nam)
            molrep.set_visible(m, rep, True)
            if len(clusters) > 1:
                molrep.modrep(m, rep, color="ColorID %d" % i)

#==============================================================================

def show_frame_clusters(molids, data, ligands):
    """
    Given a molid and frame, shows all clusters represented in
    that frame, colored differently. The molid and frame
    are obtain from the current top molecule.

    Args:
        clustertable: Output from generate_cluster_representations
        data (numpy): msmbuilder clustering results
    """
    pos = molecule.listall().index(molecule.get_top())
    stride = len(data)/molecule.num()
    c = [ d[molecule.get_frame(molecule.get_top())] 
          for d in data[stride*pos:stride*pos+len(ligands)] ]
    print("Clusters present in frame %d: %s"
          % (molecule.get_frame(molecule.get_top()), c))
    clustertable = generate_cluster_representations(molids, data, ligands, c)
    show_clusters(c, clustertable)

#==============================================================================

def set_representations(ids, ligands, clear=True):
    """
    Sets the representation of each molecule and alprenolol

    Args:
        ids (list of int): Molecule IDs
        ligands (list of int): Ligand IDs, in order
        clear (bool): Whether to delete previous representations
    """

    for m in ids:
        # Clear existing representations
        if clear:
            clear_representations(m)

        molrep.addrep(m, style="NewRibbons", selection="protein", color="ColorID 6")
        for l in ligands:
            molrep.addrep(m, style="Licorice", selection="noh and residue %d" % l,
                          color="User")
            molrep.set_colorupdate(m, molrep.num(m)-1, True)
            evaltcl("mol scaleminmax %d %d 0.0 1.0" % (m, molrep.num(m)-1))

#==============================================================================

def get_msm_clusters(molids, msm, clusters, ligands):
    """
    Generates and shows clusters selected to resample

    Args:
        molids (list of int): Molecules to view
        msm (MarkovStateModel): Built model from previous generation
        clusters: Pickled clusters
        ligands (list of int): Ligand IDs, in order

    Returns:
        mins (list of int): Cluster numbers to resample
        clustl (list): Representations for 50 least sampled clusters
    """
    msm_to_clust = dict((v,k) for k,v in msm.mapping_.iteritems())
    mins = [msm_to_clust[x] for x in msm.populations_.argsort()[:50]]

    for m in mins:
        frames = {k:v for k,v in {i:np.ravel(np.where(c==m)) \
                                  for i,c in enumerate(clusters)}.iteritems() \
                  if len(v)}
    clustl = generate_cluster_representations(molids, clusters, ligands, mins)
    return mins, clustl

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                  LOADERS                                   #
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_data(filenames, identifiers, topology, stride):
    """
    Loads a given set of trajectories

    Args:
        filenames (list of str): Path of files to load
        identifiers (list of str): Molecule name for each file
        topology (str): Topology to load
        stride (int): Number of frames to downsample

    Returns:
        (dict str->int) Molecule name and VMD molid
    """

    loaded = {}
    for i,f in enumerate(filenames):
       filetype = f.split('.')[-1]
       if filetype == "nc": filetype="netcdf"
       loaded[identifiers[i]] = molecule.load(topology[-3:], topology)
       molecule.read(loaded[identifiers[i]], filetype, f, skip=stride, waitfor=-1)
       molecule.rename(loaded[identifiers[i]], identifiers[i])
    return loaded

#==============================================================================

def load_topology(filename, topology):
    """
    Loads the topology as a mdtraj object

    Args:
        filename (str): A trajectory file to attach topology to
        topology (str): The topology to load

    Returns:
        (mdtraj Topology) topology
    """
    t = md.load(filename, top=topology, stride=1000)
    return t.topology

#==============================================================================

def load_sampled(topology, eqdir):
    """
    Loads all equilibrated files so you can see what's been sampled.

    Args:
        topology (str): Path to the psf file
        eqdir (str): Path to the equilibration directory
    """
    for gen in glob("%s/*" % eqdir):
        for rep in glob("%s/*" % gen):
            name = "G%s-r%s" % (gen, rep)
            print("Loading: %s" % name)
            mid = molecule.load("psf", topology)
            molecule.read(mid, "netcdf", "%s/Eq_5.nc" % rep, skip=1000, waitfor=-1)
            molecule.rename(mid, name)

#==============================================================================

