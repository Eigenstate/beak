import math
import mdtraj as md
import molecule
import molrep
import numpy as np
import trans
import vmdnumpy
from atomsel import atomsel
from glob import glob
from msmbuilder.msm import MarkovStateModel
from VMD import evaltcl, graphics

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
        sels = [ atomsel("same fragment as residue %d" % l, m) for l in ligands ]
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
                # Skip trimmed trajectories bc cant correlate them
                if len(data[mx*len(ligands)+i]) != molecule.numframes(m):
                    continue
                if c=="nan":
                    frames = [_ for _, d in enumerate(data[mx*len(ligands)+i]) if np.isnan(d)]
                else:
                    frames = [_ for _, d in enumerate(data[mx*len(ligands)+i]) if d==c]
                if not len(frames): continue

                # Add a representation with all these frames
                molrep.addrep(m, style="Licorice",
                              selection="same fragment as residue %d and noh" % l,
                              color="Type")
                molrep.set_visible(m, molrep.num(m)-1, False)
                repname = molrep.get_repname(m, molrep.num(m)-1)
                evaltcl("mol drawframes %d %d %s"
                        % (m, molrep.num(m)-1, ','.join(str(_) for _ in frames)))

                # Put this pair on the stack of cluster datas
                clusters[c].append((m,repname))

    return clusters

#==============================================================================

def display_msm(molids, clust, ligands, msm, states=None):
    """
    Creates a visual representation of an msm

    Args:
        molids (list of int): Molecule IDs loaded
        clust (list of ndarray): Cluster data MSM was built from
        ligands (list of int): Residue numbers of ligands
        msm (MarkovStateModel): The model to visualise
        states (list of int): Which states to visualise. If None,
            will show all states. Should match the clustering,
            not the internal MSM states.

    Returns:
        int: Geometry fake molecule that has all the lines on it
    """
    # Clear any representations or displayed graphics primitives
    for m in molids:
        clear_representations(m)
        for g in graphics.listall(m): graphics.delete(m, g)
    geom = molecule.load("graphics", "msm_vis")

    if not states:
        states = msm.mapping_.keys()

    # Show the protein as ribbons
    molrep.addrep(molids[0], style="NewRibbons",
                  selection="protein", color="ColorID 6")

    # Generate a representative cluster for each state
    rms = []; reps = []
    handles = {} # Will hold representative state coordinates
    for c in states:
        msmstate = msm.mapping_[c]
        lg, r = get_representative_ligand(c, clust, molids, ligands)
        rms.append(r)
        if lg is None:
            print("Cluster %d unrepresented" % c)
            continue
        m,f,l = lg
        atomsel("same fragment as residue %d" % l, molid=m).set("user", math.log(msm.populations_[msmstate]))
        molrep.addrep(m, style="Licorice 0.1 12.0 12.0" % rms, color="User",
                      selection="noh and same fragment as residue %d" % l) 
        reps.append((m, molrep.num(m)-1))
        evaltcl("mol scaleminmax %d %d %f %f" % (m, molrep.num(m)-1, math.log(min(msm.populations_)), math.log(max(msm.populations_)) ))
        evaltcl("mol drawframes %d %d %s"
                % (m, molrep.num(m)-1, f))

        # Get a handle on this state and add it
        atoms = atomsel("residue %d" % l, molid=m).get('index')
        handles["%s-in" % c] = vmdnumpy.timestep(m,f)[atoms[0]]
        handles["%s-out" % c] = vmdnumpy.timestep(m,f)[atoms[1]]

    # Normalize to show width
    print rms
    rms = [r/sum(rms) * len(states)/5. for r in rms]
    for i, m in enumerate(reps):
        molrep.modrep(m[0], m[1], "Licorice %f 12.0 12.0" % max(0.1,rms[i]))

    # Draw lines between states with txn probability above threshold
    graphics.color(geom, "red")
    for i, c1 in enumerate(states):
        for j, c2 in enumerate(states[i+1:]):
            cin = c2; cout = c1; tx = msm.transmat_[c1][c2]
            if tx < msm.transmat_[c2][c1]:
                cin = c1; cout = c2
                tx = msm.transmat_[c2][c1]
            #p = 255.*msm.transmat_[c1][c2]
            #graphics.color(geom, (p,p,p))
            if tx > 0.05: # TODO better threshold
                slope = handles["%s-in"% cin]-handles["%s-out" % cout]
                yval = handles["%s-in"%cin]-0.3*slope
                graphics.cylinder(geom, tuple(handles["%s-out" % cout]),
                                  tuple(yval),
                                  radius=2*tx,
                                  resolution=10, filled=True)
                graphics.cone(geom, tuple(yval), tuple(handles["%s-in" % cin]),
                              radius=5*tx,
                              resolution=10)
                #graphics.sphere(geom, handles["%s-in" % c1], radius=0.1)

    return geom

#==============================================================================

def get_representative_ligand(cluster, data, molids, ligands):
    """
    Visualizes and generates a representative ligand for the given cluster.
    Finds the average coordinates of the cluster and then finds the closest
    ligand to that.

    Args:
        cluster (int): Which cluster
        data (list of ndarray): Cluster data
        molids (list of int): Molecule IDs to search
        ligands (list of int): Ligand IDs, in order

    Returns:
        tuple (int, int, int), float: Molid, frame, ligand ID of representative
            ligand, and RMSD to average structure
    """
    masks = dict((l,vmdnumpy.atomselect(molids[0], 0, "residue %d and noh" % l)) for l in ligands)

    # Get the average of the cluster
    count = 0
    molframes = []
    ligheavyatoms = len(atomsel("residue %d and noh" % ligands[0], molids[0]))
    mean = np.zeros((ligheavyatoms, 3)) # TODO get natoms in ligand
    for mx, m in enumerate(molids):
        for i, l in enumerate(ligands):
            if len(data[mx*len(ligands)+i]) != molecule.numframes(m):
                print("MISMATCH %d, %d" % (len(data[mx*len(ligands)+i]), molecule.numframes(m)))
                continue
            if cluster == "nan":
                frames = [_ for _, d in enumerate(data[mx*len(ligands)+i]) if np.isnan(d)]
            else:
                frames = [_ for _, d in enumerate(data[mx*len(ligands)+i]) if d==cluster]

            # Add frames to running total
            for f in frames:
                count += 1
                molframes.append((m,f,l))
                mean += np.compress(masks[l], vmdnumpy.timestep(m, f), axis=0)
    if not count:
        return None
    mean /= float(count)

    # Find the closest frame to that one
    rmsd = []
    for molid, frame, ligand in molframes:
        rmsd.append(np.sum(np.sqrt(np.sum((np.compress(masks[ligand], vmdnumpy.timestep(molid, frame), axis=0)-mean)**2, axis=1)) ))

    return molframes[np.argmin(rmsd)], min(rmsd)

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
            molrep.addrep(m, style="Licorice 0.3 12.0 12.0", selection="noh and same fragment as residue %d" % l,
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

