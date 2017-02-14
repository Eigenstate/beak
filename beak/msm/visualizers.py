"""
Contains functions that operate on Sampler objects to
change what is visualized
"""
import math
import numpy as np
from msmbuilder.tpt import hub_scores, top_path, net_fluxes
try:
    from vmd import *
    atomsel = atomsel.atomsel
except:
    import vmd
    from atomsel import atomsel
    from VMD import evaltcl
    import graphics
    import molecule, molrep
    import vmdnumpy

#==========================================================================

def get_representative_ligand(samp, cluster, data=None):
    """
    Visualizes and generates a representative ligand for the given cluster.
    Finds the average coordinates of the cluster and then finds the closest
    ligand to that.

    Args:
        samp (Sampler): Sampler to evaluate
        cluster (int): Which cluster
        data (list of list of int): Cluster information, defaults to
            samp.mclust

    Returns:
        tuple (int, int, int), float: Molid, frame, ligand ID of representative
            ligand, and RMSD to average structure
    """
    # Set default
    if data is None:
        data = samp.mclust

    # Get the average of the cluster
    count = 0
    molframes = []
    ligids = sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                                molid=samp.molids[0]).get("residue")))
    ligheavyatoms = len(atomsel("noh and same fragment as residue %d"
                        % ligids[0], molid=samp.molids[0]))
    masks = {}

    mean = np.zeros((ligheavyatoms, 3)) # TODO get natoms in ligand
    for mx, m in enumerate(samp.molids):
        ligands = sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                                     molid=m).get("residue")))
        masks.update(dict(("%d-%d" % (m,l),
                           vmdnumpy.atomselect(m, 0,
                          "noh and same fragment as residue %d" % l))
                           for l in ligands))

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
                mean += np.compress(masks["%d-%d"%(m,l)], vmdnumpy.timestep(m, f), axis=0)
    if not count:
        return None
    mean /= float(count)

    # Find the closest frame to that one
    rmsd = []
    for molid, frame, ligand in molframes:
        rmsd.append(np.sum(np.sqrt(np.sum((np.compress(masks["%d-%d" %(molid,ligand)], vmdnumpy.timestep(molid, frame), axis=0)-mean)**2, axis=1)) ))

    return molframes[np.argmin(rmsd)], min(rmsd)

#==============================================================================

def get_msm_clusters(msm, clust, samp, scores=None):
    """
    Generates and shows clusters selected to resample

    Args:
        msm (MarkovStateModel): MSM to get clusters of
        clust (clusters): Clusters to visualize
        samp (Sampler): MSM trajecotry thing
        scores (list of float): Cluster hub scores, if none will calculate

    Returns:
        mins (list of int): Cluster numbers to resample
        clustl (list): Representations for 50 least sampled clusters
        scores (list of float): Scores for clusters
    """
    if scores is None or not len(scores):
        scores = hub_scores(msm)

    mins = list(msm.inverse_transform(scores.argsort())[0])

    for m in mins:
        frames = {k:v for k,v in {i:np.ravel(np.where(c==m)) \
                                  for i,c in enumerate(clust)}.items() \
                  if len(v)}
    clustl = generate_cluster_representations(samp, mins, clust)
    return mins, clustl, scores

#==========================================================================

def generate_cluster_representations(samp, clustvis, clust=None):
    """
    Generates sets of representations for each cluster. The resulting
    data from these sets can be used to visualized.

    Args:
        samp (Sampler): MSM loaded trajectory thing
        clustvis (list of int): Which clusters to display
        clust: Loaded clusters, defaults to samp.mclust

    Returns:
        (dict int -> list of tuple): Cluster dictionary, connecting cluster
            name to list of representations, which are each a tuple consisting
            first of molid and then unique representation name. There is
            also an entry "-1" for the basic protein representation
    """
    if clust is None:
        clust = samp.mclust

    assert len(clust) == samp.num_ligands*len(samp.molids)

    clusters = {}
    clusters[-1] = []
    for c in clustvis:
        clusters[c] = []

    protvis = False
    for mx, m in enumerate(samp.molids):
        # Get ligand ids for this molecule
        ligands = sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                                     molid=m).get("residue")))

        # Clear representations for this molid
        while molrep.num(m):
           molrep.delrep(m, 0)

        # Create the basic set of representations
        if not protvis:
            molrep.addrep(m, style="NewRibbons",
                          selection="(protein or resname ACE NMA) and not same "
                                    "fragment as resname %s" % " ".join(samp.ligands),
                          color="ColorID 6")
            protvis = True
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
                # Skip trimmed trajectories bc cant correlate them
                if len(clust[mx*len(ligands)+i]) != molecule.numframes(m):
                    continue
                if c=="nan":
                    frames = [_ for _, d in enumerate(clust[mx*samp.num_ligands+i]) if np.isnan(d)]
                else:
                    frames = [_ for _, d in enumerate(clust[mx*samp.num_ligands+i]) if d==c]
                if not len(frames): continue

                # Add a representation with all these frames
                molrep.addrep(m, style="Licorice 0.3 12.0 12.0",
                              selection="noh and same fragment as residue %d " % l,
                              color="Type",
                              material="AOShiny")
                molrep.set_visible(m, molrep.num(m)-1, False)
                repname = molrep.get_repname(m, molrep.num(m)-1)
                evaltcl("mol drawframes %d %d %s"
                        % (m, molrep.num(m)-1, ','.join(str(_) for _ in frames)))

                # Put this pair on the stack of cluster datas
                clusters[c].append((m,repname))

    return clusters

#==============================================================================

def color_ligands(samp, data, featidx):
    """
    Sets the user field of each molecule according to the
    values in data, on a per frame basis.

    Args:
        samp (Sampler): MSM trajectory object
        data (np thing): Featurization data
        featidx (int): Feature index to show by
    """
    assert len(data) == samp.num_ligands*len(samp.molids)

    # Normalize the feature across all trajectories
    minl = min(min(d[:,featidx]) for d in data)
    rge = max(max(d[:,featidx]) for d in data) - minl

    for mx,m in enumerate(sorted(samp.molids)):
        ligands = sorted(set(atomsel("resname %s" % " ".join(ligands)),
                                     molid=m).get("residue"))
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

#==========================================================================

def clear_representations(samp):
    """
    Deletes all representations for this sampler

    Args:
        samp (Sampler): Sampler to delete reps for
    """
    for molid in samp.molids:
        while molrep.num(molid):
           molrep.delrep(molid, 0)

#==============================================================================

def display_msm(samp, msm=None, clust=None, states=None):
    """
    Creates a visual representation of an msm

    Args:
        samp (Sampler): MSM trajectory object
        msm (MarkovStateModel): Model to visualize, default samp.mmsm
        clust (list of ndarray): Cluster data for msm, default samp.mclust
        states (list of int): Which states to visualise. If None,
            will show all states. Should match the clustering,
            not the internal MSM states.

    Returns:
        int: Geometry fake molecule that has all the lines on it
    """
    # Set default options
    if msm is None:
        msm = samp.mmsm
    if clust is None:
        clust = samp.mclust
    if states is None:
        states = msm.mapping_.keys()

    # Clear any representations or displayed graphics primitives
    clear_representations(samp)
    for m in samp.molids:
        for g in graphics.listall(m): graphics.delete(m, g)
    geom = molecule.load("graphics", "msm_vis")

    # Show the protein as ribbons
    molrep.addrep(samp.molids[0], style="NewRibbons",
                  selection="protein", color="ColorID 6",
                  material="Translucent")

    # Generate a representative cluster for each state
    rms = []; reps = []
    handles = {} # Will hold representative state coordinates
    for c in states:
        msmstate = msm.mapping_[c]
        lg, r = get_representative_ligand(samp, c, clust)
        rms.append(r)
        if lg is None:
            print("Cluster %d unrepresented" % c)
            continue
        m,f,l = lg
        atomsel("same fragment as residue %d" % l, molid=m).set("user", math.log(msm.populations_[msmstate]))
        molrep.addrep(m, style="Licorice 0.3 12.0 12.0" % rms, color="User",
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

def show_frame_clusters(sampler, clust=None):
    """
    Shows all clusters present in the given frame of the top
    molecule

    Args:
        sampler (Sampler): MSM trajectory object
        clust (np arrays): Clustered data to view, defaults to samp.mclust

    Returns:
        c (list of int): Indices of present clusters
        clustertable (list): Representations for those clusters
    """
    if clust is None:
        clust = sampler.mclust

    pos = sampler.molids.index(molecule.get_top())
    c = [ d[molecule.get_frame(molecule.get_top())]
          for d in clust[sampler.num_ligands*pos:(1+pos)*sampler.num_ligands] ]
    print("Clusters present in frame %d: %s"
          % (molecule.get_frame(molecule.get_top()), c))
    clustertable = generate_cluster_representations(sampler, c, clust)
    show_clusters(c, clustertable, sampler)
    return c, clustertable

#==============================================================================

def show_clusters(clusters, clustertable, samp):
    """
    Sets the representations for each molecule to show representations
    for the specified cluster

    Args:
        cluster (list of int): The cluster(s) to display
        clustertable: Output from generate_cluster_representations
        samp (Sampler): MSM trajectory thing
    """

    # Hide all representations
    assert clustertable.get(-1) is not None
    #for m,_ in clustertable.get(-1):
    for m in samp.molids:
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

def set_representations(samp, clear=True):
    """
    Sets the representation of each molecule and ligand
    in the default way

    Args:
        samp (Sampler): MSM trajectory object
        clear (bool): Whether to delete previous representations
    """
    if clear: clear_representations(samp)

    for m in samp.molids:
        molrep.addrep(m, style="NewRibbons", selection="protein", color="ColorID 6")
        for l in sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                                    molid=m).get("residue"))):
            molrep.addrep(m, style="Licorice 0.3 12.0 12.0",
                          selection="noh and same fragment as residue %d" % l,
                          color="User")
            molrep.set_colorupdate(m, molrep.num(m)-1, True)
            evaltcl("mol scaleminmax %d %d 0.0 1.0" % (m, molrep.num(m)-1))

#==============================================================================

def draw_representative_ligands(samp, clust, states, colors=None):
    """
    Draws a representative ligand for each cluster frame

    Args:
        samp (Sampler): MSM trajectory object
        clust (clusters): Which clusters to show
        states (list of int): Which clusters to show
        colors (list of float): Number by which to color clusters
    """
    clear_representations(samp)
    if colors is not None:
        realc = [_/sum(colors) for _ in colors] # Normalize colors

    # Draw protein once
    molrep.addrep(samp.molids[0], style="NewRibbons",
                  selection="(protein or resname ACE NMA) and not same "
                            "fragment as resname %s" % " ".join(samp.ligands),
                  color="ColorID 6")
    for i, c in enumerate(states):
        lg, r = get_representative_ligand(samp, c, clust)
        if lg is None:
            print("Cluster %d unrepresented" % c)
            continue
        m,f,l = lg
        if colors is None:
            molrep.addrep(m, style="Licorice", material="Opaque", color="ColorID %d" %i,
                          selection="noh and same fragment as residue %d" % l)
        else:
            atomsel("same fragment as residue %d" % l, molid=m).set("user", realc[i])
            molrep.addrep(m, style="Licorice", material="Opaque", color="User",
                          selection="noh and same fragment as residue %d" % l)
        evaltcl("mol drawframes %d %d %s"
                % (m, molrep.num(m)-1, f))

#==============================================================================

def closest_to_bound(samp, clust, msm, truesel, trueid):
    """
    Visualizes and returns the closest cluster to the bound pose.

    Args:
        samp (Sampler): MSM trajectory object
        clust (cluster): Which clusters to consider
        msm (MarkovStateModel): Markov model
        truesel (str): Atom selection for true bound pose
        trueid (int): VMD molecule ID of molecule with bound pose

    Returns:
        (list of float/int): Cluster labels, in order, of closest to bound
    """
    # Get the number of heavy atoms in one ligand
    ligids = sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                                molid=samp.molids[0]).get("residue")))
    ligheavyatoms = len(atomsel("noh and same fragment as residue %d"
                        % ligids[0], molid=samp.molids[0]))

    # Find average structure for each cluster
    clustmeans = {}; num = {}
    for mx, molid in enumerate(sorted(samp.molids)):
        for i, l in enumerate(sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                molid=molid).get("residue")))):
            mask = vmdnumpy.atomselect(molid, 0,
                                       "noh and same fragment as residue %d" % l)
            for frame in range(molecule.numframes(molid)):
                c = clust[mx*samp.num_ligands+i][frame]

                # Handle uninitialized cluster
                if clustmeans.get(c) is None:
                    clustmeans[c] = np.zeros((ligheavyatoms, 3))
                    num[c] = 0.
                clustmeans[c] += np.compress(mask, vmdnumpy.timestep(molid, frame), axis=0)
                num[c] += 1.

    # Now the sum is finished, do the division
    for cl in clustmeans:
        clustmeans[cl] = np.divide(clustmeans[cl], num[cl])

    # Get the atom selection mask for the known bound ligand
    tmask = vmdnumpy.atomselect(trueid, 0, "noh and (%s)" % truesel)
    bound = np.compress(tmask, vmdnumpy.timestep(trueid, 0), axis=0)

    # Find the closest frame to that one
    rmsds = {}
    for c in clustmeans:
        r = np.sum(np.sqrt(np.sum((bound-clustmeans[c])**2, axis=1)))
        rmsds[r] = c

    return [rmsds[x] for x in np.sort(rmsds.keys())]

#==============================================================================

def show_binding_pathway(samp, bound, clust, msm, scores=None):
    """
    Visualizes the clusters along the binding pathway compared to
    a known bound pose. Pathway is the most probable one to bulk solvent.

    Args:
        samp (Sampler): MSM trajectory object
        bound (cluster): Which cluster corresponds to the bound pose
        clust (clusters): Which clusters to consider
        msm (MarkovStateModel): Which MSM to consider
        scores (list of float): Hub scores, or will be computed if none

    Returns:
        (list of int): Clusters along the binding pathway, including
            the closest one to the known bound pose, not including bulk solvent.
    """
    # Compute scores if undefined
    if scores is None:
        scores = hub_scores(msm)

    # Identify the solvent cluster and the bound cluster
    solvent = np.argmax(scores)

    # Get the top pathway and remove solvent from it
    flux = net_fluxes(sources=[solvent], sinks=[bound], msm=msm)
    pathway = list(top_path(sources=[solvent], sinks=[bound], net_flux=flux)[0])
    print(pathway)

    # Visualize representative ligands in pathway
    draw_representative_ligands(samp, clust, pathway,
                                colors=range(len(pathway)))
    return pathway

#==============================================================================

