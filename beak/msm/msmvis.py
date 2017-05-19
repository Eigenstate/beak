"""
Contains methods for generally visualizing MSMs that are
loaded into DensitySamplers, usually
"""
from vmd import molecule, atomsel

#==============================================================================

def add_frames(samp, cluster, num=10):
    """
    Adds representative frames of the given molecule to the visualizer
    session.

    Args:
        samp (DensitySampler): Sampler to add frames to
        cluster (int): Cluster label to visualize
        num (int): Number of frames to load

    """
    for _ in range(num):
        samp._load_frame(cluster)

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

    for mx,mids in enumerate(sorted(samp.molids)):
        for m in mids:
            ligands = sorted(set(atomsel("resname %s" % " ".join(samp.ligands),
                                         molid=m).get("residue")))
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
