from vmd import atomsel

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                        ALL PURPOSE METHODS                                 +
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_ligand_residues(molid, ligands):
    """
    Gets the VMD resids corresponding to the mdtraj ligands, in order.
    This order is the same as featurization

    Args:
        molid (int): VMD molecule ID to colour
          up ligands
        ligands (list of str): Ligand resnames to consider
    Returns:
        (list of int): Residue numbers of each ligands, in order
    """
    ligids = sorted(set(atomsel("resname %s" % " ".join(ligands)),
                                molid=molid).get("residue"))
    return ligids

