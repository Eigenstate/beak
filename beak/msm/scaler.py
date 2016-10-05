import vmd, molecule
from atomsel import *
import numpy as np

def get_scaler(prefix, steepness=0.5):
    """
    Returns a scaling function appropriate for the given system.
    The function will smoothly switch from the normal contact
    distance to the maximal possible z distance based on box size
    at the point corresponding to the z dimension of the lowest
    lipid.

    Args:
        molid (int): VMD molecule ID to obtain scaler for
        steepness (float): Steepness of the switching function,
            or the width over which switching will be applied.
    Returns:
        (function handle): Scaling function to pass to featurizer
    """

    min_z = min(atomsel("lipid", molid=molid).get('z'))
    zdim = max(atomsel(molid=molid).get('z')) \
           - min(atomsel(molid=molid).get('z'))

    def scaler(ligand_com, raw_dists):
        if len(ligand_com) != len(raw_dists):
            raise ValueError("Array size mismatch in scaling function")

        scale_factor = -0.5*np.tanh(steepness*ligand_com[:,2]-min_z)+0.5
        return raw_dists*(1.-scale_factor) + zdim*scale_factor

    return scaler

