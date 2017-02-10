# These are useful python functions that I commonly use in notebooks
from glob import glob
import vmd, molecule
import numpy
import vmdnumpy
import pandas as pd
from atomsel import atomsel

#===============================================================================

def align_on_tm(molid, ref):
    """
    Aligns on the transmembrane helices, assuming that the Ballasteros-Weinstein
    numbering is present in the mass field of the psf file. Molecule is
    aligned in place.

    Args:
        molid (int): VMD molecule ID to align
        ref (int): VMD molecule ID of reference structure to align to.
    """

    refsel = atomsel("protein backbone and mass > 0", molid=ref) # Align on TM helices
    sel = atomsel("protein backbone and mass > 0", molid=molid)
    
    for i in range(molecule.numframes(molid)):
        molecule.set_frame(molid, i)
        sel.update()
        diff = sel.fit(refsel)
        atomsel("all", molid=molid).move(diff)

#===============================================================================

def load_condition(psf, ref, skip, align):
    """
    Loads all replicates corresponding to an experimental condition.
    Automatically aligns on the transmembrane helices.

    Args:
        psf (str): Psf file to load. Revision and production directory will
                   be inferred from this path, so you need to be using my
                   directory structure, and pass the _trans.psf file.
        ref (int): Reference molecule ID for alignment
        skip (int): Number of frames to skip
        align (bool): Whether to align on the TM helices
    
    Returns:
        list of tuple: (molid, replicate) of all replicates loaded, with
                   their corresponding replicate number, so you don't have
                   to match it up with paths later.
    """
        
    revision = (psf.split('/')[-1]).split("_")[0].replace("inp","")
    prods = glob("/".join(psf.split("/")[:-1])+"/production/"+revision+
                 "/*/Reimaged_Eq6*_skip_1.nc")
    molids = []
    for p in sorted(prods, key=lambda x: int(x.split('/')[-2])):
        print("LOADING: %s" % p)
        a = molecule.load('psf',psf)
        molecule.read(a, 'netcdf', p, skip=skip, waitfor=-1)
        if (align):
            align_on_tm(a, ref)
        molids.append((a, int(p.split('/')[-2])))
    return molids

#===============================================================================

def sliding_mean(data_array, window=5):
    """
    Smooths an array of data with the given window size

    Args:
        data_array (numpy array): The 1D data array to smooth
        window (int): Size of the smoothing window

    Returns:
        numpy array or list: The smoothed data set

    Raises:
        ValueError if the data array isn't a numpy array or list
        ValueError if the data array is not one dimensional
    """

#    if not isinstance(data_array, (numpy.ndarray, list)):
#        raise ValueError("data_array must be numpy array or list")
#    if data_array.ndim != 1:
#        raise ValueError("data_array must be one dimensional!")
    if not window & 1:
        raise ValueError("Need odd number window")

    idx = numpy.array(data_array.index)

    box = int((window-1)/2.)
    new_list = numpy.empty((len(data_array)))
    for i in range(len(idx)):
        indices = range(max(i - box, 0),
                        min(i + box + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[idx[j]]
        avg /= len(indices)
        new_list[i] = avg

    return pd.Series(new_list, index=idx)

#===============================================================================

def calc_average_structure(molids, psf, minframe=0):
    """
    Calculates the average structure for a given trajectory and psf
    
    Args:
        molids (list of int): Trajectory IDs to average
        psf (str): Path to psf file describing this topology
        minframe (int): Frame to start computation from
    """
    data = []
    start_frame = minframe
    for m in molids:
        if start_frame >= molecule.numframes(m[0]): continue
        for f in range(start_frame, molecule.numframes(m[0])):
            data.append(vmdnumpy.timestep(m[0], f))
    avg = numpy.mean(numpy.stack(data), axis=0)

    # Now we have average coords, so set them in a new molecule
    if "_trans" in psf:
        pdb = psf.replace("_trans.psf", ".pdb")
    else:
        pdb = psf.replace(".psf", ".pdb")

    outid = molecule.load('psf', psf, 'pdb', pdb)
    atomsel("all", outid).set('x', avg[:,0])
    atomsel("all", outid).set('y', avg[:,1])
    atomsel("all", outid).set('z', avg[:,2])
    return outid

#===============================================================================

def calc_rmsf_to_average(molid, avg, selstr, minframe=0):
    """
    Calculates the RMSF of an atom selection 

    Args:
        molid (int): VMD molecule ID to calculate
        avg (int): VMD molecule ID of average structure
        selstr (str): Selection to compute RMSF over
        minframe (int): Frame to start computation from
    """
    mask = vmdnumpy.atomselect(avg, 0, selstr)
    ref = numpy.compress(mask, vmdnumpy.timestep(avg,0), axis=0)
    
    if molecule.numframes(molid) <= minframe:
        print("Only %d frames in %d" % (molecule.numframes(molid), molid))
        return None
    rmsf = numpy.zeros(len(ref))
    
    for f in range(minframe, molecule.numframes(molid)):
        frame = numpy.compress(mask, vmdnumpy.timestep(molid, f), axis=0)
        rmsf += numpy.sqrt(numpy.sum((frame-ref)**2, axis=1))
        
    rmsf /= (molecule.numframes(molid)-minframe)
    rmsf = numpy.sqrt(rmsf)
   
    return rmsf 

#===============================================================================
