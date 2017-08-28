import os
import sys
from beak.msm import utils
from configparser import ConfigParser
from itertools import repeat
from multiprocessing import Pool
from vmd import atomsel, molecule

def process_traj(traj, lignames, config):
    """
    Processes one trajectory file.

    Args:
        traj (str): Trajectory file to process
        lignames (list of str): Ligand names
        config (str): Path to sampling config file
    Returns:
        (list of int): Atom indices within 3A of ligand
    """

    molid = utils.load_trajectory(filename=traj,
                                  config=config)
    contacts = []

    csel = atomsel("protein and within 3 of "
                   "same fragment as resname %s" % " ".join(lignames))

    for frame in range(molecule.numframes(molid)):
        molecule.set_frame(molid, frame)
        csel.update()
        contacts.extend(csel.get("index"))

    # Handle non-stripped trajectory
    minp = min(atomsel("resname ACE").get("index"))
    for i in range(len(contacts)):
        contacts[i] -= minp

    molecule.delete(molid)
    print("Done with trajectory %s" % traj)
    sys.stdout.flush()

    return contacts

def get_coverage(config, outname, generation):
    """
    Sets the beta field of the reference structure depending on how
    much the ligand contacts the protein

    Args:
        config (str): Path to config file
        outname (str): Name of output file where beta will be set
    """
    cfg = ConfigParser(interpolation=None)
    cfg.read(config)

    p = Pool(int(os.environ.get("SLURM_NTASKS", 4)))
    prodfiles =  utils.get_prodfiles(generation,
                                     rootdir=cfg["system"]["rootdir"],
                                     equilibration=cfg.getboolean("model",
                                                                  "include_equilibration"))

    dats = p.starmap(process_traj,
                     zip(prodfiles, repeat(cfg["system"]["ligands"].split(",")),
                         repeat(config))
                    )

    if "psf" in cfg["system"]["reference"]:
        refid = molecule.load("psf", cfg["system"]["reference"],
                              "pdb", cfg["system"]["reference"].replace("psf",
                                                                        "pdb"))
    else:
        refid = molecule.load("parm7", cfg["system"]["reference"],
                              "crdbox", cfg["system"]["reference"].replace("prmtop",
                                                                           "inpcrd"))

    # A little inelegant but selecting by index is pretty fast
    atomsel("all", refid).set("mass", 0)
    for d in dats:
        for l in d:
            sel = atomsel("index %d" % l, refid)
            sel.set("mass", float(sel.get("mass")[0])+1.)

    molecule.write(refid, "mae", outname)
