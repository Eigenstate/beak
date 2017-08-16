import os
import sys
from beak.msm import utils
from configparser import ConfigParser
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
    dats = p.map(
                 lambda x: process_traj(traj=x,
                                        lignames=cfg["system"]["ligands"].split(","),
                                        config=config),
                 utils.get_prodfiles(generation,
                                     rootdir=cfg["system"]["rootdir"],
                                     equilibration=cfg.getboolean("model",
                                                                  "include_equilibration"))
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
    atomsel("all", refid).set("beta", 0)
    for d in dats:
        sel = atomsel("index %d" % d, refid)
        sel.set("beta", int(sel.get("beta")[0])+1)

    molecule.write(refid, "mae", outname)
