#!/usr/bin/env python

from beak import MinDistanceAnalyzer, TrajectorySet

trajs = []
sel1 = "(resname UNC7 and name N1) or (resname UNC8 and name N3)"
sel2 = "resname ASP and resid 114 and name OD1 OD2"

ldir = "/home/robin/Work/Projects/D2_dopamine/ligand_simulations/"
sdir = "/home/robin/Work/Projects/D2_dopamine/sherlock_scratch/ligand_simulations/"
trajs.append(TrajectorySet(name="UNC3279 inactive",
                           psf=ldir+"UNC3279_inactive/inp04_02D2_inactive_unc3279_dabbled_trans.psf",
                           directory=sdir+"unc3279_inactive/production/04",
                           ligand="UNC7",
                           stride=10, save_frequency=125.,
                           color="blue"))

trajs.append(TrajectorySet(name="UNC3286 inactive",
                           psf=ldir+"UNC3286_inactive/inp04_02D2_inactive_unc3286_dabbled_trans.psf",
                           directory=sdir+"unc3286_inactive/production/04",
                           ligand="UNC8",
                           stride=10, save_frequency=125.,
                           color="red"))

trajs.append(TrajectorySet(name="UNC3279 active",
                           psf=ldir+"UNC3279_active/inp07_06D2_active_unc3279_dabbled_trans.psf",
                           directory=sdir+"unc3279_active/production/07",
                           ligand="UNC7",
                           stride=10, save_frequency=200.,
                           color="purple"))

trajs.append(TrajectorySet(name="UNC3286 active",
                           psf=ldir+"UNC3286_active/inp07_06D2_active_unc3286_dabbled_trans.psf",
                           directory=sdir+"unc3286_active/production/07",
                           ligand="UNC8",
                           stride=10, save_frequency=200.,
                           color="orange"))

a = MinDistanceAnalyzer(data=trajs, selection1=sel1, selection2=sel2)
fig = a.plot(title="Asp 114 Salt Bridge", ylabel="Pip N - Asp O distance (A)")
fig.show()

# Keep figure up
raw_input()
