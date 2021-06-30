import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sys

# sys.path.append("/home/groups/copperma/ZuckermanLab/msmWE")
# import msm_we
import h5py
import pickle
import os
import subprocess
import mdtraj as md

sys.path.append("/Users/russojd/Research/molssi_2021/msmWE_cc/msm_we")
from msm_we import msm_we
"""
This code

1. Initializes a model with the given PDBs and file globs

2. Opens the associated H5 file, in append mode

3. Reads through each of the trajectory files, putting the coordinates into the H5 file's auxdata/coord (if they don't already exist)
"""

# fileSpecifier='/home/groups/ZuckermanLab/mostofia/ProteinFolding/NTL9/WestPA/Gamma05ps/Pcoord1D/OpenMM/Run03/west_prev.h5'
# refPDBfile='/home/groups/ZuckermanLab/copperma/msmWE/NTL9/reference.pdb'
# initPDBfile='/home/groups/ZuckermanLab/copperma/msmWE/NTL9/coor0.pdb'
# modelName='NTL9_Run03_jan9'
fileSpecifier = sys.argv[1]
refPDBfile = sys.argv[2]
initPDBfile = sys.argv[3]
modelName = sys.argv[4]
WEfolder = sys.argv[5]

# Model is loaded to get the number of iterations and segments
model = msm_we.modelWE()
print("initializing...")
model.initialize(fileSpecifier, refPDBfile, initPDBfile, modelName)
model.get_iterations()

f = h5py.File(model.fileList[0], "a")

# For each iteration, look at all the trajectory segments that were generated and add them to the h5

# Loop over iterations
for n_iter in range(1, model.maxIter + 1):

    # Print some logging output every 10 steps
    if n_iter % 10 == 0:
        sys.stdout.write("copying coords into westfile iteration " + str(n_iter) + "\n")

    nS = model.numSegments[n_iter - 1].astype(int)

    # Coords will be an array indexable by
    #   (Segment, Parent/Child, Atom, Position)
    coords = np.zeros((nS, 2, model.nAtoms, 3))

    # Loop over all segments in this iteration
    for iS in range(nS):

        # Load up the file corresponding to this segment, and extract the relevant coordinates
        trajpath = WEfolder + "/traj_segs/%06d/%06d" % (n_iter, iS)

        parent_coords = np.squeeze(
            md.load(
                trajpath + "/parent.xml", top=model.reference_structure.topology
            )._xyz
        )
        child_coords = np.squeeze(
            md.load(trajpath + "/seg.xml", top=model.reference_structure.topology)._xyz
        )
        coords[iS, 0, :, :] = parent_coords
        coords[iS, 1, :, :] = child_coords

    # Try to put the newly-obtained coordinates in the H5 file, if they're not already there.
    dsetName = "/iterations/iter_%08d/auxdata/coord" % int(n_iter)
    try:
        dset = f.create_dataset(dsetName, np.shape(coords))
        dset[:] = coords
    except:
        sys.stdout.write(
            "coords exist for iteration " + str(n_iter) + " NOT overwritten\n"
        )

f.close()
