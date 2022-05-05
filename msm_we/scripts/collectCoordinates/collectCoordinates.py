import numpy as np
import sys

import h5py
import pickle
import os
import subprocess
import mdtraj as md

import logging
log = logging.getLogger()

from msm_we import msm_we
"""
This code

1. Initializes a model with the given PDBs and file globs

2. Opens the associated H5 file, in append mode

3. Reads through each of the trajectory files, putting the coordinates into the H5 file's auxdata/coord (if they don't already exist)
"""

we_h5filename = sys.argv[1]
refPDBfile = sys.argv[2]
modelName = sys.argv[3]
WEfolder = sys.argv[4]
pcoord_ndim = sys.argv[5]

model = msm_we.modelWE()
log.info('Preparing coordinates...')

# Only need the model to get the number of iterations and atoms
# TODO: Replace this with something more lightweight, get directly from WE
log.debug(f'Doing collectCoordinates on  WE file {we_h5filename}')
model.initialize(
    [we_h5filename],
    refPDBfile,
    modelName,
    # Pass some dummy arguments -- these aren't important, this model is just created for convenience
    # in the coordinate collection. Dummy arguments prevent warnings from being raised.
    basis_pcoord_bounds=None,
    target_pcoord_bounds=None,
    tau=1,
    pcoord_ndim=pcoord_ndim,
    _suppress_boundary_warning=True,
)
model.get_iterations()

n_iter = None
for n_iter in tqdm.tqdm(range(1, model.maxIter + 1)):

    nS = model.numSegments[n_iter - 1].astype(int)
    coords = np.zeros((nS, 2, model.nAtoms, 3))
    dsetName = "/iterations/iter_%08d/auxdata/coord" % int(n_iter)

    coords_exist = False
    try:
        dset = h5file.create_dataset(dsetName, np.shape(coords))
    except (RuntimeError, ValueError):
        log.debug('coords exist for iteration ' + str(n_iter) + ' NOT overwritten')
        coords_exist = True
        continue

    for iS in range(nS):
        trajpath = WEfolder + "/traj_segs/%06d/%06d" % (n_iter, iS)

        try:
            coord0 = np.squeeze(md.load(f'{trajpath}/{parentTraj}', top=model.reference_structure.topology)._xyz)
        except OSError:
            log.warning("Parent traj file doesn't exist, loading reference structure coords")
            coord0 = np.squeeze(model.reference_structure._xyz)

        coord1 = np.squeeze(md.load(f'{trajpath}/{childTraj}', top=model.reference_structure.topology)._xyz)

        coords[iS, 0, :, :] = coord0
        coords[iS, 1, :, :] = coord1

    if not coords_exist:
        dset[:] = coords

log.debug(f"Wrote coords for {n_iter} iterations.")
