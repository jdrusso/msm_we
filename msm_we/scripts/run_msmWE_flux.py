import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("/Users/russojd/Research/molssi_2021/msmWE_cc/msm_we")
from msm_we import msm_we
import h5py
import pickle
import os
import subprocess

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("runner")
log.setLevel(logging.DEBUG)

# Parse command-line options
fileSpecifier = sys.argv[1]
refPDBfile = sys.argv[2]
initPDBfile = sys.argv[3]
modelName = sys.argv[4]
n_clusters = sys.argv[5]
last_iter = sys.argv[6]
n_clusters = int(n_clusters)
last_iter = int(last_iter)

model = msm_we.modelWE()
log.info("initializing...")

# Fire up the model object
# (Eventually this will just go in __init__)
model.initialize(fileSpecifier, refPDBfile, initPDBfile, modelName)
# Set some model parameters
# model.WEtargetp1 = 1.0  # target def on WE p1
# model.WEbasisp1_min = 9.6  # WE bin where basis structure is mapped- lower edge
# model.WEbasisp1_max = 12.5  # upper edge
# model.pcoord_ndim = 1  # number of pcoords
model.WEtargetp1 = 2.6  # target def on WE p1
model.WEbasisp1_min = 12.0  # WE bin where basis structure is mapped- lower edge
model.WEbasisp1_max = 13.0  # upper edge
model.pcoord_ndim = 1  # number of pcoords
# model.pcoord_len = 50  # number of pcoords
model.n_lag = 0
n_lag = model.n_lag
model.dimReduceMethod = "pca"  # dimensionality reduction method

last_iter_cluster = last_iter  # model.maxIter-1 #last iter often not complete
# ncoords = 10000 # TODO: what on earth is this
ncoords = 285  # TODO: this is like the total number of segments over all iterations?
ncoords = 125
i = last_iter_cluster

numCoords = 0
coordSet = np.zeros((0, model.nAtoms, 3))  # extract coordinate libraries for clustering
pcoordSet = np.zeros((0, model.pcoord_ndim))

log.debug("Loading in iteration data.. (this could take a while)")

# Keep adding iterations until the total number of segments is equal to ncoords
while numCoords < ncoords:
    print(
        f"Doing iteration {i}/{last_iter_cluster} (goes backwards to 0) | {numCoords}",
        end="\r",
    )
    # TODO: Replace this, which iterates over a set of trajectory files, with something that just looks at auxdata
    # Load data into the model's state
    model.load_iter_data(i)
    model.load_iter_coordinates()

    # TODO: Slowwwwwww appends
    indGood = np.squeeze(np.where(np.sum(np.sum(model.cur_iter_coords, 2), 1) != 0))
    coordSet = np.append(coordSet, model.cur_iter_coords[indGood, :, :], axis=0)
    pcoordSet = np.append(pcoordSet, model.pcoord1List[indGood, :], axis=0)
    numCoords = np.shape(coordSet)[0]
    i = i - 1


# Set the coords, and pcoords
# TODO: There's no need to set these as attributes of the object
model.all_coords = coordSet
model.pcoordSet = pcoordSet

log.debug(f"CoordLIST: {model.cur_iter_coords.shape}")
log.debug(f"CoordSET: {model.all_coords.shape}")

# TODO: Are first_iter and last_iter used consistently everywhere? Some places they're taken as parameters,
#   some places the current value is just pulled from state
# TODO: What does "cluster" mean?
first_iter_cluster = i
model.first_iter = first_iter_cluster
model.last_iter = last_iter_cluster

# TODO: Related to above comment, just use coordSet not model.coordSet
n_coords = np.shape(model.all_coords)[0]

model.dimReduce()

clusterFile = (
    modelName
    + "_clusters_s"
    + str(first_iter_cluster)
    + "_e"
    + str(last_iter_cluster)
    + "_nC"
    + str(n_clusters)
    + ".h5"
)
exists = os.path.isfile(clusterFile)
exists = False
log.warning("Skipping any potential cluster reloading!")

# If a cluster file with the name corresponding to these parameters exists, load clusters from it.
if exists:
    print("loading clusters...")
    model.load_clusters(clusterFile)
# Otherwise, do the clustering (which will create and save to that file)
else:
    print(
        "clustering "
        + str(n_coords)
        + " coordinates into "
        + str(n_clusters)
        + " clusters..."
    )
    model.cluster_coordinates(n_clusters)

first_iter = 1
model.get_fluxMatrix(
    n_lag, first_iter, last_iter
)  # extracts flux matrix, output model.fluxMatrixRaw
model.organize_fluxMatrix()  # gets rid of bins with no connectivity, sorts along p1, output model.fluxMatrix
model.get_Tmatrix()  # normalizes fluxMatrix to transition matrix, output model.Tmatrix
model.get_steady_state_algebraic()  # gets steady-state from eigen decomp, output model.pSS
model.get_steady_state_target_flux()  # gets steady-state target flux, output model.JtargetSS

log.info(f"Got steady state:")
log.info(model.pSS)
log.info(model.JtargetSS)

objFile = (
    modelName
    + "_s"
    + str(first_iter)
    + "_e"
    + str(last_iter)
    + "_nC"
    + str(n_clusters)
    + ".obj"
)
objFileHandler = open(objFile, "wb")
del model.clusters
print("Pickling model")
pickle.dump(model, objFileHandler)
objFileHandler.close()

log.info("Completed flux matrix calculation and steady-state estimation!")
