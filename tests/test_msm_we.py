#!/usr/bin/env python
"""Tests for `msm_we` package."""

import pytest
import numpy as np
import mdtraj as md
from copy import deepcopy
import pickle

from msm_we import msm_we


# Process coordinates override
def processCoordinates(self, coords):
    if self.dimReduceMethod == "none":
        nC = np.shape(coords)
        nC = nC[0]
        data = coords.reshape(nC, 3 * self.nAtoms)
        return data

    if self.dimReduceMethod == "pca" or self.dimReduceMethod == "vamp":
        # Dimensionality reduction

        xt = md.Trajectory(xyz=coords, topology=None)
        indCA = self.reference_structure.topology.select("name CA")
        pair1, pair2 = np.meshgrid(indCA, indCA, indexing="xy")
        indUT = np.where(np.triu(pair1, k=1) > 0)
        pairs = np.transpose(np.array([pair1[indUT], pair2[indUT]])).astype(int)
        dist = md.compute_distances(xt, pairs, periodic=True, opt=True)

        return dist


msm_we.modelWE.processCoordinates = processCoordinates


@pytest.fixture
def modelParams():
    """
    Store some parameters about the reference model
    """
    params = {
        "last_iter": 100,
        "n_cluster_centers": 100,
        # "WEtargetp1_bounds": [-np.inf, 1.0],
        # "WEbasisp1_bounds":  [9.6, 12.5],
        # "pcoord_ndim0": 2,
        # "dimReduceMethod": "pca",
        # "fixed_seed": 1337,
    }

    return params


@pytest.fixture
def ref_ntl9_hdf5_paths():
    """
    Fixture containing paths to the reference HDF5 files, from NTL9 folding.
    """
    restarts = [0, 1]
    runs = [1]

    paths = []
    for restart in restarts:
        for run in runs:
            paths.append(f"reference/1000ns_ntl9/restart{restart}/run{run}/west.h5")

    return paths


@pytest.fixture
def ref_ntl9_structure_path():
    """
    Fixture containing the path to the NTL9 reference structure.
    """

    return "reference/1000ns_ntl9/reference.pdb"


@pytest.fixture
def initialized_model():
    """
    An initialized haMSM model.
    """
    with open("reference/1000ns_ntl9/models/initialized.obj", "rb") as model_file:
        model = pickle.load(model_file)

    return model


@pytest.fixture
def loaded_model():
    """
    An initialized haMSM model.
    """
    with open("reference/1000ns_ntl9/models/loaded.obj", "rb") as model_file:
        model = pickle.load(model_file)
    return model


@pytest.fixture
def clustered_model():
    """
    An initialized haMSM model.
    """
    with open("reference/1000ns_ntl9/models/clustered.obj", "rb") as model_file:
        model = pickle.load(model_file)

    # When loading from a pickle, some of the PCA parameters are nuked, so regenerate them.
    model.coordinates.estimate(model.processCoordinates(model.all_coords))

    return model


@pytest.fixture
def organized_model():
    """
    An initialized haMSM model.
    """
    with open("reference/1000ns_ntl9/models/organized.obj", "rb") as model_file:
        model = pickle.load(model_file)

    return model


@pytest.fixture
def completed_model():
    """
    An initialized haMSM model.
    """
    with open("reference/1000ns_ntl9/models/completed.obj", "rb") as model_file:
        model = pickle.load(model_file)

    return model


@pytest.fixture
def fluxmatrix_raw():
    """
    An initialized haMSM model.
    """
    fluxmatrix_raw = np.load("reference/1000ns_ntl9/models/fluxmatrix_raw.npy")
    return fluxmatrix_raw


@pytest.fixture
def fluxmatrix():
    """
    An initialized haMSM model.
    """
    fluxmatrix = np.load("reference/1000ns_ntl9/models/fluxmatrix.npy")
    return fluxmatrix


@pytest.fixture
def tmatrix():
    """
    An initialized haMSM model.
    """
    tmatrix = np.load("reference/1000ns_ntl9/models/tmatrix.npy")
    return tmatrix


@pytest.fixture
def pSS():
    """
    An initialized haMSM model.
    """
    pSS = np.load("reference/1000ns_ntl9/models/pSS.npy")
    return pSS


@pytest.fixture
def JtargetSS():
    """
    An initialized haMSM model.
    """
    JtargetSS = np.load("reference/1000ns_ntl9/models/JtargetSS.npy")
    return JtargetSS


def test_initialize(ref_ntl9_hdf5_paths, ref_ntl9_structure_path):
    """
    Test initialization of the haMSM model from some h5 data files.
    """

    model = msm_we.modelWE()

    h5_path_string = " ".join(ref_ntl9_hdf5_paths)

    model.initialize(h5_path_string, ref_ntl9_structure_path, "test_initialize_model")

    assert model.coordsExist, "Coords were not successfully loaded"


def test_get_coord_set(initialized_model, modelParams, clustered_model):
    """
    Test loading coordinates from the h5 files.
    """

    initialized_model.get_iterations()
    initialized_model.get_coordSet(last_iter=modelParams["last_iter"])

    assert initialized_model.n_coords == clustered_model.n_coords
    assert initialized_model.first_iter == clustered_model.first_iter
    assert initialized_model.last_iter == clustered_model.last_iter
    assert (initialized_model.pcoordSet == clustered_model.pcoordSet).all()
    assert (initialized_model.all_coords == clustered_model.all_coords).all()


def test_dim_reduce_and_cluster(modelParams, clustered_model):
    """
    Test dimensionality reduction and clustering.

    I try to keep my tests separated as much as possible, but these are very closely related.
    """
    loaded_model = deepcopy(clustered_model)
    loaded_model.ndim = None
    loaded_model.coordinates = None

    # Dimensionality reduction first
    loaded_model.dimReduce()
    assert loaded_model.ndim == clustered_model.ndim

    # Make sure the PCA decomposition gave the correct result
    pca_params = loaded_model.coordinates.model.get_model_params()
    ref_params = clustered_model.coordinates.model.get_model_params()
    assert np.isclose(pca_params["mean"], ref_params["mean"]).all()
    assert np.isclose(pca_params["eigenvectors"], ref_params["eigenvectors"]).all()

    # Do the clustering
    loaded_model.cluster_coordinates(modelParams["n_cluster_centers"])

    # Make sure the clusters are what they should be
    # Be a little flexible here, because the PCA has *very* minor differences in outputs, so the cluster centers
    #   will vary by a little more.
    assert np.isclose(
        loaded_model.clusters.clustercenters,
        clustered_model.clusters.clustercenters,
        atol=1e-2,
    ).all()


def test_get_flux_matrix(fluxmatrix_raw, fluxmatrix, clustered_model):
    """
    Test flux matrix calculation and organizing.

    Again, these are distinct operations, but pretty closely related, and for now I want to minimize the amount of
    pickled haMSM objects I have until I can pare them down and store them more efficiently.
    """

    first_iter, last_iter = clustered_model.first_iter, clustered_model.last_iter

    clustered_model.get_fluxMatrix(0, first_iter, last_iter)

    assert (clustered_model.fluxMatrixRaw == fluxmatrix_raw).all()

    clustered_model.organize_fluxMatrix()

    assert (clustered_model.fluxMatrix == fluxmatrix).all()


def test_get_tmatrix(organized_model, tmatrix):
    """
    Test transition matrix construction from the organized flux matrix.
    """

    organized_model.get_Tmatrix()

    assert (organized_model.Tmatrix == tmatrix).all()


def test_get_steady_state_algebraic(completed_model, pSS):
    """
    Test calculating the steady-state distribution from the model with the transition matrix.
    """

    completed_model.pSS = None
    completed_model.get_steady_state_algebraic()

    assert np.isclose(completed_model.pSS, pSS).all()


def test_get_steady_state_target_flux(completed_model, JtargetSS):
    """
    Test calculating steady-state flux from the model with the transition matrix.
    """

    completed_model.JtargetSS = None
    completed_model.get_steady_state_target_flux()

    assert completed_model.JtargetSS == JtargetSS
