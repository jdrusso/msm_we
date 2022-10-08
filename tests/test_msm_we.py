#!/usr/bin/env python
"""Tests for `msm_we` package."""

import pytest
import numpy as np
from copy import deepcopy
import pickle
import bz2
import _pickle as cPickle
import getpass

import MDAnalysis as mda
from MDAnalysis.analysis import distances

from msm_we import modelWE

import os
import ray

BASE_PATH = os.path.dirname(__file__)


def decompress_pickle(file):
    """
    Convenience function for loading compressed pickles.
    Annoying, but necessary for some files that exceed Github's 100MB limit.
    """
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data



ref_file = BASE_PATH + '/../examples/data/2JOF.pdb'
def processCoordinates(self, coords):
    u_ref = mda.Universe(ref_file)
    u_check = mda.Universe(ref_file)

    dist_out = []

    u_check.load_new(coords)

    for frame in u_check.trajectory:
        dists = distances.dist(
            u_check.select_atoms('backbone'),
            u_ref.select_atoms('backbone')
        )[2]

        dist_out.append(dists)

    dist_out = np.array(dist_out)

    return dist_out


modelWE.processCoordinates = processCoordinates


@pytest.fixture
def modelParams():
    """
    Store some parameters about the reference model
    """
    params = {
        "last_iter": 100,
        "n_cluster_centers": 25,
        "cluster_seed": 1337,
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
            path = os.path.join(
                BASE_PATH, f"reference/1000ns_ntl9/restart{restart}/run{run}/west.h5"
            )
            paths.append(path)

    return paths


@pytest.fixture
def ref_ntl9_structure_path():
    """
    Fixture containing the path to the NTL9 reference structure.
    """

    return os.path.join(BASE_PATH, "reference/1000ns_ntl9/reference.pdb")


@pytest.fixture
def ref_cluster_structures():
    """
    Fixture containing reference cluster structures.
    """
    path = os.path.join(
        BASE_PATH, "reference/1000ns_ntl9/models/bin10_cluster_structures.npy"
    )
    return np.load(path, allow_pickle=True)


@pytest.fixture
def initialized_model():
    """
    An initialized haMSM model.
    """
    return load_model("reference/1000ns_ntl9/models/initialized.obj")

@pytest.fixture
def clustered_model():
    """
    An initialized haMSM model.
    """

    return load_model(
        "reference/1000ns_ntl9/models/clustered.obj",
        regenerate_coords=True,
        compressed=False,
    )


@pytest.fixture
def organized_model():
    """
    An initialized haMSM model.
    """
    return load_model("reference/1000ns_ntl9/models/organized.obj")


@pytest.fixture()
def ray_cluster():
    ray.init()
    yield None
    ray.shutdown()

@pytest.fixture
def completed_model():
    """
    An initialized haMSM model.
    """
    return load_model("reference/1000ns_ntl9/models/completed.obj")


def load_model(relative_path, regenerate_coords=False, compressed=False):

    path = os.path.join(BASE_PATH, relative_path)

    if compressed:
        model = decompress_pickle(path)

    else:
        with open(path, "rb") as model_file:
            model = pickle.load(model_file)

    if regenerate_coords:
        # When loading from a pickle, some of the PCA parameters are nuked, so regenerate them.
        # model.coordinates.estimate(model.processCoordinates(model.all_coords))
        pass

    # Patch paths in filelist. As constructed, they're
    old_paths = model.fileList
    new_paths = []
    for path in old_paths:
        relative_path = "/".join(path.split("/")[-3:])
        absolute_path = os.path.join(BASE_PATH, relative_path)
        new_paths.append(absolute_path)
    model.fileList = new_paths

    return model


@pytest.fixture
def fluxmatrix_raw():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/fluxmatrix_raw.npy")


@pytest.fixture
def fluxmatrix():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/fluxmatrix.npy")


@pytest.fixture
def tmatrix():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/tmatrix.npy")


@pytest.fixture
def pSS():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/pSS.npy")


@pytest.fixture
def JtargetSS():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/JtargetSS.npy")


@pytest.fixture
def cleanup_generated(generated_filename):
    """
    Fixture to automatically delete all generated h5 files in the root test directory.
    """

    # Run the test
    yield

    # Remove the generated file
    try:
        os.remove(generated_filename)
    except FileNotFoundError:
        pass


def load_numeric(relative_path):
    path = os.path.join(BASE_PATH, relative_path)
    numeric_result = np.load(path)
    return numeric_result


def test_initialize(ref_ntl9_hdf5_paths, ref_ntl9_structure_path):
    """
    Test initialization of the haMSM model from some h5 data files.
    """

    model = modelWE()

    h5_path_string = " ".join(ref_ntl9_hdf5_paths)

    model.initialize(
        h5_path_string, ref_ntl9_structure_path, "test_initialize_model", tau=10e-12
    )

    assert model.coordsExist, "Coords were not successfully loaded"


def test_get_coord_set(initialized_model, modelParams, clustered_model):
    """
    Test loading coordinates from the h5 files.
    """

    initialized_model.get_iterations()
    initialized_model.get_coordSet(last_iter=modelParams["last_iter"])

    assert initialized_model.first_iter == clustered_model.first_iter
    assert initialized_model.last_iter == clustered_model.last_iter
    assert np.isclose(
        initialized_model.pcoordSet, clustered_model.pcoordSet, equal_nan=True
    ).all()


def test_dim_reduce(clustered_model):
    """
    Test dimensionality reduction. This uses the non-streaming clustered model, but it shouldn't matter.
    """
    loaded_model = deepcopy(clustered_model)
    loaded_model.ndim = None
    loaded_model.coordinates = None
    loaded_model.clusters = None

    # Dimensionality reduction first
    loaded_model.dimReduce()
    assert loaded_model.ndim == clustered_model.ndim

    # Make sure the PCA decomposition gave the correct result
    ref_covariance = clustered_model.coordinates.get_covariance()
    test_covariance = loaded_model.coordinates.get_covariance()

    assert np.isclose(ref_covariance, test_covariance).all()


def test_streaming_stratified_clustering(clustered_model, ray_cluster, modelParams):

    loaded_model = deepcopy(clustered_model)
    loaded_model.clusters = None

    loaded_model.dimReduce()

    # Do the clustering
    loaded_model.cluster_coordinates(
        modelParams["n_cluster_centers"],
        stratified=True,
        use_ray=True,
        random_state=modelParams["cluster_seed"],
    )

    # Make sure the clusters are what they should be
    # Be a little flexible here, because the PCA has *very* minor differences in outputs, so the cluster centers
    #   will vary by a little more.
    assert np.isclose(
        loaded_model.clusters.cluster_models[3].cluster_centers_,
        clustered_model.clusters.cluster_models[3].cluster_centers_,
        atol=1e-4,
    ).all()


@pytest.mark.xfail(
    getpass.getuser() == "runner", reason="Hangs on github actions", run=False
)
@pytest.mark.parametrize(
    "generated_filename", ["initialized_model-fluxmatrix-_s1_e100_lag0_clust300.h5"]
)
def test_get_flux_matrix(
    fluxmatrix_raw, fluxmatrix, clustered_model, cleanup_generated, ray_cluster
):
    """
    Test flux matrix calculation and organizing.

    Again, these are distinct operations, but pretty closely related, and for now I want to minimize the amount of
    pickled haMSM objects I have until I can pare them down and store them more efficiently.

    This is an xfail for now, with an explicit timeout, because the subprocess calls
    may not execute on the Github Actions CI runner
    """

    clustered_model.get_fluxMatrix(n_lag=0)

    assert (clustered_model.fluxMatrixRaw == fluxmatrix_raw).all()

    clustered_model.organize_fluxMatrix()

    assert (clustered_model.fluxMatrix == fluxmatrix).all()


def test_get_tmatrix(organized_model, tmatrix):
    """
    Test transition matrix construction from the organized flux matrix.
    """

    organized_model.get_Tmatrix()

    assert (organized_model.Tmatrix == tmatrix).all()


def test_get_steady_state(organized_model, pSS):
    """
    Test calculating the steady-state distribution from the model with the transition matrix.
    """

    organized_model.pSS = None
    organized_model.get_Tmatrix()
    organized_model.get_steady_state()

    assert np.isclose(organized_model.pSS, pSS).all()


def test_get_steady_state_target_flux(organized_model, JtargetSS):
    """
    Test calculating steady-state flux from the model with the transition matrix.
    """

    organized_model.JtargetSS = None
    organized_model.get_Tmatrix()
    organized_model.get_steady_state()
    organized_model.get_steady_state_target_flux()

    assert organized_model.JtargetSS == JtargetSS


def test_get_cluster_structures(completed_model, ref_cluster_structures):
    """
    Tests obtaining the library of structures in each MSM bin.
    """

    completed_model.update_cluster_structures()
    cluster_structures = completed_model.cluster_structures

    # Just check one bin, otherwise the ref file is huge
    assert (np.array(ref_cluster_structures) == np.array(cluster_structures[10])).all()
