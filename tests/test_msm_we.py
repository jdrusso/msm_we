#!/usr/bin/env python
"""Tests for `msm_we` package."""

import pytest
import numpy as np
from copy import deepcopy

import MDAnalysis as mda
from MDAnalysis.analysis import distances

from msm_we import modelWE

import os

BASE_PATH = os.path.dirname(__file__)
ref_file = BASE_PATH + "/../examples/data/2JOF.pdb"


def processCoordinates(self, coords):
    u_ref = mda.Universe(ref_file)
    u_check = mda.Universe(ref_file)

    dist_out = []

    u_check.load_new(coords)

    for frame in u_check.trajectory:
        dists = distances.dist(
            u_check.select_atoms("backbone"), u_ref.select_atoms("backbone")
        )[2]

        dist_out.append(dists)

    dist_out = np.array(dist_out)

    return dist_out


modelWE.processCoordinates = processCoordinates


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


def test_get_coord_set(initialized_model, model_params, clustered_model):
    """
    Test loading coordinates from the h5 files.
    """

    initialized_model.get_iterations()
    initialized_model.get_coordSet(last_iter=model_params["last_iter"])

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


@pytest.mark.xfail(reason="Mini-Batch KMeans appears to have enough randomness to make this fail occasionally.")
def test_streaming_stratified_clustering(
    loaded_model, clustered_model, ray_cluster, model_params
):

    loaded_model = deepcopy(loaded_model)
    loaded_model.clusters = None

    loaded_model.dimReduce()

    # Do the clustering
    loaded_model.cluster_coordinates(
        model_params["n_cluster_centers"],
        stratified=True,
        use_ray=True,
        random_state=model_params["cluster_seed"],
    )

    # Make sure the clusters are what they should be
    # Be a little flexible here, because the PCA has *very* minor differences in outputs, so the cluster centers
    #   will vary by a little more.
    # Furthermore, minibatch Kmeans has some randomness in it, so we leave a little fudge factor for that too.
    assert np.allclose(
        loaded_model.clusters.cluster_models[3].cluster_centers_,
        clustered_model.clusters.cluster_models[3].cluster_centers_,
        atol=1e-3,
        rtol=1e-2
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

    assert np.allclose(clustered_model.fluxMatrixRaw, fluxmatrix_raw)

    clustered_model.organize_fluxMatrix()

    assert np.allclose(clustered_model.fluxMatrix, fluxmatrix)


def test_get_tmatrix(organized_model, tmatrix):
    """
    Test transition matrix construction from the organized flux matrix.
    """

    organized_model.get_Tmatrix()

    assert np.allclose(organized_model.Tmatrix, tmatrix)


def test_get_steady_state(organized_model, pSS):
    """
    Test calculating the steady-state distribution from the model with the transition matrix.
    """

    organized_model.pSS = None
    organized_model.get_Tmatrix()
    organized_model.get_steady_state()

    assert np.allclose(organized_model.pSS, pSS)


def test_get_steady_state_target_flux(organized_model, JtargetSS):
    """
    Test calculating steady-state flux from the model with the transition matrix.
    """

    organized_model.JtargetSS = None
    organized_model.get_Tmatrix()
    organized_model.get_steady_state()
    organized_model.get_steady_state_target_flux()

    assert np.isclose(organized_model.JtargetSS, JtargetSS)


def test_get_cluster_structures(organized_model, ref_cluster_structures):
    """
    Tests obtaining the library of structures in each MSM bin.
    """

    organized_model.update_cluster_structures()
    cluster_structures = organized_model.cluster_structures

    # Just check one bin, otherwise the ref file is huge
    assert (np.array(ref_cluster_structures) == np.array(cluster_structures[10])).all()
