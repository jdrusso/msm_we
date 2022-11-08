import pytest
import os
import pickle
import numpy as np
import bz2
import _pickle as cPickle
import ray

# This will point to tests/
BASE_PATH = os.path.dirname(__file__) + "/../"


@pytest.fixture
def initialized_model():
    """
    An initialized haMSM model.
    """
    return load_model("reference/1000ns_ntl9/models/initialized.obj")


@pytest.fixture
def loaded_model():
    """
    An initialized haMSM model.
    """
    return load_model("reference/1000ns_ntl9/models/loaded.obj")


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


@pytest.fixture
def completed_model():
    """
    An initialized haMSM model.
    """
    return load_model("reference/1000ns_ntl9/models/completed.obj")


@pytest.fixture
def model_params():
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


def load_model(relative_path, regenerate_coords=False, compressed=False):

    path = os.path.join(BASE_PATH, relative_path)

    if compressed:
        model = decompress_pickle(path)

    else:
        with open(path, "rb") as model_file:
            model = pickle.load(model_file)

    if regenerate_coords:
        # When loading from a pickle, some of the PCA parameters are cleared, so regenerate them.
        # model.coordinates.estimate(model.processCoordinates(model.all_coords))
        pass

    # Patch paths in filelist. As constructed, they're relative paths, from another directory.
    #   This splits off the /restartXX/runYY/west.h5 part and re-writes the paths relative to the tests.
    models = [model]
    if (
        hasattr(model, "pre_discretization_model")
        and model.pre_discretization_model is not None
    ):
        models.append(model.pre_discretization_model)
    for _model in models:
        old_paths = _model.fileList
        new_paths = []
        for path in old_paths:
            # The -3 indexing pulls out 'some_long_path / restartXX/runYY/west.h5'
            relative_path = "/".join(path.split("/")[-3:])
            absolute_path = os.path.join(BASE_PATH, relative_path)
            new_paths.append(absolute_path)
        _model.fileList = new_paths

    return model


@pytest.fixture(scope="session")
def ray_cluster():
    ray.init(num_cpus=1)
    yield None
    ray.shutdown()


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
def discrepancy():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/discrepancy.npy")


@pytest.fixture
def variance():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/variance.npy")


@pytest.fixture
def ref_clustered_optimization():
    """
    An initialized haMSM model.
    """
    return load_numeric("reference/1000ns_ntl9/models/optimized_clustered_bins.npy")


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


@pytest.fixture
def RANDOM_SEED():
    """Random seed for various stochastic estimators. Putting this as a fixture might be overkill.."""
    return 42


def decompress_pickle(file):
    """
    Convenience function for loading compressed pickles.
    Annoying, but necessary for some files that exceed Github's 100MB limit.
    """
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


def load_numeric(relative_path):
    path = os.path.join(BASE_PATH, relative_path)
    numeric_result = np.load(path)
    return numeric_result
