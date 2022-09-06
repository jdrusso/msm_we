"""Main module."""
from __future__ import division, print_function

__metaclass__ = type
import numpy as np
import tqdm.auto as tqdm
from functools import partialmethod
import sys
import h5py
import concurrent
import multiprocessing as mp
from copy import deepcopy
from westpa import analysis
from westpa.core.binning import RectilinearBinMapper, VoronoiBinMapper

from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sparse
from sklearn.decomposition import IncrementalPCA as iPCA
from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import MiniBatchKMeans as mini_kmeans

import logging
from rich.logging import RichHandler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(RichHandler())
log.propagate = False

# Using the tkinter backend makes matplotlib run better on a cluster, maybe?
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import mdtraj as md
# import pyemma.coordinates as coor
# import pyemma.coordinates.clustering as clustering
# import pyemma

# used to check connectivity
import scipy.sparse.csgraph as csgraph
from scipy.sparse.sputils import isdense

from rich.live import Live
from rich.table import Table

import ray

# If you implement a custom bin mapper that should work with stratified clustering, you can add it to this set
#   after importing msm_we
SUPPORTED_MAPPERS = {RectilinearBinMapper, VoronoiBinMapper}

def find_connected_sets(C, directed=True):
    r"""
    This implementation is taken from msmtools.estimation.sparse.connectivity, at commit 9312660.
    See the original at https://github.com/markovmodel/msmtools/blob/devel/msmtools/estimation/sparse/connectivity.py#L30

    Compute connected components for a directed graph with weights
    represented by the given count matrix.
    Parameters
    ----------
    C : scipy.sparse matrix or numpy ndarray
        square matrix specifying edge weights.
    directed : bool, optional
       Whether to compute connected components for a directed  or
       undirected graph. Default is True.
    Returns
    -------
    cc : list of arrays of integers
        Each entry is an array containing all vertices (states) in
        the corresponding connected component.
    """
    if isdense(C):
        C = csr_matrix(C)

    M = C.shape[0]
    """ Compute connected components of C. nc is the number of
    components, indices contain the component labels of the states
    """
    nc, indices = csgraph.connected_components(
        C, directed=directed, connection="strong"
    )

    states = np.arange(M)  # Discrete states

    """Order indices"""
    ind = np.argsort(indices)
    indices = indices[ind]

    """Order states"""
    states = states[ind]
    """ The state index tuple is now of the following form (states,
    indices)=([s_23, s_17,...,s_3, s_2, ...], [0, 0, ..., 1, 1, ...])
    """

    """Find number of states per component"""
    count = np.bincount(indices)

    """Cumulative sum of count gives start and end indices of
    components"""
    csum = np.zeros(len(count) + 1, dtype=int)
    csum[1:] = np.cumsum(count)

    """Generate list containing components, sort each component by
    increasing state label"""
    cc = []
    for i in range(nc):
        cc.append(np.sort(states[csum[i] : csum[i + 1]]))

    """Sort by size of component - largest component first"""
    cc = sorted(cc, key=lambda x: -len(x))

    return cc


def is_connected(matrix, source_states, target_states, directed=True):
    """
    Check for connectivity between two states.
    If directed is True, then checks for directional connectivity from source_states to target_states.

    Parameters
    ----------
    matrix: np.array, NxN
        Transition matrix
    source_states: array-like, N
        Source states
    target_states: array-like, N
        Target states
    directed: bool, default=True
        Compute directional connectivity

    Returns
    -------
    bool
    """

    return (
        np.inf
        not in csgraph.shortest_path(matrix, directed=directed, indices=source_states)[
            :, target_states
        ]
    )


def inverse_iteration(guess, matrix, mu=1):
    """
    Do one iteration of inverse iteration.

    Parameters
    ----------
    guess: array-like  (N elements)
        Vector of weights to be used as the initial guess.

    matrix: array-like (NxN elements)
        Transition matrix to use for inverse iteration.

    Returns
    -------
    The new vector of weights after one iteration of inverse iteration.
    """

    # Looking for eigenvector corresponding to eigenvalue 1
    identity = sparse.eye(guess.shape[0])

    # Inverse
    try:
        inverse = sparse.linalg.inv(matrix.T - mu * identity)
    except RuntimeError as e:
        if not mu == 1:
            filename = "bad_matrix.npy"
            log.error(
                f"Inverse iteration still failed with mu={mu} -- examine your transition matrix for why it could "
                f"be unsolvable. Saving the transition matrix to {filename}."
            )
            np.save(filename, matrix)
            raise e
        elif mu == 1:
            log.error(
                f"When solving steady-state, failed to perform inverse iteration! "
                f"Trying again with mu=0.999 instead of {mu}."
            )
            return inverse_iteration(guess, matrix, mu=0.999)

    result = inverse @ guess
    result = result.squeeze()

    # Normalize
    result /= sum(result)

    return result


class StratifiedClusters:
    """
    This is a class that provides semi-seamless integration of stratified clusters into the
    existing msm_we code.

    In "stratified clustering", clustering is done independently within each WE bin.
    Cluster models between WE bins are kept completely independent.
    This may help enforce a wider spread of cluster centers.

    The purpose of this class is to provide an object with a `.predict()` method, just like
    the regular kmeans object, that takes in a set of coordinates and assigns it to a cluster.
    However, `StratifiedClusters.predict()` does an extra step of first mapping the coords
    to the appropriate WE bin, and then using that WE bin's cluster model.
    """

    def __init__(self, bin_mapper, model, n_clusters, target_bins, **_cluster_args):
        """
        bin_mapper: westpa.core.binning.BinMapper
            Bin mapper for the current simulation being analyzed.

        model: msm_we.modelWE
            The current haMSM model object

        n_clusters: int
            Number of cluster per bin

        target_bins: array-like
            Indices of WE bins in the target.
            Anything in the target gets mapped to its own state anyway,
            so we don't want to try to cluster within it.

        **_cluster_args:
            Arguments passed through to sklearn.cluster.MiniBatchKMeans

        Notes
        =====

        A bin mapper can be obtained from an existing simulation's h5 file with
        .. code-block:: python
            import westpa.tools.binning
            mapper, _, _ = westpa.tools.binning.mapper_from_hdf5(h5file['bin_topologies'],
                    h5file['iterations/iter_00000002'].attrs['binhash'])

        """

        n_clusters_per_bin = n_clusters
        # n_total_clusters = n_clusters_per_bin * (bin_mapper.nbins - 1)

        # Set some default arguments, and overwrite them with the user's choices if provided
        cluster_args = {
            "n_clusters": n_clusters_per_bin,
            "max_iter": 100,
        }
        cluster_args.update(_cluster_args)

        self.n_clusters_per_bin = n_clusters
        self.bin_mapper = bin_mapper
        self.n_total_clusters = self.n_clusters_per_bin * (
            self.bin_mapper.nbins - len(target_bins)
        )
        log.info(
            f"Doing stratified clustering with {self.n_total_clusters} total clusters"
        )

        self.cluster_args = cluster_args
        self.model = model

        self.cluster_models = [
            mini_kmeans(**cluster_args) for x in range(self.bin_mapper.nbins)
        ]

        # Processing "from", as in the from-coords, or pcoord0List.
        self.processing_from = False
        self.toggle = False

        # These are bases/targets
        # It's only really important to ignore targets, because you may not have structures in the target b/c of recycling
        # self.target_bins = target_bins

        self.we_remap = {x: x for x in range(self.bin_mapper.nbins)}

        # This isn't really used any more, excise it at some point.
        # I need consecutive indices for each non-basis/non-target bin
        # In other words, remove the target, and then consecutively index all the remaining bins
        # legitimate_bins = []
        # for bin_index in range(self.bin_mapper.nbins):
        #     if bin_index not in target_bins:
        #         legitimate_bins.append(bin_index)
        #
        # self.legitimate_bins = legitimate_bins
        self.legitimate_bins = range(self.bin_mapper.nbins)

        self.target_bins = set()
        self.basis_bins = set()

    def predict(self, coords):
        """
        Map input coordinates to a WE bin, and then discretize using that bin's clusterer.

        Parameters
        ----------
        coords: array-like
            Coordinates to discretize

        Returns
        -------
        Cluster assignments for coordinates.

        Notes
        -----
        In general, when a set of coordinates is obtained and being discretized, it's coordinates at the end of a
        segment, and so should be binned acoording to the progress coordinate at the end of that segment (i.e.,
        modelWE.pcoord1List).
        However, there's an exception to this -- when doing fluxmatrix calculations, the initial and final clusters
        for each segment are obtained back-to-back, to get the start/end point of each transition.

        For that case, setting :code:`StratifiedClusters.toggle=True` will alternate between pcoord0List and pcoord1List every
        time a prediction is done.
        This is a pretty janky solution, but at the moment it keeps it relatively seamless between Stratified and regular
        clusters.
        :code:`StratifiedClusters.toggle` must be set to False after.
        """

        if self.processing_from:
            iter_pcoords = self.model.pcoord0List
        else:
            iter_pcoords = self.model.pcoord1List

        we_bins = self.bin_mapper.assign(iter_pcoords)
        we_bins = [self.we_remap[we_bin] for we_bin in we_bins]

        is_target = self.model.is_WE_target(iter_pcoords)
        is_basis = self.model.is_WE_basis(iter_pcoords)

        # Discretize coords according to which WE bin they're in
        discrete = []

        for i, coord in enumerate(coords):

            if is_target[i]:
                _discrete = [self.model.n_clusters + 1]

                _bin = we_bins[i]
                self.target_bins.add(_bin)

            elif is_basis[i]:
                _discrete = [self.model.n_clusters]

                _bin = we_bins[i]
                self.basis_bins.add(_bin)

            # Meanwhile, if you're NOT in the target (ignored) bin...
            else:
                consecutive_index = self.legitimate_bins.index(we_bins[i])

                # Since we cluster within each WE bin, the clusters within each bin are indexed from 0.
                # So, if we're in WE Bin N, the index of the 0th cluster in bin N is really (N-1) * (clusters per bin),
                #   not 0.
                offset = sum(
                    [
                        len(self.cluster_models[idx].cluster_centers_)
                        if hasattr(self.cluster_models[idx], "cluster_centers_")
                        else 0
                        for idx in self.legitimate_bins[:consecutive_index]
                    ]
                )

                assert hasattr(
                    self.cluster_models[we_bins[i]], "cluster_centers_"
                ), f"Not initialized in seg {i}, bin {we_bins[i]}. Coord was {coord}, coords were {coords}"
                try:
                    _discrete = [
                        self.cluster_models[we_bins[i]].predict([coord])[0] + offset
                    ]
                except Exception as e:
                    log.error(f"At seg {i}, bin {we_bins[i]}: {e}")
                    raise e
                else:
                    pass

            discrete.extend(_discrete)

        # This should be true ONLY when doing fluxmatrix calculations!
        if self.toggle:
            self.processing_from = not self.processing_from
            log.debug(
                f"Finished and toggling... Next iteration will use pcoord{not self.processing_from:d}List"
            )

        return np.array(discrete)


class modelWE:
    """
    Implementation of haMSM model building, particularly for steady-state estimation (but there are lots of extras),
    from WE sampling with basis (source) and target (sink) states with recycling.

    Set up for typical west.h5 file structure, with coordinates to be stored in west.h5 /iterations/auxdata/coord and
    basis and target definitions from progress coordinates.

    Check out run_msmWE.slurm and run_msmWE_flux.py in scripts folder for an implementation example.

    Danger
    -------
    This code currently, in general, appears to assume a 1-D progress coordinate.

    Todo
    ----
    Refactor
        In general, this class's methods generally handle data by holding state in the object.
        The functions that update state with the result of a calculation, though, tend to update a lot of state on the way.
        The state being updated along the way is usually "helper" quantities (an example would be the number of bins
        or number of walkers, which is computed "along the way" in a number of functions, and the object state updated.)

        I think it would be prudent to refactor these in such a way that these are updated in as few places as possible --
        one example of this might be setting them as properties, and then updating the value in state as part of that
        accessor if necessary.


    References
    --------
    Copperman and Zuckerman,
    *Accelerated estimation of long-timescale kinetics by combining weighted ensemble simulation with Markov model
    microstategs using non-Markovian theory*, **arXiv** (2020).

    """

    class BlockValidationError(Exception):
        pass

    def __init__(self):
        """
        Work-in-progress init function. For now, just start adding attribute definitions in here.

        Todo
        ----
        - Most logic from initialize() should be moved in here.

        - Also, comment all of these here. Right now most of them have comments throughout the code.

        - Reorganize these attributes into some meaningful structure
        """

        self.modelName = None
        """str: Name used for storing files"""
        self.fileList = None
        """list of str: List of all filenames with data"""
        self.n_data_files = None
        """int: Number of files in :code:`fileList`

        **TODO**: Deprecate this, this could just be a property"""

        self.n_lag = 0
        self.pcoord_ndim = None
        """int: Number of dimensions in the progress coordinate"""
        self.pcoord_len = None
        """int: Number of stored progress coordinates for each iteration, per-segment."""
        self.tau = None
        """float: Resampling time for weighted ensemble. (Maybe should be int? Units?)"""

        self.WEtargetp1_min = None
        self.WEtargetp1_max = None
        """float: Progress coordinate value at target state.optimization flow
        Used to check if a progress coord is in the target, and to set the RMSD of the target cluster when cleaning the
        fluxmatrix."""
        self.target_bin_center = None
        self._WEtargetp1_bounds = None

        self.WEbasisp1_min = None
        """float: Minimum progress coordinate value within basis state.
        Used to check if a progress coord is in the basis, and to set the RMSD of the basis cluster when cleaning the
        fluxmatrix."""
        self.WEbasisp1_max = None
        """float: Maximum progress coordinate value within basis state.
        Used to check if a progress coord is in the basis, and to set the RMSD of the basis cluster when cleaning the
        fluxmatrix."""
        self.basis_bin_center = None
        self._WEbasisp1_bounds = None

        self._basis_pcoord_bounds = None
        self._target_pcoord_bounds = None

        self.dimReduceMethod = None
        """str: Dimensionality reduction method. Must be one of "pca", "vamp", or "none" (**NOT** NoneType)"""

        self.vamp_lag = None
        self.vamp_dim = None

        # For optimized binning
        self.nB = None
        self.nW = None

        self.min_walkers = None
        """str: Test description for minwalkers"""

        self.binMethod = None
        self.allocationMethod = None

        self.coordsExist = None

        self.westList = None

        self.reference_structure = None
        self.reference_coord = None
        self.basis_structure = None
        # TODO: This is plural, reference_coord is singular. Intentional? Can you have multiple bases but 1 reference?
        self.basis_coords = None
        self.nAtoms = None

        self.numSegments = None
        self.maxIter = None

        # TODO: Describe segindList better.
        self.segindList = None
        """list: List of segment indices(?)"""

        self.weightList = None
        """array-like: List of segment weights in an iteration"""

        self.nSeg = None
        self.pcoord0List = None
        self.pcoord1List = None
        self.seg_weights = {}

        self.coordPairList = None
        self.transitionWeights = None
        self.departureWeights = None

        self.n_iter = None

        self.coordinates = None
        self.ndim = None

        self.n_hist = None
        """int: Number of steps of history information to use when building transitions."""

        self.n_clusters = None
        self.clusters = None
        self.clusterFile = None

        self.errorWeight = None
        self.errorCount = None
        self.fluxMatrixRaw = None

        self.targetRMSD_centers = None
        """array-like: List of RMSDs corresponding to each cluster."""
        self.fluxMatrix = None
        self.indBasis = None

        self.Tmatrix = None
        self.pSS = None
        self.lagtime = None
        self.JtargetSS = None

        self.removed_clusters = []
        self.cluster_structures = None
        self.cluster_structure_weights = None
        """dict: Mapping of cluster indices to structures in that cluster"""

        self.clustering_method = None
        self.validation_models = []

        self.pcoord_shape_warned = False

    def initialize(
        # self, fileSpecifier: str, refPDBfile: str, initPDBfile: str, modelName: str
        self,
        fileSpecifier: str,
        refPDBfile: str,
        modelName: str,
        basis_pcoord_bounds: list = None,
        target_pcoord_bounds: list = None,
        dim_reduce_method: str = "pca",
        tau: float = None,
        pcoord_ndim: int = 1,
        auxpath: str = "coord",
        _suppress_boundary_warning=False,
    ):
        """
        Initialize the model-builder.

        Parameters
        ----------
        fileSpecifier : list
            List of paths to H5 files to analyze.

        refPDBfile : string
            Path to PDB file that defines topology.

        modelName : string
            Name to use in output filenames.

        basis_pcoord_bounds: list
            List of [[pcoord0 lower bound, pcoord1 upper bound], [pcoord1 lower bound, pcoord1 upper bound], ...]
            in pcoord-space for the basis state

        target_pcoord_bounds: list
            List of [[pcoord0 lower bound, pcoord1 upper bound], [pcoord1 lower bound, pcoord1 upper bound], ...]
            in pcoord-space for the target state

        dim_reduce_method: str
            Dimensionality reduction method. "pca", "vamp", or "none".

        tau: float
            Resampling time (i.e. time of 1 WE iteration). Used to map fluxes to physical times.

        pcoord_ndim: int
            Defaults to 1. Dimensionality of progress coordinates.

        Returns
        -------
        None


        Todo
        ----
        Some of this logic should be broken into a constructor, and default arguments handled in the constructor's
        function signature.
        """

        log.debug("Initializing msm_we model")

        self.modelName = modelName

        if type(fileSpecifier) is list:
            fileList = fileSpecifier
        elif type(fileSpecifier) is str:
            fileList = fileSpecifier.split(" ")
            log.warning(
                "HDF5 file paths were provided in a string -- this is now deprecated, please pass as a list "
                "of paths."
            )

        self.pcoord_ndim = pcoord_ndim
        self.pcoord_len = 2

        if basis_pcoord_bounds is None:
            if not _suppress_boundary_warning:
                log.warning(
                    "No basis coord bounds provided to initialize(). "
                    "You can manually set this for now, but that will be deprecated eventually."
                )
        else:
            self.basis_pcoord_bounds = basis_pcoord_bounds

        self._target_pcoord_bounds = None
        if target_pcoord_bounds is None:
            if not _suppress_boundary_warning:
                log.warning(
                    "No target coord bounds provided to initialize(). "
                    "You can manually set this for now, but that will be deprecated eventually."
                )
        else:
            log.debug("Setting basis pcoord bounds")
            # self.WEtargetp1_bounds = target_pcoord_bounds
            self.target_pcoord_bounds = target_pcoord_bounds

        # self._basis_pcoord_bounds = None
        self.auxpath = auxpath
        self.fileList = fileList
        self.n_data_files = len(fileList)
        #####

        if tau is None:
            log.warning("No tau provided, defaulting to 1.")
            tau = 1.0

        self.tau = float(tau)

        # This is really only used for nAtoms
        self.set_topology(refPDBfile)
        # self.set_basis(initPDBfile)

        if dim_reduce_method is None:
            log.warning(
                "No dimensionality reduction method provided to initialize(). Defaulting to pca."
                "You can manually set this for now, but that will be deprecated eventually."
            )
            self.dimReduceMethod = "pca"
        else:
            self.dimReduceMethod = dim_reduce_method

        self.vamp_lag = 10
        self.vamp_dim = 10
        self.nB = 48  # number of bins for optimized WE a la Aristoff
        self.nW = 40  # number of walkers for optimized WE a la Aristoff
        self.min_walkers = 1  # minimum number of walkers per bin
        self.binMethod = "adaptive"  # adaptive for dynamic k-means bin edges, uniform for equal spacing on kh
        self.allocationMethod = (
            "adaptive"  # adaptive for dynamic allocation, uniform for equal allocation
        )

        try:
            self.load_iter_data(1)
            self.load_iter_coordinates0()
            self.coordsExist = True

        # Nothing is raised here because this might be fine, depending on what you're doing.
        except KeyError:
            if not _suppress_boundary_warning:
                log.warning("Model initialized, but coordinates do not exist yet.")
            self.coordsExist = False

        log.debug("msm_we model successfully initialized")

    # TODO: Deprecate this for an N-dimensional version
    @property
    def WEbasisp1_bounds(self):
        return self.basis_pcoord_bounds

    # TODO: Deprecate this for an N-dimensional version
    @WEbasisp1_bounds.setter
    def WEbasisp1_bounds(self, bounds):
        """
        Set the boundaries for the basis state in pcoord1, and also set the bin center based on those.

        Parameters
        ----------
        bounds
        """

        log.warning(
            "WEbasisp1_bounds is a deprecated attribute. "
            "Set pcoord boundaries using basis_pcoord_bounds instead"
        )

        self.basis_pcoord_bounds = bounds

    @property
    def basis_pcoord_bounds(self):
        return self._basis_pcoord_bounds

    @basis_pcoord_bounds.setter
    def basis_pcoord_bounds(self, bounds):
        """
        Set the boundaries for the basis state in an arbitrary dimension pcoord space.

        Parameters
        ----------
        bounds: array-like, (pcoord_ndim x 2)
            Array of [lower, upper] bounds for each progress coordinate.
        """
        bounds = np.array(bounds)

        # In case it's a 1D pcoord provided as a simple list [min, max],
        #   reshape it to be consistent with N-D pcoord boundaries as  [[min, max]]
        if len(bounds.shape) == 1:
            log.warning(
                "Please provide 1-D boundaries as a list of lists or 2-D array"
                " [[lower bound, upper bound]]. Automatically doing conversion for now."
            )
            bounds = np.reshape(bounds, (1, 2))

        assert bounds.shape == (
            self.pcoord_ndim,
            2,
        ), f"Shape of bounds was {bounds.shape}, should've been ({self.pcoord_ndim}, 2)"

        assert np.all(
            [bound[0] < bound[1] for bound in bounds]
        ), "A boundary has a lower bound larger than its upper bound"

        self._basis_pcoord_bounds = bounds

        basis_bin_centers = np.full(self.pcoord_ndim, fill_value=np.nan)
        for i, (bound_min, bound_max) in enumerate(bounds):

            # If neither of the bin boundaries are infinity, then the bin center is their mean.
            if not abs(bound_min) == np.inf and not abs(bound_max) == np.inf:
                basis_bin_centers[i] = np.mean([bound_min, bound_max])

            # If one of them IS infinity, their "bin center" is the non-infinity one.
            else:
                # Janky indexing, if p1_max == inf then that's True, and True == 1 so it picks the second element
                basis_bin_centers[i] = [bound_min, bound_max][abs(bound_min) == np.inf]
        self.basis_bin_centers = basis_bin_centers

    @property
    def n_lag(self):
        return self._n_lag

    @n_lag.setter
    def n_lag(self, lag):

        if not lag == 0:
            raise NotImplementedError(
                "Only a lag of 1 tau (n_lag = 0) is currently supported"
            )

        else:
            self._n_lag = lag

    # TODO: Deprecate this for an N-dimensional version
    @property
    def WEtargetp1_bounds(self):
        return self.target_pcoord_bounds

    # TODO: Deprecate this for an N-dimensional version
    @WEtargetp1_bounds.setter
    def WEtargetp1_bounds(self, bounds):
        """
        Set the boundaries for the target state in pcoord1, and also set the bin center based on those.

        Parameters
        ----------
        bounds
        """
        if None in bounds:
            raise Exception("A target boundary has not been correctly provided")

        log.warning(
            "WEbasisp1_bounds is a deprecated attribute. "
            "Set pcoord boundaries using basis_pcoord_bounds instead"
        )

        self.target_pcoord_bounds = bounds

    def progress_disable(self):
        """Disable all progress bars"""
        tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)

    def progress_enable(self):
        """Enable all progress bars"""
        tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=False)

    @property
    def target_pcoord_bounds(self):
        return self._target_pcoord_bounds

    @target_pcoord_bounds.setter
    def target_pcoord_bounds(self, bounds):
        """
        Set the boundaries for the target state in an arbitrary dimension pcoord space.

        Parameters
        ----------
        bounds: array-like, (pcoord_ndim x 2)
            Array of [lower, upper] bounds for each progress coordinate.
        """
        bounds = np.array(bounds)

        # In case it's a 1D pcoord provided as a simple list [min, max],
        #   reshape it to be consistent with N-D pcoord boundaries as  [[min, max]]
        if len(bounds.shape) == 1:
            log.warning(
                "Please provide 1-D boundaries as a list of lists or 2-D array"
                " [[lower bound, upper bound]]. Automatically doing conversion for now."
            )
            bounds = np.reshape(bounds, (1, 2))

        assert bounds.shape == (
            self.pcoord_ndim,
            2,
        ), "Shape of bounds was {bounds.shape}, should've been ({self.pcoord_ndim}, 2)"

        assert np.all(
            [bound[0] < bound[1] for bound in bounds]
        ), "A boundary has a lower bound larger than its upper bound"

        self._target_pcoord_bounds = bounds

        target_bin_centers = np.full(self.pcoord_ndim, fill_value=np.nan)
        for i, (bound_min, bound_max) in enumerate(bounds):

            # If neither of the bin boundaries are infinity, then the bin center is their mean.
            if not abs(bound_min) == np.inf and not abs(bound_max) == np.inf:
                target_bin_centers[i] = np.mean([bound_min, bound_max])

            # If one of them IS infinity, their "bin center" is the non-infinity one.
            else:
                # Janky indexing, if p1_max == inf then that's True, and True == 1 so it picks the second element
                target_bin_centers[i] = [bound_min, bound_max][abs(bound_min) == np.inf]
        self.target_bin_centers = target_bin_centers

    @staticmethod
    def check_connect_ray():

        assert ray.is_initialized(), (
            "Ray cluster has not been initialized! "
            "Launch from the code calling this with ray.init()."
        )
        resources = ray.cluster_resources()

        try:
            log.info(f"Using Ray cluster with {resources['CPU']} CPUs!")
        except KeyError as e:
            log.error(f"Total cluster resources were {resources}")
            log.error(f"However, available resources are {ray.available_resources()}")
            raise e

    def initialize_from_h5(self, refPDBfile, initPDBfile, modelName):
        """
        Like initialize, but sets state without
        Parameters
        ----------
        refPDBfile
        initPDBfile
        modelName

        Returns
        -------

        """
        pass

    def is_WE_basis(self, pcoords):
        """
        Checks if the input progress coordinates are in the basis state.

        Parameters
        ----------
        pcoords : numpy.ndarray(num_segments, num_pcoords)
            Array of progress coordinates for each segment.

        Returns
        -------
        True or False : bool

        Todo
        ----
        This only checks the 0th progress coordinate
        """

        in_basis = np.full_like(pcoords, fill_value=np.nan)

        for pcoord_dimension in range(self.pcoord_ndim):
            in_basis[:, pcoord_dimension] = np.logical_and(
                pcoords[:, pcoord_dimension]
                > self.basis_pcoord_bounds[pcoord_dimension, 0],
                pcoords[:, pcoord_dimension]
                < self.basis_pcoord_bounds[pcoord_dimension, 1],
            )

        in_basis = np.all(in_basis, axis=1)

        return in_basis

    def is_WE_target(self, pcoords):
        """
        Checks if the input progress coordinates are in the target state.

        Parameters
        ----------
        pcoords : numpy.ndarray(num_segments, num_pcoords)
            Array of progress coordinates for each segment.

        Returns
        -------
        True or False : bool

        Todo
        ----
        This only checks the 0th progress coordinate

        This also assumes you need a small pcoord!

        """

        in_target = np.full_like(pcoords, fill_value=np.nan)

        for pcoord_dimension in range(self.pcoord_ndim):
            in_target[:, pcoord_dimension] = np.logical_and(
                pcoords[:, pcoord_dimension]
                > self.target_pcoord_bounds[pcoord_dimension, 0],
                pcoords[:, pcoord_dimension]
                < self.target_pcoord_bounds[pcoord_dimension, 1],
            )

        in_target = np.all(in_target, axis=1)

        return in_target

    @staticmethod
    def do_step(table, row, step, args=[], kwargs={}, in_subprocess=False):

        step_text = table.columns[1]._cells[row]

        table.columns[0]._cells[row] = "[bold black][ [bold yellow]* [bold black]]"
        table.columns[1]._cells[row] = f"[bold black]{step_text}"

        try:
            if not in_subprocess:
                step(*args, **kwargs)

            else:
                # print(f"Calling {step} with args={args} and kwargs={kwargs}")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("fork")
                ) as executor:
                    executor.submit(step, *args, **kwargs).result()

        except Exception as e:
            table.columns[0]._cells[row] = "[bold black] [[bold red]x[bold black]]"
            table.columns[1]._cells[row] = f"[black]{step_text}"
            table.columns[2]._cells[row] = f"{getattr(e, 'message', repr(e))}"
            raise e

        table.columns[0]._cells[row] = "[bold black] [[bold green]✓[bold black]]"
        table.columns[1]._cells[row] = f"[black]{step_text}"

    @staticmethod
    def set_note(table, row, text):
        table.columns[2]._cells[row] = text

    @staticmethod
    def new_table():
        table = Table(title="haMSM Progress")
        table.add_column("Status")
        table.add_column("Step")
        table.add_column("Notes")

        steps = [
            "Ray initialization",
            "Model initialization",
            "Loading iterations",
            "Loading coordinates",
            "Computing dimensionality reduction",
            "Clustering",
            "Flux matrix",
            "Cleaning",
            "Transition matrix",
            "Steady-state distribution",
            "Steady-state target flux",
            "Cross-validation",
        ]

        for step in steps:
            table.add_row(" [ ]", f"{step}", "")

        return table

    def build_analyze_model(
        self,
        file_paths,
        ref_struct,
        modelName,
        basis_pcoord_bounds,
        target_pcoord_bounds,
        dimreduce_method,
        tau,
        n_clusters,
        ray_kwargs={},
        max_coord_iter=-1,
        stratified=True,
        streaming=True,
        use_ray=True,
        fluxmatrix_iters=[1, -1],
        fluxmatrix_iters_to_use=None,
        cross_validation_groups=2,
        cross_validation_blocks=4,
        show_live_display=True,
        step_kwargs={}
    ):
        """
        One-shot function to build the model and analyze all at once. This provides a convenient interface for running
        the blockwise estimation.

        This may not be desirable for very long workflows, or workflows still being debugged, where it might make sense
        to run the individual steps one by one.

        Parameters
        ----------
        file_paths: list
            List of paths to H5 files to analyze.
        ref_struct: string
            Path to PDB file that defines topology.
        modelName: string
            Name to use in output filenames.
        basis_pcoord_bounds: list
            List of [[pcoord0 lower bound, pcoord1 upper bound], [pcoord1 lower bound, pcoord1 upper bound], ...]
            in pcoord-space for the basis state
        target_pcoord_bounds: list
            List of [[pcoord0 lower bound, pcoord1 upper bound], [pcoord1 lower bound, pcoord1 upper bound], ...]
            in pcoord-space for the target state
        dimreduce_method: str
            Dimensionality reduction method. "pca", "vamp", or "none".
        tau: float
            Resampling time (i.e. time of 1 WE iteration). Used to map fluxes to physical times.
        n_clusters: int
            Number of clusters to use when clustering. This is clusters per bin for stratified, or total clusters for
            aggregate.
        ray_kwargs: dict
            Keyword arguments passed to ray.init(). Useful for specifying num_cpus. You could also use this to connect
            to an existing Ray cluster.
        max_coord_iter: int, optional (Default = model.maxIter, so all)
            Last iteration to obtain coordinates from. Useful for excluding the end of some data.

        stratified: bool, optional (Default = True)
            Enables stratified clustering, where clustering is performed independently within each WE bin.
        streaming: bool, optional (Default = True)
            Enables streaming over input data, rather than batch processing. Substantially improves memory efficiency,
            at a potential small performance hit.
        use_ray: bool, optional (Default = True)
            Enable parallelization, using Ray. This provides substantial speedup in discretization and fluxmatrix
            calculations.

        fluxmatrix_iters: list, optional (Default = [1, -1])
            List of [first, last] iteration to use when calculating fluxmatrix. Defaults to using all iterations.
        fluxmatrix_iters_to_use: list, optional (Default = None)
            Specific range of iterations to use, as opposed to bounds like fluxmatrix_iters. Note that either this OR
            fluxmatrix_iters

        cross_validation_groups: int, optional (Default = 2)
            Number of independent models to build when doing cross-validation. Each group contains (blocks / groups)
            blocks.
        cross_validation_blocks: int, optional (Default = 4)
            Number of blocks to split your data into, before building the independent models.

        show_live_display

        Returns
        -------

        """

        table = self.new_table()

        # Clean up any existing Ray instances
        if use_ray:
            ray.shutdown()

        with Live(table, refresh_per_second=4, auto_refresh=show_live_display) as live:

            # If live updating was disabled, write to the table once now. (Doesn't do anything if it was enabled)
            live.refresh()

            model = self

            # # Launch Ray
            # TODO: Do I actually want to do this in here? Previously, I assumed the user would set it up.
            step_idx = 0
            if use_ray:
                table.columns[0]._cells[
                    0
                ] = "[bold black][ [bold yellow]* [bold black]]"
                table.columns[1]._cells[0] = "Ray initialization"
                table.columns[2]._cells[0] = ""

                self.do_step(table, step_idx, ray.init, kwargs=ray_kwargs)
                self.set_note(
                    table, step_idx, f"{ray.available_resources()['CPU']} CPUs"
                )

            # # Initialize model
            step_idx += 1
            self.do_step(
                table,
                step_idx,
                step=model.initialize,
                kwargs={
                    "fileSpecifier": file_paths,
                    "refPDBfile": ref_struct,
                    "modelName": modelName,
                    "basis_pcoord_bounds": basis_pcoord_bounds,
                    "target_pcoord_bounds": target_pcoord_bounds,
                    "dim_reduce_method": dimreduce_method,
                    "tau": tau,
                    **step_kwargs.get('initialize', {})
                },
            )
            self.set_note(table, step_idx, "")

            # # Get number of iterations
            step_idx += 1
            self.do_step(
                table, step_idx, step=model.get_iterations,
            )
            self.set_note(table, step_idx, f"{model.maxIter} iterations exist")

            # # Load coordinates
            step_idx += 1
            _max_coord_iter = [max_coord_iter, model.maxIter][max_coord_iter == -1]
            self.do_step(
                table, step_idx, step=model.get_coordSet, args=[_max_coord_iter],
            )
            self.set_note(
                table, step_idx, f"Got coords for {_max_coord_iter} iterations"
            )

            # # Dimensionality reduction
            step_idx += 1
            self.set_note(table, step_idx, f"Method: {model.dimReduceMethod}")
            self.do_step(
                table, step_idx, step=model.dimReduce,
                kwargs={**step_kwargs.get('dimReduce', {})}
            )

            # # Clustering
            step_idx += 1
            self.do_step(
                table,
                step_idx,
                step=model.cluster_coordinates,
                kwargs={
                    "n_clusters": n_clusters,
                    "streaming": streaming,
                    "use_ray": use_ray,
                    "stratified": stratified,
                    **step_kwargs.get('clustering', {})
                },
            )

            # # Flux matrix
            step_idx += 1
            _fluxmatrix_iters = fluxmatrix_iters
            if fluxmatrix_iters[1] == -1:
                _fluxmatrix_iters[1] = model.maxIter
            self.do_step(
                table,
                step_idx,
                step=model.get_fluxMatrix,
                kwargs={
                    "n_lag": 0,
                    "first_iter": _fluxmatrix_iters[0],
                    "last_iter": _fluxmatrix_iters[1],
                    "iters_to_use": fluxmatrix_iters_to_use,
                    "use_ray": use_ray,
                    **step_kwargs.get('fluxmatrix', {})
                },
            )
            self.set_note(
                table,
                step_idx,
                f"Fluxmatrix built from iters "
                f"{[f'{_fluxmatrix_iters[0]} - {_fluxmatrix_iters[1]}', fluxmatrix_iters_to_use][fluxmatrix_iters_to_use is not None]}",
            )

            # # Cleaning
            step_idx += 1
            original_clusters = model.fluxMatrixRaw.shape[0]
            self.do_step(
                table,
                step_idx,
                step=model.organize_fluxMatrix,
                kwargs={"use_ray": use_ray,
                        **step_kwargs.get('organize', {})
                        },
            )
            final_clusters = model.fluxMatrix.shape[0]
            self.set_note(
                table,
                step_idx,
                f"{original_clusters} clusters cleaned to {final_clusters}",
            )

            # # Transition matrix
            step_idx += 1
            self.do_step(table, step_idx, step=model.get_Tmatrix)

            # # Steady state
            step_idx += 1
            self.do_step(table, step_idx, step=model.get_steady_state)

            # # Steady-state flux
            step_idx += 1
            self.do_step(table, step_idx, step=model.get_steady_state_target_flux)
            self.set_note(table, step_idx, f"Target flux: {model.JtargetSS:.2e}")

            # # Cross-validation
            # TODO
            # use `post_cluster_model`, which is the discretized model right before the fluxmatrix was calculated
            # for now just do something useless w/ the post_cluster_model so linter doesn't complain
            step_idx += 1
            self.do_step(
                table,
                step_idx,
                step=model.do_block_validation,
                kwargs={
                    "cross_validation_groups": cross_validation_groups,
                    "cross_validation_blocks": cross_validation_blocks,
                    "use_ray": use_ray,
                    **step_kwargs.get('block_validation', {})
                },
            )
            flux_text = ""
            for group, _validation_model in enumerate(self.validation_models):
                flux_text += f"Group {group} flux: {_validation_model.JtargetSS:.2e}\n"
            self.set_note(table, step_idx, flux_text)

            # If live updating was disabled, write to the table once now. (Doesn't do anything if it was enabled)
            live.refresh()

    def do_block_validation(
        self, cross_validation_groups, cross_validation_blocks, use_ray=True
    ):

        assert (
            hasattr(self, "post_cluster_model") and self.post_cluster_model is not None
        ), (
            "Perform clustering with cluster_coordinates() before attempting"
            "block validation -- self.post_cluster_model is not set."
        )

        # TODO: Best way to move around post_cluster_model? Copying it will get massive.
        # TODO: copying AGAIN seems suboptimal...
        validation_models = [
            deepcopy(self.post_cluster_model) for _ in range(cross_validation_groups)
        ]

        # Get the number of iterations in each block
        iters_per_block = self.post_cluster_model.maxIter // cross_validation_blocks
        block_iterations = [
            [start_iter, start_iter + iters_per_block]
            for start_iter in range(1, self.post_cluster_model.maxIter, iters_per_block)
        ]

        # Otherwise, this may be maxIters + 1
        block_iterations[-1][-1] = block_iterations[-1][-1] - 1

        # Get the iterations corresponding to each group
        group_blocks = [
            range(start_idx, cross_validation_blocks, cross_validation_groups,)
            for start_idx in range(cross_validation_groups)
        ]

        validation_iterations = []

        for group in range(cross_validation_groups):

            group_iterations = []

            for block in group_blocks[group]:
                group_iterations.extend(range(*block_iterations[block]))

            validation_iterations.append(group_iterations)

            # You're looking at this massive try block and judging me -- but don't worry.
            #   The purpose of this is just to catch ANY error, and preface it with an explicit heads-up that it's coming
            #   from the block validation. This is useful because errors may crop up only in the block-validation, and it
            #   should be clear at a glance that it's not from the main model building, but only when the data is split up.
            try:
                log.info(
                    f"Beginning analysis of cross-validation group {group + 1}/{cross_validation_groups}."
                )

                _model = validation_models[group]

                # Get the flux matrix
                _model.get_fluxMatrix(
                    0, iters_to_use=validation_iterations[group], use_ray=use_ray
                )

                # Clean it
                _model.organize_fluxMatrix(use_ray=use_ray)

                # Get tmatrix
                _model.get_Tmatrix()

                # Get steady-state
                _model.get_steady_state()

                # Get target flux
                _model.get_steady_state_target_flux()

                # Get FPT distribution?
                pass

            except Exception as e:

                log.error("Error during block validation!")
                log.exception(e)


                # TODO: Would be nice to gracefully handle this and move on to the next validation group.
                #   However, validation models are used in a number of places, and leaving a model with uninitialized
                #   parameters will cause problems there.
                #   Maybe a solution is to only populate self.validation_models with successfully generated ones, though
                #   make sure having the length possibly change there is handled well.
                raise modelWE.BlockValidationError(e)

        # Store the validation models, in case you want to analyze them.
        self.validation_iterations = validation_iterations
        self.validation_models = validation_models

    def load_iter_data(self, n_iter: int):
        """
        Update state with the data (including pcoord but not including coords) corresponding to an iteration.

        Object fields updated with the information from the selected iteration:
            - `self.westList`
            - `self.segindList`
            - `self.weightList`
            - `self.n_segs`
            - `self.pcoord0List`
            - `self.pcoord1List`

        Parameters
        ----------
        n_iter : int
            Iteration to get data for.

        Returns
        -------
        None

        Todo
        ----
        May want to rework the logic here, depending on how this is used.
        Seems like some of this iteration can be removed/optimized.
        """

        # log.debug("Getting iteration data")

        self.n_iter = n_iter
        westList = np.array([])
        segindList = np.array([])
        weightList = np.array([])
        pcoord0List = np.empty((0, self.pcoord_ndim))
        pcoord1List = np.empty((0, self.pcoord_ndim))

        seg_weights = np.array([])

        n_segs = 0

        # Iterate through each file index, trying to find files that contains the iteration of interest
        # TODO: Can replace this with `for if, fileName in enumerate(self.\\)`
        for file_idx in range(self.n_data_files):
            fileName = self.fileList[file_idx]
            try:
                # Try to find the h5 data file associated with this iteration
                dataIn = h5py.File(fileName, "r")
                dsetName = "/iterations/iter_%08d/seg_index" % int(n_iter)

                # Check if the dataset
                dataset_exists = dsetName in dataIn

                # Check to make sure this isn't the last iteration -- last iterations have incomplete data
                is_not_last_iteration = (
                    "/iterations/iter_%08d/seg_index" % int(n_iter + 1) in dataIn
                )

                log.debug(f"From file {fileName}, loading iteration {n_iter}")

                if dataset_exists and is_not_last_iteration:

                    dset = dataIn[dsetName]
                    newSet = dset[:]
                    n_segs_in_file = np.shape(newSet)
                    n_segs_in_file = n_segs_in_file[0]
                    dsetNameP = "/iterations/iter_%08d/pcoord" % int(n_iter)
                    dsetP = dataIn[dsetNameP]
                    pcoord = dsetP[:]
                    weights = dset["weight"]
                    seg_weights = np.append(seg_weights, weights)

                    if not pcoord.shape[2] == self.pcoord_ndim and not self.pcoord_shape_warned:
                        log.warning(f"Dimensions of pcoord in {fileName} ({pcoord.shape[2]}) do not match specified "
                                    f"pcoord dimensionality self.pcoord_ndim ({self.pcoord_ndim}). "
                                    f"MSM-WE will only load up to dimension {self.pcoord_ndim}. "
                                    f"This pcoord is just used for stratification, and this is expected behavior if "
                                    f"you're extending your pcoord (i.e. in an optimization flow). ")
                        self.pcoord_shape_warned = True

                    # Iterate over segments in this dataset
                    for seg_idx in range(n_segs_in_file):
                        # if np.sum(pcoord[seg_idx,self.pcoord_len-1,:])==0.0:
                        # # intentionally using this to write in dummy pcoords,
                        # # this is a good thing to have for post-analysis though!
                        #    raise ValueError('Sum pcoord is 0, probably middle of WE iteration, not using iteration') f
                        westList = np.append(westList, file_idx)
                        segindList = np.append(segindList, seg_idx)
                        weightList = np.append(weightList, newSet[seg_idx][0])
                        pcoord0List = np.append(
                            pcoord0List,
                            np.expand_dims(pcoord[seg_idx, 0, :self.pcoord_ndim], 0),
                            axis=0,
                        )
                        pcoord1List = np.append(
                            pcoord1List,
                            np.expand_dims(pcoord[seg_idx, self.pcoord_len - 1, :self.pcoord_ndim], 0),
                            axis=0,
                        )
                        n_segs = n_segs + 1
                dataIn.close()
            except Exception as dataset_exists:
                sys.stdout.write("error in " + fileName + str(sys.exc_info()[0]) + "\n")
                raise dataset_exists

        # log.debug(f"Found {n_segs} segments in iteration {n_iter}")

        self.westList = westList.astype(int)

        # This is a list of the segment indices
        self.segindList = segindList.astype(int)
        self.seg_weights[n_iter] = seg_weights
        self.weightList = weightList
        self.nSeg = n_segs
        self.pcoord0List = pcoord0List
        self.pcoord1List = pcoord1List

    def get_iterations(self):
        """
        Updates internal state with the maximum number of iterations, and the number of segments in each section.

        Note
        ----
        This updates :code:`numSegments` -- :code:`numSegments` is actually a *list* of the number of segments in each iteration.

        Returns
        -------
        None
        """

        log.debug("Getting number of iterations and segments")

        numSegments = np.array([])
        nSeg = 1
        n_iter = 1

        # Loop over nSegs
        # TODO: Not sure I understand the logic in this loop
        while nSeg > 0:
            nSeg = 0

            # Iterate through each filename in fileList, and see if it contains the iteration we're looking for
            # TODO: This loop is pretty common, this should be refactored into a find_iteration() or something
            for file_index in range(self.n_data_files):
                fileName = self.fileList[file_index]
                try:
                    dataIn = h5py.File(fileName, "r")
                    dsetName = "/iterations/iter_%08d/seg_index" % int(n_iter)
                    dataset_exists = dsetName in dataIn

                    is_not_last_iteration = (
                        "/iterations/iter_%08d/seg_index" % int(n_iter + 1) in dataIn
                    )

                    if dataset_exists and is_not_last_iteration:
                        # If this file does contain the iteration of interest
                        # if dataset_exists:
                        dset = dataIn[dsetName]
                        newSet = dset[:]
                        nS = np.shape(newSet)
                        nSeg = nS[0] + nSeg
                    dataIn.close()
                except Exception as e:
                    log.error(e)
                    log.error(f"No segments in {fileName} {str(sys.exc_info()[0])}")

            if nSeg > 0:
                numSegments = np.append(numSegments, nSeg)
                log.debug(
                    "Iteration " + str(n_iter) + " has " + str(nSeg) + " segments...\n"
                )

            n_iter = n_iter + 1

        # Warning: These are not defined until this is run for the first time
        self.numSegments = numSegments
        self.maxIter = numSegments.size

    def get_iterations_iters(self, first_iter: int, last_iter: int):
        """
        Updates internal state with the maximum number of iterations, and the number of segments in each section.

        Parameters
        ----------
        first_iter : int
        last_iter : int

        Returns
        -------
        None

        Warning
        ----
        This is potentially deprecated or unnecessary. Currently unused.

        """

        numSegments = np.array([])

        for n_iter in range(first_iter, last_iter + 1):
            nSeg = 0
            for iF in range(self.n_data_files):
                fileName = self.fileList[iF]
                try:
                    dataIn = h5py.File(fileName, "r")
                    dsetName = "/iterations/iter_%08d/seg_index" % int(n_iter)
                    dataset_exists = dsetName in dataIn
                    if dataset_exists:
                        dset = dataIn[dsetName]
                        newSet = dset[:]
                        nS = np.shape(newSet)
                        nSeg = nS[0] + nSeg
                    dataIn.close()
                except Exception as e:
                    log.error(e)
                    log.error(f"No segments in {fileName} {str(sys.exc_info()[0])}")

            if nSeg > 0:
                numSegments = np.append(numSegments, nSeg)
                sys.stdout.write(
                    "Iteration " + str(n_iter) + " has " + str(nSeg) + " segments...\n"
                )
        self.numSegments = numSegments
        self.maxIter = last_iter

    def set_topology(self, topology):
        """
        Updates internal state with a new topology.

        Parameters
        ----------
        topology : str, md.Trajectory, dict
            Path to a file containing the PDB with the topology,
            an mdtraj Trajectory object describing the new basis structure,
            or a dictionary with keys
                'basis_coords': the coordinates of the basis structure and
                'nAtoms': the number of features in the coordinates

        Returns
        -------
        None
        """

        if type(topology) is str:

            log.debug(
                "Input reference topology was provided as a path, trying to load with mdtraj"
            )

            if topology[-3:] == "dat":
                self.reference_coord = np.loadtxt(topology)
                self.nAtoms = 1
                self.coord_ndim = 3
                return

            elif topology[-6:] == "prmtop":
                struct = md.load_prmtop(topology)
                # self.reference_structure = struct
                self.nAtoms = struct.n_atoms
                self.coord_ndim = 3
                return

            elif not topology[-3:] == "pdb":
                log.critical(
                    "Topology is not a recognized type (PDB)! Proceeding, but no guarantees."
                )

            struct = md.load(topology)
            # self.reference_structure = struct
            self.reference_coord = np.squeeze(struct._xyz)
            self.nAtoms = struct.topology.n_atoms
            self.coord_ndim = 3
            return

        elif type(topology) in [md.Trajectory, md.Topology]:
            log.debug(
                "Input reference topology  was provided as an mdtraj structure, loading that"
            )

            struct = topology
            # self.reference_structure = struct
            self.reference_coord = np.squeeze(struct._xyz)
            self.nAtoms = struct.topology.n_atoms
            self.coord_ndim = 3

        elif type(topology) == dict:

            self.reference_coord = topology['coords']
            self.nAtoms = topology['nAtoms']
            self.coord_ndim = topology['coord_ndim']

        else:
            raise NotImplementedError('Unsupported topology')

    def set_basis(self, basis):
        """
        Updates internal state with a new basis.

        Parameters
        ----------
        basis : str, mdtraj.Trajectory, or dict
            Path to a file containing the PDB with the new basis state,
            an mdtraj Trajectory object describing the new basis structure,
            or a dictionary with key 'basis_coords': the coordinates of the basis structure

        Returns
        -------
        None
        """

        if type(basis) is str:

            log.debug(
                "Input basis state topology was provided as a path, trying to load with mdtraj"
            )

            if basis[-3:] == "dat":
                self.basis_coords = np.loadtxt(basis)
            elif basis[-3:] == "pdb":
                struct = md.load(basis)
                # self.basis_structure = struct
                self.basis_coords = np.squeeze(struct._xyz)
            else:
                log.critical(
                    "Basis is not a recognized type! Proceeding, but no guarantees."
                )
                # raise NotImplementedError("Basis coordinates are not a recognized filetype")

        elif type(basis) in [md.Trajectory, md.Topology]:
            log.debug(
                "Input reference topology  was provided as an mdtraj structure, loading that"
            )

            struct = basis
            # self.basis_structure = struct
            self.basis_coords = np.squeeze(struct._xyz)

        elif type(basis) == dict:

            self.basis_coords = basis['coords']

        else:
            raise NotImplementedError('Unsupported topology')

    def get_transition_data(self, n_lag):
        """
        This function analyzes pairs of coordinates at the current iteration, set by :code:`self.n_iter`, and at some
            lag in the past, :code:`self.n_iter - n_lag`.

        Segments where a walker was warped (recycled) use the basis coords as the lagged coords.

        Parameters
        ----------
        n_lag : int
            Number of lags to use for transitions.

        Returns
        -------
        None
        """

        log.warning(
            "Getting transition data at arbitrary lags > 0 is not yet supported! Use at your own risk."
        )

        # get segment history data at lag time n_lag from current iter
        if n_lag > self.n_iter:
            sys.stdout.write(
                "too much lag for iter... n_lag reduced to: " + str(self.n_iter) + "\n"
            )
            n_lag = self.n_iter
        if n_lag >= self.n_hist:
            sys.stdout.write("too much lag for stored history... recalculating...\n")
            self.get_seg_histories(n_lag)

        self.n_lag = n_lag
        segindList_lagged = self.seg_histories[:, n_lag]

        # TODO: What exactly is this a list of?
        # seg_histories is a list of indices of the segments
        warpList = self.seg_histories[:, 0:n_lag]  # check for warps
        warpList = np.sum(warpList < 0, 1)

        # Get the weights for the lagged and current iterations
        # If something was split/merged between n_iter and n_iter-n_lag , then the weights may have changed, so
        #   check a particular segment's weight.
        # TODO: Does this effectively get the parent weight if it was split from something else? Then weight_histories
        #    would need to be tracking parents/children
        weightList_lagged = self.weight_histories[:, n_lag]
        # TODO: Does this copy need to be made?
        weightList = self.weightList

        # This will become a list of (lagged iter coord, current iter coord)
        coordPairList = np.zeros((self.nSeg, self.nAtoms, self.coord_ndim, 2))

        prewarpedStructures = np.zeros((self.nSeg, self.nAtoms, self.coord_ndim))
        nWarped = 0

        # Go through each segment, and get pairs of coordinates at current iter (n_iter) and
        # lagged iter (n_iter-n_lag)
        for seg_idx in range(self.nSeg):

            # FIXME: Try statements should encompass the smallest amount of code
            #  possible - anything could be tripping this
            # try:
            if seg_idx == 0:
                westFile = self.fileList[self.westList[seg_idx]]
                dataIn = h5py.File(westFile, "r")
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_current = dset[:]
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter - n_lag),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_lagged = dset[:]
            elif self.westList[seg_idx] != self.westList[seg_idx - 1]:
                # FIXME: I think you can just move this close to an if statement in the beginning, and then remove
                #   this whole if/elif. Everything after that close() seems to be duplicated.
                dataIn.close()
                westFile = self.fileList[self.westList[seg_idx]]
                dataIn = h5py.File(westFile, "r")

                # Load the data for the current iteration
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_current = dset[:]

                # Load the lagged data for (iteration - n_lag)
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter - n_lag),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_lagged = dset[:]

            coordPairList[seg_idx, :, :, 1] = coords_current[
                self.segindList[seg_idx], 1, :, :
            ]

            # If this segment has no warps, then add the lagged coordinates
            if warpList[seg_idx] == 0:
                # Try to set the previous coord in the transition pair to the segment's lagged coordinates
                try:
                    lagged_seg_index = segindList_lagged[seg_idx]
                    coordPairList[seg_idx, :, :, 0] = coords_lagged[
                        lagged_seg_index, 0, :, :
                    ]

                # If this fails, then there were no lagged coordinates for this structure.
                except IndexError as e:
                    log.critical(
                        f"Lagged coordinates do not exist for the structure in segment {seg_idx}"
                    )
                    raise e
                    weightList_lagged[seg_idx] = 0.0
                    weightList[
                        seg_idx
                    ] = 0.0  # set transitions without structures to zero weight

            # If something was recycled during this segment, then instead of using the lagged cooordinates,
            #   just use the basis coords.
            # But, also save the original structure before the warp!
            elif warpList[seg_idx] > 0:

                # St
                prewarpedStructures[nWarped, :, :] = coords_lagged[
                    segindList_lagged[seg_idx], 0, :, :
                ]

                assert self.basis_coords is not None

                coordPairList[seg_idx, :, :, 0] = self.basis_coords
                nWarped = nWarped + 1

            # # TODO: What triggers this? When that critical log hits, come update this comment and the main docstring.
            # # This triggers on index out of bounds... But that's a huge try statement!
            # except Exception as e:
            #     log.critical("Document whatever's causing this exception!")
            #     log.warning(e)
            #     raise e
            #     weightList_lagged[seg_idx] = 0.0
            #     weightList[
            #         seg_idx
            #     ] = 0.0  # set transitions without structures to zero weight

        # Squeeze removes axes of length 1 -- this helps with np.where returning nested lists of indices
        # FIXME: is any of this necessary? This kinda seems like it could be replaced with
        #  something like indWarped = np.squeeze(np.where(warpList > 0)).astype(int)
        indWarped = np.squeeze(np.where(warpList > 0))
        indWarpedArray = np.empty(nWarped)
        indWarpedArray[0:nWarped] = indWarped
        indWarped = indWarpedArray.astype(int)

        # Get the current and lagged weights
        # This returns (the wrong? none at all?) weights for segments with warps, which is corrected below
        transitionWeights = weightList.copy()
        departureWeights = weightList_lagged.copy()

        # Get correct weights for segments that warped
        for iW in range(nWarped):

            # The coord pair here is (pre-warped structure, reference topology) instead of
            #   (lagged struture, current structure)
            coordPair = np.zeros((1, self.nAtoms, self.coord_ndim, 2))
            coordPair[0, :, :, 0] = prewarpedStructures[iW, :, :]
            coordPair[0, :, :, 1] = self.reference_coord
            coordPairList = np.append(coordPairList, coordPair, axis=0)

            # TODO: iterWarped appears to be the iteration the warp happened at
            iterWarped = np.squeeze(np.where(self.seg_histories[indWarped[iW], :] < 0))
            try:
                nW = np.shape(iterWarped)
                iterWarped = iterWarped[0]
                sys.stdout.write(
                    "    segment "
                    + str(indWarped[iW])
                    + " warped "
                    + str(nW)
                    + " times\n"
                )
            # TODO: What exception is this handling? Why would this be tripped?
            except Exception as e:
                log.critical(f"Exception was {e}")
                log.critical("UPDATE THIS EXCEPTION HANDLER!")

                sys.stdout.write(
                    "    segment " + str(indWarped[iW]) + " warped 1 time\n"
                )

            # The "current" weights are taken at the warp iteration, so the final pair of weights are
            #   (the lagged weights, the weights at the warp)
            transitionWeights = np.append(
                transitionWeights, self.weight_histories[indWarped[iW], iterWarped]
            )
            departureWeights = np.append(
                departureWeights, self.weight_histories[indWarped[iW], n_lag]
            )

        self.coordPairList = coordPairList
        self.transitionWeights = transitionWeights
        self.departureWeights = departureWeights

    def get_transition_data_lag0(self):
        """
        Get coordinate pairs at the beginning and end of this iteration.

        Updates:
            - self.coordPairList, a list of  parent/child coordinate pairs
            - self.transitionWeights, a copy of self.weightList
            - self.departureWeights, a copy of self.weightList
        """

        weightList = self.weightList
        coordPairList = np.zeros((self.nSeg, self.nAtoms, self.coord_ndim, 2))

        log.debug(
            f"Getting transition data for {self.nSeg} segs in iteration {self.n_iter}, at a lag of 0"
        )

        # the segments in this iteration may be split across a number of different files
        # "Traditionally", we store a reference for  each segment  of which WEST file it's in
        # But what if we flip that, and for each west file, get which segments are in it?

        seg_west_files = self.westList[range(self.nSeg)]
        west_file_idxs = np.unique(seg_west_files)
        west_files = [self.fileList[idx] for idx in west_file_idxs]

        for idx, west_file in enumerate(west_files):

            segs_contained = np.where(seg_west_files == idx)[0]

            with h5py.File(west_file, "r") as data_file:

                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )

                try:
                    dset = data_file[dsetName]
                except KeyError as e:
                    raise e

                coords = dset

                coordPairList[segs_contained, :, :, 0] = coords[:, 0, :, :]
                coordPairList[segs_contained, :, :, 1] = coords[
                    :, self.pcoord_len - 1, :, :
                ]

                # Check for NaNs in segments
                nan_segments = segs_contained[
                    np.argwhere(
                        np.isnan(coordPairList[segs_contained]).any(axis=(1, 2, 3))
                    )
                ]

                if nan_segments.shape[0] > 0:
                    log.warning(
                        f"Bad coordinates for segments {nan_segments}, setting weights to 0"
                    )
                    weightList[nan_segments] = 0.0

        transitionWeights = weightList.copy()
        departureWeights = weightList.copy()

        self.coordPairList = coordPairList
        self.transitionWeights = transitionWeights
        self.departureWeights = departureWeights

    def get_warps_from_parent(self, first_iter, last_iter):
        """
        Get all warps and weights over a range of iterations.

        Parameters
        ----------
        first_iter: int
            First iteration in range.
        last_iter: int
            Last iteration in range.

        Returns
        -------
        warpedWeights: list
            List of weights for each warp.

        """
        warpedWeights = []
        for iS in range(first_iter + 1, last_iter + 1):
            self.load_iter_data(iS + 1)
            self.get_seg_histories(2)
            parentList = self.seg_histories[:, 1]
            warpList = np.where(parentList < 0)
            warpedWeights.append(self.weightList[warpList])
        return warpedWeights

    def get_warps_from_pcoord(
        self, first_iter, last_iter
    ):  # get all warps and weights over set of iterations
        warpedWeights = []
        for iS in range(first_iter, last_iter + 1):
            self.load_iter_data(iS)
            pcoord = self.pcoord1List[:, 0]
            # warpList = np.where(pcoord < self.WEtargetp1)
            warpList = np.where(self.is_WE_target(pcoord))
            warpedWeights.append(self.weightList[warpList])
            meanJ = (
                np.mean(self.weightList[warpList]) / self.tau / np.sum(self.weightList)
            )
            sys.stdout.write("Jdirect: " + str(meanJ) + " iter: " + str(iS) + "\n")
        return warpedWeights

    def get_direct_target_flux(self, first_iter, last_iter, window):
        nIterations = last_iter - first_iter
        Jdirect = np.zeros(nIterations - 1)
        f = h5py.File(self.modelName + ".h5", "a")
        dsetName = (
            "/s"
            + str(first_iter)
            + "_e"
            + str(last_iter)
            + "_w"
            + str(window)
            + "/Jdirect"
        )
        e = dsetName in f
        if not e:
            warpedWeights = self.get_warps_from_pcoord(first_iter, last_iter)
            self.warpedWeights = warpedWeights
            JdirectTimes = np.zeros(nIterations - 1)
            for iS in range(nIterations - 1):
                end = iS + 1
                start = iS - window
                if start < 0:
                    start = 0
                nI = end - start
                warpedWeightsI = np.array([])
                for i in range(start, end):
                    warpedWeightsI = np.append(warpedWeightsI, warpedWeights[i])
                nWarped = warpedWeightsI.size
                particles = (nWarped * warpedWeightsI) / nI
                Jdirect[iS] = np.mean(particles)
                JdirectTimes[iS] = (first_iter + iS) * self.tau
            Jdirect = Jdirect / self.tau
            dsetP = f.create_dataset(dsetName, np.shape(Jdirect))
            dsetP[:] = Jdirect
            dsetName = (
                "/s"
                + str(first_iter)
                + "_e"
                + str(last_iter)
                + "_w"
                + str(window)
                + "/JdirectTimes"
            )
            dsetP = f.create_dataset(dsetName, np.shape(JdirectTimes))
            dsetP[:] = JdirectTimes
        elif e:
            dsetP = f[dsetName]
            Jdirect = dsetP[:]
            dsetName = (
                "/s"
                + str(first_iter)
                + "_e"
                + str(last_iter)
                + "_w"
                + str(window)
                + "/JdirectTimes"
            )
            dsetP = f[dsetName]
            JdirectTimes = dsetP[:]
        f.close()
        self.Jdirect = Jdirect / self.n_data_files  # correct for number of trees
        self.JdirectTimes = JdirectTimes

    def get_seg_histories(self, n_hist):
        """
        **TODO: What does this do exactly?**

        Updates:
            - self.seg_histories
            - self.weight_histories
            - self.n_hist

        Parameters
        ----------
        n_hist : int
            Number of steps of history information to include.

        Returns
        -------
        None

        """

        log.debug(f"Getting seg histories for {self.nSeg} segments")

        if n_hist > self.n_iter:
            sys.stdout.write(
                "we have too much history... n_hist reduced to: "
                + str(self.n_iter)
                + "\n"
            )
            n_hist = self.n_iter

        # FIXME: The only other place this is used is in get_transition_data(). May need to see how it's handled there
        #   i.e. make sure it's initialized
        self.n_hist = n_hist

        seg_histories = np.zeros((self.nSeg, self.n_hist + 1))
        weight_histories = np.zeros((self.nSeg, self.n_hist))

        # Loop over all segments
        for iS in range(self.nSeg):

            # Print a message every 100 segments
            # if iS % 100 == 0:
            #     sys.stdout.write(
            #         "        getting history for iteration "
            #         + str(self.n_iter)
            #         + " segment "
            #         + str(iS)
            #         + "...\n"
            #     )

            # Open the relevant datafile for reading
            if iS == 0:
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")
            elif self.westList[iS] != self.westList[iS - 1]:
                # If you're past the 0th segment, then close the previous file.
                # FIXME: Close the file after finished working with it, at the *end* of the loop. As-is, the last loop
                #       leaves the file open.
                dataIn.close()
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")

            seg_histories[iS, 0] = self.segindList[iS]
            warped = 0

            # Iterate over history lengths from 1 to n_hist
            for iH in range(1, self.n_hist + 1):

                indCurrentSeg = seg_histories[iS, iH - 1]

                if indCurrentSeg < 0 and warped == 0:
                    sys.stdout.write(
                        "Segment "
                        + str(iS)
                        + " warped last iter: History must end NOW!\n"
                    )
                    warped = 1

                elif indCurrentSeg >= 0 and warped == 0:
                    dsetName = "/iterations/iter_%08d/seg_index" % int(
                        self.n_iter - iH + 1
                    )

                    dset = dataIn[dsetName]
                    seg_histories[iS, iH] = dset[indCurrentSeg][1]
                    weight_histories[iS, iH - 1] = dset[indCurrentSeg][0]

                    if seg_histories[iS, iH] < 0:
                        sys.stdout.write(
                            "            segment "
                            + str(iS)
                            + " warped "
                            + str(iH)
                            + " iters ago\n"
                        )
        self.seg_histories = seg_histories[:, :-1].astype(int)
        self.weight_histories = weight_histories
        # FIXME:
        #   weight histories and segment histories go in reverse order,
        #   so final current iter is first of 0 index

    def collect_iter_coordinates(self):  # grab coordinates from WE traj_segs folder
        """
        Goes through the generated trajectory segments, and adds data from the segments to an H5 file.

        This should be implemented by the user, and this implementation assumes a really specific configuration.
        This is left in mostly as an example.

        Returns
        -------
        None

        Todo
        ----
        Generalize to different filetypes. This appears to be AMBER specific and relies on loading rst7 files
        """

        log.critical(
            "If you're calling this function, be absolutely sure it does what you want -- this is "
            "very specifically written for the output of certain simulations."
        )

        nS = self.nSeg
        westFile = self.fileList[self.westList[0]]
        dataIn = h5py.File(westFile, "a")
        coords = np.zeros((0, 2, self.nAtoms, self.coord_ndim))
        for iS in range(self.nSeg):
            # FIXME: Replace strings with Pathlib paths
            westFile = self.fileList[self.westList[iS]]
            WEfolder = westFile.replace("west.h5", "")
            trajpath = WEfolder + "traj_segs/%06d/%06d" % (self.n_iter, iS)
            coord0 = np.squeeze(
                md.load(
                    trajpath + "/parent.rst7", top=self.reference_structure.topology
                )._xyz
            )
            coord1 = np.squeeze(
                md.load(
                    trajpath + "/seg.rst7", top=self.reference_structure.topology
                )._xyz
            )
            coordT = np.array([coord0, coord1])
            coordT = coordT[np.newaxis, :, :, :]
            coords = np.append(coords, coordT, axis=0)
            try:
                if iS > 0:
                    if self.westList[iS] != self.westList[iS - 1] and iS < nS - 1:
                        dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                            int(self.n_iter),
                            self.auxpath,
                        )
                        try:
                            # TODO: Why exclude the last point?
                            dset = dataIn.create_dataset(
                                dsetName, np.shape(coords[:-1, :, :, :])
                            )
                            dset[:] = coords[:-1, :, :, :]
                        except (RuntimeError, ValueError):
                            del dataIn[dsetName]
                            dset = dataIn.create_dataset(
                                dsetName, np.shape(coords[:-1, :, :, :])
                            )
                            dset[:] = coords[:-1, :, :, :]
                            log.warning(
                                "coords exist for iteration "
                                + str(self.n_iter)
                                + " overwritten\n"
                            )

                        dataIn.close()
                        coords = np.zeros((0, 2, self.nAtoms, self.coord_ndim))
                        coords = np.append(coords, coordT, axis=0)
                        dataIn = h5py.File(westFile, "a")

                    # If it's the last segment, don't exclude the last point (why?)
                    elif iS == nS - 1:
                        dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                            int(self.n_iter),
                            self.auxpath,
                        )
                        try:
                            dset = dataIn.create_dataset(dsetName, np.shape(coords))
                            dset[:] = coords
                        except (RuntimeError, ValueError):
                            del dataIn[dsetName]
                            dset = dataIn.create_dataset(dsetName, np.shape(coords))
                            dset[:] = coords
                            log.warning(
                                "coords exist for iteration "
                                + str(self.n_iter)
                                + " overwritten\n"
                            )

                        dataIn.close()

            except Exception as e:
                log.error(
                    "error collecting coordinates from "
                    + WEfolder
                    + " , iter "
                    + str(self.n_iter)
                    + " segment "
                    + str(self.segindList[iS])
                    + "\n"
                )
                log.error(e)

    def get_iter_coordinates(self, iteration):
        """
        Return the valid coordinates for a certain iteration

        Parameters
        ----------
        iteration: int
            The iteration to return coordinates for

        Returns
        -------
        Array of coordinates for all atoms at the current iteration
        """

        self.load_iter_data(iteration)
        self.load_iter_coordinates()
        # indGood = np.squeeze(np.where(np.sum(np.sum(self.cur_iter_coords, 2), 1) != 0))

        bad_coords = np.isnan(self.cur_iter_coords).any(axis=(1, 2))
        good_coords = ~bad_coords

        # iter_coords = self.cur_iter_coords[indGood]
        iter_coords = self.cur_iter_coords[good_coords]

        return iter_coords

    def load_iter_coordinates(self):
        """

        Returns
        -------

        """

        cur_iter_coords = np.full((self.nSeg, self.nAtoms, self.coord_ndim), fill_value=np.nan)

        log.debug(
            f"Getting coordinates for {self.nSeg} segs in iteration {self.n_iter}, at a lag of 0"
        )

        # the segments in this iteration may be split across a number of different files
        # "Traditionally", we store a reference for  each segment  of which WEST file it's in
        # But flip that, and for each west file, get which segments are in it

        seg_west_files = self.westList[range(self.nSeg)]
        west_file_idxs = np.unique(seg_west_files)
        west_files = [self.fileList[idx] for idx in west_file_idxs]

        for idx, west_file in enumerate(west_files):

            segs_contained = np.where(seg_west_files == idx)[0]

            with h5py.File(west_file, "r") as data_file:

                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )

                try:
                    dset = data_file[dsetName]
                except (RuntimeError, ValueError, KeyError):
                    log.error(
                        f"Error getting coordinates from {west_file}, in iteration {self.n_iter}"
                    )

                    cur_iter_coords[segs_contained, :, :] = np.full(
                        (self.nAtoms, self.coord_ndim), fill_value=np.nan
                    )
                    self.coordsExist = False

                    # raise e

                else:
                    coords = dset
                    cur_iter_coords[segs_contained, :, :] = coords[:, 1, :, :]

        self.cur_iter_coords = cur_iter_coords

    def load_iter_coordinates0(self):  # get iteration initial coordinates
        coordList = np.full((self.nSeg, self.nAtoms, self.coord_ndim), fill_value=np.nan)
        for iS in range(self.nSeg):
            if iS == 0:
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coord = dset[:]
            elif self.westList[iS] != self.westList[iS - 1]:
                dataIn.close()
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coord = dset[:]
            coordList[iS, :, :] = coord[self.segindList[iS], 0, :, :]
        self.cur_iter_coords = coordList

    def get_coordinates(self, first_iter, last_iter):
        """
        Unused

        Parameters
        ----------
        first_iter
        last_iter

        Returns
        -------
        """

        log.warning("This function is not tested or supported, use at your own risk!")

        self.first_iter = first_iter
        self.last_iter = last_iter
        iters = range(self.first_iter, self.last_iter + 1)
        coordSet = np.zeros((0, self.nAtoms, self.coord_ndim))
        for iter in iters:
            if iter % 50 == 0:
                sys.stdout.write(
                    "    gathering structures from iteration " + str(iter) + "...\n"
                )
            self.load_iter_data(iter)
            self.load_iter_coordinates()
            if self.coordsExist:
                coordSet = np.append(coordSet, self.cur_iter_coords, axis=0)
        self.all_coords = coordSet

    def get_coordSet(self, last_iter, streaming=None):
        """
        Loads all coordinates and progress coordinates into memory for later usage.

        If streaming, then this only loads pcoords

        Todo
        ----
        I want to avoid loading full coordinates into memory as much as possible.
        That means trying to replace usage of all_coords here.

        Parameters
        ----------
        last_iter

        Returns
        -------

        """

        if streaming is None and self.dimReduceMethod == "vamp":
            streaming = False
        elif streaming is None:
            streaming = True

        log.debug(f"CoordSet loading up to {last_iter}")
        total_segments = int(sum(self.numSegments[:last_iter]))

        if not streaming:
            coordSet = np.full((total_segments, self.nAtoms, self.coord_ndim), fill_value=np.nan)
        pcoordSet = np.full((total_segments, self.pcoord_ndim), fill_value=np.nan)

        last_seg_idx = total_segments

        # Update iterations N+1 -> 1
        for i in tqdm.tqdm(range(last_iter, 0, -1), desc="Getting coordSet"):

            self.load_iter_data(i)
            self.load_iter_coordinates()

            first_seg_idx = last_seg_idx - len(self.segindList)
            assert first_seg_idx >= 0, "Referencing a segment that doesn't exist"

            # Get the indices of all "good" coordinates, where a  valid coordinate has been obtained
            # Boolean mask for coords that have been successfully initialized
            bad_coords = np.isnan(self.cur_iter_coords).any(axis=(1, 2))
            good_coords = ~bad_coords

            if bad_coords.any():
                log.warning(
                    f"Walker {np.argwhere(bad_coords).squeeze()} has bad coords in iteration(s) {self.n_iter}"
                )

            if not streaming:
                coordSet[first_seg_idx:last_seg_idx][
                    good_coords
                ] = self.cur_iter_coords[good_coords, :, :]

            pcoordSet[first_seg_idx:last_seg_idx][good_coords] = self.pcoord1List[
                good_coords, :
            ]

            last_seg_idx = first_seg_idx

        # Set the coords, and pcoords
        if not streaming:
            self.all_coords = coordSet
        else:
            pass

        self.pcoordSet = pcoordSet

        first_iter_cluster = i
        self.first_iter = first_iter_cluster
        self.last_iter = last_iter

    def get_traj_coordinates(self, from_iter, traj_length):
        if traj_length > from_iter:
            traj_length = from_iter - 1
            sys.stdout.write(
                "trajectory length too long: set to " + str(traj_length) + "\n"
            )
        self.load_iter_data(from_iter)
        self.get_seg_histories(traj_length)
        traj_iters = np.zeros((traj_length, self.nSeg, self.nAtoms, self.coord_ndim))
        ic = traj_length - 1
        iH = 0
        nS = self.nSeg
        westList = self.westList.copy()
        for i in range(from_iter, from_iter - traj_length, -1):
            sys.stdout.write(
                "    gathering structures from iteration " + str(i) + "...\n"
            )
            self.load_iter_data(i)
            self.load_iter_coordinates()
            seg_history = self.seg_histories[:, iH]
            seg_history_index = np.zeros(nS).astype(int)
            last_index = np.zeros(nS).astype(
                int
            )  # list of final iterations for warped particles
            for iS in range(nS):
                if seg_history[iS] > 0:
                    seg_history_index[iS] = np.where(
                        np.logical_and(
                            self.segindList == seg_history[iS],
                            self.westList == westList[iS],
                        )
                    )[0][
                        0
                    ]  # segment indexes are local to the westfile, not global for the analysis set
                    last_index[iS] = 0
                elif seg_history[iS] < 0:
                    last_index[iS] = i - (from_iter - traj_length) - 1
            traj_iters[ic, :, :, :] = self.cur_iter_coords[seg_history_index, :, :]
            ic = ic - 1
            iH = iH + 1
        self.load_iter_data(from_iter)
        traj_iters = np.swapaxes(traj_iters, 0, 1)
        self.trajSet = [None] * self.nSeg
        for iS in range(self.nSeg):
            self.trajSet[iS] = traj_iters[iS, last_index[iS] :, :, :]

    def do_pca(self, arg):
        rough_pca, iteration, processCoordinates = arg
        iter_coords = self.get_iter_coordinates(iteration)

        # If  no good coords in this iteration, skip it
        if iter_coords.shape[0] == 0:
            return rough_pca

        processed_iter_coords = processCoordinates(iter_coords)
        rough_pca.partial_fit(processed_iter_coords)

        log.debug(f"{rough_pca.n_samples_seen_} samples seen")

        return rough_pca

    def do_full_pca(self, arg):

        ipca, iteration, processCoordinates, components_for_var = arg

        iter_coords = self.get_iter_coordinates(iteration)

        used_iters = 0
        # Keep adding coords until you have more than your components
        while iter_coords.shape[0] <= components_for_var:

            used_iters += 1

            _iter_coords = self.get_iter_coordinates(iteration + used_iters)
            if _iter_coords.shape[0] == 0:
                continue

            iter_coords = np.append(iter_coords, _iter_coords, axis=0)

        processed_iter_coords = processCoordinates(iter_coords)
        log.debug(
            f"About to run iPCA on  {processed_iter_coords.shape} processed coords"
        )
        ipca.partial_fit(processed_iter_coords)

        return ipca, used_iters

    def dimReduce(
        self,
        first_iter=1,
        first_rough_iter=None,
        last_iter=None,
        rough_stride=10,
        fine_stride=1,
        variance_cutoff=0.95
    ):
        """
        Dimensionality reduction using the scheme specified in initialization.

        This just defines the dimensionality reduction scheme and builds the model -- it does NOT actually transform
        the data!

        Transforming the data is performed via reduceCoordinates(), which uses self.coordinates as set
            by this.

        Updates:
            - `self.coordinates`
            - `self.ndim`

        Returns
        -------
        None
        """

        log.debug(f"Running dimensionality reduction -- method: {self.dimReduceMethod}")

        # log.debug(self.coordSet)
        if self.dimReduceMethod == "pca":

            # Do this in a streaming way, iteration by iteration
            # First, do a "rough" PCA on the last 10% of the data to get the number of components that explain the
            #   variance cutoff.
            # This is necessary because with incremental PCA, there's no way to do this ahead of time.

            if last_iter is None:
                last_iter = self.maxIter

            rough_ipca = iPCA()

            if first_rough_iter is None:
                total_num_iterations = last_iter
                # Stride every 10th frame, so you're only doing the "rough" pca on 10% of the data
                if total_num_iterations > 100:
                    rough_iters = range(1, total_num_iterations, rough_stride)

                # But if you only have 100 frames or fewer, then just do the last-half.
                else:
                    rough_iters = range(total_num_iterations // 2, total_num_iterations)
            else:
                rough_iters = range(first_rough_iter, last_iter, rough_stride)

            for iteration in tqdm.tqdm(rough_iters, desc="Initial iPCA"):

                # TODO: Allow  chunking here so you don't have  to  go 1  by  1, but N by N
                # If you don't use 'fork' context here, this will break in Jupyter.
                # That's because processCoordinates is monkey-patched in. With 'spawn' (i.e. without fork), the module
                #   is re-imported in the child process. In the reimported  module, processCoordinates is undefined.
                # With 'fork', it preserves the monkey-patched version.
                # Additionally, 'fork' is a little faster than  spawn. Ironically, that's usually at the cost  of memory
                #   usage. But here, the memory being used by the main thread (and therefore being copied here) isn't
                #   that great -- the memory issue stems from it not being freed up between successive calls.
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("fork")
                ) as executor:
                    rough_ipca = executor.submit(
                        self.do_pca, [rough_ipca, iteration, self.processCoordinates]
                    ).result()

            components_for_var = (
                np.argmax(
                    np.cumsum(rough_ipca.explained_variance_ratio_) > variance_cutoff
                )
                + 1
            )
            log.debug(f"Keeping {components_for_var} components")
            components_for_var = min(
                components_for_var, (last_iter - first_iter) // fine_stride - 1
            )

            # Now do the PCA again, with that many components, using all the iterations.
            ipca = iPCA(n_components=components_for_var)

            extra_iters_used = 0
            for iteration in tqdm.tqdm(
                range(first_iter, last_iter, fine_stride), desc="iPCA"
            ):

                if extra_iters_used > 0:
                    extra_iters_used -= 1
                    log.debug(f"Already processed  iter  {iteration}")
                    continue

                # Try some stuff to help memory management. I think  a lot of memory is not being explicitly released
                #   here when I'm looping, because I can watch my swap usage steadily grow while it's running this loop.
                # https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python has some good
                #   details  on how memory may be freed by Python, but not necessarily recognized  as free by the OS.
                # One "guaranteed" way to free  memory back to the OS that's been released by Python is to do  the memory
                #   intensive operation  in a subprocess. So, maybe I need to do my partial fit in a subprocess.
                # In fact, I first moved partial_fit alone to a subprocess, but that didn't help. The issue isn't
                #   partial_fit, it's actually loading the coords.
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("fork")
                ) as executor:
                    ipca, extra_iters_used = executor.submit(
                        self.do_full_pca,
                        [ipca, iteration, self.processCoordinates, components_for_var],
                    ).result()

            self.coordinates = ipca
            self.ndim = components_for_var

        elif self.dimReduceMethod == "vamp":
            # TODO: I  don't think trajSet is initialized by itself -- you need to manually call get_traj_coordinates
            log.warning(
                "VAMP dimensionality reduction requires you to *manually* call get_traj_coordinates first, "
                "or self.trajSet will be all None. Make sure you've done that!"
            )

            ntraj = len(self.trajSet)
            data = [None] * ntraj
            for itraj in range(ntraj):
                data[itraj] = self.processCoordinates(self.trajSet[itraj])
            self.coordinates = coor.vamp(
                data,
                lag=self.vamp_lag,
                dim=self.vamp_dim,
                scaling=None,
                right=False,
                stride=1,
                skip=0,
            )
            self.ndim = self.coordinates.dimension()

        elif self.dimReduceMethod == "none":
            self.ndim = int(3 * self.nAtoms)

            # TODO: Why is this unused?
            # data = self.all_coords.reshape(-1, self.ndim)
            self.coordinates = self.Coordinates()
            # self.coordinates.transform = self.processCoordinates

    class Coordinates(object):
        """
        Fake Coordinates class, in case you don't want to use either PCA or VAMP for dimensionality reduction
        """

        # The class "constructor" - It's actually an initializer
        def __init__(self):
            self.explanation = "coordinate object"

        def transform(self, coords):
            return coords

    def reduceCoordinates(self, coords):
        """
        This performs the dimensionality reduction.

        dimReduce() defines self.coordinates, which is an object that has a .transform() function that produces the
        reduced data.

        reduceCoordinates() actually uses that coordinates object, transforms the coordinates,
        and returns the reduced data.

        The reduced coordinates are then stored in /auxdata for each iteration.

        Parameters
        ----------
        coords: array-like
            Array of coordinates to reduce.

        Returns
        -------
        Reduced data
        """

        log.debug("Reducing coordinates")

        if (
            self.dimReduceMethod == "none"
            or self.dimReduceMethod == "pca"
            or self.dimReduceMethod == "vamp"
        ):
            coords = self.processCoordinates(coords)
            coords = self.coordinates.transform(coords)
            return coords

        raise Exception("dimReduceMethod undefined in reduceCoordinates")

    def update_cluster_structures(self):
        """
        Find structures (i.e. sets of coordinates) corresponding to each clusters.

        Returns
        -------
        A dictionary where the keys are cluster indices, and the values are lists of coordinates (structures)
            in that cluster.
        """

        log.debug("Obtaining cluster structures...")

        assert self.clusters is not None, "Clusters have not been computed!"

        log.debug(f"Dtrajs len: {len(self.dtrajs)}, [0] shape: {self.dtrajs[0].shape}")

        cluster_structures = dict()
        cluster_structure_weights = dict()

        # Move this elsewhere, WE segment weights are useful to have outside of this
        all_seg_weights = np.full(int(sum(self.numSegments)), fill_value=None)

        i = 0
        # total_num_iterations = len(self.numSegments)
        total_num_iterations = self.maxIter
        # Don't include the last iteration, where dynamics didn't run
        for _iter in range(1, total_num_iterations - 1):
            iter_weights = self.seg_weights[_iter]

            # This is an array, not a dict, so index by... well, index, and not iter_number
            num_segs_in_iter = int(self.numSegments[_iter - 1])

            # log.debug(f"Found {num_segs_in_iter} in iter {_iter}")
            # log.debug(f"Updating indices {i} :  {i + num_segs_in_iter}")

            assert None not in iter_weights, f"None in iter {_iter}, {iter_weights}"

            all_seg_weights[i : i + num_segs_in_iter] = iter_weights

            i += num_segs_in_iter

        log.debug(f"Got {all_seg_weights.shape} seg weights")

        # Assign each segment to a cluster by iterating over coords
        # Track the "absolute" segment index because all_seg_weights is a flat list
        seg_idx = 0
        for iteration in range(1, total_num_iterations - 1):

            iter_coords = self.get_iter_coordinates(iteration)
            segs_in_iter = int(self.numSegments[iteration - 1])

            for _seg in range(segs_in_iter):

                # log.debug(f"Iteration {iteration}, segment {_seg}, segs in iter {segs_in_iter}")
                # iteration-1 because dtrajs has n_iterations-1 elements

                try:
                    cluster_idx = self.dtrajs[iteration - 1][_seg]

                # This *should* trip if there was no data for this segments
                except IndexError:
                    log.error(
                        f"Tried to get dtraj for seg {_seg} at iteration {iteration-1}"
                    )
                    continue

                if cluster_idx in self.removed_clusters:

                    raise Exception(
                        "This dtraj point was in a removed cluster -- this should never happen!"
                    )
                    # log.debug(f"Skipping cluster {cluster_idx}")
                    # continue

                if cluster_idx not in cluster_structures.keys():
                    cluster_structures[cluster_idx] = []
                    cluster_structure_weights[cluster_idx] = []

                seg_coords = iter_coords[_seg]
                cluster_structures[cluster_idx].append(seg_coords)
                cluster_structure_weights[cluster_idx].append(all_seg_weights[seg_idx])

                seg_idx += 1

        assert len(list(cluster_structures.keys())) == len(
            list(cluster_structure_weights.keys())
        ), "Structures and weights have different numbers of bins?"

        test_cluster = list(cluster_structures.keys())[0]
        assert len(cluster_structures[test_cluster]) == len(
            cluster_structure_weights[test_cluster]
        ), "First MSM bin has different numbers of structures and weights"

        self.cluster_structures = cluster_structures
        self.cluster_structure_weights = cluster_structure_weights

        log.debug("Cluster structure mapping completed.")
        log.debug(f"Cluster keys are {cluster_structures.keys()}")

    def do_clustering(self, arg):

        kmeans_model, iters_to_use, cluster_args, processCoordinates = arg

        # Cast to a list in case it's a np array
        iters_to_use = list(iters_to_use)

        iteration = iters_to_use[0]

        min_coords = 1

        # The first time we cluster, we need at least n_clusters datapoints.
        # Before the first clustering has been done, cluster_centers_ is unset, so use that to know if  we're  on the
        #   first round.
        if not hasattr(kmeans_model, "cluster_centers_"):
            log.debug(
                f"First batch to k-means, need a minimum of {cluster_args['n_clusters']} segments"
            )
            min_coords = cluster_args["n_clusters"]

        iter_coords = self.get_iter_coordinates(iteration)

        used_iters = 0
        # Keep adding coords until you have more than your components
        while iter_coords.shape[0] <= min_coords:

            iteration = iters_to_use.pop(0)

            used_iters += 1

            _iter_coords = self.get_iter_coordinates(iteration)
            if _iter_coords.shape[0] == 0:
                continue

            iter_coords = np.append(iter_coords, _iter_coords, axis=0)

            log.debug(f"Have {iter_coords.shape[0]}, need {min_coords}")

        transformed_coords = self.coordinates.transform(processCoordinates(iter_coords))

        kmeans_model.partial_fit(transformed_coords)

        return kmeans_model, used_iters

    def do_discretization(self, arg):

        kmeans_model, iteration, processCoordinates = arg

        iter_coords = self.get_iter_coordinates(iteration)

        used_iters = 0
        # Keep adding coords until you have more than your components
        # This actually isn't necessary here like it is for the clustering, I could just return an empty array
        while iter_coords.shape[0] < 1:

            used_iters += 1

            _iter_coords = self.get_iter_coordinates(iteration + used_iters)
            if _iter_coords.shape[0] == 0:
                continue

            iter_coords = np.append(iter_coords, _iter_coords, axis=0)

        transformed_coords = self.coordinates.transform(processCoordinates(iter_coords))

        dtrajs = kmeans_model.predict(transformed_coords)

        # log.debug(f"Took {used_iters} extra iterations to process iteration {iteration}")

        return dtrajs, used_iters

    @ray.remote
    def do_ray_discretization(model, kmeans_model, iteration, processCoordinates):

        # model_id, kmeans_model_id, iteration, processCoordinates_id = arg

        # self = ray.get(model_id)
        # kmeans_model = ray.get(kmeans_model_id)
        # processCoordinates = ray.get(processCoordinates_id)
        self = model

        # Need to do this so the model's transformation array is writable -- otherwise predict chokes
        #   with 'buffer source array is read-only'.
        kmeans_model = deepcopy(kmeans_model)

        iter_coords = self.get_iter_coordinates(iteration)

        # If there are no coords for this iteration, return None
        if iter_coords.shape[0] == 0:
            return None, 0, iteration

        # Otherwise, apply the k-means model and discretize
        transformed_coords = self.coordinates.transform(processCoordinates(iter_coords))
        dtrajs = kmeans_model.predict(transformed_coords)

        return dtrajs, 1, iteration

    def cluster_coordinates(
        self,
        n_clusters,
        streaming=False,
        first_cluster_iter=None,
        use_ray=False,
        stratified=True,
        iters_to_use=None,
        **_cluster_args,
    ):

        self.clustering_method = None

        log.info(
            "Be aware: Number of cluster centers is an important parameter, and can drastically affect model "
            "quality. We recommend examining block-validation results with a range of numbers of clusters, to "
            "check for overfitting."
        )

        if stratified:
            log.info("Beginning stratified clustering.")
            self.clustering_method = "stratified"
            self.cluster_stratified(
                n_clusters=n_clusters,
                streaming=streaming,
                first_cluster_iter=first_cluster_iter,
                use_ray=use_ray,
                iters_to_use=iters_to_use,
                **_cluster_args,
            )

        # Make sure you know what you're doing if using this!
        else:
            log.info(
                "Beginning aggregate clustering. (Beware! This can produce poor clustering)"
            )
            self.clustering_method = "aggregated"
            self.cluster_aggregated(
                n_clusters=n_clusters,
                streaming=streaming,
                first_cluster_iter=first_cluster_iter,
                use_ray=use_ray,
                iters_to_use=iters_to_use,
                **_cluster_args,
            )

        # Do this so you can use it for building the validation models.
        #   We directly modify the clusters in cleaning, so this is the easiest way of recreating the original,
        #   "unclean" validation model.
        self.post_cluster_model = deepcopy(self)

    def cluster_aggregated(
        self,
        n_clusters,
        streaming=False,
        first_cluster_iter=None,
        use_ray=False,
        iters_to_use=None,
        **_cluster_args,
    ):
        """
        Use k-means to cluster coordinates into `n_clusters` cluster centers, and saves the resulting cluster object
        to a file.

        Saved cluster file is named
        "<`self.model_name`>_clusters_s<`self.first_iter`>_e<`self.last_iter`>_nC<`self.n_clusters`>.h5"

        Updates:
            - `self.n_clusters`
            - `self.clusters`
            - `self.clusterFile`

        Parameters
        ----------
        n_clusters: int
            Number of cluster centers to use.

        streaming: boolean
            Whether to stream k-means clustering, or load all from memory.
            Not supported for vamp dimensionality reduction.

        **_cluster_args:
            Keyword arguments that will be passed directly to cluster_kmeans

        Returns
        -------

        """

        log.debug(f"Doing clustering on {n_clusters} clusters")

        # Have to do this instead of setting a default argument so that we can specify custom iterations to use
        if first_cluster_iter is None:
            first_cluster_iter = 1

        self.n_clusters = n_clusters
        self.first_cluster_iter = first_cluster_iter
        # streaming = False

        # This is a bit of a gnarly chain of if statements, but it handles checking whether a user explicitly passed
        #   these arguments.
        # If you haven't specified a range of iterations to use, then go with first_cluster_iter
        if iters_to_use is None and first_cluster_iter is None:
            first_cluster_iter = 1
            iters_to_use = range(first_cluster_iter, self.maxIter)
        elif iters_to_use is None and first_cluster_iter is not None:
            iters_to_use = range(first_cluster_iter, self.maxIter)
        elif iters_to_use is not None and first_cluster_iter is not None:
            log.error(
                "Conflicting parameters -- either iters_to_use OR first_cluster_iter should be provided, not both."
            )
        else:
            # iters_to_use was provided, and first_cluster_iter was not
            pass

        if "metric" in _cluster_args.keys() or "k" in _cluster_args.keys():
            log.error(
                "You're passing pyemma-style arguments to k-means. K-means now uses sklearn, please update"
                "your code accordingly."
            )
            raise Exception(
                "PyEmma style arguments passed to kmeans, which is now based on sklearn"
            )

        # Set some default arguments, and overwrite them with the user's choices if provided
        cluster_args = {
            "n_clusters": n_clusters,
            "max_iter": 100,
        }
        cluster_args.update(_cluster_args)

        if streaming and not self.dimReduceMethod == "vamp":
            cluster_model = mini_kmeans(**cluster_args)
            # TODO: Any mini_batch_kmeans specific arguments?

        elif streaming and self.dimReduceMethod == "vamp":
            log.warning(
                "Streaming clustering is not supported for dimReduceMethod 'none'. Using standard k-means."
            )
            cluster_model = kmeans(**cluster_args)
            streaming = False

        else:
            cluster_model = kmeans(**cluster_args)

        if self.dimReduceMethod == "none":

            if not streaming:

                # TODO: Fix this to correctly use processCoordinates, like in the streaming case
                raise NotImplementedError(
                    "Non-streaming dimreduce None is currently broken"
                )

                _data = [
                    self.get_iter_coordinates(iteration).reshape(-1, self.coord_ndim * self.nAtoms)
                    for iteration in iters_to_use
                ]
                stacked_data = np.vstack(_data)

                if self.nAtoms > 1:
                    self.clusters = cluster_model.fit(stacked_data)

                # Else here is a little sketchy, but fractional nAtoms is useful for some debugging hacks.
                else:
                    self.clusters = cluster_model.fit(stacked_data)

                self.dtrajs = [
                    cluster_model.predict(iter_trajs) for iter_trajs in _data
                ]

            else:
                self.dtrajs = []

                extra_iters_used = 0
                for iter_idx, iteration in enumerate(
                    tqdm.tqdm(iters_to_use, desc="Clustering")
                ):

                    if extra_iters_used > 0:
                        extra_iters_used -= 1
                        log.debug(f"Already processed  iter  {iteration}")
                        continue

                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=1, mp_context=mp.get_context("fork")
                    ) as executor:
                        cluster_model, extra_iters_used = executor.submit(
                            self.do_clustering,
                            [
                                cluster_model,
                                [iters_to_use[iter_idx]],
                                cluster_args,
                                self.processCoordinates,
                            ],
                        ).result()

                self.clusters = cluster_model

                # Now compute dtrajs from the final model
                # TODO: Do I want to always discretize the full data?
                for iteration in range(1, self.maxIter):
                    iter_coords = self.get_iter_coordinates(iteration)
                    transformed_coords = self.coordinates.transform(
                        self.processCoordinates(iter_coords)
                    )

                    self.dtrajs.append(cluster_model.predict(transformed_coords))

        elif self.dimReduceMethod == "pca" and streaming:

            self.dtrajs = []

            # continued = False
            extra_iters_used = 0
            for iter_idx, iteration in enumerate(
                tqdm.tqdm(iters_to_use, desc="Clustering")
            ):

                if extra_iters_used > 0:
                    extra_iters_used -= 1
                    log.debug(f"Already processed  iter  {iteration}")
                    continue

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("fork")
                ) as executor:
                    cluster_model, extra_iters_used = executor.submit(
                        self.do_clustering,
                        [
                            cluster_model,
                            iters_to_use[iter_idx:],
                            cluster_args,
                            self.processCoordinates,
                        ],
                    ).result()

            self.clusters = cluster_model

            # Now compute dtrajs from the final model
            extra_iters_used = 0

            # If we're not using Ray, then calculate serially
            if not use_ray:
                for iteration in tqdm.tqdm(
                    range(1, self.maxIter), desc="Discretization"
                ):

                    if extra_iters_used > 0:
                        extra_iters_used -= 1
                        log.debug(f"Already processed  iter  {iteration}")
                        continue

                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=1, mp_context=mp.get_context("fork")
                    ) as executor:
                        dtrajs, extra_iters_used = executor.submit(
                            self.do_discretization,
                            [cluster_model, iteration, self.processCoordinates],
                        ).result()

                    self.dtrajs.append(dtrajs)

            # If we want to use Ray
            else:

                # First, connect to the ray cluster
                self.check_connect_ray()

                # Submit all the discretization tasks to the cluster
                task_ids = []

                model_id = ray.put(self)
                cluster_model_id = ray.put(cluster_model)
                process_coordinates_id = ray.put(self.processCoordinates)

                # max_inflight = 50
                for iteration in tqdm.tqdm(
                    range(1, self.maxIter), desc="Submitting discretization tasks"
                ):

                    # if len(task_ids) > max_inflight:
                    #
                    #     # The number that need to be ready before we can submit more
                    #     num_ready = iteration - max_inflight
                    #     ray.wait(task_ids, num_returns=num_ready)

                    _id = self.do_ray_discretization.remote(
                        model_id, cluster_model_id, iteration, process_coordinates_id
                    )
                    task_ids.append(_id)

                # As they're completed, add them to dtrajs
                dtrajs = [None] * (self.maxIter - 1)

                # Do these in bigger batches, dtrajs aren't very big

                with tqdm.tqdm(
                    total=len(task_ids), desc="Retrieving discretized trajectories"
                ) as pbar:
                    while task_ids:
                        result_batch_size = 50
                        result_batch_size = min(result_batch_size, len(task_ids))

                        # Returns the first ObjectRef that is ready, with a 20s timeout
                        finished, task_ids = ray.wait(
                            task_ids, num_returns=result_batch_size, timeout=20
                        )
                        results = ray.get(finished)

                        for dtraj, _, iteration in results:
                            dtrajs[iteration - 1] = dtraj
                            pbar.update(1)
                            pbar.refresh()

                # Remove all empty elements from dtrajs and assign to self.dtrajs
                self.dtrajs = [dtraj for dtraj in dtrajs if dtraj is not None]

        elif self.dimReduceMethod == "vamp" and streaming:

            raise NotImplementedError("VAMP + streaming clustering is not supported.")

        elif (
            self.dimReduceMethod == "pca"
            or self.dimReduceMethod == "vamp"
            and not streaming
        ):

            if self.dimReduceMethod == "pca":
                transformed_data = []

                for iteration in iters_to_use:
                    iter_coords = self.get_iter_coordinates(iteration)

                    # Skip if  this is an empty iteration
                    if iter_coords.shape[0] == 0:
                        continue

                    transformed_data.append(
                        self.coordinates.transform(self.processCoordinates(iter_coords))
                    )
                    log.debug(f"Appended trajs for iter {iteration}")

                stacked_data = np.vstack(transformed_data)
                self.clusters = cluster_model.fit(stacked_data)
                self.dtrajs = [
                    cluster_model.predict(iter_trajs) for iter_trajs in transformed_data
                ]

            # TODO: Make sure this is returned in the shape (iteration, ...)
            elif self.dimReduceMethod == "vamp":
                log.warning("Clustering VAMP-reduced data still very experimental!")

                transformed_data = self.coordinates.get_output()

                self.clusters = cluster_model.fit(transformed_data)
                self.dtrajs = [
                    cluster_model.predict(iter_trajs) for iter_trajs in transformed_data
                ]

        self.clusterFile = (
            self.modelName
            + "_clusters_s"
            + str(self.first_iter)
            + "_e"
            + str(self.last_iter)
            + "_nC"
            + str(self.n_clusters)
            + ".h5"
        )

        # self.dtrajs = self.clusters.dtrajs
        assert self.dtrajs is not None

        log.debug("Clustering completed.")
        # log.debug(f"Dtrajs: {self.clusters.dtrajs}")
        # self.clusters.save(self.clusterFile, save_streaming_chain=True, overwrite=True)

    def cluster_stratified(
        self,
        n_clusters,
        streaming=True,
        first_cluster_iter=None,
        use_ray=True,
        bin_iteration=2,
        iters_to_use=None,
        user_bin_mapper=None,
        **_cluster_args,
    ):
        """
        Perform full-stratified clustering, enforcing independent clustering for trajectories within each WE bin.

        Parameters
        ----------
        n_clusters: int
            Number of cluster centers per WE bin.

        streaming: bool (default True)
            Must be True for now.

        first_cluster_iter: int
            Iteration to start clustering on.
            Discretization is always done on all iteration, but it may not be necessary or desirable to use all to
            cluster.

        use_ray: bool (default True)
            Must be True for now.

        bin_iteration: int (default 2)
            Iteration to obtain bin definitions from.

        iters_to_use: array-like (default [1, model.maxIter])
            Specific set of iterations to cluster on.

        **_cluster_args:
            Arguments passed through to sklearn.cluster.MiniBatchKMeans

        Returns
        -------

        TODO
        ----
        Allow directly passing in a bin mapper or equivalent here. Way easier than trying to reverse engineer one.
        Require it if the existing is not supported.
        """

        # This provides backwards compatibility when loading WESTPA data from simulations where binning lived in westpa.binning
        # It's a little sketchy, but it import westpa.tools.binning under westpa.binning
        import importlib
        import westpa

        westpa.binning = importlib.import_module("westpa.tools.binning", "westpa.tools")

        if user_bin_mapper is not None:
            log.info("Loading user-specified bin mapper for stratified clustering.")
            bin_mapper = user_bin_mapper
        else:
            log.info(
                f"Obtaining bin definitions from iteration {bin_iteration} in file {self.fileList[0]}"
            )
            iteration = analysis.Run(self.fileList[0]).iteration(bin_iteration)
            log.info("Loading pickled bin mapper from H5 for stratified clustering...")
            bin_mapper = iteration.bin_mapper

            # Problem: I need a consistent set of bins, and some bin mappers may not return that! Some may re-calculate bins
            #    on-the-fly whenever data is passed to them, which will produce different bins for each iteration.
            #   In particular, ones that may are really any functional mapper:
            #   -   MABBinMapper will by construction never return bins that have everything populated, and by default,
            #       calculates boundaries on the fly
            #   -   PiecewiseBinMapper/VectorizingFuncBinMapper/FuncBinMapper are also function-based so they may do whatever
            #   Ones that won't:
            #   -   VoronoiBinMapper takes in a set of centers when it's initialized, so it'll be consistent
            #   -   Rectilinear is of course always consistent
            # Ideally, the best approach here would be to go through your bin mapper, recursing as necessary, and replacing
            #   any of the above functional bin mapper types with static bin mappers.

            if type(bin_mapper) not in SUPPORTED_MAPPERS:
                log.warning(
                    f"{type(bin_mapper)} mapper loaded, but supported mappers are {SUPPORTED_MAPPERS} and others may"
                    f"produce inconsistent bins between iterations. Please provide a supported user_bin_mapper."
                )
                raise Exception

            # TODO: Before moving on, make sure it's actually possible to populate each bin with enough segments to
            #       cluster at least once from the given set of iterations.
            #   In other words, just load up pcoords iteration by iteration, and count the total number in each bin.
            #   Note that it's possible you'll end up clustering in a bin once, but then pulling too few in that bin for
            #       the rest of the iterations after that, or something like that.
            #   The alternative to this would be in do_stratified_clustering(), if a bin isn't populated, check every
            #       subsequent iteration and see if it's possible to populate it from them.
            #   This is a little more flexible/dynamic, however, I think it may potentially require much more looping
            #       through iterations.

        # Clustering will not be performed in these bins
        ignored_bins = []

        if not streaming or not use_ray:
            log.error(
                "Stratified clustering currently MUST run in streaming mode (and with ray enabled for "
                "discretization). Enabling both and continuing..."
            )
            streaming = True
            use_ray = True

        # Do the streaming clustering, but cluster each bin individually.
        stratified_clusters = StratifiedClusters(
            bin_mapper, self, n_clusters, ignored_bins, **_cluster_args
        )

        # This is a bit of a gnarly chain of if statements, but it handles checking whether a user explicitly passed
        #   these arguments.
        # If you haven't specified a range of iterations to use, then go with first_cluster_iter
        if iters_to_use is None and first_cluster_iter is None:
            first_cluster_iter = 1
            iters_to_use = range(first_cluster_iter, self.maxIter)
        elif iters_to_use is None and first_cluster_iter is not None:
            iters_to_use = range(first_cluster_iter, self.maxIter)
        elif iters_to_use is not None and first_cluster_iter is not None:
            log.error(
                "Conflicting parameters -- either iters_to_use OR first_cluster_iter should be provided, not both."
            )
        else:
            # iters_to_use was provided, and first_cluster_iter was not
            pass

        # ## Build the clustering model
        self.dtrajs = []
        extra_iters_used = 0
        all_filled_bins = set()
        all_unfilled_bins = set()
        for iter_idx, iteration in enumerate(
            tqdm.tqdm(iters_to_use, desc="Clustering")
        ):

            if extra_iters_used > 0:
                extra_iters_used -= 1
                log.debug(f"Already processed  iter  {iteration}")
                continue

            ignored_bins = []

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("fork")
            ) as executor:
                (
                    stratified_clusters,
                    extra_iters_used,
                    filled_bins,
                    unfilled_bins,
                ) = executor.submit(
                    self.do_stratified_clustering,
                    [
                        self,
                        stratified_clusters,
                        iters_to_use[iter_idx:],
                        self.processCoordinates,
                        ignored_bins,
                    ],
                ).result()

                all_filled_bins.update(filled_bins)
                all_unfilled_bins.update(unfilled_bins)

        # all_filled_bins holds every bin that was clustered in
        # all_unfilled_bins holds any bin that was ever attempted, but unfilled
        # so, get the unfilled that were unfilled always, and never got clustered in
        true_unfilled = np.setdiff1d(range(bin_mapper.nbins), list(all_filled_bins))
        log.debug(f"Filled bins are {all_filled_bins}")
        log.debug(f"Unfilled bins were {all_unfilled_bins}")
        log.debug(f"True unfilled bins are {true_unfilled}")

        for unfilled_bin_idx in true_unfilled:
            remap_bin = self.find_nearest_bin(
                bin_mapper, unfilled_bin_idx, list(all_filled_bins)
            )
            stratified_clusters.we_remap[unfilled_bin_idx] = remap_bin
            log.debug(f"Remapped {unfilled_bin_idx} to {remap_bin}")

        # make sure this doesn't mess up we_remap later.. I think it should be fine, when I write to we_remap in the cleaning
        #       it should identify a superset of the bins identified here.

        self.clusters = stratified_clusters
        self.clusters.model = self

        ## The below -1 was removed, because now I may specify a custom set of WE bins
        ## -1 because we don't cluster in the target

        self.n_clusters = n_clusters * (bin_mapper.nbins)

        self.clusters.toggle = False
        self.launch_ray_discretization()

    @staticmethod
    def find_nearest_bin(bin_mapper, bin_idx, filled_bins):
        """
        Given a bin mapper, find the bin closest to bin_idx (that isn't bin_idx).

        Do this Voronoi-style by obtaining a set of bin centers, and finding which center bin_idx is closest to.

        Parameters
        ----------
        bin_mapper
        bin_idx

        Returns
        -------
        Index of the closest bin.

        TODO
        ----
        Note in the documentation that this can be overriden for finer control over empty bin mapping, if so desired.
        """

        assert len(filled_bins) > 0, (
            "Can't find nearest populated bin -- no WE bins are populated with clusters! "
            "Try fewer clusters/bin."
        )

        assert type(bin_mapper) in [
            VoronoiBinMapper,
            RectilinearBinMapper,
        ], f"{type(bin_mapper)} is unsupported!"

        if type(bin_mapper) is VoronoiBinMapper:

            centers = bin_mapper.centers
            distance_function = bin_mapper.dfunc

        elif type(bin_mapper) is RectilinearBinMapper:

            def _rmsd(point, _centers):
                return np.sqrt(np.mean(np.power(point - _centers, 2), axis=1))

            distance_function = _rmsd

            bounds = np.array(bin_mapper.boundaries)
            _centers = []
            for dim in bounds:
                _centers.append(dim[:-1] + (dim[1:] - dim[:-1]) / 2)
            centers = (
                np.array(np.meshgrid(*_centers)).T.squeeze().reshape(-1, len(bounds))
            )

        # Remove both the bin you're looking at, and any other unfilled bins
        # all_ignored = np.concatenate([[bin_idx], unfilled_bins])

        # Ignore any bin that isn't an explicitly provided filled bin
        all_ignored = np.setdiff1d(range(centers.shape[0]), filled_bins)
        other_centers = np.delete(centers, all_ignored, axis=0)

        closest = np.argmin(distance_function(centers[bin_idx], other_centers))

        # Increment index if it's past the ones we deleted
        for _bin_idx in sorted(all_ignored):
            if closest >= _bin_idx:
                closest += 1

        return closest

    def do_stratified_clustering(self, arg):
        """
        Perform the full-stratified clustering.

        This works as follows:

            1. Pull coordinates from the first iteration to process
            2. Assign each segment to a WE bin, using its pcoord and the bin_mapper associated with this StratifiedCluster
                object.
            3a. If any of the seen WE bins have fewer segments than cluster centers, and there are more iterations left
                to process, repeat from 1.
                - Note that this may add NEW seen bins, and those new seen bins may not be full yet -- so more iterations
                    may be required.
            3b. If any seen WE bins have fewer segments than cluster centers, **but no iterations are left to process**,
                then assign each structure in an "unfilled" WE bin to the "filled" WE bin with the closest index.,

            At this point, we have a set of structures and the WE bins they're associated with, and each WE bin has a
            number of structures equal to or greater than the target number of cluster centers.

            Within each WE bin:

            4. Apply dimensionality reduction to structures
            5. Update clustering for that bin using this set of dimensionality reduced coordinates.
        """

        self, kmeans_models, iters_to_use, processCoordinates, ignored_bins = arg

        # Cast to a list in case it's a np array
        iters_to_use = list(iters_to_use)

        bin_mapper = kmeans_models.bin_mapper

        min_coords = kmeans_models.cluster_args["n_clusters"]

        # The number of populated bins is the number of total bins - 1
        all_bins_have_segments = False

        # Until all bins are populated
        used_iters = -1
        iter_coords = []
        unique_bins = np.array([])
        counts = np.array([])
        we_bin_assignments = []

        # Maybe not even necessary to track iter_coords, just _iter_coords.
        # The problem with it as it stands is that assert -- imagine if I have to grab a second iteration
        while not all_bins_have_segments:
            unfilled_bins = []

            # This may cover the same use case as iteration > self.maxIter below
            try:
                iteration = iters_to_use.pop(0)
            except IndexError:
                log.warning(
                    f"At iteration {iteration} (pulled {used_iters} extra), couldn't get segments in all bins, and no "
                    f"iterations left."
                )

                # Which bin didn't have enough clusters?
                unfilled_bins = unique_bins[counts < min_coords]
                filled_bins = np.setdiff1d(unique_bins, unfilled_bins)

                # Find the nearest non-empty bin
                for unfilled_bin in unfilled_bins:

                    nearest_filled_bin = self.find_nearest_bin(
                        bin_mapper, unfilled_bin, list(filled_bins)
                    )

                    unfilled_bin_indices = np.where(we_bin_assignments == unfilled_bin)
                    log.warning(
                        f"Remapping {len(unfilled_bin_indices)} segments from unfilled bin {unfilled_bin} to "
                        f"{nearest_filled_bin} for stratified clustering"
                    )
                    we_bin_assignments[unfilled_bin_indices] = nearest_filled_bin

                # Remove unfilled bins from unique_bins
                unique_bins = filled_bins

                break

            if used_iters > -1:
                log.debug(
                    f"Still missing segs in bin {unique_bins[counts < min_coords]}. Pulled {used_iters+1} extra iter"
                )
                log.debug(f"Had {list(zip(unique_bins, counts))}, need {min_coords}")

            if iteration > self.maxIter:
                # TODO: Is this always a deal-breaker?
                log.warning(
                    f"At iteration {iteration} (pulled {used_iters} extra), couldn't get segments in all bins, and no "
                    f"iterations left"
                )
                break

            used_iters += 1
            _iter_coords = self.get_iter_coordinates(iteration)
            if used_iters == 0:
                iter_coords = _iter_coords
                pcoords = [x for x in self.pcoord0List]
            else:
                iter_coords = np.append(iter_coords, _iter_coords, axis=0)
                pcoords.extend(self.pcoord0List)
                log.debug(
                    f"After extension, pcoords: {len(pcoords)}, iter_coords: {iter_coords.shape}. Ignored: {ignored_bins}. Mapper: {bin_mapper}"
                )

            # Map coords to WE bins
            pcoord_array = np.array(pcoords)
            assert pcoord_array.shape[0] == iter_coords.shape[0], f"{pcoord_array.shape}, {iter_coords.shape}"

            # Ignore any segments that are in the basis or target
            pcoord_is_target = self.is_WE_target(pcoord_array)
            pcoord_is_basis = self.is_WE_basis(pcoord_array)
            pcoord_array = pcoord_array[~(pcoord_is_target | pcoord_is_basis)]

            if len(pcoord_array) > 0:
                we_bin_assignments = bin_mapper.assign(pcoord_array)
            else:
                log.debug(
                    "No coordinates outside of basis/target, we_bin_assignments will be empty and clustering"
                    f" will be skipped for this iteration. ({sum(pcoord_is_target)} in target, {sum(pcoord_is_basis)} in basis)"
                )
                we_bin_assignments = np.array([])

            unique_bins, counts = np.unique(we_bin_assignments, return_counts=True)
            all_bins_have_segments = np.all(counts >= min_coords)

        # By now, I have some segments in each WE bin.
        # Now, do clustering within each WE bin, and only cluster in WE bins that things were actually assigned to
        for i, _bin in enumerate(unique_bins):

            segs_in_bin = np.argwhere(we_bin_assignments == _bin)

            log.debug(f"Clustering {segs_in_bin} segments in WE bin {_bin}.")

            transformed_coords = self.coordinates.transform(
                processCoordinates(np.squeeze(iter_coords[segs_in_bin]))
            )

            try:
                kmeans_models.cluster_models[_bin].partial_fit(transformed_coords)
            except ValueError as e:
                log.info(f"Was on bin {_bin}")
                log.error(f"Error fitting k-means to bin {_bin}")
                raise e

        return kmeans_models, used_iters, unique_bins, unfilled_bins

    def organize_stratified(self, use_ray=True):
        """
        Alternative to organize_fluxMatrix, for stratified clustering.

        Parameters
        ----------
        use_ray: bool
            If True, use Ray when calculating the cleaned fluxmatrix.

        Returns
        -------

        """

        # Find disconnected states
        fmatrix_original = self.fluxMatrixRaw.copy()

        fmatrix = self.fluxMatrixRaw.copy()
        fmatrix[-1, -2] = 1.0
        connected_sets = find_connected_sets(fmatrix, directed=True)

        start_cleaning_idx = 1

        if len(connected_sets) == 1:
            log.info("Nothing to clean")
            #         return
            states_to_remove = []
        else:

            # # Experimental
            # # I've seen cases where I have two large disconnected sets of similar size, with problems that arise from cleaning
            # #   one entirely. I think maybe this is like two sides of a barrier, with weak connection between, so when you
            # #   do the cleaning basically everything on one side gets lumped together.
            # # To try and ameliorate this, what if we group together large clusters, and bank on the fact that after we clean
            # #   all the small isolated clusters, those large ones will be connected?
            # # This doesn't work :( After cleaning, these sets remain disconnected
            #
            # set_sizes = np.array([len(_set) for _set in connected_sets])
            # primary_set_size = set_sizes[0]
            #
            # # Keep any sets that are at least half the size of the primary set
            # start_cleaning_idx = np.argmin(set_sizes > (primary_set_size//2)).flatten()[0]
            # log.info(f"Merging sets of size {set_sizes[:start_cleaning_idx]} and cleaning the rest")
            # #

            log.debug(
                f"Cleaning states {np.concatenate(connected_sets[start_cleaning_idx:])}"
            )
            states_to_remove = np.concatenate(connected_sets[start_cleaning_idx:])

        # Does NOT include the basis/target -- none of these numbers of cluster centers should,
        #   since they're explicitly not clustered in
        pre_cleaning_n_clusters_per_bin = [
            len(cluster_model.cluster_centers_)
            if hasattr(cluster_model, "cluster_centers_")
            else 0
            for cluster_model in self.clusters.cluster_models
        ]

        log.debug(
            f"Pre-cleaning clusters per bin: "
            f"{list(zip(range(len(pre_cleaning_n_clusters_per_bin)), pre_cleaning_n_clusters_per_bin))}"
        )

        empty_we_bins = set()
        # target, basis = self.clusters.target_bins, self.clusters.basis_bins
        # Go through each WE bin, finding which clusters within it are not in the connected set
        #    and removing them.
        for we_bin in range(self.clusters.bin_mapper.nbins):

            consecutive_index = we_bin

            offset = sum(pre_cleaning_n_clusters_per_bin[:we_bin])

            # Get all the clusters that would be contained in this bin
            # clusters_in_bin = range(offset, offset + self.clusters.n_clusters_per_bin)
            n_clusters_in_bin = 0
            if hasattr(
                self.clusters.cluster_models[consecutive_index], "cluster_centers_"
            ):
                n_clusters_in_bin = len(
                    self.clusters.cluster_models[consecutive_index].cluster_centers_
                )
            # Otherwise, this is an uninitialized bin.
            # Bins can be uninitialized in 2 cases -- completely unvisited, or a basis/target
            # Segments in a basis/target should not be mapped to other bins -- however, this is short-circuited in
            #   StratifiedClusters.predict(), so even if they're mapped to those cluster centers, it's fine.

            clusters_in_bin = range(offset, offset + n_clusters_in_bin,)
            log.debug(f"Cluster models len: {len(self.clusters.cluster_models)}")
            log.debug(
                f"WE Bin {we_bin} (consec. index {consecutive_index}) contains {n_clusters_in_bin} clusters {clusters_in_bin}"
            )

            # Find which of the removed clusters are in this
            bin_clusters_to_clean = np.intersect1d(states_to_remove, clusters_in_bin)

            # Before actually cleaning, check if we're not cleaning anything, or if we're cleaning everything

            # If not cleaning anything, just move on
            if len(bin_clusters_to_clean) == 0:
                log.debug(f"not cleaning any clusters from bin {we_bin}")

                if len(clusters_in_bin) == 0:
                    empty_we_bins.add(we_bin)

                continue

            # If cleaning EVERYTHING, handle this bin differently
            # We'll just re-map it to a "good" adjacent WE bin
            elif len(bin_clusters_to_clean) == len(clusters_in_bin):
                empty_we_bins.add(we_bin)

            else:
                log.debug(
                    f"Cleaning {len(bin_clusters_to_clean)} clusters {bin_clusters_to_clean} from WE bin {we_bin}."
                )

            self.clusters.cluster_models[we_bin].cluster_centers_ = np.delete(
                self.clusters.cluster_models[we_bin].cluster_centers_,
                bin_clusters_to_clean - offset,
                0,
            )

        log.info(
            f"Started with {self.n_clusters} clusters, and removed {len(states_to_remove)}"
        )
        self.n_clusters = self.n_clusters - len(states_to_remove)
        log.debug(f"n_clusters is now {self.n_clusters}")

        assert (
            self.n_clusters > 1
        ), "All clusters would be cleaned! You probably need more data, fewer clusters, or both."

        # If a WE bin was completely emptied of cluster centers, map it to the nearest non-empty bin
        populated_we_bins = np.setdiff1d(
            range(self.clusters.bin_mapper.nbins), list(empty_we_bins)
        )

        if len(empty_we_bins) > 0:
            log.warning(f"All clusters were cleaned from bins {empty_we_bins}")

        for empty_we_bin in empty_we_bins:

            # Find the nearest non-empty bin
            nearest_populated_bin = self.find_nearest_bin(
                self.clusters.bin_mapper, empty_we_bin, populated_we_bins
            )

            # Replace self.clusters.cluster_models[empty_we_bin].cluster_centers_ with
            #   self.clusters.cluster_models[nearest_nonempty_we_bin].cluster_centers_
            self.clusters.we_remap[empty_we_bin] = nearest_populated_bin

        _running_total = 0
        for we_bin in range(self.clusters.bin_mapper.nbins):

            try:
                clusters_in_bin = len(
                    self.clusters.cluster_models[
                        self.clusters.we_remap[we_bin]
                    ].cluster_centers_
                )
            except AttributeError as e:
                log.error(
                    f"Error obtaining clusters for WE bin {we_bin}, remapped to {self.clusters.we_remap[we_bin]}. "
                    f"Target {self.clusters.target_bins}, basis {self.clusters.basis_bins}"
                )
                raise e

            _running_total += clusters_in_bin
            log.debug(
                f"{clusters_in_bin} in bin {we_bin}. Running total: {_running_total}"
            )

        # Now re-discretize
        self.clusters.toggle = False
        self.clusters.processing_from = False
        self.launch_ray_discretization()

        pcoord_sort_indices = self.get_cluster_centers()

        # And recalculate the flux matrix
        self.clusters.toggle = True
        self.clusters.processing_from = True
        self.get_fluxMatrix(*self._fluxMatrixParams, use_ray=use_ray)
        self.clusters.processing_from = False
        self.clusters.toggle = False

        # Set the cleaned matrix as fluxMatrix
        fluxMatrix = self.fluxMatrixRaw.copy()
        fluxMatrix = fluxMatrix[pcoord_sort_indices, :]
        fluxMatrix = fluxMatrix[:, pcoord_sort_indices]
        self.fluxMatrix = fluxMatrix / np.sum(fluxMatrix)

        self.fluxMatrixRaw = fmatrix_original

        self.indBasis = np.array([self.n_clusters])
        self.indTargets = np.array([self.n_clusters + 1])
        self.nBins = self.n_clusters + 2

        self.cluster_mapping = {x: x for x in range(self.n_clusters + 2)}

        # Sanity check that cleaning worked
        fmatrix = self.fluxMatrix.copy()
        fmatrix[-1, -2] = 1.0
        connected_sets = find_connected_sets(fmatrix, directed=True)
        log.debug(
            f"After cleaning, shape is {fmatrix.shape} and disconnected sets are: {connected_sets[1:]}"
        )
        assert (
            len(connected_sets[start_cleaning_idx:]) == 0
        ), "Still not clean after cleaning!"

    def launch_ray_discretization(self):
        """
        Apply discretization in parallel, through Ray

        Returns
        -------

        Todo
        ====
        Replace manual discretization calls in cluster_aggregated with this
        Add flag to toggle between stratified and regular do_ray_discretization
        """

        self.check_connect_ray()

        self.dtrajs = []

        # Submit all the discretization tasks to the cluster
        task_ids = []

        model_id = ray.put(self)
        cluster_model_id = ray.put(self.clusters)
        process_coordinates_id = ray.put(self.processCoordinates)

        # max_inflight = 50
        for iteration in tqdm.tqdm(
            range(1, self.maxIter), desc="Submitting discretization tasks"
        ):

            _id = self.do_stratified_ray_discretization.remote(
                model_id, cluster_model_id, iteration, process_coordinates_id
            )
            task_ids.append(_id)

        # As they're completed, add them to dtrajs
        dtrajs = [None] * (self.maxIter - 1)

        # Do these in bigger batches, dtrajs aren't very big

        with tqdm.tqdm(
            total=len(task_ids), desc="Retrieving discretized trajectories"
        ) as pbar:
            while task_ids:
                result_batch_size = 50
                result_batch_size = min(result_batch_size, len(task_ids))

                # Returns the first ObjectRef that is ready, with a 20s timeout
                finished, task_ids = ray.wait(
                    task_ids, num_returns=result_batch_size, timeout=20
                )
                results = ray.get(finished)

                for dtraj, _, iteration, target_bins, basis_bins in results:

                    self.clusters.target_bins.update(target_bins)
                    self.clusters.basis_bins.update(basis_bins)

                    dtrajs[iteration - 1] = dtraj
                    pbar.update(1)
                    pbar.refresh()

        # Remove all empty elements from dtrajs and assign to self.dtrajs
        self.dtrajs = [dtraj for dtraj in dtrajs if dtraj is not None]

        log.info("Discretization complete")

    def get_cluster_centers(self):
        """
        Standalone method to obtain average pcoords of all segments in each cluster.

        This functionality is contained in organize_fluxMatrix.

        TODO
        ====
        Remove duplicate code in organize_fluxMatrix.
        """

        # Get the average pcoord of trajs in each bin
        cluster_pcoord_centers = np.zeros((self.n_clusters + 2, self.pcoord_ndim))
        target_cluster_index = self.n_clusters + 1  # Target at -1
        basis_cluster_index = self.n_clusters  # basis at -2
        log.debug(f"Basis, target are {basis_cluster_index}, {target_cluster_index}")
        cluster_pcoord_centers[target_cluster_index] = self.target_bin_center
        cluster_pcoord_centers[basis_cluster_index] = self.basis_bin_center

        cluster_pcoord_range = np.zeros((self.n_clusters + 2, self.pcoord_ndim, 2))
        cluster_pcoord_range[target_cluster_index] = [
            self.target_bin_center,
            self.target_bin_center,
        ]
        cluster_pcoord_range[basis_cluster_index] = [
            self.basis_bin_center,
            self.basis_bin_center,
        ]

        cluster_pcoord_all = [[] for _ in range(self.n_clusters + 2)]

        for cluster in range(self.n_clusters):
            idx_traj_in_cluster = [np.where(dtraj == cluster) for dtraj in self.dtrajs]

            # Get the pcoord points that correspond to these dtraj points
            offset = 0
            pcoord_indices = []
            for i, idxs in enumerate(idx_traj_in_cluster):
                pcoord_idxs = offset + idxs[0]
                pcoord_indices.extend(pcoord_idxs)
                offset += len(self.dtrajs[i])

            if len(self.pcoordSet[pcoord_indices, 0]) == 0:
                target_bin_index = self.n_clusters + 1

                if not cluster == target_bin_index:
                    # log.warning(idx_traj_in_cluster)
                    log.warning(
                        f"No trajectories in cluster {cluster}! (Target was {target_bin_index})"
                    )

                cluster_pcoord_centers[cluster] = np.nan
                cluster_pcoord_range[cluster] = [np.nan, np.nan]
                cluster_pcoord_all.append([None])
                continue

            cluster_pcoord_centers[cluster] = np.nanmean(
                self.pcoordSet[pcoord_indices, 0], axis=0
            )
            cluster_pcoord_range[cluster] = [
                np.nanmin(self.pcoordSet[pcoord_indices, 0], axis=0),
                np.nanmax(self.pcoordSet[pcoord_indices, 0], axis=0),
            ]
            cluster_pcoord_all[cluster] = self.pcoordSet[pcoord_indices, 0]

        pcoord_sort_indices = np.argsort(cluster_pcoord_centers[:, 0])

        self.targetRMSD_centers = cluster_pcoord_centers[pcoord_sort_indices]
        self.targetRMSD_minmax = cluster_pcoord_range[pcoord_sort_indices]
        self.targetRMSD_all = np.array(cluster_pcoord_all)[pcoord_sort_indices]

        return pcoord_sort_indices

    @ray.remote
    def do_stratified_ray_discretization(
        model, kmeans_model, iteration, processCoordinates
    ):

        # model_id, kmeans_model_id, iteration, processCoordinates_id = arg

        # import sys
        # import westpa.core.binning

        # sys.modules["westpa.binning"] = sys.modules["westpa.core.binning"]
        # This is silly -- I need to import westpa.core.binning so it's loaded into sys.modules but the linter
        #   complains that it's unused... so, use it.
        # log.debug(f"Loaded {westpa.core.binning}")

        # self = ray.get(model_id)
        # kmeans_model = ray.get(kmeans_model_id)
        # processCoordinates = ray.get(processCoordinates_id)
        self = model

        # Need to do this so the model's transformation array is writable -- otherwise predict chokes
        #   with 'buffer source array is read-only'.
        kmeans_model = deepcopy(kmeans_model)
        kmeans_model.model = self

        # for i, cluster_model in enumerate(kmeans_model.cluster_models):
        #     print(f"Model {i}: \t ", end=" ")
        #     try:
        #         print(cluster_model.cluster_centers_)
        #     except AttributeError:
        #         print("No cluster centers!")

        iter_coords = self.get_iter_coordinates(iteration)

        # If there are no coords for this iteration, return None
        if iter_coords.shape[0] == 0:
            return None, 0, iteration

        # Otherwise, apply the k-means model and discretize
        transformed_coords = self.coordinates.transform(processCoordinates(iter_coords))

        try:
            dtrajs = kmeans_model.predict(transformed_coords)
        except AttributeError as e:
            log.error("Cluster center was not initialized and not remapped")
            log.error(kmeans_model.we_remap)
            raise e
            # TODO: Remap to nearest visited

        return dtrajs, 1, iteration, kmeans_model.target_bins, kmeans_model.basis_bins

    def load_clusters(self, clusterFile):
        """
        Load clusters from a file.

        Updates:
            - `self.clusters`
            - `self.n_clusters`

        Parameters
        ----------
        clusterFile: str
            Filename to load clusters from.

        Returns
        -------
        None
        """

        log.debug(f"Found saved clusters -- loading from {clusterFile}")

        self.clusters = pyemma.load(clusterFile)
        self.n_clusters = np.shape(self.clusters.clustercenters)[0]

    @ray.remote
    def get_iter_fluxMatrix_ray(model, processCoordinates, n_iter):

        # model_id, n_iter, processCoordinates_id = args

        # self = ray.get(model_id)
        # processCoordinates = ray.get(processCoordinates_id)
        self = model

        self.processCoordinates = processCoordinates

        self.clusters = deepcopy(self.clusters)
        self.clusters.model = self

        iter_fluxmatrix = self.get_iter_fluxMatrix(n_iter)

        return iter_fluxmatrix, n_iter

    def get_iter_fluxMatrix(self, n_iter):
        """
        Get the flux matrix for an iteration.

        1. Update state with data from the iteration you want to compute the flux matrix for
        2. Load transition data at the requested lag

        Parameters
        ----------
        n_iter

        Returns
        -------

        TODO
        ----
        This function is slow because of the call to get_transition_data_lag0(). See that function for more info.
        """

        # 1. Update state with data from the iteration you want to compute the flux matrix for
        # sys.stdout.write("iteration " + str(n_iter) + ": data \n")
        self.load_iter_data(n_iter)

        #  2. Load transition data at the requested lag
        if self.n_lag > 0:
            # If you're using more than 1 lag, obtain segment histories at that lag
            #   This means you have to look back one or more iterations

            # sys.stdout.write("segment histories \n")
            self.get_seg_histories(self.n_lag + 1)
            # sys.stdout.write(" transition data...\n")
            self.get_transition_data(self.n_lag)
        elif self.n_lag == 0:
            # If you're using a lag of 0, your coordinate pairs are just the beginning/end of the iteration.
            #   In that, it's not possible to have warps/recycles, since those are done handled between iterations.
            try:
                self.get_transition_data_lag0()
            except KeyError:
                log.warning(
                    f"No coordinates for iter {n_iter}, skipping this iter in fluxmatrix calculation"
                )
                return np.zeros(shape=(self.n_clusters + 2, self.n_clusters + 2))

        log.debug(f"Getting flux matrix for iter {n_iter} with {self.nSeg} segments")
        # If you used a lag of 0, transitions weights are just weightList
        # If you used a lag > 0, these include the weight histories from previous iterations
        # num_transitions = np.shape(self.transitionWeights)[0]

        # Create dedicated clusters for the target and basis states,
        # and reassign any points within the target or basis to those
        target_cluster_index = self.n_clusters + 1
        basis_cluster_index = self.n_clusters

        # (Segment, Atom, [lagged, current coord])
        log.debug(f"Coord pairlist shape is {self.coordPairList.shape}")

        # Get which coords are not NaN
        good_coords = ~np.isnan(self.coordPairList).any(axis=(1, 2, 3))

        # Assign a cluster to the lagged and the current coords
        reduced_initial = self.reduceCoordinates(
            self.coordPairList[good_coords, :, :, 0]
        )

        reduced_final = self.reduceCoordinates(self.coordPairList[good_coords, :, :, 1])

        # Wrap this in a try to make sure we unset the toggle
        # Note: This toggle is only used for stratified clusteirng, but it's set either way.
        self.clusters.toggle = True
        self.clusters.processing_from = True
        try:
            start_cluster = self.clusters.predict(reduced_initial)
            end_cluster = self.clusters.predict(reduced_final)
        except Exception as e:
            log.error(reduced_initial)
            log.error(reduced_final)
            self.clusters.processing_from = False
            self.clusters.toggle = False
            raise e
        else:
            self.clusters.processing_from = False
            self.clusters.toggle = False

        log.debug(f"Cluster 0 shape: {start_cluster.shape}")

        # Record every point where you're in the target
        ind_end_in_target = np.where(self.is_WE_target(self.pcoord1List))

        if ind_end_in_target[0].size > 0:
            log.debug(
                "Number of post-transition target1 entries: "
                + str(ind_end_in_target[0].size)
                + "\n"
            )
        else:
            log.debug(f"No target1 entries. {ind_end_in_target}")

        # Get the index of every point
        ind_start_in_basis = np.where(self.is_WE_basis(self.pcoord0List[good_coords]))
        if ind_start_in_basis[0].size > 0:
            log.debug(
                "Number of pre-transition points in basis0: "
                + str(ind_start_in_basis[0].size)
                + "\n"
            )

        ind_end_in_basis = np.where(self.is_WE_basis(self.pcoord1List[good_coords]))
        if ind_end_in_basis[0].size > 0:
            log.debug(
                "Number of post-transition points in basis1: "
                + str(ind_end_in_basis[0].size)
                + "\n"
            )

        log.debug(
            f"Target cluster index is: {target_cluster_index},  basis cluster index is: {basis_cluster_index}"
        )

        # Re-assign points that were in either the target or the basis to the target or basis clusters
        end_cluster[ind_end_in_target] = target_cluster_index
        start_cluster[ind_start_in_basis] = basis_cluster_index
        end_cluster[ind_end_in_basis] = basis_cluster_index

        # coo_matrix takes in (Data, (x, y)) and then gives you a matrix, with the point at Data[i]
        #   placed at (x,y)[i]
        # Data here is just the number of segments since each segment is associated with 1 transition
        try:
            fluxMatrix = coo_matrix(
                (self.transitionWeights[good_coords], (start_cluster, end_cluster)),
                shape=(self.n_clusters + 2, self.n_clusters + 2),
            ).todense()
        except ValueError as e:
            log.error(
                f"Iter_fluxmatrix failed. Transition was from {start_cluster} -> {end_cluster} "
                f"\n\t(Total {self.n_clusters + 2} clusters)"
                f"\n\t(End in target: {ind_end_in_target})"
            )
            raise e

        # While the sparse matrix implementation is nice and efficient, using the np.matrix type is a little weird
        #   and fragile, and it's possible it'll be deprecated in the future.
        fluxMatrix = fluxMatrix.A

        return fluxMatrix

    def get_pcoord1D_fluxMatrix(self, n_lag, first_iter, last_iter, binbounds):
        self.n_lag = n_lag
        nBins = binbounds.size - 1
        fluxMatrix = np.zeros((nBins, nBins))
        nI = 0
        f = h5py.File(self.modelName + ".h5", "w")
        dsetName = (
            "/s"
            + str(first_iter)
            + "_e"
            + str(last_iter)
            + "_lag"
            + str(n_lag)
            + "_b"
            + str(nBins)
            + "/pcoord1D_fluxMatrix"
        )
        e = dsetName in f

        # Again, disable this file
        if True or not e:
            dsetP = f.create_dataset(dsetName, np.shape(fluxMatrix))
            for iS in range(first_iter + 1, last_iter + 1):
                if n_lag > 0:
                    fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix(iS, binbounds)
                elif n_lag == 0:
                    log.debug(f"Calling with {iS}, {binbounds}")
                    fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix_lag0(iS, binbounds)
                fluxMatrixI = fluxMatrixI / np.sum(
                    self.weightList
                )  # correct for multiple trees
                fluxMatrix = fluxMatrix + fluxMatrixI
                nI = nI + 1
                dsetP[:] = fluxMatrix / nI
                dsetP.attrs["iter"] = iS
            fluxMatrix = fluxMatrix / nI
        elif e:
            dsetP = f[dsetName]
            fluxMatrix = dsetP[:]
            try:
                nIter = dsetP.attrs["iter"]
            except Exception as e:
                nIter = first_iter + 1
                fluxMatrix = np.zeros((nBins, nBins))
                log.error(e)
            nI = 1
            if nIter < last_iter:
                for iS in range(nIter, last_iter + 1):
                    if n_lag > 0:
                        # TODO: Is this even implemented..?
                        fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix(iS, binbounds)
                    if n_lag == 0:
                        fluxMatrixI = self.get_iter_pcoord1D_fluxMatrix_lag0(
                            iS, binbounds
                        )
                    fluxMatrix = fluxMatrix + fluxMatrixI
                    nI = nI + 1
                    dsetP[:] = fluxMatrix / nI
                    dsetP.attrs["iter"] = iS
                fluxMatrix = fluxMatrix / nI
        f.close()
        self.pcoord1D_fluxMatrix = fluxMatrix

    def get_iter_pcoord1D_fluxMatrix_lag0(self, n_iter, binbounds):
        """
        Compute a flux-matrix in the space of the 1D progress coordinate, at a given iteration.

        Parameters
        ----------
        n_iter: integer
            Iteration to compute flux matrix for

        binbounds: array-like
            Array of progress coordinate bin boundaries

        Returns
        -------
        The fluxmatrix at iteration n_iter.
        """

        # log.info(n_iter)
        # log.info(binbounds)

        log.debug("iteration " + str(n_iter) + ": solving fluxmatrix \n")

        self.load_iter_data(n_iter)

        weightList = self.weightList.copy()

        num_segments_in_iter = np.shape(self.weightList)[0]

        nBins = binbounds.size - 1
        fluxMatrix = np.zeros((nBins, nBins))

        # Lists of all parent and child pcoords
        pcoord0 = self.pcoord0List[:, 0]
        pcoord1 = self.pcoord1List[:, 0]

        # Assign parent/child pcoords to bins
        bins0 = np.digitize(pcoord0, binbounds)
        bins1 = np.digitize(pcoord1, binbounds)

        for seg_idx in range(num_segments_in_iter):
            # I THINK WHAT'S HAPPENING HERE IS:
            # The lowest binbound provided here should be smaller than the smallest possible
            #   value in the trajectory.
            # I.e., because of the way bins are defined for WE, you might have bin bounds like
            #   [0,1,2,3], where x>3 is in the basis and 0<x<1 is in the target. However, digitize
            #   would assign a point of 0.5, which should be in the target, to index 1.
            # I think this -1 corrects for that.

            from_bin_index = bins0[seg_idx] - 1
            to_bin_index = bins1[seg_idx] - 1

            # Set the weight of any segment that jumps more than 12 bins to 0?
            #   This seems super risky, and also originally this function didn't copy weightList, so it
            #   modified weightList in state for anything that runs after.
            # In particular, I think that's bad because if you have more than 12 bins, going from the target to
            #   the basis is going to be set to 0 by this logic.
            # So, I'm going to disable this for now...
            if False and np.abs(from_bin_index - to_bin_index) > 12:
                weightList[seg_idx] = 0.0

            fluxMatrix[from_bin_index, to_bin_index] = (
                fluxMatrix[from_bin_index, to_bin_index] + weightList[seg_idx]
            )

        return fluxMatrix

    def get_fluxMatrix(
        self, n_lag, first_iter=1, last_iter=None, iters_to_use=None, use_ray=False
    ):
        """
        Compute the matrix of fluxes at a given lag time, for a range of iterations.

        Checks if a file has been written named
        "<`self.modelName`>_s<`first_iter`>_e<`last_iter`>_lag<`n_lag`>_clust<`self.n_clusters`>.h5".
        If this file exists, load it and recalculate if it was calculated at an earlier iteration.
        Otherwise, write it.

        Updates:
            - `self.n_lag`
            - `self.errorWeight`
            - `self.errorCount`
            - `self.fluxMatrixRaw`

        Parameters
        ----------
        n_lag: int
            Number of lags to use.
        first_iter: int
            First iteration to use.
        last_iter: int
            Last iteration to use.

        Returns
        -------
        None

        """

        self._fluxMatrixParams = [n_lag, first_iter, last_iter, iters_to_use]

        if iters_to_use is not None:
            log.debug(
                "Specific iterations to use were provided for fluxmatrix calculation, using those."
            )
        elif iters_to_use is None:

            if last_iter is None:
                last_iter = self.maxIter
            iters_to_use = range(first_iter + 1, last_iter + 1)

        # Else, if iters_to_use is Not none, and Not (first_iter is None and last_iter is None)
        else:
            log.error(
                "Both specific iterations to use AND first/last were provided to fluxmatrix calculation. Specify"
                " one or the other, not both."
            )
            raise Exception

        log.debug("Computing all flux matrices")

        self.n_lag = n_lag
        self.errorWeight = 0.0
        self.errorCount = 0

        # +2 because the basis and target states are the last two indices
        fluxMatrix = np.zeros((self.n_clusters + 2, self.n_clusters + 2))
        nI = 0

        fileName = (
            self.modelName
            + "-fluxmatrix-_s"
            + str(first_iter)
            + "_e"
            + str(last_iter)
            + "_lag"
            + str(n_lag)
            + "_clust"
            + str(self.n_clusters)
        )

        # Overwrite this file, don't try to read from it.  Hence the "w" flag
        # TODO: Maybe in the future return to this,
        #  but it caused more problems than it was worth when doing multiple runs.
        try:
            f = h5py.File(fileName + ".h5", "w")
        except BlockingIOError:
            # Janky -- if the file exists, pick a random name
            import random

            fileName += f"_{int(random.random()*1000):04d}"
            f = h5py.File(fileName + ".h5", "w")

        dsetName = "fluxMatrix"

        # FIXME: name this something descriptive or just use the 'in' statement in the if/elif
        fluxmatrix_exists_in_h5 = dsetName in f
        always_overwrite_fluxmatrix = True

        # If this data file does not contain a fluxMatrix entry, create it
        # For now, don't use a saved fluxmatrix, annoying to debug
        if always_overwrite_fluxmatrix or not fluxmatrix_exists_in_h5:
            # Create the fluxMatrix dataset
            dsetP = f.create_dataset(dsetName, np.shape(fluxMatrix))
            dsetP[:] = fluxMatrix

            # Add up the flux matrices for each iteration to get the flux matrix.
            # Then, save that matrix to the data file, along with the number of iterations used
            # FIXME: Duplicated code
            # The range is offset by 1 because you can't calculate fluxes for the 0th iteration
            if not use_ray:
                for iS in tqdm.tqdm(iters_to_use, desc="Constructing flux matrix",):
                    log.debug("getting fluxMatrix iter: " + str(iS) + "\n")

                    # fluxMatrixI = self.get_iter_fluxMatrix(iS)
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=1, mp_context=mp.get_context("fork")
                    ) as executor:
                        fluxMatrixI = executor.submit(
                            self.get_iter_fluxMatrix, iS,
                        ).result()

                    fluxMatrix = fluxMatrix + fluxMatrixI
                    nI = nI + 1

                    # f = h5py.File(fileName + ".h5", "a")
                    dsetP = f[dsetName]
                    dsetP[:] = fluxMatrix / nI
                    dsetP.attrs["iter"] = iS
                    log.debug(f"Completed flux matrix for iter {iS}")

            # If we're running through Ray..
            else:

                # First, connect to the ray cluster
                self.check_connect_ray()

                # Submit all the tasks for iteration fluxmatrix calculations
                task_ids = []

                model_id = ray.put(self)
                processCoordinates_id = ray.put(self.processCoordinates)

                # max_inflight = 70
                for iteration in tqdm.tqdm(
                    iters_to_use, desc="Submitting fluxmatrix tasks",
                ):

                    # Allow 1000 in flight calls
                    # This is buggy, so disable for now.
                    # For example, if i = 5000, this call blocks until that
                    # 4000 of the object_refs in result_refs are ready
                    # and available.
                    # See: https://docs.ray.io/en/latest/ray-design-patterns/limit-tasks.html
                    # if len(task_ids) > max_inflight:
                    #
                    #     # The number that need to be ready before we can submit more
                    #     num_ready = iteration - max_inflight
                    #     log.info(f"At iteration {iteration}, waiting to submit more jobs until {num_ready} are ready")
                    #     ready, notready = ray.wait(task_ids, num_returns=num_ready)
                    #     log.info(f"At iteration {iteration}, {len(ready)} jobs are ready {len(notready)} not, submitting more.")

                    # log.debug(f"Submitted fluxmatrix task iteration {iteration}")
                    _id = self.get_iter_fluxMatrix_ray.remote(
                        model_id, processCoordinates_id, iteration
                    )

                    task_ids.append(_id)

                # Wait for them to complete
                # Process results as they're ready, instead of in submission order
                #  See: https://docs.ray.io/en/latest/ray-design-patterns/submission-order.html
                # Additionally, this batches rather than getting them all at once, or one by one.

                with tqdm.tqdm(
                    total=len(iters_to_use), desc="Retrieving flux matrices"
                ) as pbar:
                    while task_ids:
                        result_batch_size = 50
                        result_batch_size = min(result_batch_size, len(task_ids))
                        log.debug(
                            f"Waiting for {result_batch_size} results ({len(task_ids)} total remain)"
                        )

                        # Returns the first ObjectRefs that are ready, with a 60s timeout.
                        finished, task_ids = ray.wait(
                            task_ids, num_returns=result_batch_size, timeout=20
                        )
                        results = ray.get(finished)
                        log.debug(f"Obtained {len(results)} results")

                        # Add each matrix to the total fluxmatrix
                        for _fmatrix, _iter in results:
                            fluxMatrix = fluxMatrix + _fmatrix
                            pbar.update(1)
                            pbar.refresh()

                log.info("Fluxmatrices all obtained")

                # Write the H5. Can't do this per-iteration, because we're not guaranteed to be going sequentially now

                nI = len(iters_to_use)
                dsetP = f[dsetName]
                dsetP[:] = fluxMatrix / nI
                dsetP.attrs["iter"] = nI

            f.close()

            # Normalize the flux matrix by the number of iterations that it was calculated with
            fluxMatrix = fluxMatrix / nI

        # If this datafile DOES contain a fluxMatrix entry...
        # HACK: This explicitly never runs now, but keep it around in case we decide to re-add this later.
        # TODO: Update this to work with the specific iterations
        elif fluxmatrix_exists_in_h5:

            log.info("Fluxmatrix already exists in h5 file, loading saved.")

            # Load the existing fluxMatrix
            dsetP = f[dsetName]
            fluxMatrix = dsetP[:]
            nIter = dsetP.attrs["iter"]
            # f.close()
            nI = 1

            # If the flux matrix was calculated at an earlier iteration than the current/requested last iteration, then
            #   recalculate it and update it.
            # TODO: Check if first_iter changed too? That's not stored anywhere

            #  For now, just always overwrite any saved data.
            # if nIter < last_iter:
            if True:
                log.warning(f"Ignoring any saved fluxmatrix in {fileName}.h5")
                for iS in range(nIter, last_iter + 1):
                    fluxMatrixI = self.get_iter_fluxMatrix(iS)
                    fluxMatrix = fluxMatrix + fluxMatrixI
                    nI = nI + 1
                    # f = h5py.File(fileName + ".h5", "a")
                    dsetP = f[dsetName]
                    dsetP[:] = fluxMatrix / nI
                    dsetP.attrs["iter"] = iS
                    # f.close()
                    sys.stdout.write("getting fluxMatrix iter: " + str(iS) + "\n")
                f.close()
                fluxMatrix = fluxMatrix / nI

        f.close()

        # Update state with the new, updated, or loaded from file fluxMatrix.
        self.fluxMatrixRaw = fluxMatrix

    def organize_fluxMatrix(self, use_ray=False, **args):

        if not hasattr(self, "clustering_method"):
            log.warning(
                "self.clustering_method is not set. This may be a model saved before stratified was "
                "implemented, or you may not have run cluster_coordinates! "
                "Assuming the former and setting to aggregated."
            )

            self.clustering_method = "aggregated"

        if self.clustering_method == "stratified":
            self.organize_stratified(use_ray)

            # TODO: Respect do_cleaning=False for blockwise stratified

        elif self.clustering_method == "aggregated":

            # If the user has manually specified states to keep, skip the normal cleaning, and just do what they said
            if "states_to_keep" in args.keys():
                self.organize_aggregated(use_ray=use_ray, **args)
                return

            # self.organize_aggregated(use_ray, **args)

            fmatrix = self.fluxMatrixRaw.copy()
            # Add recycling to avoid detecting the target as a sink
            fmatrix[-1, -2] = 1.0

            flat_raw_islands = np.sort(
                [i for s in find_connected_sets(fmatrix, directed=True)[1:] for i in s]
            )

            new_args = {k: v for k, v in args.items()}
            new_args["do_cleaning"] = False
            states_to_keep = self.organize_aggregated(
                use_ray=use_ray, **new_args
            ).astype(bool)

            regular_clean = np.argwhere(~states_to_keep)

            if len(flat_raw_islands) > 0:
                states_to_keep[flat_raw_islands] = False

            states_to_keep[[self.n_clusters, self.n_clusters + 1]] = True

            modified_clean = np.argwhere(~states_to_keep)

            log.debug(
                f"Modified cleaning added states {np.setdiff1d(modified_clean, regular_clean)}"
            )

            # If you didn't want to do cleaning, you've gone far enough, return the list of good state indices
            if "do_cleaning" in args.keys() and not args["do_cleaning"]:
                return np.argwhere(states_to_keep)

            # Otherwise, carry on and actually clean
            new_args = {k: v for k, v in args.items()}
            new_args["states_to_keep"] = np.argwhere(states_to_keep)
            self.organize_aggregated(use_ray=use_ray, **new_args)

            return np.argwhere(states_to_keep)

        else:
            raise Exception(
                f"Unrecognized clustering_method (Had: {self.clustering_method})"
            )

    def organize_aggregated(
        self, use_ray=False, do_cleaning=True, states_to_keep=None, rediscretize=True
    ):
        """
        Do some cleaning on the flux matrix, and update state with the cleaned flux matrix.

        Namely:
            - Remove unvisited clusters
            - Remove bins with no connectivity
            - Sort along the bins' projection in pcoord 1

        Parameters
        ----------
        ray_args : dict
            Dictionary whose values are currently unused, but containing an 'address' and 'password' key will enable
            Ray.

        do_cleaning : bool
            If true, clean self.fluxMatrixRaw and re-discretize to produce self.fluxMatrix. If false, just return
            the list of good states to keep.

        Returns
        -------
        None

        TODO
        ----
        Break this up into find_traps() and clean_traps(list_of_traps).
        """

        original_fluxmatrix = self.fluxMatrixRaw.copy()

        log.debug("Cleaning flux matrix")

        # Discretize trajectories via clusters
        # Get the indices of the target and basis clusters
        target_cluster_index = self.n_clusters + 1  # Target at -1
        basis_cluster_index = self.n_clusters  # basis at -2

        # This tracks which clusters are going to be cleaned from the flux matrix.
        # A 0 means it'll be cleaned, a 1 means it'll be kept.
        good_clusters = np.ones(self.n_clusters + 2)

        cluster_pcoord_centers = np.zeros((self.n_clusters + 2, self.pcoord_ndim))
        # cluster_pcoord_centers[indTargetCluster]=self.target_rmsd
        cluster_pcoord_centers[target_cluster_index] = self.target_bin_center
        # cluster_pcoord_centers[indBasisCluster]=self.get_reference_rmsd(self.basis_coords)
        cluster_pcoord_centers[basis_cluster_index] = self.basis_bin_center

        # Just initialize this to some positive nonzero value to kick off the while loop
        nTraps = 1000
        fluxMatrixTraps = self.fluxMatrixRaw.copy()
        while nTraps > 0:
            nTraps = 0
            for cluster_index in range(self.n_clusters):
                # Get the indices of the dtraj points in this cluster
                idx_traj_in_cluster = [
                    np.where(dtraj == cluster_index) for dtraj in self.dtrajs
                ]

                # Get the pcoord points that correspond to these dtraj points
                offset = 0
                pcoord_indices = []
                for i, idxs in enumerate(idx_traj_in_cluster):
                    pcoord_idxs = offset + idxs[0]
                    pcoord_indices.extend(pcoord_idxs)
                    offset += len(self.dtrajs[i])

                # Get the number of dtraj points in this cluster
                n_in_cluster = np.sum([x[0].shape for x in idx_traj_in_cluster])

                # If this cluster is unvisited in the trajectories, set good_cluster to False
                # if np.shape(idx_traj_in_cluster)[1] == 0:
                if n_in_cluster == 0:
                    good_clusters[cluster_index] = 0

                # Otherwise, get the average pcoord of the points in the cluster
                elif n_in_cluster > 0:
                    # cluster_pcoord_centers[iC]=np.mean(self.get_reference_rmsd(self.coordSet[idx_traj_in_cluster[0],:,:]))
                    # The coordinate of this cluster center is the average pcoord of all points in it

                    try:
                        cluster_pcoord_centers[cluster_index] = np.nanmean(
                            self.pcoordSet[pcoord_indices, 0], axis=0
                        )
                    except IndexError as e:
                        log.error(f"Offset is {offset}")
                        log.error(pcoord_indices)
                        raise e

                # Get the total flux along the row and col of this index
                net_flux = np.sum(fluxMatrixTraps[:, cluster_index]) + np.sum(
                    fluxMatrixTraps[cluster_index, :]
                )

                # If both the row and column are all zero, this is an unvisited state, so set good_clusters to 0/False
                if net_flux == 0.0:
                    good_clusters[cluster_index] = 0

                # If the sum along the row and column are nonzero
                if net_flux > 0:
                    # Get all the clusters that *aren't* the one we're looking at
                    all_other_cluster_indices = np.setdiff1d(
                        range(self.n_clusters), cluster_index
                    )

                    # Look at all the flux FROM other clusters
                    total_flux_in = np.sum(
                        fluxMatrixTraps[cluster_index, all_other_cluster_indices]
                    )

                    # And look at all the flux TO other clusters
                    total_flux_out = np.sum(
                        fluxMatrixTraps[all_other_cluster_indices, cluster_index]
                    )

                    # If either the flux from or the flux to other clusters are all zero,
                    #   then this is a source or sink respectively.
                    # So, clean it
                    if total_flux_in == 0.0 or total_flux_out == 0.0:
                        nTraps = nTraps + 1
                        good_clusters[cluster_index] = 0
                        fluxMatrixTraps[:, cluster_index] = 0.0
                        fluxMatrixTraps[cluster_index, :] = 0.0

            # Make sure we don't clean the target or basis clusters
            good_clusters[basis_cluster_index] = 1
            good_clusters[target_cluster_index] = 1

        # If a list of states to keep has been provided, forget everything we just did, and only clean those.
        # Still run the above, rather than putting the else for this if clause around it,
        #   so that we still get the cluster pcoords from it.
        if states_to_keep is not None:
            good_clusters[:] = 0
            good_clusters[states_to_keep] = 1

        if not do_cleaning:
            return good_clusters

        # Store this array. 1 if a cluster is good, 0 otherwise.
        clusters_good = np.copy(good_clusters)

        # Get all the visited and non-trap clusters, which we want to keep, and make fluxMatrix from those
        good_clusters = np.squeeze(np.where(good_clusters > 0))
        fluxMatrix = self.fluxMatrixRaw[good_clusters, :]
        fluxMatrix = fluxMatrix[:, good_clusters]

        # Get the RMSD centers for all of the clusters we want to keep
        cluster_pcoord_centers = cluster_pcoord_centers[good_clusters]
        # Get the indices that sort it, in pcoord 0
        pcoord_sort_indices = np.argsort(cluster_pcoord_centers[:, 0])
        # And update the model's RMSD cluster centers to just include the sorted clusters to keep
        self.targetRMSD_centers = cluster_pcoord_centers[pcoord_sort_indices]

        # Sort fluxmatrix using the sorted indices, columns and then rows
        fluxMatrix = fluxMatrix[pcoord_sort_indices, :]
        fluxMatrix = fluxMatrix[:, pcoord_sort_indices]

        # Renormalize the new flux matrix
        self.fluxMatrix = fluxMatrix / np.sum(
            fluxMatrix
        )  # average weight transitioning or staying put should be 1

        # Update the index of the basis and target states to account for their position in the new sorted clusters
        originalClusters = good_clusters[pcoord_sort_indices]
        # log.debug(f"Good clusters: {good_clusters}")
        self.indBasis = np.where(originalClusters == basis_cluster_index)[0]
        self.indTargets = np.where(originalClusters == target_cluster_index)[0]
        log.debug(f"indBasis:  {self.indBasis}, indTargets: {self.indTargets}")
        log.debug(
            f"Sanity check -- basis:  {self.targetRMSD_centers[self.indBasis]}, target: {self.targetRMSD_centers[self.indTargets]}"
        )

        # Save the new, sorted clusters
        self.originalClusters = originalClusters

        # Update binCenters with the new, sorted centers
        self.binCenters = cluster_pcoord_centers[pcoord_sort_indices]
        self.nBins = np.shape(self.binCenters)[0]

        # Remove the cluster structure dict entries corresponding to removed clusters
        removed_clusters = np.argwhere(clusters_good == 0).squeeze()

        # log.debug(f"Good clusters are {clusters_good}")
        log.debug(f"Removed clusters were {removed_clusters}")
        self.removed_clusters = removed_clusters

        # TODO: Since I re-discretize below, this is no longer necessary.
        #  For now, set it to 1:1 for backwards compatibility
        # After cleaning, the cluster indices may no longer be consecutive.
        #  So, when I do the cleaning, I need to build a mapping of old, nonconsecutive cluster indices from the
        #  non-cleaned matrix, to new, consecutive indices.
        # That's like collapsing the nonconsecutive list. In other words, imagine I started with 5 clusters [0,1,2,3,4]
        #   and clean cluster 2. Now, I'll still have clusters labeled as [0,1,3,4], but my steady-state distribution
        #   is only 4 elements. So indexing element 4 won't do anything.
        cluster_mapping = {x: x for x in range(self.n_clusters + 2)}
        n_removed = 0
        for key in cluster_mapping.keys():
            if key in self.removed_clusters:
                n_removed += 1
            cluster_mapping[key] = cluster_mapping[key] - n_removed

        # TODO: This is unused, which is why it's just set to a 1:1. This is legacy, but left in for compatibility
        #   with the restart plugin which expects it.
        # log.debug(f"New cluster mapping is  {cluster_mapping}")
        self.cluster_mapping = cluster_mapping
        self.cluster_mapping = {
            x: x for x in range(self.clusters.cluster_centers_.shape[0])
        }

        # Remove the bad clusters from the set of cluster centers. Thus, they won't be used by self.clusters.predict
        #   in the future.
        self.clusters.cluster_centers_ = np.delete(
            self.clusters.cluster_centers_, removed_clusters, 0
        )

        # TODO: Don't duplicate code from cluster_coordinates!!
        # TODO: You don't actually need to rediscretize every point -- just the removed ones.  Do this  later to make
        #   this more efficient.
        self.dtrajs = []

        first_iter, last_iter = 1, self.maxIter

        if rediscretize and not use_ray:
            extra_iters_used = 0
            for iteration in tqdm.tqdm(
                range(first_iter, last_iter), desc="Post-cleaning rediscretization"
            ):

                if extra_iters_used > 0:
                    extra_iters_used -= 1
                    log.debug(f"Already processed  iter  {iteration}")
                    continue

                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("fork")
                ) as executor:
                    dtrajs, extra_iters_used = executor.submit(
                        self.do_discretization,
                        [self.clusters, iteration, self.processCoordinates],
                    ).result()

                    self.dtrajs.append(dtrajs)

        # If we're using Ray...
        elif rediscretize and use_ray:

            # First, connect to the ray cluster
            self.check_connect_ray()

            # Submit all the discretization tasks to the cluster
            task_ids = []

            model_id = ray.put(self)
            cluster_model_id = ray.put(self.clusters)
            process_coordinates_id = ray.put(self.processCoordinates)

            # max_inflight = 50
            for iteration in tqdm.tqdm(
                range(first_iter, last_iter), desc="Submitting discretization tasks"
            ):
                # if len(task_ids) > max_inflight:
                #
                #     # The number that need to be ready before we can submit more
                #     num_ready = iteration - max_inflight
                #     ray.wait(task_ids, num_returns=num_ready)

                _id = self.do_ray_discretization.remote(
                    model_id, cluster_model_id, iteration, process_coordinates_id
                )
                task_ids.append(_id)

            # As they're completed, add them to dtrajs
            dtrajs = [None] * (last_iter - first_iter)

            # Do these in bigger batches, dtrajs aren't very big

            with tqdm.tqdm(
                total=len(task_ids), desc="Retrieving discretized trajectories"
            ) as pbar:
                while task_ids:
                    result_batch_size = 50
                    result_batch_size = min(result_batch_size, len(task_ids))

                    # Returns the first ObjectRef that is ready, with a 20s timeout
                    finished, task_ids = ray.wait(
                        task_ids, num_returns=result_batch_size, timeout=20
                    )
                    results = ray.get(finished)

                    for dtraj, _, iteration in results:
                        dtrajs[iteration - 1] = dtraj
                        pbar.update(1)
                        pbar.refresh()

            # Remove all empty elements from dtrajs and assign to self.dtrajs
            self.dtrajs = [dtraj for dtraj in dtrajs if dtraj is not None]

        self.removed_clusters = []

        # Update self.n_clusters to account for any removed clusters
        self.n_clusters -= n_removed

        # Rebuild the fluxmatrix with whatever params were originally provided
        self.get_fluxMatrix(*self._fluxMatrixParams, use_ray=use_ray)

        # # The new, organized fluxmatrix is the result of computing the new fluxmatrix, on the new set of bins.
        new_fluxMatrix = self.fluxMatrixRaw.copy()

        # Do the same sorting and organization we did above on the "raw" fluxmatrix, for consistency.
        new_fluxMatrix = new_fluxMatrix[pcoord_sort_indices, :]
        new_fluxMatrix = new_fluxMatrix[:, pcoord_sort_indices]

        # Renormalize the new flux matrix
        new_fluxMatrix = new_fluxMatrix / np.sum(
            new_fluxMatrix
        )  # average weight transitioning or staying put should be 1

        self.fluxMatrixRaw = original_fluxmatrix
        self.fluxMatrix = new_fluxMatrix

    def get_model_clusters(self):
        """
        Used by get_iter_aristoffian(). Untested and un-debugged, use at your own risk.

        Updates:
        - self.model_clusters
        """
        # define new clusters from organized flux matrix corresponding to model

        log.critical(
            "This function is untested, and may rely on other untested parts of this code. Use with extreme caution."
        )

        clustercenters = np.zeros((self.n_clusters + 2, self.ndim))
        clustercenters[0 : self.n_clusters, :] = self.clusters.clustercenters
        if self.dimReduceMethod == "none":
            coords = np.array([self.basis_coords, self.reference_coord])
            clustercenters[self.n_clusters :, :] = self.reduceCoordinates(
                coords
            )  # add in basis and target
            self.model_clusters = clustering.AssignCenters(
                self.reduceCoordinates(clustercenters[self.originalClusters, :]),
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        if self.dimReduceMethod == "pca" or self.dimReduceMethod == "vamp":
            coords = np.array(
                [
                    self.reduceCoordinates(self.basis_coords),
                    self.reduceCoordinates(self.reference_coord),
                ]
            )
            clustercenters[self.n_clusters :, :] = np.squeeze(
                coords
            )  # add in basis and target

            self.model_clusters = clustering.AssignCenters(
                clustercenters[self.originalClusters, :],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )

    def get_Tmatrix(self):
        """
        Compute the transition matrix from the flux matrix.
        Corrects the "target" states to be true sink states.

        More specifically:
            - row-normalizes the flux matrix,
            - sets any states with 0 flux ot (i.e. sinks) to have 1.0
            - sets target bins to uniformly recycle into basis bins

        Updates:
            - `self.Tmatrix`

        Returns
        -------
        None
        """

        log.debug("Computing transition matrix")

        # Get a copy of the flux matrix
        fluxmatrix = self.fluxMatrix.copy()
        # Get the dimension of the flux matrix
        fluxmatrix_shape = np.shape(fluxmatrix)
        # Add up the total flux on each row, i.e. from each state
        fluxes_out = np.sum(fluxmatrix, 1)

        # For each state
        for state_idx in range(fluxmatrix_shape[0]):
            # For positive definite flux, set the matrix elements based on normalized fluxes
            if fluxes_out[state_idx] > 0:
                fluxmatrix[state_idx, :] = (
                    fluxmatrix[state_idx, :] / fluxes_out[state_idx]
                )

            # If the flux is zero, then consider it all self-transition
            # FIXME: this if can be an elif
            if fluxes_out[state_idx] == 0.0:
                fluxmatrix[state_idx, state_idx] = 1.0

        # Make the transition matrix a steady-state matrix
        # Identify the bins corresponding to target states.
        sinkBins = self.indTargets  # np.where(avBinPnoColor==0.0)

        # Get the number of sink bins
        n_sink_bins = np.shape(sinkBins)
        n_sink_bins = n_sink_bins[0]

        # TODO: The goal here is to correct for the transition probabilities out of the sink state. Not
        #   sure I fully understand this implementation, but that's the goal.
        # Correction: I think this sets all the sink bins to recycle to the basis
        sinkRates = np.zeros((1, self.nBins))
        sinkRates[0, self.indBasis] = 1.0 / self.indBasis.size
        tmatrix = fluxmatrix.copy()
        tmatrix[sinkBins, :] = np.tile(sinkRates, (n_sink_bins, 1))

        self.Tmatrix = tmatrix

    def get_eqTmatrix(self):
        Mt = self.fluxMatrix.copy()
        n = np.shape(Mt)[0]
        indSpace = np.arange(n).astype(int)
        indSpace = np.setdiff1d(indSpace, np.append(self.indTargets, self.indBasis))
        Mt = Mt[indSpace, :]
        Mt = Mt[:, indSpace]
        nR = np.shape(Mt)
        sM = np.sum(Mt, 1)
        for iR in range(nR[0]):
            if sM[iR] > 0:
                Mt[iR, :] = Mt[iR, :] / sM[iR]
            if sM[iR] == 0.0:
                Mt[iR, iR] = 1.0
        self.Tmatrix = Mt

    def get_steady_state(self, flux_fractional_convergence=1e-4, max_iters=10):
        """"
        Get the steady-state distribution for the transition matrix.
        Uses scipy eigensolver to obtain an initial guess, then refines that using inverse iteration.

        Parameters
        ----------
        flux_fractional_convergence: (optional, default=1e-4) float
            Convergence of the

        max_iters: (optional, default=100) int

        Notes
        -----
        Transition matrices generated from WE data may have entries spanning many orders of magnitude, and may have
        extremely high condition numbers.
        Furthermore, the smallest entries may be those near the target state, which are also the most important
        for computing target fluxes, meaning values near machine precision can't just be truncated.
        All this means that floating-point error may substantially affect the results of eigensolvers, and may produce
        bad/negative/inaccurate values for small probability bins.

        In order to obtain better estimates, sparse matrices are used to reduce the number of floating point operations
        being performed.
        A stationary distribution is first estimated using scipy's eigensolver for sparse matrices.
        This is then used as an initial guess for the inverse iteration method, to further refine it.
        Convergence of the inverse iteration is determined using change in the flux estimate.
        """

        # Cast the matrix to a sparse matrix, to reduce floating point operations
        sparse_mat = sparse.csr_matrix(self.Tmatrix)

        algebraic_pss = self.get_steady_state_algebraic(
            max_iters=10, check_negative=False, set=False
        )

        # Get an initial flux estimate using the numpy algebraic solver
        # Call with _set=False so you don't actually update self.JtargetSS
        last_flux = self.get_steady_state_target_flux(pSS=algebraic_pss, _set=False)

        # ## Next, use that as an initial guess  for inverse iteration
        last_pSS = algebraic_pss

        log.debug(f"Initial flux: {last_flux}\n")
        flux_warned = False

        for N in range(max_iters):

            iterated = inverse_iteration(matrix=sparse_mat, guess=last_pSS)

            # Compute change in pSS
            pSS_change = np.sqrt(np.mean(np.power(iterated - last_pSS, 2)))
            log.debug(f"\t Change in SS: {pSS_change:.2e}")
            last_pSS = iterated

            # Compute change in target flux with the new pSS
            new_flux = self.get_steady_state_target_flux(pSS=last_pSS, _set=False)

            flux_change = new_flux - last_flux
            log.debug(
                f"\t Change in flux estimate: {flux_change:.2e} \t ({new_flux:.2e} raw)"
            )
            last_flux = new_flux

            # Set the convergence criterion after the iteration, in case the original result is really junk
            # Do it after so the first time this is calculated, it's using the result of the first iteration.
            # If the initial numpy estimate was really bad, it'll never be used.
            flux_convergence_criterion = last_flux * flux_fractional_convergence
            log.debug(f"\t Flux convergence criterion is {flux_convergence_criterion}")

            if N > 0:
                if last_flux == 0 and not flux_warned:
                    log.warning(
                        "Flux is 0, so steady-state solver will only converge after max iterations (using flux as "
                        "convergence criterion is not meaningful if it's 0!). "
                        "If you're looking for equilibrium, this is probably OK."
                        " Otherwise, take a look at why you have 0 flux."
                    )
                    flux_warned = True

            if abs(flux_change) < flux_convergence_criterion:
                log.info(
                    f"\nFlux converged to {last_flux:.4e} after {N + 1} iterations of inverse iteration."
                )
                break

            elif N == max_iters - 1 and not last_flux == 0:
                log.warning("Flux is nonzero and did not converge!")

        assert (last_pSS >= 0).all(), "Negative elements in pSS"
        assert last_flux >= 0, "Negative flux estimate from this pSS"

        log.info("Done with steady-state estimation.")
        self.pSS = last_pSS

    def get_steady_state_algebraic(self, max_iters=1000, check_negative=True, set=True):
        """
        Compute the steady-state distribution as the eigenvectors of the transition matrix.

        Updates:
            - `self.pSS`

        Parameters
        ----------
        max_iters: int, optional
            Defaults to 1000. Number of power-method iterations to run if the numpy eigensolver returns negative elements.

        check_negative: bool, optional
            Defaults to True. If True, then raise an Exception if there are negative elements in the normalized pSS.

        Returns
        -------
        None
        """

        # log.warning(
        #     "get_steady_state_algebraic() will be deprecated soon. Use get_steady_state() instead, which has"
        #     " a more robust eigensolver."
        # )

        log.debug("Computing steady-state from eigenvectors")

        eigenvalues, eigenvectors = np.linalg.eig(np.transpose(self.Tmatrix))

        pSS = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])

        pSS = pSS.squeeze()

        assert not np.isclose(np.sum(pSS), 0), "Steady-state distribution sums to 0!"
        pSS = pSS / np.sum(pSS)

        # The numpy eigensolver is iterative, and approximate. Given that data from WE often spans many orders of
        #   magnitude, we'll sometimes run into situations where our populations span more than machine precision.
        #   This causes hiccups in the eigensolver. However, we can't just zero these out (as attempted above),
        #   because these values are often important.
        # So, if there are any negative elements, try to correct the NP eigensolver result using the matrix method
        if sum(pSS < 0) > 0 and max_iters > 0:
            log.info(
                "Negative elements in pSS after normalization, attempting to correct with matrix power method."
            )
            pSS_last = pSS
            _tmatrix = self.Tmatrix.copy()

            for N in range(max_iters):

                pSS_new = _tmatrix.T @ pSS_last
                num_negative_elements = sum(pSS_new < 0)
                if num_negative_elements == 0:
                    log.info(f"Corrected to semidefinite pSS in {N} iterations")
                    break

                pSS_last = pSS_new
                _tmatrix = np.matmul(self.Tmatrix, _tmatrix)

            if N == max_iters - 1:
                log.warning(
                    "Power method did NOT obtain semidefinite pSS. Some negative values remain. This is weird, and you"
                    " should try to figure out why this is happening."
                )
            else:
                pSS = pSS_new

        elif max_iters == 0:
            log.debug(
                "Negative elements in pSS after normalization, "
                "but max_iters=0 so not attempting to correct with power method."
            )

        if not np.all(pSS >= 0):

            if check_negative:
                assert np.all(
                    pSS >= 0
                ), f"Some negative elements in steady-state distribution: {pSS}"
            else:
                log.warning(
                    "Some negative  elements in pSS... Ignoring, and setting model.pSS anyways."
                )

        if set:
            self.pSS = pSS
        else:
            return pSS

    def get_steady_state_matrixpowers(self, conv):
        """
        Compute the steady-state distribution using the matrix power method.

        Updates:
            - `self.pSS`

        Parameters
        ----------
        conv: numeric
            Convergence criterion for iteration.

        Returns
        -------
        None
        """
        max_iters = 10000
        Mt = self.Tmatrix.copy()
        dconv = 1.0e100
        N = 1
        pSS = np.mean(Mt, 0)
        pSSp = np.ones_like(pSS)
        while dconv > conv and N < max_iters:
            Mt = np.matmul(self.Tmatrix, Mt)
            N = N + 1
            if N % 10 == 0:
                pSS = np.mean(Mt, 0)
                pSS = pSS / np.sum(pSS)
                dconv = np.sum(np.abs(pSS - pSSp))
                pSSp = pSS.copy()
                sys.stdout.write("N=" + str(N) + " dconv: " + str(dconv) + "\n")
                self.pSS = pSS.copy()

    def get_steady_state_target_flux(self, pSS=None, _set=True):
        """
        Get the total flux into the target state(s).

        Updates:
            - `self.lagtime`
            - `self.JtargetSS`

        Parameters
        ----------
        pSS: (optional) array-like
            Steady-state distribution. If nothing provided, then use self.pSS

        _set: (optional) boolean
            If True, then update self.JtargetSS and self.lagtime. If False, then just return the value of JtargetSS.

        Returns
        -------
        None
        """

        # If this matrix isn't connected from source to target, then it's not possible to get an MFPT directly out of it
        sparse_mat = sparse.csr_matrix(self.Tmatrix)
        connected = is_connected(
            sparse_mat, self.indBasis, self.indTargets, directed=True
        )
        if not connected:
            log.critical(
                "There is no path in this matrix from the basis to the target, so no MFPT can be calculated."
            )
            return -1

        Mss = self.Tmatrix

        # If no pSS was provided, then pull from self
        if pSS is None:
            pSS = np.squeeze(np.array(self.pSS))

        lagtime = self.tau * (self.n_lag + 1)

        # Get a list of all the states that AREN'T targets, since we want to sum up
        nTargets = self.indTargets.size
        indNotTargets = np.setdiff1d(range(self.nBins), self.indTargets)
        # log.debug(f"Non-target states are those with index {indNotTargets}")

        Jt = 0.0
        # Add up the total flux into each of the targets
        for j in range(nTargets):
            log.debug(
                f"Processing flux into target state with index {self.indTargets[j]}"
            )

            jj = self.indTargets[j]

            Jt = Jt + np.sum(
                np.multiply(
                    pSS[indNotTargets],
                    np.squeeze(
                        np.array(Mss[indNotTargets, jj * np.ones_like(indNotTargets)])
                    ),
                )
            )

        if _set:
            self.lagtime = lagtime
            self.JtargetSS = Jt / self.lagtime
        else:
            return Jt / lagtime

    def evolve_probability(
        self, nEvolve, nStore
    ):  # iterate nEvolve times, storing every nStore iterations, initial condition at basis
        nIterations = np.ceil(nEvolve / nStore).astype(int) + 1
        self.nEvolve = nEvolve
        self.nStore = nStore
        Mss = self.Tmatrix
        nBins = self.nBins
        binCenters = self.binCenters
        probBasis = np.zeros((1, self.nBins))
        probBasis[0, self.indBasis] = 1.0
        pSS = probBasis.copy()
        pSSPrev = np.ones_like(pSS)
        iT = 1
        probTransient = np.zeros((nIterations, nBins))  # while dConv>.00001:
        probTransient[0, :] = probBasis
        for i in range(nEvolve):
            pSS = np.matmul(pSS, Mss)
            dConv = np.sum(np.abs(pSS - pSSPrev))
            pSSPrev = pSS.copy()
            if i % nStore == 0:
                sys.stdout.write("SS conv: " + str(dConv) + " iter: " + str(i))
                try:
                    plt.plot(
                        binCenters,
                        np.squeeze(pSS),
                        "-",
                        color=plt.cm.Greys(float(i) / float(nEvolve)),
                    )
                    plt.yscale("log")
                except Exception as e:
                    log.error(e)
                    try:
                        plt.plot(
                            binCenters,
                            pSS.A1,
                            "-",
                            color=plt.cm.Greys(float(i) / float(nEvolve)),
                        )
                    # ????? why these nested excepts? What's so fragile here? Maybe the shape of pSS?
                    except Exception as e2:
                        log.error(e2)
                        pass
                probTransient[iT, :] = np.squeeze(pSS)
                iT = iT + 1
        probTransient = probTransient[0:iT, :]
        self.probTransient = probTransient
        pSS = np.squeeze(np.array(pSS))
        self.pSS = pSS / np.sum(pSS)
        try:
            plt.pause(4)
            plt.close()
        except Exception as e:
            log.error(e)
            pass

    def evolve_probability2(
        self, nEvolve, nStore
    ):  # iterate nEvolve times, storing every nStore iterations, initial condition spread for everything at RMSD higher than basis
        nIterations = np.ceil(nEvolve / nStore).astype(int) + 1
        self.nEvolve = nEvolve
        self.nStore = nStore
        Mss = self.Tmatrix
        nBins = self.nBins
        binCenters = self.binCenters
        probBasis = np.zeros((1, self.nBins))
        probBasis[
            0, self.indBasis[0] :
        ] = 1.0  # assign initial probability to everything at RMSD higher than the basis, for case when nothing observed leaving exact basis
        probBasis = probBasis / np.sum(probBasis)
        pSS = probBasis.copy()
        pSSPrev = np.ones_like(pSS)
        iT = 1
        probTransient = np.zeros((nIterations, nBins))  # while dConv>.00001:
        probTransient[0, :] = probBasis
        for i in range(nEvolve):
            pSS = np.matmul(pSS, Mss)
            dConv = np.sum(np.abs(pSS - pSSPrev))
            pSSPrev = pSS.copy()
            if i % nStore == 0:
                sys.stdout.write("SS conv: " + str(dConv) + " iter: " + str(i))
                try:
                    plt.plot(
                        binCenters,
                        np.squeeze(pSS),
                        "-",
                        color=plt.cm.Greys(float(i) / float(nEvolve)),
                    )
                    plt.yscale("log")
                except Exception as e:
                    log.error(e)
                    try:
                        plt.plot(
                            binCenters,
                            pSS.A1,
                            "-",
                            color=plt.cm.Greys(float(i) / float(nEvolve)),
                        )
                    except Exception as e2:
                        log.error(e2)
                        pass
                # plt.ylim([1e-100,1])
                # plt.title(str(iT)+' of '+str(nIterations))
                # plt.pause(.1)
                probTransient[iT, :] = np.squeeze(pSS)
                iT = iT + 1
        probTransient = probTransient[0:iT, :]
        self.probTransient = probTransient
        pSS = np.squeeze(np.array(pSS))
        self.pSS = pSS / np.sum(pSS)
        try:
            plt.pause(4)
            plt.close()
        except Exception as e:
            log.error(e)
            pass

    def evolve_probability_from_initial(
        self, p0, nEvolve, nStore
    ):  # iterate nEvolve times, storing every nStore iterations, initial condition provided
        nIterations = np.ceil(nEvolve / nStore).astype(int) + 1
        self.nEvolve = nEvolve
        self.nStore = nStore
        Mss = self.Tmatrix
        nBins = self.nBins
        binCenters = self.binCenters
        probBasis = np.zeros((1, self.nBins))
        if np.shape(probBasis)[1] == np.shape(p0)[0]:
            probBasis[0, :] = p0
        else:
            probBasis = p0
        pSS = probBasis.copy()
        pSSPrev = np.ones_like(pSS)
        iT = 1
        probTransient = np.zeros((nIterations, nBins))  # while dConv>.00001:
        probTransient[0, :] = probBasis
        for i in range(nEvolve):
            pSS = np.matmul(pSS, Mss)
            dConv = np.sum(np.abs(pSS - pSSPrev))
            pSSPrev = pSS.copy()
            if i % nStore == 0:
                sys.stdout.write("SS conv: " + str(dConv) + " iter: " + str(i))
                try:
                    plt.plot(
                        binCenters,
                        np.squeeze(pSS),
                        "-",
                        color=plt.cm.Greys(float(i) / float(nEvolve)),
                    )
                    plt.yscale("log")
                except Exception as e:
                    log.error(e)
                    try:
                        plt.plot(
                            binCenters,
                            pSS.A1,
                            "-",
                            color=plt.cm.Greys(float(i) / float(nEvolve)),
                        )
                    except Exception as e2:
                        log.error(e2)
                        pass
                probTransient[iT, :] = np.squeeze(pSS)
                iT = iT + 1
        probTransient = probTransient[0:iT, :]
        self.probTransient = probTransient
        pSS = np.squeeze(np.array(pSS))
        self.pSS = pSS / np.sum(pSS)
        try:
            plt.pause(4)
            plt.close()
        except Exception as e:
            log.error(e)
            pass

    def get_flux(self):
        """
        Get the measured flux (i.e. from the flux matrix) into the target.
        """

        # J = np.zeros_like(self.binCenters)
        nBins = np.shape(self.targetRMSD_centers)[0]
        J = np.zeros(nBins)
        fluxMatrix = self.fluxMatrix.copy()
        for i in range(0, nBins - 1):
            indBack = range(i + 1)
            indForward = range(i + 1, nBins)
            JR = 0.0
            JF = 0.0
            for j in indBack:
                JR = JR + np.sum(fluxMatrix[indForward, j * np.ones_like(indForward)])
            for j in indForward:
                JF = JF + np.sum(fluxMatrix[indBack, j * np.ones_like(indBack)])
            J[i] = JR - JF
            self.J = J

    def get_flux_committor(self):
        """
        Get the flux binned according to committors

        Returns
        -------

        """

        # J = np.zeros_like(self.targetRMSD_centers)
        # nBins = np.shape(self.binCenters)[0]
        nBins = np.shape(self.targetRMSD_centers)[0]
        J = np.zeros(nBins)
        fluxMatrix = self.fluxMatrix.copy()
        indq = np.argsort(np.squeeze(1.0 - self.q))
        fluxMatrix = fluxMatrix[indq, :]
        fluxMatrix = fluxMatrix[:, indq]

        for i in tqdm.tqdm(
            range(0, nBins - 1), desc="Obtaining pseudocommittor-fluxes"
        ):
            indBack = range(i + 1)
            indForward = range(i + 1, nBins)
            JR = 0.0
            JF = 0.0
            for j in indBack:
                JR = JR + np.sum(fluxMatrix[indForward, j * np.ones_like(indForward)])
            for j in indForward:
                JF = JF + np.sum(fluxMatrix[indBack, j * np.ones_like(indBack)])
            J[i] = JR - JF
            self.Jq = J.squeeze() / self.tau
            # sys.stdout.write("%s " % i)

    def plot_flux_committor_pcoordcolor(
        self, nwin=1, ax=None, pcoord_to_use=0, **_plot_args,
    ):

        _models = [self]
        _model_labels = ["main_model"]

        plot_args = {
            "linewidth": 2,
            "s": 50,
            "marker": ">",
            "cmap": plt.cm.rainbow.reversed(),
            "alpha": 0.7,
        }

        plot_args.update(_plot_args)

        if ax is None:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

        for i, (_model, _label) in enumerate(zip(_models[::-1], _model_labels[::-1])):

            if _model is None:
                continue

            if not hasattr(_model, "q"):
                log.warning(
                    f"Committors have not yet been generated for {_label}, generating now."
                )
                _model.get_committor()

            if not hasattr(_model, "Jq"):
                log.warning(
                    f"Committor-fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux_committor()

            n_bins = _model.targetRMSD_centers.shape[0]
            Jq_avg = _model.Jq.copy()
            Jq_std = np.zeros_like(Jq_avg)

            q_avg = np.zeros_like(Jq_avg)

            indq = np.argsort(np.squeeze(1.0 - _model.q))
            for _i in range(n_bins - 1, nwin - 1, -1):
                iav = _i - nwin
                ind = range(_i - nwin, _i)
                Jq_avg[iav] = np.mean(_model.Jq[ind])
                Jq_std[iav] = np.std(_model.Jq[ind])
                q_avg[iav] = np.mean(_model.q[indq[ind]])

            indPlus = np.where(Jq_avg > 0.0)

            lines = ax.scatter(
                q_avg[indPlus],
                np.squeeze(Jq_avg[indPlus]),
                c=_model.targetRMSD_centers[indPlus, pcoord_to_use],
                label=f"{_label} flux toward target",
                **plot_args,
            )

        print("Plotting committor")
        ax.figure.colorbar(lines, label=f"Progress Coordinate {pcoord_to_use}")

        ax.set_xlim([-0.1, 1.1])

        ax.set_title("Full-data model")
        ax.set_yscale("log")
        ax.set_xlabel("Pseudocommittor")
        ax.set_ylabel("Flux (weight/second)")
        self.print_pseudocommittor_warning()

        return ax, lines

    def plot_flux_committor(
        self,
        nwin=1,
        ax=None,
        save=False,
        suppress_validation=False,
        _from_colors=None,
        _to_colors=None,
        **_plot_args,
    ):

        _models = [self]
        _models.extend(self.validation_models)

        _model_labels = ["main_model"]
        _model_labels.extend(
            [f"validation_model_{n}" for n in range(len(self.validation_models))]
        )

        cross_validation_done = (
            self.validation_models is not None and len(self.validation_models) > 1
        )
        if not cross_validation_done:
            log.critical(
                "No cross-validation models have been generated! Do this before making plots."
            )
            if not suppress_validation:
                raise Exception("Perform cross-validation before plotting results.")

        own_ax = False
        if ax is None:
            own_ax = True
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

        plot_args = {"linewidth": 2, "markersize": 10}
        plot_args.update(_plot_args)

        n_models = len(_models)
        from_colors = [
            plt.get_cmap("cool")(0.25 + (0.75 * i / n_models)) for i in range(n_models)
        ]
        to_colors = [
            plt.get_cmap("hot")(0.25 + (0.5 * i / n_models)) for i in range(n_models)
        ]

        for i, (_model, _label) in enumerate(zip(_models, _model_labels)):

            if _model is None:
                continue

            if not hasattr(_model, "q"):
                log.warning(
                    f"Committors have not yet been generated for {_label}, generating now."
                )
                _model.get_committor()

            if not hasattr(_model, "Jq"):
                log.warning(
                    f"Committor-fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux_committor()

            n_bins = _model.targetRMSD_centers.shape[0]
            Jq_avg = _model.Jq.copy()
            Jq_std = np.zeros_like(Jq_avg)

            q_avg = np.zeros_like(Jq_avg)

            indq = np.argsort(np.squeeze(1.0 - _model.q))
            for _i in range(n_bins - 1, nwin - 1, -1):
                iav = _i - nwin
                ind = range(_i - nwin, _i)
                Jq_avg[iav] = np.mean(_model.Jq[ind])
                Jq_std[iav] = np.std(_model.Jq[ind])
                q_avg[iav] = np.mean(_model.q[indq[ind]])

            indPlus = np.where(Jq_avg > 0.0)
            indMinus = np.where(Jq_avg < 0.0)

            if _from_colors is not None:
                plot_args["color"] = _from_colors[i]
            else:
                plot_args["color"] = from_colors[i]

            ax.plot(
                q_avg[indMinus],
                -np.squeeze(Jq_avg[indMinus]),
                "<",
                # color=to_target_colors[i%len(to_target_colors)],
                label=f"{_label} flux toward source/basis",
                **plot_args,
            )

            if _to_colors is not None:
                plot_args["color"] = _to_colors[i]
            else:
                plot_args["color"] = to_colors[i]

            ax.plot(
                q_avg[indPlus],
                np.squeeze(Jq_avg[indPlus]),
                ">",
                # color=from_target_colors[i%len(from_target_colors)],
                label=f"{_label} flux toward target",
                **plot_args,
            )

        ax.set_yscale("log")
        ax.set_xscale("linear")
        ax.set_xlim([-0.1, 1.1])
        ax.set_xlabel("Pseudocommittor")
        ax.set_ylabel("Flux (weight/second)")
        self.print_pseudocommittor_warning()

        if own_ax:
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            fig.tight_layout()

        if save:
            plot_filename = f"{self.modelName}_flux_committor.pdf"
            log.info(f"Saving flux-committor plot to {plot_filename}")
            plt.savefig(plot_filename)

        return ax

    def plot_flux(
        self,
        custom_name=None,
        ax=None,
        save=False,
        suppress_validation=False,
        _from_colors=None,
        _to_colors=None,
        pcoord_to_use=0,
        **_plot_args,
    ):
        """
        Make, and save, a plot of the fluxes along the RMSD.  get_flux() must be run before this.

        Parameters
        ----------
        custom_name : str (optional)
            The name for the saved plot. Defaults to flux_s<first iter>_e<last iter>.png

        Returns
        -------

        """

        _models = [self]
        _models.extend(self.validation_models)

        _model_labels = ["main_model"]
        _model_labels.extend(
            [f"validation_model_{n}" for n in range(len(self.validation_models))]
        )

        cross_validation_done = (
            self.validation_models is not None and len(self.validation_models) > 1
        )
        if not cross_validation_done:
            log.critical(
                "No cross-validation models have been generated! Do this before making plots."
            )
            if not suppress_validation:
                raise Exception("Perform cross-validation before plotting results.")

        own_ax = False
        if ax is None:
            own_ax = True
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

        plot_args = {"linewidth": 2, "markersize": 10}
        plot_args.update(_plot_args)

        n_models = len(_models)
        from_colors = [
            plt.get_cmap("cool")(0.25 + (0.75 * i / n_models)) for i in range(n_models)
        ]
        to_colors = [
            plt.get_cmap("hot")(0.25 + (0.5 * i / n_models)) for i in range(n_models)
        ]

        # Draw the basis/target boundaries in this pcoord
        [
            ax.axvline(
                bound, color="r", linestyle="--", label=["", "Target boundary"][i == 0]
            )
            for i, bound in enumerate(self.target_pcoord_bounds[pcoord_to_use, :])
        ]
        [
            ax.axvline(
                bound,
                color="b",
                linestyle="--",
                label=["", "Basis/Source boundary"][i == 0],
            )
            for i, bound in enumerate(self.basis_pcoord_bounds[pcoord_to_use, :])
        ]

        for i, (_model, _label) in enumerate(zip(_models, _model_labels)):

            if _model is None:
                continue

            if not hasattr(_model, "J"):
                log.warning(
                    f"Fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux()

            J = _model.J / _model.tau

            binCenters = _model.targetRMSD_centers[:, pcoord_to_use]
            indPlus = np.where(J > 0.0)
            indMinus = np.where(J < 0.0)

            if _to_colors is not None:
                plot_args["color"] = _to_colors[i]
            else:
                plot_args["color"] = to_colors[i]

            ax.plot(
                binCenters[indPlus],
                np.squeeze(J[indPlus]),
                ">",
                label=f"{_label} flux toward target",
                **plot_args,
            )

            if _from_colors is not None:
                plot_args["color"] = _from_colors[i]
            else:
                plot_args["color"] = from_colors[i]

            ax.plot(
                binCenters[indMinus],
                -np.squeeze(J[indMinus]),
                "<",
                label=f"{_label} flux toward source/basis",
                **plot_args,
            )

        ax.set_yscale("log")
        ax.set_xlabel(f"Pcoord {pcoord_to_use}")
        ax.set_ylabel("Flux (weight/second)")

        if own_ax:
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            fig.tight_layout()

        if save:
            if custom_name is not None:
                plot_filename = custom_name
            else:
                plot_filename = f"{self.modelName}_flux.pdf"
            log.info(f"Saving flux plot to {plot_filename}")
            plt.savefig(plot_filename)

        return ax

    def evolve_target_flux(self):
        Mss = self.Tmatrix
        probTransient = self.probTransient
        nT = np.shape(probTransient)[0]
        Jtarget = np.zeros(nT)
        self.lagtime = self.tau * (self.n_lag + 1)
        nTargets = self.indTargets.size
        indNotTargets = np.setdiff1d(range(self.nBins), self.indTargets)
        JtargetTimes = np.zeros(nT)
        for iT in range(nT):
            Jt = 0.0
            for j in range(nTargets):
                jj = self.indTargets[j]
                Jt = Jt + np.sum(
                    np.multiply(
                        probTransient[iT, indNotTargets],
                        Mss[indNotTargets, jj * np.ones_like(indNotTargets)],
                    )
                )
            Jtarget[iT] = Jt
            JtargetTimes[iT] = iT * self.nStore * self.lagtime
        self.Jtarget = Jtarget / self.lagtime
        self.JtargetTimes = JtargetTimes

    def get_hflux(self, conv):
        convh = conv
        convf = conv
        max_iters = 50000
        nTargets = self.indTargets.size
        indNotTargets = np.setdiff1d(range(self.nBins), self.indTargets)
        Mt = self.Tmatrix.copy()
        dconvh = 1.0e100
        dconvf = 1.0e100
        fTotal = np.zeros((self.nBins, 1))
        fSSp = 0.0
        hp = np.zeros_like(fTotal)
        N = 1
        while dconvh > convh or dconvf > convf and N < max_iters:
            f = np.zeros((self.nBins, 1))
            for i in range(self.nBins):
                Jt = 0.0
                Pt = Mt[i, :]
                for j in range(nTargets):
                    jj = self.indTargets[j]
                    Jt = Jt + np.sum(
                        np.multiply(
                            Pt[0, indNotTargets],
                            Mt[indNotTargets, jj * np.ones_like(indNotTargets)],
                        )
                    )
                f[i, 0] = Jt / self.tau
            fTotal = fTotal + f
            fSS = np.mean(f[indNotTargets, 0])
            ht = fTotal - N * fSS
            dconvh = np.max(np.abs(hp - ht)) / np.max(ht)
            dconvf = np.abs(fSS - fSSp) / fSS
            sys.stdout.write(
                "N="
                + str(N)
                + " dh: "
                + str(dconvh)
                + " df: "
                + str(dconvf)
                + " Jss:"
                + str(fSS)
                + "\n"
            )
            hp = ht.copy()
            fSSp = fSS
            self.h = ht.copy()
            Mt = np.matmul(Mt, self.Tmatrix)
            N = N + 1

    def get_model_aristoffian(self):
        kh = np.matmul(self.Tmatrix, self.h)
        kh_sq = np.power(kh, 2)
        hsq = np.power(self.h, 2)
        k_hsq = np.matmul(self.Tmatrix, hsq)
        varh = k_hsq - kh_sq
        # val=np.sqrt(varh)
        self.varh = varh
        self.kh = kh

    def get_model_steady_state_aristoffian(self):
        nB = int(self.nB)
        if self.binMethod == "adaptive":
            self.kh_clusters = coor.cluster_kmeans(self.kh, k=nB, metric="euclidean")
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
            khbins_centers_unique, ind_unique = np.unique(
                khbins_centers, return_index=True
            )
            if khbins_centers_unique.size != nB:
                khbins = np.squeeze(
                    np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
                )  # equal spacing if not enough for k-means
                khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
        elif self.binMethod == "uniform":
            khbins = np.squeeze(
                np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
            )  # equal spacing if not enough for k-means
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "log_uniform":
            transformedBins = np.geomspace(
                np.abs(np.min(self.kh)) / np.max(self.kh),
                1.0 + 2.0 * np.abs(np.min(self.kh)) / np.max(self.kh),
                self.nB + 1,
            )
            khbins_binEdges_log = transformedBins * np.max(self.kh) - 2.0 * np.abs(
                np.min(self.kh)
            )
            khbins = khbins_binEdges_log  # equal log-spacing
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "optimized":
            try:
                khbins_centers = np.loadtxt("khbins_binCenters.dat")
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            except Exception as e:
                log.error(e)
                sys.stdout.write(
                    "khbins (khbins_binCenters.dat) not found: initializing\n"
                )
                self.get_initial_khbins_equalAlloc()
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            if not hasattr(self, "kh_clusters"):
                sys.stdout.write("giving up: log uniform kh bins")
                self.get_initial_khbins()
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
        dtraj_kh_clusters = self.kh_clusters.assign(self.kh)
        alloc = np.zeros(
            nB
        )  # get bin objective function, value and allocation over set of bins
        value = np.zeros(nB)
        bin_kh_var = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            if indBin[0].size == 0:
                alloc[i] = 0.0
                bin_kh_var[i] = 0.0
            else:
                # n = indBin[0].size
                bin_kh_var[i] = np.var(self.kh[indBin])
                wt = np.sum(self.pSS[indBin])
                vw = np.sum(np.multiply(self.pSS[indBin] / wt, self.varh[indBin]))
                alloc[i] = wt * (vw) ** 0.5
                value[i] = vw ** 0.5
        if self.allocationMethod == "uniform":
            alloc = np.ones_like(alloc)
        alloc = alloc / np.sum(alloc)
        self.alloc = alloc
        # base_walkers=self.min_walkers*np.ones_like(alloc)
        # nBase=np.sum(base_walkers)
        # nAdapt=self.nW-nBase
        # if nAdapt<0:
        #    nAdapt=0
        # walkers=np.round(alloc*nAdapt)
        # walkers=walkers+base_walkers
        # indZero=np.where(walkers==0.0)
        # walkers[indZero]=1.0
        # walkers=walkers.astype(int)
        # binEdges=np.zeros(self.nB+1)
        # binEdges[0]=-np.inf
        # binEdges[-1]=np.inf
        # ind=np.argsort(self.kh_clusters.clustercenters[:,0]) #note sorting makes kh_clusters indexes differen
        # self.khbins_binCenters=self.kh_clusters.clustercenters[ind,0]
        # binEdges[1:-1]=0.5*(self.khbins_binCenters[1:]+self.khbins_binCenters[0:-1])
        # self.khbins_binEdges=binEdges
        # self.walkers_per_bin=walkers[ind]
        # self.bin_kh_var=bin_kh_var[ind]
        gamma = self.alloc.copy()  # asymptotic particle distribution in bins
        # asymptotic particle distribution after mutation
        rho = np.zeros_like(gamma)
        rhov = np.zeros((self.nB, self.nB))
        for v in range(self.nB):
            indBinv = np.where(dtraj_kh_clusters == v)
            wv = np.sum(self.pSS[indBinv])
            sys.stdout.write("sum v: " + str(v) + "\n")
            for u in range(self.nB):
                indBinu = np.where(dtraj_kh_clusters == u)
                for p in indBinv[0]:
                    for q in indBinu[0]:
                        rhov[u, v] = (
                            rhov[u, v]
                            + self.alloc[v] * (self.pSS[p] / wv) * self.Tmatrix[p, q]
                        )
        rho = np.sum(rhov, 1)
        pOccupied = 1.0 - np.power(1.0 - rho, self.nW)
        nOccupied = nB - np.sum(np.power(1.0 - rho, self.nW))
        nAdditional = (self.nW - nOccupied) * self.alloc
        nT = nAdditional + pOccupied
        # nT=np.zeros(nB)
        # for i in range(nB):
        #    nT[i]=np.max(np.array([1,nAdditional[i]+pOccupied[i]]))
        bin_mutV = np.zeros(nB)
        bin_selV = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            wi = np.sum(self.pSS[indBin])
            bin_mutV[i] = ((wi ** 2) / (nT[i])) * np.sum(
                np.multiply(self.pSS[indBin] / wi, self.varh[indBin])
            )
            bin_selV[i] = ((wi ** 2) / (nT[i])) * np.sum(
                np.multiply(self.pSS[indBin] / wi, np.power(self.kh[indBin], 2))
                - np.power(np.multiply(self.pSS[indBin] / wi, self.kh[indBin]), 2)
            )
        self.binObjective = np.sum(bin_mutV + bin_selV)
        binEdges = np.zeros(self.nB + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        ind = np.argsort(
            self.kh_clusters.clustercenters[:, 0]
        )  # note sorting makes kh_clusters indexes different
        self.khbins_binCenters = self.kh_clusters.clustercenters[ind, 0]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        # self.walkers_per_bin=walkers[ind]
        self.bin_kh_var = bin_kh_var[ind]
        base_walkers = self.min_walkers * np.ones_like(alloc)
        nBase = nOccupied  # estimated from occupied bins a la Aristoff notes, was np.sum(base_walkers)
        nAdapt = self.nW - nBase
        if nAdapt < 0:
            nAdapt = 0
        walkers = np.round(alloc * nAdapt)
        walkers = walkers + base_walkers
        indZero = np.where(walkers == 0.0)
        walkers[indZero] = 1.0
        walkers = walkers.astype(int)
        self.walkers_per_bin = walkers[ind]
        self.bin_mutV = bin_mutV[ind]
        self.bin_selV = bin_selV[ind]
        self.nOccupancySS = nT[ind]
        self.nOccupied = nOccupied
        self.nAdapt = nAdapt
        self.rhomutation = rho[ind]
        self.value = value

    def get_initial_khbins(self):  # log-uniform kh bins
        transformedBins = np.geomspace(
            np.abs(np.min(self.kh)) / np.max(self.kh),
            1.0 + 2.0 * np.abs(np.min(self.kh)) / np.max(self.kh),
            self.nB + 1,
        )
        khbins_binEdges_log = transformedBins * np.max(self.kh) - 2.0 * np.abs(
            np.min(self.kh)
        )
        khbins = khbins_binEdges_log  # equal log-spacing
        khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
        self.kh_clusters = clustering.AssignCenters(
            khbins_centers[:, np.newaxis],
            metric="euclidean",
            stride=1,
            n_jobs=None,
            skip=0,
        )
        binEdges = np.zeros(self.nB + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        ind = np.argsort(
            self.kh_clusters.clustercenters[:, 0]
        )  # note sorting makes kh_clusters indexes different
        self.khbins_binCenters = self.kh_clusters.clustercenters[ind, 0]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        np.savetxt("khbins_binCenters.dat", self.khbins_binCenters)

    def get_initial_khbins_equalAlloc(self):  # kh bins approximately of equal value
        if not hasattr(self, "kh"):
            self.get_model_aristoffian()
        binMethod_use = self.binMethod
        allocationMethod_use = self.allocationMethod
        nB_use = self.nB
        self.binMethod = "uniform"
        self.allocationMethod = "adaptive"
        points = np.linspace(0, 1, self.nB)
        # resN=np.round(abs(np.max(self.kh)/np.min(np.abs(self.kh)))).astype(int)
        resN = 10000
        self.nB = resN
        self.get_model_steady_state_aristoffian()
        dist = self.alloc.copy()
        dist = dist / np.sum(dist)
        dist = np.cumsum(dist)
        dist_unique, ind_unique = np.unique(dist, return_index=True)
        kh_unique = self.khbins_binCenters[ind_unique]
        xB = np.zeros_like(points)
        for i in range(xB.size):
            indm = np.argmin(np.abs(dist_unique - points[i]))
            xB[i] = kh_unique[indm]
            dist_unique[indm] = np.inf
        khbins_centers = xB.copy()
        self.nB = nB_use
        self.binMethod = binMethod_use
        self.allocationMethod = allocationMethod_use
        self.kh_clusters = clustering.AssignCenters(
            khbins_centers[:, np.newaxis],
            metric="euclidean",
            stride=1,
            n_jobs=None,
            skip=0,
        )
        binEdges = np.zeros(self.nB + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        ind = np.argsort(
            self.kh_clusters.clustercenters[:, 0]
        )  # note sorting makes kh_clusters indexes different
        self.khbins_binCenters = self.kh_clusters.clustercenters[ind, 0]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        np.savetxt("khbins_binCenters.dat", self.khbins_binCenters)

    def get_bin_kh_var(self, x):
        nB = self.nB
        self.kh_clusters = clustering.AssignCenters(
            x[:, np.newaxis], metric="euclidean", stride=1, n_jobs=None, skip=0
        )
        dtraj_kh_clusters = self.kh_clusters.assign(self.kh)
        # alloc=np.zeros(nB) #get bin objective function, value and allocation over set of bins
        bin_kh_var = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            if indBin[0].size == 0:
                # alloc[i]=0.0
                bin_kh_var[i] = 0.0
            else:
                # n = indBin[0].size
                bin_kh_var[i] = np.var(self.kh[indBin])
                # wt=np.sum(self.pSS[indBin])
                # vw=np.sum(np.multiply(self.pSS[indBin]/wt,self.varh[indBin]))
                # alloc[i]=wt*(vw)**.5
        self.bin_kh_var = bin_kh_var
        self.total_bin_kh_var = np.sum(bin_kh_var)
        return self.total_bin_kh_var

    def get_bin_total_var(self, x):
        # nB = self.nB
        self.kh_clusters = clustering.AssignCenters(
            x[:, np.newaxis], metric="euclidean", stride=1, n_jobs=None, skip=0
        )
        self.binMethod = "optimized"
        self.get_model_steady_state_aristoffian()
        return self.binObjective

    def get_iter_aristoffian(self, iter):

        log.critical(
            "This function is untested, and may rely on other untested parts of this code. Use with extreme caution."
        )

        self.load_iter_data(iter)
        if not hasattr(self, "model_clusters"):
            self.get_model_clusters()
        #        if self.pcoord_is_kh:
        # wt=np.sum(self.pSS[indBin])
        # vw=np.sum(np.multiply(self.pSS[indBin]/wt,self.varh[indBin]))
        # alloc[i]=wt*(vw)**.5
        #            self.khList=np.array(self.pcoord1List[:,1]) #kh is pcoord 2 from optimized WE sims
        #            self.khList=self.khList[:,np.newaxis]
        #        else:
        self.load_iter_coordinates()
        dtraj_iter = self.model_clusters.assign(
            self.reduceCoordinates(self.cur_iter_coords)
        )
        kh_iter = self.kh[dtraj_iter]
        self.khList = np.array(kh_iter[:, 0])  # get k-means bins defined over walkers
        nB = self.nB
        khList_unique = np.unique(self.khList)
        if khList_unique.size > 2.0 * nB and self.binMethod == "adaptive":
            self.kh_clusters = coor.cluster_kmeans(
                khList_unique, k=nB, metric="euclidean"
            )
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
            khbins_centers_unique, ind_unique = np.unique(
                khbins_centers, return_index=True
            )
            if khbins_centers_unique.size != nB:
                khbins = np.squeeze(
                    np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
                )  # equal spacing if not enough for k-means
                khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
        elif self.binMethod == "uniform":
            khbins = np.squeeze(
                np.linspace(np.min(self.kh), np.max(self.kh), nB + 1)
            )  # equal spacing if not enough for k-means
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "log_uniform":
            transformedBins = np.geomspace(
                np.abs(np.min(self.kh)) / np.max(self.kh),
                1.0 + 2.0 * np.abs(np.min(self.kh)) / np.max(self.kh),
                self.nB,
            )
            khbins_binEdges_log = transformedBins * np.max(self.kh) - 2.0 * np.abs(
                np.min(self.kh)
            )
            khbins = khbins_binEdges_log  # equal log-spacing
            khbins_centers = 0.5 * (khbins[1:] + khbins[0:-1])
            self.kh_clusters = clustering.AssignCenters(
                khbins_centers[:, np.newaxis],
                metric="euclidean",
                stride=1,
                n_jobs=None,
                skip=0,
            )
        elif self.binMethod == "optimized":
            try:
                khbins_centers = np.loadtxt("khbins_binCenters.dat")
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            except Exception as e:
                log.error(e)
                log.debug("khbins (khbins_binCenters.dat) not found: initializing\n")
                self.get_initial_khbins_equalAlloc()
                self.kh_clusters = clustering.AssignCenters(
                    khbins_centers[:, np.newaxis],
                    metric="euclidean",
                    stride=1,
                    n_jobs=None,
                    skip=0,
                )
            if not hasattr(self, "kh_clusters"):
                sys.stdout.write("giving up: log uniform kh bins")
                self.get_initial_khbins()
            khbins_centers = self.kh_clusters.clustercenters[:, 0]
        dtraj_kh_clusters = self.kh_clusters.assign(self.khList)
        varh_iter = self.varh[dtraj_iter]
        alloc = np.zeros(
            nB
        )  # get bin objective function, value and allocation over set of bins
        bin_kh_var = np.zeros(nB)
        for i in range(nB):
            indBin = np.where(dtraj_kh_clusters == i)
            if indBin[0].size == 0:
                alloc[i] = 0.0
                bin_kh_var[i] = 0.0
            else:
                # n = indBin[0].size
                bin_kh_var[i] = np.var(self.khList[indBin])
                wt = np.sum(self.weightList[indBin])
                vw = np.sum(np.multiply(self.weightList[indBin], varh_iter[indBin]))
                alloc[i] = (wt * vw) ** 0.5
        if self.allocationMethod == "uniform":
            alloc = np.ones_like(alloc)
        alloc = alloc / np.sum(alloc)
        self.alloc = alloc
        base_walkers = self.min_walkers * np.ones_like(alloc)
        nBase = np.sum(base_walkers)
        if hasattr(self, "nAdapt"):
            nAdapt = self.nAdapt
        else:
            nAdapt = self.nW - nBase
        if nAdapt < 0:
            nAdapt = 0
        walkers = np.round(alloc * nAdapt)
        walkers = walkers + base_walkers
        indZero = np.where(walkers == 0.0)
        walkers[indZero] = 1.0
        walkers = walkers.astype(int)
        khbins_centers = self.kh_clusters.clustercenters[:, 0]
        khbins_centers_unique, ind_unique = np.unique(khbins_centers, return_index=True)
        walkers = walkers[ind_unique]
        bin_kh_var = bin_kh_var[ind_unique]
        binEdges = np.zeros(khbins_centers_unique.size + 1)
        binEdges[0] = -np.inf
        binEdges[-1] = np.inf
        # ind=np.argsort(khbins_centers_unique) #note sorting makes kh_clusters indexes different
        self.khbins_binCenters = khbins_centers_unique  # [ind]
        binEdges[1:-1] = 0.5 * (
            self.khbins_binCenters[1:] + self.khbins_binCenters[0:-1]
        )
        self.khbins_binEdges = binEdges
        self.walkers_per_bin = walkers  # [ind]
        self.bin_kh_var = bin_kh_var  # [ind]
        self.binObjective = np.sum(bin_kh_var)

    def write_iter_kh_pcoord(self):  # grab coordinates from WE traj_segs folder
        nS = self.nSeg
        if not hasattr(self, "model_clusters"):
            self.get_model_clusters()
        self.load_iter_coordinates()  # post coordinates
        dtraj_iter = self.model_clusters.assign(
            self.reduceCoordinates(self.cur_iter_coords)
        )
        kh_iter = self.kh[dtraj_iter]
        khList1 = np.array(kh_iter[:, 0])  # post pcoord
        self.load_iter_coordinates0()  # pre coordinates
        dtraj_iter = self.model_clusters.assign(
            self.reduceCoordinates(self.cur_iter_coords)
        )
        kh_iter = self.kh[dtraj_iter]
        khList0 = np.array(kh_iter[:, 0])  # pre pcoord
        westFile = self.fileList[self.westList[0]]
        dataIn = h5py.File(westFile, "a")
        pcoords = np.zeros(
            (0, 2, 2)
        )  # this is explicitly set up for p1 (target,basis def) and p2 (kh-aristoffian)
        for iS in range(self.nSeg):
            westFile = self.fileList[self.westList[iS]]
            pcoord = np.zeros((1, 2, 2))
            pcoord[0, 0, 0] = self.pcoord0List[iS, 0]
            pcoord[0, 1, 0] = self.pcoord1List[iS, 0]
            pcoord[0, 0, 1] = khList0[iS, 0]
            pcoord[0, 1, 1] = khList1[iS, 0]
            pcoords = np.append(pcoords, pcoord, axis=0)
            try:
                if iS > 0:
                    if self.westList[iS] != self.westList[iS - 1] and iS < nS - 1:
                        dsetName = "/iterations/iter_%08d/pcoord" % int(self.n_iter)
                        del dataIn[dsetName]
                        dset = dataIn.create_dataset(
                            dsetName, np.shape(pcoords[:-1, :, :])
                        )
                        dset[:] = pcoords[:-1, :, :]
                        dataIn.close()
                        pcoords = np.zeros(
                            (0, 2, 2)
                        )  # this is explicitly set up for p1 (target,basis def) and p2 (kh-aristoffian)
                        pcoords = np.append(pcoords, pcoord, axis=0)
                        dataIn = h5py.File(westFile, "a")
                    elif iS == nS - 1:
                        dsetName = "/iterations/iter_%08d/pcoord" % int(self.n_iter)
                        del dataIn[dsetName]
                        dset = dataIn.create_dataset(dsetName, np.shape(pcoords))
                        dset[:] = pcoords
                        sys.stdout.write(
                            "pcoords for iteration "
                            + str(self.n_iter)
                            + " overwritten\n"
                        )
                        dataIn.close()
            except Exception as e:
                log.error(e)
                log.error(
                    "error overwriting pcoord from "
                    + westFile
                    + " , iter "
                    + str(self.n_iter)
                    + " segment "
                    + str(self.segindList[iS])
                    + "\n"
                )

    def get_committor(self, conv=1e-5):
        """
        Iteratively obtain an estimate of the committor.

        1. Take the flux matrix, and normalize it into a transition matrix.

        2. Apply two-sided absorbing boundary conditions by setting self-transition probabilities for the basis and
            target states to 1.0, and all transitions out to 0.0.

        3. Starting with an initial committor "guess" of all 1s, iteratively multiply the guess by the transition matrix
            until convergence is below conv.

        Updates:
            - self.q

        Parameters
        ----------
        conv: numerical
            Convergence criteria for committor calculation. Calculation stops when the total difference between q_p and q
            is less than this.

        Returns
        -------

        """

        self.print_pseudocommittor_warning()

        _fluxMatrix = self.fluxMatrix.copy()

        # Number of bins/states in the fluxmatrix

        num_bins = np.shape(_fluxMatrix)
        flux_out = np.sum(_fluxMatrix, 1)

        for bin_idx in range(num_bins[0]):

            # If the flux out is positive semidefinite, then normalize that row to get transition probabilities
            if flux_out[bin_idx] > 0:
                _fluxMatrix[bin_idx, :] = _fluxMatrix[bin_idx, :] / flux_out[bin_idx]

            # If the flux out is zero, then just set the self-transition to 1.0
            # (This probably keeps a calculation later clean?)
            if flux_out[bin_idx] == 0.0:
                _fluxMatrix[bin_idx, bin_idx] = 1.0

        #  The basis states are set to have 0 flux out, and only self-transition = 1.0.
        #   These are the two-sided absorbing BCs for the committor.
        sinkBins = self.indBasis
        for ii in sinkBins:
            _fluxMatrix[ii, :] = np.zeros((1, self.nBins))
            _fluxMatrix[ii, ii] = 1.0

        q = np.zeros((self.nBins, 1))

        # Committor to the target state is 1 by definition
        q[self.indTargets, 0] = 1.0

        dconv = 100.0
        progress = 0
        qp = np.ones_like(q)

        # Iteratively update the committor estimate until it converges to stationarity
        # (The committor is the stationary distribution for two-sided absorbing boundary conditions)
        with tqdm.tqdm(total=-np.log10(conv)) as pbar:
            while dconv > conv:
                q[self.indTargets, 0] = 1.0
                q[self.indBasis, 0] = 0.0
                q = np.matmul(_fluxMatrix, q)
                dconv = np.sum(np.abs(qp - q))

                # Update with progress since last iter
                log.debug("convergence: " + str(dconv) + "\n")
                pbar.update(max(0, -np.log10(dconv)) - progress)
                progress = max(0, -np.log10(dconv))

                qp = q.copy()
                self.q = q

        self.q = q.squeeze()

    # TODO: This should probably just be a call to get_committor, followed by self.q = 1 - self.q
    def get_backwards_committor(self, conv):
        Mt = self.fluxMatrix.copy()
        nR = np.shape(Mt)
        sM = np.sum(Mt, 1)
        for iR in range(nR[0]):
            if sM[iR] > 0:
                Mt[iR, :] = Mt[iR, :] / sM[iR]
            if sM[iR] == 0.0:
                Mt[iR, iR] = 1.0
        sinkBins = self.indTargets  # np.where(avBinPnoColor==0.0)
        nsB = np.shape(sinkBins)
        nsB = nsB[0]
        for ii in sinkBins:
            Mt[ii, :] = np.zeros((1, self.nBins))
            Mt[ii, ii] = 1.0
        Mt = np.transpose(Mt)  # time reversal
        q = np.zeros((self.nBins, 1))
        q[self.indBasis, 0] = 1.0
        dconv = 100.0
        qp = np.ones_like(q)
        while dconv > conv:
            q[self.indBasis, 0] = 1.0
            q[self.indTargets, 0] = 0.0
            q = np.matmul(Mt, q)
            dconv = np.sum(np.abs(qp - q))
            sys.stdout.write("convergence: " + str(dconv) + "\n")
            qp = q.copy()
            self.qm = q
        self.q = q.copy()

    def plot_committor(self):
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(self.targetRMSD_centers[:, 0], self.q, s=15, c="black")
        plt.yscale("log")
        plt.ylabel("Pseudocommittor to target", fontsize=12)
        plt.xlabel("Average microstate pcoord", fontsize=12)

        self.print_pseudocommittor_warning()

        plt.pause(1)
        fig.savefig(
            self.modelName
            + "_s"
            + str(self.first_iter)
            + "_e"
            + str(self.last_iter)
            + "committor.png"
        )

    @staticmethod
    def print_pseudocommittor_warning():
        log.warning(
            "Note that, if steady-state weighted ensemble data is being analyzed, this is a 'pseudocommittor' "
            "and not a true committor as a result of being constructed from a one-way ensemble."
        )
