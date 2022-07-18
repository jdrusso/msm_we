import numpy as np

import logging

import westpa.core.binning
from rich.logging import RichHandler

from . import msm_we
from sklearn.cluster import KMeans
import pickle

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(RichHandler())
log.propagate = False


def solve_discrepancy(tmatrix, pi, B):
    """
    Given a transition matrix, solves for the discrepancy function.

    The Poisson equation for the discrepancy function is
    .. math::
        (I - K)h = 1_B - \pi(B), \:\: h \cdot \pi = 0

    however, since :math:`I-K` is singular, we instead solve
    .. math::
        (I - K + \pi \pi^T / || \pi ||^2_2)h = 1_B - \pi(B), \:\: h \cdot \pi = 0
    where :math:`h` is a volumn vector, `1_B` is an indicator function which is 1 in B and 0 everywhere
    else, :math:`\pi` is the steady-state solution of :math:`K`, and `\pi(B)` is a column vector with
    the steady-state value of :math:`\pi(B)` in every element.

    Parameters
    ----------
    tmatrix, 2D array-like: Transition matrix
    pi, array-like: Steady-state distribution for the input transition matrix
    B, array-like: Indices of target states B

    Returns
    --------
    (discrepancy, variance)
    """

    log.info("Computing pi matrix")
    norm = np.dot(pi, pi.T)
    pi_matrix = pi @ pi.T / norm

    b_indicator = np.zeros_like(pi)
    b_indicator[B] = 1.0

    pi_b = np.ones_like(pi)
    pi_b[:] = sum(pi[B])

    discrepancy = np.linalg.solve(np.identity(tmatrix.shape[0]) - tmatrix + pi_matrix,
                                  b_indicator - pi_b)

    variance = np.sqrt(
        np.dot(tmatrix, discrepancy**2) -
        np.dot(tmatrix, discrepancy)**2
    )

    return discrepancy, variance


def get_uniform_mfpt_bins(variance, steady_state, n_desired_we_bins):
    """
    Implements the MFPT-binning strategy described in [1], where bins are groups of microstates that are uniformly
    spaced in the integral of pi * v

    Parameters
    ----------
    variance, array-like: Variance function
    steady_state, array-like: Steady-state distribution
    n_bins int: Number of macrobins

    Returns
    -------

    References
    ----------
    [1] Aristoff, D., Copperman, J., Simpson, G., Webber, R. J. & Zuckerman, D. M.
    Weighted ensemble: Recent mathematical developments. Arxiv (2022).

    """

    # The last two elements of this are the basis and target states respectively
    pi_v = steady_state * variance

    spacing = sum(pi_v) / n_desired_we_bins

    bin_states = np.full_like(steady_state, fill_value=np.nan)
    for i in range(n_desired_we_bins):
        lower, upper = spacing * i, spacing * (i+1)
        log.debug(f"Checking for states with pi_v between {lower}, {upper}")

        states_in_bin = np.argwhere(
            (lower < np.cumsum(pi_v)) &
            (np.cumsum(pi_v) <= upper)
        )

        log.info(f"Found that bin {i} contains microstates {states_in_bin}")
        bin_states[states_in_bin] = i

    return bin_states


def get_clustered_mfpt_bins(variance, steady_state, n_desired_we_bins):
    """
    Implements the MFPT-binning strategy described in [1], where bins are groups of microstates that are uniformly
    spaced in the integral of pi * v

    Parameters
    ----------
    variance, array-like: Variance function
    steady_state, array-like: Steady-state distribution
    n_bins int: Number of macrobins

    Returns
    -------

    References
    ----------
    [1] Aristoff, D., Copperman, J., Simpson, G., Webber, R. J. & Zuckerman, D. M.
    Weighted ensemble: Recent mathematical developments. Arxiv (2022).

    """

    # The last two elements of this are the basis and target states respectively
    pi_v = steady_state * variance

    clusterer = KMeans(n_clusters=n_desired_we_bins)

    # -2 so you don't cluster the basis/target states
    # TODO: Handle those in a more general way, though from msm_we you're guaranteed they'll be the last two states
    # we_bin_assignments = clusterer.fit_predict(pi_v.reshape(-1, 1))
    we_bin_assignments = clusterer.fit_predict(pi_v[:-2].reshape(-1, 1))

    bin_states = np.full_like(steady_state, fill_value=np.nan)
    for i in range(n_desired_we_bins):
        states_in_bin = np.argwhere(we_bin_assignments == i).squeeze()
        bin_states[states_in_bin] = i
        log.info(f"Found that bin {i} contains microstates {states_in_bin}")

    return bin_states


class OptimizedBinMapper(westpa.core.binning.FuncBinMapper):

    def __init__(self, *args, **kwargs):

        if len(args) + len(kwargs) == 1:
            log.info("Single argument provided to binmapper initializer, attempting to unpickle")
            self.unpickle(args[0])

        elif len(args) + len(kwargs) > 1:
            log.info("Multiple arguments provided to binmapper initializer, creating new object")
            self.create_new(*args, **kwargs)

    def unpickle(self, bin_dict):

        loaded = pickle.loads(bin_dict['bytestring'])

        assert type(loaded) is type(self), "Invalid pickled object provided"

        # TODO: Probably just update the attributes I explicitly care about, this seems a little insecure
        for k, v in vars(loaded).items():
            setattr(self, k, v)

    def create_new(self,
                 nbins: int,
                 n_original_pcoord_dims: int,
                 target_pcoord_bounds,
                 basis_pcoord_bounds,
                 previous_binmapper,
                 microstate_mapper: dict,
                 stratified_clusterer: msm_we.StratifiedClusters,
                 *args,
                 **kwargs
                 ):
        """
        Creates an OptimizedBinMapper, suitable for use with the optimization workflow

        Parameters
        ----------
        nbins: int, Number of WE bins
        n_original_pcoord_dims: int, Number of dimensions in the original user-supplied progress coordinate
        microstate_mapper: dict, Mapping of microstates to WE bins
        """

        super().__init__(func=self.mapper, nbins=nbins, args=args, kwargs=kwargs)

        self.microstate_mapper = microstate_mapper
        self.n_original_pcoord_dims = n_original_pcoord_dims

        self.base_mapper = previous_binmapper

        self.simple_model = msm_we.modelWE()
        self.simple_model.pcoord_ndim = n_original_pcoord_dims
        self.simple_model.basis_pcoord_bounds = basis_pcoord_bounds
        self.simple_model.target_pcoord_bounds = target_pcoord_bounds

        self.clusterer = stratified_clusterer
        self.clusterer.model = self.simple_model

        self.clusterer.model.n_clusters = 2
        for cluster_model in self.clusterer.cluster_models:
            if hasattr(cluster_model, 'cluster_centers_'):
                self.clusterer.model.n_clusters += len(cluster_model.cluster_centers_)
        log.info(f"Clusterer has {self.clusterer.model.n_clusters} total clusters")
        log.info(f"Clusterer remap is {self.clusterer.we_remap}")

    def mapper(self, coords, mask, output, *args, **kwargs):

        # TODO: When analyzing augmented coordinates, this needs to pull just the child pcoord -- but when analyzing
        #   non-augmented, that won't exist
        # final_coords = coords[:,-1]
        if len(coords.shape) == 3:
            final_coords = coords[:, -1]
        else:
            final_coords = coords

        log.debug(f"Mapping pcoords {final_coords}")

        # To use stratified clustering, first load the ORIGINAL pcoords into stratified.pcoord1List, then call
        #   stratified.predict().
        # Segments in the basis will be assigned to whatever n_clusters is, and targets will be assigned to whatever
        #   n_clusters + 1 is.
        #  This isn't actually used for anything else, and no clustering happens for these, so I can actually
        #  set these arbitrarily.
        original_pcoords = final_coords[:, :self.n_original_pcoord_dims]

        basis_we_bin_idx, target_we_bin_idx = self.nbins-2, self.nbins-1

        log.debug(f"Original pcoords dimensionality was {self.n_original_pcoord_dims}")
        log.debug(f"Original pcoords had shape {original_pcoords.shape}")
        self.clusterer.model.pcoord1List = original_pcoords

        base_bins = self.base_mapper.assign(original_pcoords)
        log.debug(f"Base bin mapper mapped to {base_bins}")

        # Now, do stratified clustering on the rest of the coordinates.
        # Each segment will be
        #   1. Assigned a WE bin, based on its pcoords and the provided bin mapper
        #   2. Discretized, according to the k-means model associated with that WE bin
        log.debug(f"About to cluster coords of shape {final_coords.shape}")
        stratified_cluster_assignments = self.clusterer.predict(final_coords)

        log.debug(f"Got microstate assignments {stratified_cluster_assignments}")

        # TODO: Map microstates to new WE bins, and populate we_bin_assignments
        # I have a microstate for each segment now -- I need to refer to my mapping of microstates to WE bins, which
        #   just comes from my optimization step
        log.debug(f"Mapping microstates to WE bins using {self.microstate_mapper}")

        we_bin_assignments = np.array([int(self.microstate_mapper[microstate])
                                       if microstate < len(self.microstate_mapper) else np.nan
                                       for microstate in stratified_cluster_assignments
                                      ])

        log.debug(f"Basis WE bin is labeled {basis_we_bin_idx}, target WE bin is labeled {target_we_bin_idx}")
        log.debug(f"WE bin assignments before correcting basis/target are {we_bin_assignments}")

        we_bin_assignments[self.clusterer.model.is_WE_target(final_coords)] = target_we_bin_idx
        we_bin_assignments[self.clusterer.model.is_WE_basis(final_coords)] = basis_we_bin_idx

        zipped_assignments = np.array(list(zip(original_pcoords.reshape(-1), we_bin_assignments)))
        zip_sort = np.argsort(original_pcoords.reshape(-1))

        log.debug(f"WE bin assignments are {zipped_assignments[zip_sort]}")

        for i in range(len(output)):
            output[i] = we_bin_assignments[i]

        return output