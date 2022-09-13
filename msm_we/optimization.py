import numpy as np

import logging

import westpa.core.binning
from rich.logging import RichHandler

from . import msm_we
from sklearn.cluster import KMeans
import pickle
import base64

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

    pi_matrix = pi * pi.T.reshape(-1,1) / norm

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

    if np.isnan(variance).any():
        log.warning("NaN elements in variance!")

    # TODO: Verify this is the correct sanity check
    assert np.isclose(discrepancy @ pi, 0), "Discrepancy solution failed normalization sanity check!"

    return discrepancy, variance


def get_uniform_mfpt_bins(variance, discrepancy, steady_state, n_desired_we_bins):
    """
    Implements the MFPT-binning strategy described in [1], where bins are groups of microstates that are uniformly
    spaced in the integral of pi * v

    Parameters
    ----------
    variance, array-like: Variance function
    discrepancy, array-like: Discrepancy function
    steady_state, array-like: Steady-state distribution
    n_desired_we_bins int: Number of WE macrobins to assign microstates to -- typically the total number of bins,
        less any recycling or basis bins

    Returns
    -------

    References
    ----------
    [1] Aristoff, D., Copperman, J., Simpson, G., Webber, R. J. & Zuckerman, D. M.
    Weighted ensemble: Recent mathematical developments. Arxiv (2022).

    """

    # The last two elements of this are the basis and target states respectively
    pi_v = steady_state * variance
    pi_v_sort = np.argsort(discrepancy).squeeze()
    cumsum = np.cumsum(pi_v[pi_v_sort])

    n_active_bins = n_desired_we_bins-2

    bin_bounds = np.linspace(0, cumsum[-1], n_active_bins + 1)[1:]
    bin_assignments = np.digitize(cumsum, bin_bounds, right=True)
    bin_states = bin_assignments[np.argsort(pi_v_sort)]

    return bin_states


def get_clustered_mfpt_bins(variance, discrepancy, steady_state, n_desired_we_bins):
    """
    Implements the MFPT-binning strategy described in [1], where bins are groups of microstates that are uniformly
    spaced in the integral of pi * v

    Parameters
    ----------
    variance, array-like: Variance function
    discrepancy, array-like: Discrepancy function
    steady_state, array-like: Steady-state distribution
    n_desired_we_bins int: Number of WE macrobins to assign microstates to -- typically the total number of bins,
        less any recycling or basis bins

    Returns
    -------

    References
    ----------
    [1] Aristoff, D., Copperman, J., Simpson, G., Webber, R. J. & Zuckerman, D. M.
    Weighted ensemble: Recent mathematical developments. Arxiv (2022).

    """

    # The last two elements of this are the basis and target states respectively
    pi_v = steady_state * variance
    n_active_bins = n_desired_we_bins-2
    pi_v_sort = np.argsort(discrepancy).squeeze()
    cumsum = np.cumsum(pi_v[pi_v_sort])

    clusterer = KMeans(n_clusters=min(n_active_bins, len(cumsum)))

    # we_bin_assignments = clusterer.fit_predict(pi_v.reshape(-1, 1))
    we_bin_assignments = clusterer.fit_predict(cumsum.reshape(-1, 1))

    bin_states = np.full_like(cumsum, fill_value=np.nan)
    for i in range(n_active_bins):
        indices = np.argwhere(we_bin_assignments == i).squeeze()
        states_in_bin = pi_v_sort[indices]
        bin_states[states_in_bin] = i
        log.debug(f"Found that bin {i} contains microstates {states_in_bin}")

    return bin_states


class OptimizedBinMapper(westpa.core.binning.FuncBinMapper):

    def __init__(self, *args, **kwargs):

        if len(args) + len(kwargs) == 1:
            log.info("Single argument provided to binmapper initializer, attempting to unpickle")
            self.unpickle(kwargs['bytestring'])

        elif len(args) + len(kwargs) > 1:
            log.info("Multiple arguments provided to binmapper initializer, creating new object")
            self.create_new(*args, **kwargs)

    def unpickle(self, bytestring):

        # TODO: This may need to be provided as a b64-encoded bytestring, rather than plain bytes,
        #  in which case we need to quickly encode it back to bytes. This can be done with
        #   pickle.loads(base64.b64decode(collection.metadata['bin_mapper']))

        loaded = pickle.loads(base64.b64decode(bytestring))

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
                 cluster_on_pcoord: bool = False,
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

        self.cluster_on_pcoord = cluster_on_pcoord

        self.clusterer.model.n_clusters = 2
        for cluster_model in self.clusterer.cluster_models:
            if hasattr(cluster_model, 'cluster_centers_'):
                self.clusterer.model.n_clusters += len(cluster_model.cluster_centers_)
        log.info(f"Clusterer has {self.clusterer.model.n_clusters} total clusters (include 1 for basis and 1 for target)")
        log.debug(f"Clusterer remap is {self.clusterer.we_remap} "
                 f"(last two correspond to basis, target states and can be ignored)")

    def mapper(self, coords, mask, output, *args, **kwargs):

        # TODO: When analyzing augmented coordinates, this needs to pull just the child pcoord -- but when analyzing
        #   non-augmented, that won't exist
        # TODO: ^ What does that mean? What exactly is happening here, and how can I check for it more gracefully?
        if len(coords.shape) == 3:
            final_coords = coords[:, -1]
        else:
            final_coords = coords

        log.debug(f"Mapping original coords of shape {coords.shape}")
        log.debug(f"Mapping pcoords {final_coords}")
        log.debug(f"Mapping pcoords of shape {final_coords.shape}")

        # To use stratified clustering, first load the ORIGINAL pcoords into stratified.pcoord1List, then call
        #   stratified.predict().
        # Segments in the basis will be assigned to whatever n_clusters is, and targets will be assigned to whatever
        #   n_clusters + 1 is.
        #  This isn't actually used for anything else, and no clustering happens for these, so I can actually
        #  set these arbitrarily.
        original_pcoords = final_coords[:, :self.n_original_pcoord_dims]

        if not hasattr(self, 'cluster_on_pcoord'):
            self.cluster_on_pcoord = False

        if not self.cluster_on_pcoord:
            extended_pcoords = final_coords[:, self.n_original_pcoord_dims:]
        else:
            extended_pcoords = final_coords[:, :self.n_original_pcoord_dims]

        basis_we_bin_idx, target_we_bin_idx = self.nbins-2, self.nbins-1

        log.debug(f"Original pcoords dimensionality was {self.n_original_pcoord_dims}")
        log.debug(f"Original pcoords had shape {original_pcoords.shape}")
        self.clusterer.model.pcoord1List = original_pcoords

        # TODO: Get the basis and target state from this, and then I don't need to use the bounds here
        base_bins = self.base_mapper.assign(original_pcoords)
        log.debug(f"Base bin mapper mapped to {base_bins}")


        # Now, do stratified clustering on the rest of the coordinates.
        # Each segment will be
        #   1. Assigned a WE bin, based on its pcoords and the provided bin mapper
        #   2. Discretized, according to the k-means model associated with that WE bin
        log.debug(f"About to cluster coords of shape {extended_pcoords.shape}")

        # TODO: Do I just want final_coords[:, self.n_original_pcoord_dims:]?

        stratified_cluster_assignments = self.clusterer.predict(extended_pcoords)

        log.debug(f"Got microstate assignments {stratified_cluster_assignments}")

        # TODO: Map microstates to new WE bins, and populate we_bin_assignments
        # I have a microstate for each segment now -- I need to refer to my mapping of microstates to WE bins, which
        #   just comes from my optimization step
        log.debug(f"Mapping microstates to WE bins using {self.microstate_mapper}")

        we_bin_assignments = np.array([float(self.microstate_mapper[microstate])
                                       if microstate < len(self.microstate_mapper) else -1
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

        assert not np.isnan(output).any(), "Some segments couldn't get assigned WE bins"
        assert all(output >= 0), "Some target/basis microstates didn't get correctly remapped"

        return output.astype(int)
