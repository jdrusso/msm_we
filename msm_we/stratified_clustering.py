from sklearn.cluster import MiniBatchKMeans as mini_kmeans
import numpy as np
from ._logging import log


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

        total_clusters = sum(
            [
                len(self.cluster_models[idx].cluster_centers_)
                if hasattr(self.cluster_models[idx], "cluster_centers_")
                else 0
                for idx in self.legitimate_bins
            ]
        )

        for i, coord in enumerate(coords):

            # Note that for target/basis states, the discrete cluster index isn't really used anywhere other than in
            #   self.cluster_structures.
            # Fluxmatrix calculations explicitly short-circuit this.
            #   (Maybe it doesn't need to, if this is correctly indexed?)
            # However, this ensures self.dtrajs is correctly indexed.
            if is_target[i]:
                _discrete = [total_clusters + 1]

                _bin = we_bins[i]
                self.target_bins.add(_bin)

            elif is_basis[i]:
                _discrete = [total_clusters]

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

                    # log.info(f"Doing prediction with n threads {self.cluster_models[we_bins[i]]._n_threads}")

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
