from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import MiniBatchKMeans as mini_kmeans
import numpy as np
from copy import deepcopy
import ray
import concurrent
import tqdm.auto as tqdm
import multiprocessing as mp
from westpa import analysis
from westpa.core.binning import RectilinearBinMapper, VoronoiBinMapper
from msm_we.stratified_clustering import StratifiedClusters
from msm_we.utils import find_connected_sets
from msm_we._logging import log

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msm_we import modelWE

# If you implement a custom bin mapper that should work with stratified clustering, you can add it to this set
#   after importing msm_we
SUPPORTED_MAPPERS = {RectilinearBinMapper, VoronoiBinMapper}


class ClusteringMixin:

    n_clusters = None
    clusters = None
    clusterFile = None
    use_weights_in_clustering = False

    targetRMSD_centers = None
    targetRMSD_minmax = None
    targetRMSD_all = None

    pcoord_cache = None
    cluster_structures = None
    cluster_structure_weights = None

    def do_clustering(self: "modelWE", arg):

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

    def do_discretization(self: "modelWE", arg):

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
    def do_ray_discretization(
        model: "modelWE", kmeans_model, iteration, processCoordinates
    ):

        # model_id, kmeans_model_id, iteration, processCoordinates_id = arg

        # self = ray.get(model_id)
        # kmeans_model = ray.get(kmeans_model_id)
        # processCoordinates = ray.get(processCoordinates_id)
        # self = model

        # Need to do this so the model's transformation array is writable -- otherwise predict chokes
        #   with 'buffer source array is read-only'.
        kmeans_model = deepcopy(kmeans_model)

        iter_coords = model.get_iter_coordinates(iteration)

        # If there are no coords for this iteration, return None
        if iter_coords.shape[0] == 0:
            return None, 0, iteration

        # Otherwise, apply the k-means model and discretize
        transformed_coords = model.coordinates.transform(
            processCoordinates(iter_coords)
        )
        dtrajs = kmeans_model.predict(transformed_coords)

        return dtrajs, 1, iteration

    def cluster_coordinates(
        self: "modelWE",
        n_clusters,
        streaming=False,
        first_cluster_iter=None,
        use_ray=False,
        stratified=True,
        iters_to_use=None,
        store_validation_model=False,
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
        if store_validation_model:
            self.post_cluster_model = deepcopy(self)

    def cluster_aggregated(
        self: "modelWE",
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
                    self.get_iter_coordinates(iteration).reshape(
                        -1, self.coord_ndim * self.nAtoms
                    )
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
        self: "modelWE",
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

                try:
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

                except AssertionError as e:
                    # If we succeeded in passing this loop at least once, then all our bins have *something* in them.
                    # TODO: The better way to handle this is as long as you've clustered something, you can do piecemeal
                    #   (i.e. don't have to populate every bin after the first time)
                    if iter_idx == 0:
                        log.info(
                            f"Failed with {iter_idx} + {extra_iters_used} vs len {(len(iters_to_use))}"
                        )
                        raise e
                    else:
                        log.info(
                            "Clustering couldn't use last iteration, not all bins filled."
                        )

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

    def do_stratified_clustering(self: "modelWE", arg):
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
            _seg_weights = self.seg_weights[iteration]

            if used_iters == 0:
                iter_coords = _iter_coords
                seg_weights = _seg_weights
                pcoords = [x for x in self.pcoord0List]
            else:
                iter_coords = np.append(iter_coords, _iter_coords, axis=0)
                pcoords.extend(self.pcoord0List)
                if self.use_weights_in_clustering:
                    seg_weights = np.append(seg_weights, _seg_weights, axis=0)
                log.debug(
                    f"After extension, pcoords: {len(pcoords)}, iter_coords: {iter_coords.shape}. Ignored: {ignored_bins}. Mapper: {bin_mapper}"
                )

            # Map coords to WE bins
            pcoord_array = np.array(pcoords)
            # seg_weight_array = np.array(seg_weights)
            assert (
                pcoord_array.shape[0] == iter_coords.shape[0]
            ), f"{pcoord_array.shape}, {iter_coords.shape}"

            if self.use_weights_in_clustering:
                assert (
                    seg_weights.shape[0] == seg_weights.shape[0]
                ), f"{seg_weights.shape}, {iter_coords.shape}"

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

            transformed_coords = self.coordinates.transform(
                processCoordinates(np.squeeze(iter_coords[segs_in_bin]))
            )

            if self.use_weights_in_clustering:
                weights = seg_weights[segs_in_bin].squeeze()
            else:
                weights = None

            if weights is not None:
                log.debug(weights.shape)
            else:
                log.debug("Weights are None")

            try:
                kmeans_models.cluster_models[_bin].partial_fit(
                    transformed_coords, sample_weight=weights
                )

            except ValueError as e:
                log.info(f"Was on bin {_bin}")
                log.error(f"Error fitting k-means to bin {_bin}")
                raise e

        return kmeans_models, used_iters, unique_bins, unfilled_bins

    def organize_stratified(self: "modelWE", use_ray=True):
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

            clusters_in_bin = range(
                offset,
                offset + n_clusters_in_bin,
            )
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

    def launch_ray_discretization(self: "modelWE"):
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

        if self.pre_discretization_model is None:
            self.pre_discretization_model = deepcopy(self)
        else:
            log.info("Using cached model for discretization")

        model_id = ray.put(self.pre_discretization_model)

        clusters = deepcopy(self.clusters)
        # It's set inside do_stratified_ray_discretization, though I could do it in either place.
        clusters.model = None  # self.pre_discretization_model
        cluster_model_id = ray.put(clusters)

        process_coordinates_id = ray.put(self.processCoordinates)

        # max_inflight = 50
        for iteration in tqdm.tqdm(
            range(1, self.maxIter), desc="Submitting discretization tasks"
        ):

            _id = self.do_stratified_ray_discretization.remote(
                model_id,
                cluster_model_id,
                iteration,
                process_coordinates_id
                # self, self.clusters, iteration, self.processCoordinates
            )
            task_ids.append(_id)

        # As they're completed, add them to dtrajs
        dtrajs = [None] * (self.maxIter - 1)
        pair_dtrajs = [None, None] * (self.maxIter - 1)

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

                for (
                    (parent_dtraj, child_dtraj),
                    _,
                    iteration,
                    target_bins,
                    basis_bins,
                ) in results:

                    self.clusters.target_bins.update(target_bins)
                    self.clusters.basis_bins.update(basis_bins)

                    dtrajs[iteration - 1] = child_dtraj

                    pair_dtrajs[iteration - 1] = list(zip(parent_dtraj, child_dtraj))

                    pbar.update(1)
                    pbar.refresh()

                del results
                del finished
        del model_id
        del cluster_model_id

        # Remove all empty elements from dtrajs and assign to self.dtrajs
        self.dtrajs = [dtraj for dtraj in dtrajs if dtraj is not None]

        self.pair_dtrajs = [dtraj for dtraj in pair_dtrajs if dtraj is not None]

        log.info("Discretization complete")

    @ray.remote
    def do_stratified_ray_discretization(
        model: "modelWE", kmeans_model, iteration, processCoordinates
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

        # iter_coords = self.get_iter_coordinates(iteration)

        kmeans_model.model.load_iter_data(iteration)
        kmeans_model.model.get_transition_data_lag0()
        # print(f"After loading coordPairList in iter {iteration}, shape is {kmeans_model.model.coordPairList.shape}")
        parent_coords, child_coords = (
            kmeans_model.model.coordPairList[..., 0],
            self.coordPairList[..., 1],
        )

        # If there are no coords for this iteration, return None
        if child_coords.shape[0] == 0:
            return None, 0, iteration

        # Otherwise, apply the k-means model and discretize
        transformed_parent_coords = kmeans_model.model.coordinates.transform(
            processCoordinates(parent_coords)
        )
        transformed_child_coords = kmeans_model.model.coordinates.transform(
            processCoordinates(child_coords)
        )

        try:
            kmeans_model.processing_from = True
            try:
                parent_dtrajs = kmeans_model.predict(transformed_parent_coords)
            except IndexError as e:

                print("Problem ===== ")
                print(
                    f"Parent pcoords are shape {kmeans_model.model.pcoord0List.shape}"
                )
                print(f"Parent coords are shape {transformed_parent_coords.shape}")
                print(f"Child pcoords are shape {kmeans_model.model.pcoord1List.shape}")
                print(f"Child coords are shape {transformed_child_coords.shape}")
                print("===== ")

                raise e

            kmeans_model.processing_from = False
            child_dtrajs = kmeans_model.predict(transformed_child_coords)
        except AttributeError as e:
            log.error("Cluster center was not initialized and not remapped")
            log.error(kmeans_model.we_remap)
            raise e
            # TODO: Remap to nearest visited

        return (
            (parent_dtrajs, child_dtrajs),
            1,
            iteration,
            kmeans_model.target_bins,
            kmeans_model.basis_bins,
        )

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

    def update_cluster_structures(self: "modelWE", build_pcoord_cache=False):
        """
        Find structures (i.e. sets of coordinates) corresponding to each clusters.

        Parameters
        ----------
        build_pcoord_cache: bool
            If True, builds self.pcoord_cache, which has elements [cluster_idx][seg_idx] holding the pcoord for
            the seg_idx'th segment in MSM cluster cluster_idx.

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

        pcoord_cache = None
        if build_pcoord_cache:
            pcoord_cache = {}

        i = 0
        # total_num_iterations = len(self.numSegments)
        total_num_iterations = self.maxIter
        # Don't include the last iteration, where dynamics didn't run
        for _iter in range(1, total_num_iterations - 1):

            if _iter not in self.seg_weights.keys():
                self.load_iter_data(_iter)

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
        # NOTE: This is *NOT* the within-iteration index, this is an absolute index.
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

                if build_pcoord_cache:
                    cluster_cache = pcoord_cache.setdefault(cluster_idx, [])
                    cluster_cache.append(self.pcoord1List[_seg])

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
        self.pcoord_cache = pcoord_cache

        log.debug("Cluster structure mapping completed.")
        log.debug(f"Cluster keys are {cluster_structures.keys()}")

    def get_cluster_centers(self: "modelWE"):
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
