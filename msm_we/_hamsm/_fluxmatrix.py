import numpy as np
import ray
from scipy.sparse import coo_matrix
import concurrent
import multiprocessing as mp
from msm_we.utils import find_connected_sets
from msm_we._logging import log, ProgressBar
import tqdm.auto as tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msm_we import modelWE


class FluxMatrixMixin:

    fluxMatrixRaw = None
    fluxMatrix = None

    def get_iter_fluxMatrix(self: "modelWE", n_iter):

        self.load_iter_data(n_iter)
        parent_pcoords = self.pcoord0List.copy()
        child_pcoords = self.pcoord1List.copy()

        self.get_transition_data_lag0()
        transition_weights = self.transitionWeights.copy()

        index_pairs = np.array(self.pair_dtrajs[n_iter - 1])

        # Record every point where you're in the target
        ind_end_in_target = np.where(self.is_WE_target(child_pcoords))

        if ind_end_in_target[0].size > 0:
            log.debug(
                "Number of post-transition target1 entries: "
                + str(ind_end_in_target[0].size)
                + "\n"
            )
        else:
            log.debug(f"No target1 entries. {ind_end_in_target}")

        # Get the index of every point
        ind_start_in_basis = np.where(self.is_WE_basis(parent_pcoords))
        if ind_start_in_basis[0].size > 0:
            log.debug(
                "Number of pre-transition points in basis0: "
                + str(ind_start_in_basis[0].size)
                + "\n"
            )

        ind_end_in_basis = np.where(self.is_WE_basis(child_pcoords))
        if ind_end_in_basis[0].size > 0:
            log.debug(
                "Number of post-transition points in basis1: "
                + str(ind_end_in_basis[0].size)
                + "\n"
            )

        return (
            self.build_flux_matrix(
                self.n_clusters,
                index_pairs,
                ind_start_in_basis,
                ind_end_in_basis,
                ind_end_in_target,
                transition_weights,
            )
            .todense()
            .A
        )

    @ray.remote
    def build_flux_matrix_remote(
        n_clusters,
        index_pairs,
        ind_start_in_basis,
        ind_end_in_basis,
        ind_end_in_target,
        transition_weights,
        n_iter,
    ):

        return (
            FluxMatrixMixin.build_flux_matrix(
                n_clusters,
                index_pairs,
                ind_start_in_basis,
                ind_end_in_basis,
                ind_end_in_target,
                transition_weights,
            ),
            n_iter,
        )

    @staticmethod
    def build_flux_matrix(
        n_clusters,
        index_pairs,
        ind_start_in_basis,
        ind_end_in_basis,
        ind_end_in_target,
        transition_weights,
    ):
        """
        Build the flux matrix for an iteration.

        Returns
        -------
        """

        # Create dedicated clusters for the target and basis states,
        # and reassign any points within the target or basis to those
        target_cluster_index = n_clusters + 1
        basis_cluster_index = n_clusters

        try:
            start_cluster, end_cluster = index_pairs.T.copy()
        except Exception as e:
            log.error(index_pairs)
            raise e
        else:
            pass

        # good_coords = np.arange(0, parent_pcoords.shape[0]).astype(int)

        log.debug(f"Cluster 0 shape: {start_cluster.shape}")

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
                (transition_weights, (start_cluster, end_cluster)),
                shape=(n_clusters + 2, n_clusters + 2),
            )  # .todense()
        except ValueError as e:
            log.error(
                f"Iter_fluxmatrix failed. Transition was from {start_cluster} -> {end_cluster} "
                f"\n\t(Total {n_clusters + 2} clusters)"
                f"\n\t(End in target: {ind_end_in_target})"
                f"\n\t(Weights len: {len(transition_weights)})"
                f"\n\t(Good Weights len: {len(transition_weights)})"
                # f"\n\t(Good Coords len: {len(good_coords)})"
                f"\n\t(start cluster len: {len(start_cluster)})"
                f"\n\t(end_cluster len: {len(end_cluster)})"
            )
            raise e

        # While the sparse matrix implementation is nice and efficient, using the np.matrix type is a little weird
        #   and fragile, and it's possible it'll be deprecated in the future.
        # fluxMatrix = fluxMatrix.A

        return fluxMatrix

    def get_fluxMatrix(
        self: "modelWE",
        n_lag,
        first_iter=1,
        last_iter=None,
        iters_to_use=None,
        use_ray=False,
        result_batch_size=5,
        progress_bar=None,
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
            iters_to_use = range(first_iter + 1, last_iter)

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

        # Add up the flux matrices for each iteration to get the flux matrix.
        # Then, save that matrix to the data file, along with the number of iterations used
        # FIXME: Duplicated code
        # The range is offset by 1 because you can't calculate fluxes for the 0th iteration
        if not use_ray:
            with ProgressBar(progress_bar) as progress_bar:
                task = progress_bar.add_task(
                    description="Constructing flux matrix", total=len(iters_to_use)
                )

                for iS in iters_to_use:
                    log.debug("getting fluxMatrix iter: " + str(iS) + "\n")

                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=1, mp_context=mp.get_context("fork")
                    ) as executor:
                        fluxMatrixI = executor.submit(
                            self.get_iter_fluxMatrix,
                            iS,
                        ).result()

                    fluxMatrix = fluxMatrix + fluxMatrixI
                    nI = nI + 1

                    log.debug(f"Completed flux matrix for iter {iS}")
                    progress_bar.update(task, advance=1)

        # If we're running through Ray..
        else:

            # First, connect to the ray cluster
            self.check_connect_ray()

            # Submit all the tasks for iteration fluxmatrix calculations
            task_ids = []

            with ProgressBar(progress_bar) as progress_bar:
                submit_task = progress_bar.add_task(
                    description="Submitting fluxmatrix tasks", total=len(iters_to_use)
                )
                for iteration in iters_to_use:

                    self.load_iter_data(iteration)
                    parent_pcoords = self.pcoord0List.copy()
                    child_pcoords = self.pcoord1List.copy()

                    ind_end_in_target = np.where(self.is_WE_target(child_pcoords))
                    ind_start_in_basis = np.where(self.is_WE_basis(parent_pcoords))
                    ind_end_in_basis = np.where(self.is_WE_basis(child_pcoords))

                    self.get_transition_data_lag0()
                    transition_weights = self.transitionWeights.copy()

                    index_pairs = np.array(self.pair_dtrajs[iteration - 1])

                    _id = self.build_flux_matrix_remote.remote(
                        self.n_clusters,
                        index_pairs,
                        ind_start_in_basis,
                        ind_end_in_basis,
                        ind_end_in_target,
                        transition_weights,
                        iteration,
                    )

                    task_ids.append(_id)
                    progress_bar.update(submit_task, advance=1)
                progress_bar.update(submit_task, visible=False)

                # Wait for them to complete
                # Process results as they're ready, instead of in submission order
                #  See: https://docs.ray.io/en/latest/ray-design-patterns/submission-order.html
                # Additionally, this batches rather than getting them all at once, or one by one.
                retrieve_task = progress_bar.add_task(
                    description="Retrieving fluxmatrix tasks", total=len(iters_to_use)
                )
                while task_ids:
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
                        fluxMatrix = fluxMatrix + _fmatrix.todense().A
                        progress_bar.update(retrieve_task, advance=1)

                    # Try to free up some memory used by Ray for these objects
                    # See: https://github.com/ray-project/ray/issues/15058
                    # I was running into issues with objects spilling from the object store during fluxmatrix
                    #   calculation. None of the individual calculations should really get that big, so maybe
                    #   something wasn't getting freed from memory when it should've.
                    del finished
                    del results

            log.debug("Fluxmatrices all obtained")
            del task_ids
            nI = len(iters_to_use)

        # Normalize the flux matrix by the number of iterations that it was calculated with
        fluxMatrix = fluxMatrix / nI

        # Update state with the new, updated, or loaded from file fluxMatrix.
        self.fluxMatrixRaw = fluxMatrix

    def organize_fluxMatrix(self: "modelWE", use_ray=False, progress_bar=None, **args):
        """
        This cleaning step removes all clusters that aren't in the largest connected set, then rediscretizes all the
        trajectories according to the new reduced set of clusters.
        """

        if not hasattr(self, "clustering_method"):
            log.warning(
                "self.clustering_method is not set. This may be a model saved before stratified was "
                "implemented, or you may not have run cluster_coordinates! "
                "Assuming the former and setting to aggregated."
            )

            self.clustering_method = "aggregated"

        if self.clustering_method == "stratified":
            self.organize_stratified(use_ray, progress_bar)

            # TODO: Respect do_cleaning=False for blockwise stratified

        elif self.clustering_method == "aggregated":

            # If the user has manually specified states to keep, skip the normal cleaning, and just do what they said
            if "states_to_keep" in args.keys():
                self.organize_aggregated(use_ray=use_ray, **args)
                return

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
        self: "modelWE",
        use_ray=False,
        do_cleaning=True,
        states_to_keep=None,
        rediscretize=True,
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

        # I don't think this should be used any more... Check to be sure
        raise DeprecationWarning(
            "organize_aggregated() is not maintained, results may be unexpected!"
        )

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
        cluster_pcoord_centers[target_cluster_index] = self.target_bin_center
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
