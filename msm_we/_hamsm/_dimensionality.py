from sklearn.decomposition import IncrementalPCA as iPCA
import concurrent
import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA
from deeptime.decomposition import TICA, VAMP
from msm_we._logging import log, ProgressBar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msm_we import modelWE

SUPPORTED_METHODS = ["none", "pca", "vamp", "tica", "batch-pca"]


class DimensionalityReductionMixin:
    dimReduceMethod = None
    """str: Dimensionality reduction method. Must be one of _dimensionality.SUPPORTED_METHODS (**NOT** NoneType)"""

    vamp_lag = None
    vamp_dim = None

    class Coordinates(object):
        """
        Fake Coordinates class, in case you don't want to use either PCA or VAMP for dimensionality reduction
        """

        # The class "constructor" - It's actually an initializer
        def __init__(self):
            self.explanation = "coordinate object"

        def transform(self, coords):
            return coords

    def reduceCoordinates(self: "modelWE", coords):
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

        # log.debug("Reducing coordinates")

        # TODO: This list should not be stored here, this should be a class attribute or something

        if self.dimReduceMethod in SUPPORTED_METHODS:
            coords = self.processCoordinates(coords)
            coords = self.coordinates.transform(coords)
            return coords

        raise Exception("dimReduceMethod undefined in reduceCoordinates")

    def do_pca(self: "modelWE", arg):
        rough_pca, iteration, processCoordinates = arg
        iter_coords = self.get_iter_coordinates(iteration)

        # If  no good coords in this iteration, skip it
        if iter_coords.shape[0] == 0:
            return rough_pca

        processed_iter_coords = processCoordinates(iter_coords)
        rough_pca.partial_fit(processed_iter_coords)

        log.debug(f"{rough_pca.n_samples_seen_} samples seen")

        return rough_pca

    def do_full_pca(self: "modelWE", arg):

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
        self: "modelWE",
        first_iter=1,
        first_rough_iter=None,
        last_iter=None,
        rough_stride=10,
        fine_stride=1,
        variance_cutoff=0.95,
        use_weights=True,
        progress_bar=None,
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

            with ProgressBar(progress_bar) as progress_bar:
                task = progress_bar.add_task(
                    description="Initial iPCA", total=len(rough_iters)
                )

                for iteration in rough_iters:
                    # TODO: Allow  chunking here so you don't have  to  go 1  by  1, but N by N
                    # If you don't use 'fork' context here, this will break in Jupyter.
                    # That's because processCoordinates is monkey-patched in. With 'spawn' (i.e. without fork), the module
                    #   is re-imported in the child process. In the reimported  module, processCoordinates is undefined.
                    # With 'fork', it preserves the monkey-patched version.
                    # Additionally, 'fork' is a little faster than  spawn. Ironically, that's usually at the cost  of memory
                    #   usage. But here, the memory being used by the main thread (and therefore being copied here) isn't
                    #   that great -- the memory issue stems from it not being freed up between successive calls.

                    # TODO: This, and the call below to do_full_pca, both sometimes indefinitely hang.
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=1, mp_context=mp.get_context("fork")
                    ) as executor:
                        rough_ipca = executor.submit(
                            self.do_pca,
                            [rough_ipca, iteration, self.processCoordinates],
                        ).result()

                    progress_bar.update(task, advance=1)

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
            iterations = range(first_iter, last_iter, fine_stride)
            with ProgressBar(progress_bar) as progress_bar:
                task = progress_bar.add_task(
                    total=len(iterations), completed=0, description="iPCA"
                )

                for iteration in iterations:

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
                            [
                                ipca,
                                iteration,
                                self.processCoordinates,
                                components_for_var,
                            ],
                        ).result()

                    progress_bar.advance(task, 1 + extra_iters_used)

            self.coordinates = ipca
            self.ndim = components_for_var

        elif self.dimReduceMethod in ["tica", "vamp", "batch-pca"]:

            # TODO: Streaming implementation, a la
            #  https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.TICA.html#deeptime.decomposition.TICA.partial_fit

            # TODO: Pre-allocate these.. but how can I get their sizes? And, they're ragged
            trajs_start = []
            trajs_end = []
            weights = []

            if last_iter is None:
                last_iter = self.maxIter

            iterations = range(first_iter, last_iter, fine_stride)
            with ProgressBar(progress_bar) as progress_bar:
                task = progress_bar.add_task(
                    total=len(iterations),
                    completed=0,
                    description=f"Loading data for {self.dimReduceMethod.upper()}",
                )
                for iteration in range(first_iter, last_iter, fine_stride):
                    # iter_coords = self.get_iter_coordinates(iteration)
                    self.load_iter_data(iteration)
                    self.get_transition_data_lag0()

                    coords_from = self.coordPairList[:, :, :, 0]
                    coords_to = self.coordPairList[:, :, :, 1]

                    # If  no good coords in this iteration, skip it
                    # if iter_coords.shape[0] == 0:
                    #     continue

                    processed_start = self.processCoordinates(coords_from)
                    processed_end = self.processCoordinates(coords_to)
                    trajs_start.extend(processed_start)
                    trajs_end.extend(processed_end)
                    weights.extend(self.weightList)

                    progress_bar.update(task, advance=1)

            weights = np.array(weights)

            if self.dimReduceMethod == "tica":
                self.coordinates = TICA(
                    lagtime=1, var_cutoff=variance_cutoff, scaling="kinetic_map"
                )
            elif self.dimReduceMethod == "vamp":
                self.coordinates = VAMP(
                    lagtime=1, var_cutoff=variance_cutoff, scaling="kinetic_map"
                )
            elif self.dimReduceMethod == "batch-pca":
                self.coordinates = PCA(n_components=None)

            # self.coordinates.fit(trajs)
            log.info(f"Performing weighted {self.dimReduceMethod}")
            # print(f"Performing weighted TICA with weights {weights.shape} and trajs {trajs.shape}")

            # Weights are not currently supported in VAMP
            #   See: https://github.com/deeptime-ml/deeptime/blob/main/deeptime/covariance/util/_running_moments.py#L247
            if not use_weights or self.dimReduceMethod == "vamp":
                weights = None

            if self.dimReduceMethod in ["vamp", "tica"]:
                self.coordinates.fit_from_timeseries(
                    (np.array(trajs_start), np.array(trajs_end)), weights=weights
                )

                # Note: ndim is only used in one place, and it's a deprecated obsolete function
                self.ndim = self.coordinates.model.output_dimension

                log.info(
                    f" {self.dimReduceMethod} will reduce "
                    f"{self.coordinates._model._instantaneous_coefficients.shape[0]} to {self.ndim} components."
                )

            elif self.dimReduceMethod in ["batch-pca"]:

                data_array = np.concatenate(
                    [np.array(trajs_start), np.array(trajs_end)]
                )
                print(f"Attempting to fit PCA to array of shape {data_array.shape}")
                self.coordinates.fit(data_array)

                self.ndim = self.coordinates.n_components_

                log.info(
                    f" {self.dimReduceMethod} will reduce {self.coordinates.n_features_} "
                    f"to {self.coordinates.n_components_} components."
                )

        elif self.dimReduceMethod == "none":
            self.ndim = int(3 * self.nAtoms)

            # TODO: Why is this unused?
            # data = self.all_coords.reshape(-1, self.ndim)
            self.coordinates = self.Coordinates()
            # self.coordinates.transform = self.processCoordinates

        else:
            raise NotImplementedError
