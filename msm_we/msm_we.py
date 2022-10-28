"""haMSM estimation and analysis"""
from __future__ import division, print_function

__metaclass__ = type

import numpy as np
import tqdm.auto as tqdm
from functools import partialmethod
import concurrent
import multiprocessing as mp
from copy import deepcopy
import mdtraj as md
from rich.live import Live
from rich.table import Table
from rich.console import Group
import ray
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

# My mixin structure may be a little strange here, because these aren't really mixins that are meant to be
#   reused, it's just to break these out.
# Alternatively, I think I could drop the functions straight in those modules without the mixin classes,
#   and within my modelWE definition just do (for example)
#   class modelWE:
#       from ._clustering import *

from ._hamsm import ClusteringMixin
from ._hamsm import DimensionalityReductionMixin
from ._hamsm import PlottingMixin
from ._hamsm import AnalysisMixin
from ._hamsm import DataMixin
from ._hamsm import FluxMatrixMixin
from ._logging import log, DefaultProgress, ProgressBar


class modelWE(
    ClusteringMixin,
    DimensionalityReductionMixin,
    PlottingMixin,
    AnalysisMixin,
    DataMixin,
    FluxMatrixMixin,
):
    """
    History-augmented Markov state model estimation from WE data

    Implementation of haMSM model building, particularly for steady-state estimation
    from recycling WE sampling with basis (source) and target (sink) states.

    Set up for typical west.h5 file structure, with coordinates to be stored in west.h5 /iterations/auxdata/coord and
    basis and target definitions in progress coordinate space.

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
        Todo
        ----
        - Document all of these attributes

        - Reorganize these attributes into some meaningful structure
        """

        # TODO: In general, it's not clear if this needs to be strictly 1...
        #  However, oversubscribing causes very difficult to diagnose problems
        #  (like hanging during clustering / k-means fitting), and 1 seems to be safe.
        if not _openmp_effective_n_threads() == 1:
            log.critical(
                "Set $OMP_NUM_THREADS=1 for proper msm-we functionality! "
                "Other values may cause strange problems such as silent hanging during "
                "discretization or ray-parallel steps."
            )

        self.modelName = None
        """str: Name used for storing files"""

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

        self.reference_structure = None
        self.reference_coord = None
        self.basis_structure = None
        # TODO: This is plural, reference_coord is singular. Intentional? Can you have multiple bases but 1 reference?
        self.basis_coords = None
        self.nAtoms = None

        self.coordinates = None
        self.ndim = None

        self.targetRMSD_centers = None
        """array-like: List of RMSDs corresponding to each cluster."""
        self.indBasis = None

        self.removed_clusters = []
        self.cluster_structures = None
        self.cluster_structure_weights = None
        """dict: Mapping of cluster indices to structures in that cluster"""

        self.clustering_method = None
        self.validation_models = []

        self.pcoord_shape_warned = False

        self.pre_discretization_model = None

        # self.progress_bar = Progress()

    def initialize(
        # self, fileSpecifier: str, refPDBfile: str, initPDBfile: str, modelName: str
        self,
        fileSpecifier: str,
        refPDBfile: str,
        modelName: str,
        basis_pcoord_bounds: list = None,
        target_pcoord_bounds: list = None,
        dim_reduce_method: str = "none",
        tau: float = None,
        pcoord_ndim: int = 1,
        auxpath: str = "coord",
        _suppress_boundary_warning=False,
        use_weights_in_clustering=False,
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

        dim_reduce_method: str, default="none"
            Dimensionality reduction method. "pca", "vamp", or "none".

        tau: float
            Resampling time (i.e. time of 1 WE iteration). Used to map fluxes to physical times.

        pcoord_ndim: int, default=1
            Defaults to 1. Dimensionality of progress coordinates.

        auxpath: str, default="coord"
            Augmented coordinates used for MSM construction are stored in west.h5 under auxdata/auxpath

        Returns
        -------
        None


        Todo
        ----
        This should probably just be merged into the constructor.
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

        try:
            self.load_iter_data(1)
            self.load_iter_coordinates0()
            self.coordsExist = True

        # Nothing is raised here because this might be fine, depending on what you're doing.
        except KeyError:
            if not _suppress_boundary_warning:
                log.warning("Model initialized, but coordinates do not exist yet.")
            self.coordsExist = False

        self.use_weights_in_clustering = use_weights_in_clustering

        log.debug("msm_we model successfully initialized")

    @property
    def WEbasisp1_bounds(self):
        return self.basis_pcoord_bounds

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
            log.debug(f"Using Ray cluster with {resources['CPU']} CPUs!")
        except KeyError as e:
            log.error(f"Total cluster resources were {resources}")
            log.error(f"However, available resources are {ray.available_resources()}")
            raise e

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
        allow_validation_failure=False,
        step_kwargs={},
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

        allow_validation_failure: bool, optional (Default = False)
            If True, then raise a warning but don't fail if construction of a cross-validation model fails.

        Returns
        -------

        """

        progress_bar = DefaultProgress()
        table = self.new_table()

        renderable_group = Group(table, progress_bar)

        # Clean up any existing Ray instances
        if use_ray:
            ray.shutdown()

        with Live(
            renderable_group, refresh_per_second=10, auto_refresh=show_live_display
        ) as live:

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

                log.info(f"Initializing Ray cluster with keywords {ray_kwargs}")

                self.do_step(table, step_idx, ray.init, kwargs=ray_kwargs)
                self.set_note(
                    table, step_idx, f"{ray.available_resources()['CPU']} CPUs"
                )

                log.info(f"Initialized Ray with resources {ray.available_resources()}")

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
                    **step_kwargs.get("initialize", {}),
                },
            )
            self.set_note(table, step_idx, "")

            # # Get number of iterations
            step_idx += 1
            self.do_step(
                table,
                step_idx,
                step=model.get_iterations,
            )
            self.set_note(table, step_idx, f"{model.maxIter} iterations exist")

            # # Load coordinates
            step_idx += 1
            _max_coord_iter = [max_coord_iter, model.maxIter][max_coord_iter == -1]
            self.do_step(
                table,
                step_idx,
                step=model.get_coordSet,
                args=[_max_coord_iter],
                kwargs={
                    "progress_bar": progress_bar,
                },
            )
            self.set_note(
                table, step_idx, f"Got coords for {_max_coord_iter} iterations"
            )

            # # Dimensionality reduction
            step_idx += 1
            self.set_note(table, step_idx, f"Method: {model.dimReduceMethod}")
            self.do_step(
                table,
                step_idx,
                step=model.dimReduce,
                kwargs={
                    "progress_bar": progress_bar,
                    **step_kwargs.get("dimReduce", {}),
                },
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
                    "store_validation_model": cross_validation_groups > 0,
                    "progress_bar": progress_bar,
                    **step_kwargs.get("clustering", {}),
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
                    "progress_bar": progress_bar,
                    **step_kwargs.get("fluxmatrix", {}),
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
                kwargs={
                    "use_ray": use_ray,
                    "progress_bar": progress_bar,
                    **step_kwargs.get("organize", {}),
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
            if cross_validation_groups > 0:
                try:
                    self.do_step(
                        table,
                        step_idx,
                        step=model.do_block_validation,
                        kwargs={
                            "cross_validation_groups": cross_validation_groups,
                            "cross_validation_blocks": cross_validation_blocks,
                            "use_ray": use_ray,
                            "progress_bar": progress_bar,
                            **step_kwargs.get("block_validation", {}),
                        },
                    )
                except Exception as e:
                    log.error(e)
                    if not allow_validation_failure:
                        raise e
                    # TODO: Print for the one that does pass
                    self.set_note(
                        table, step_idx, "At least one validation model failed"
                    )
                else:
                    flux_text = ""
                    for group, _validation_model in enumerate(self.validation_models):
                        flux_text += (
                            f"Group {group} flux: {_validation_model.JtargetSS:.2e}\n"
                        )
                    self.set_note(table, step_idx, flux_text)

            # If live updating was disabled, write to the table once now. (Doesn't do anything if it was enabled)
            live.refresh()

    def do_block_validation(
        self,
        cross_validation_groups,
        cross_validation_blocks,
        use_ray=True,
        progress_bar=None,
    ):
        """
        One way to estimate the uncertainty of your model is to split your data into blocks, compute models over
        groups of the blocks, and assess consistency between the groups.

        The procedure here chops your data up into uniform blocks, by iteration. For example, with 100 iterations and 4
        blocks, the blocks consist of iterations `[0-25), [25-50), [50-75), [75-100)`.
        If the above blocks were assigned to 2 groups, the groups would consist of iterations `( [0-25), [50-75) )` and
        `(  [25-50), [75-100) )`

        Parameters
        ----------
        cross_validation_groups: int
            Number of groups to assign blocks over

        cross_validation_blocks: int
            Number of blocks to split data into
        """

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
            range(
                start_idx,
                cross_validation_blocks,
                cross_validation_groups,
            )
            for start_idx in range(cross_validation_groups)
        ]

        validation_iterations = []

        with ProgressBar(progress_bar) as progress_bar:

            task = progress_bar.add_task(description="Block validation", total=2)

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
                        0,
                        iters_to_use=validation_iterations[group],
                        use_ray=use_ray,
                        progress_bar=progress_bar,
                    )

                    # Clean it
                    _model.organize_fluxMatrix(
                        use_ray=use_ray, progress_bar=progress_bar
                    )

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

                progress_bar.advance(task, 1)

        # Store the validation models, in case you want to analyze them.
        self.validation_iterations = validation_iterations
        self.validation_models = validation_models

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
                self.reference_structure = struct
                self.nAtoms = struct.n_atoms
                self.coord_ndim = 3
                return

            elif not topology[-3:] == "pdb":
                log.critical(
                    "Topology is not a recognized type (PDB)! Proceeding, but no guarantees."
                )

            struct = md.load(topology)
            self.reference_structure = struct
            self.reference_coord = np.squeeze(struct._xyz)
            self.nAtoms = struct.topology.n_atoms
            self.coord_ndim = 3
            return

        elif type(topology) in [md.Trajectory, md.Topology]:
            log.debug(
                "Input reference topology  was provided as an mdtraj structure, loading that"
            )

            struct = topology
            self.reference_structure = struct
            self.reference_coord = np.squeeze(struct._xyz)
            self.nAtoms = struct.topology.n_atoms
            self.coord_ndim = 3

        elif type(topology) == dict:

            self.reference_coord = topology["coords"]
            self.nAtoms = topology["nAtoms"]
            self.coord_ndim = topology["coord_ndim"]

        else:
            raise NotImplementedError("Unsupported topology")

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

            self.basis_coords = basis["coords"]

        else:
            raise NotImplementedError("Unsupported topology")
