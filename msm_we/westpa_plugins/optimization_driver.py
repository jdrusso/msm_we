"""Plugin for automated WE hyperparameter optimization."""
import msm_we.msm_we
import westpa
from westpa.core import extloader
from westpa.cli.core import w_run
from msm_we import optimization
import numpy as np
import pickle
from rich.progress import Progress
import ray
from synd.westpa.propagator import get_segment_parent_index
from westpa.core.data_manager import create_dataset_from_dsopts


@ray.remote
class GlobalModelActor:
    """
    Ray-parallel Actor that loads a model and holds it in memory. Used by the PcoordCalculator.
    """

    def __init__(self, model, processCoordinates, synd_model, original_pcoord_ndim):
        msm_we.msm_we.modelWE.processCoordinates = processCoordinates
        self.model = model
        assert hasattr(self.model, "processCoordinates")

        self.backmap = synd_model.backmap
        self.original_pcoord_ndim = original_pcoord_ndim

    def get_model(self):
        return self.model

    def get_original_pcoord(self, state_index):
        return self.backmap(state_index)[: self.original_pcoord_ndim]


@ray.remote
class PcoordCalculator:
    """
    Ray-parallel Actor that computes the extended progress coordinate
    (original progress coordinate + dimensionality-reduce MSM features) for a structure.
    """

    def __init__(self, model_actor, processCoordinates):
        msm_we.msm_we.modelWE.processCoordinates = processCoordinates
        self.model_actor = model_actor
        self.model = ray.get(self.model_actor.get_model.remote())

    def compute_new_structure_pcoord(self, structure, state_index):
        reduceCoordinates = self.model.reduceCoordinates

        # Take the zero index, because we're just passing a single structure
        reduced_coords = reduceCoordinates(structure)[0]

        original_pcoord = ray.get(
            self.model_actor.get_original_pcoord.remote(state_index)
        )

        new_pcoord = np.concatenate([original_pcoord, reduced_coords])

        return new_pcoord, state_index


class OptimizationDriver:
    """
    WESTPA plugin to automatically handle performing optimization.

    Using an haMSM, updates binning and allocation according to user-specified optimization algorithms.
    An OptimizedBinMapper is constructed from the optimized binning and allocation, and WE is continued with the new
    mapper.

    Can be used by including the following entries in your west.cfg::

        west:
            plugins:
            - plugin: msm_we.westpa_plugins.optimization_driver.OptimizationDriver
              full_coord_map: A pickled dictionary mapping discrete states to full-coordinate structures
              max_iters: Number of total iterations. WE will run for west.system.max_iters, perform optimization, and
                        continue for another west.system.max_iters, up to this value.
              # The following parameters are optional, and provided as an example.
              binning_strategy: An arbitrary python function defining a bin optimization scheme.
                    Takes in an msm_we.msm_we.modelWE and returns an array-like of length n_msm_states, where each
                    element is the index of the WE bin that MSM state will be assigned to by the OptimizedMapper.
              allocation_strategy: An arbitrary python function defining an allocation optimization scheme.
                    Takes in an msm_we.msm_we.modelWE and returns an array of integer walker allocations for the WE bins.
    """

    def __init__(self, sim_manager, plugin_config):

        westpa.rc.pstatus("Initializing optimization plugin")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager
        self.we_driver = westpa.rc.get_we_driver()
        self.propagator = westpa.rc.get_propagator()

        self.synd_model = westpa.rc.get_propagator().synd_model

        self.plugin_config = plugin_config

        coord_map_path = plugin_config.get("full_coord_map")
        with open(coord_map_path, "rb") as infile:
            self.coord_map = pickle.load(infile)

        # Big number is low priority -- this should run before anything else
        self.priority = plugin_config.get("priority", 3)
        
        self.cluster_on_pcoord = plugin_config.get("cluster_on_pcoord", False)

        self.original_pcoord_dim = westpa.rc.config.get(
            ["west", "system", "system_options", "pcoord_ndim"]
        )

        sim_manager.register_callback(
            sim_manager.finalize_run, self.do_optimization, self.priority
        )

    def do_optimization(self):
        """
        Update WESTPA with an optimized bin mapper, bin allocation, and extend the progress coordinate. Then, continue
        the WE for more iterations.
        """

        # 1. Discrepancy calculation
        westpa.rc.pstatus("Updating bin mapper")
        we_bin_mapper = self.compute_optimized_bins()
        self.we_driver.bin_mapper = we_bin_mapper

        # 2. Update allocation
        # "This is where I'd put my updated allocation... IF I HAD ONE"
        westpa.rc.pstatus("Updating allocation")
        we_allocation = self.compute_optimized_allocation()
        self.we_driver.bin_target_counts = we_allocation

        # 3. Update pcoord
        # TODO: This is SynMD specific -- how can I make extending the progress coordinate generic?
        #   Maybe I could wrap the progress coordinate calculation as the original progress coordinate calculation,
        #   whatever it may be, and then additionally the result of `model.reduceCoordinates` on the full-coord
        #   structure (not sure the best way to get that, it is eventually stored in auxdata)
        westpa.rc.pstatus("Updating pcoord map")
        new_pcoord_map = self.compute_new_pcoord_map()
        self.update_westpa_pcoord(new_pcoord_map)

        # 4. Continue WE, with optimized parameters
        # No need to re-initialize/restart, just extend max iterations and continue
        remaining_iters = (
            self.plugin_config.get("max_iters") - self.sim_manager.max_total_iterations
        )
        if remaining_iters > 0:
            new_iters = min(
                remaining_iters,
                westpa.rc.config.get(["west", "propagation", "max_total_iterations"]),
            )
            self.sim_manager.max_total_iterations += new_iters

            westpa.rc.pstatus(
                f"\n\n=== Applying optimization and continuing for {new_iters} more iterations ===\n"
            )
            w_run.run_simulation()

        else:
            westpa.rc.pstatus("No more iterations for optimization, completing.")

    @staticmethod
    def default_allocation_optimizer(model):
        """A (trivial) example allocation optimization function, which returns an array with the target number of
        walkers in each bin."""

        westpa.rc.pstatus("\tNot updating allocation")
        return westpa.rc.we_driver.bin_target_counts

    def compute_optimized_allocation(self):
        """
        Compute the optimal allocation.

        If `plugin.allocation_strategy` is None or not provided, the allocation is not updated.

        Otherwise, the constructed haMSM is passed to an arbitrary function that returns an array-like describing the
        new walker allocation over the WE bins.
        """

        allocation_strategy = self.plugin_config.get("allocation_strategy", None)

        if allocation_strategy is None:
            westpa.rc.pstatus("\tNot updating allocation")
            allocation_optimizer = self.default_allocation_optimizer
        else:
            westpa.rc.pstatus(f"\tUsing {allocation_strategy} to update allocation")
            allocation_optimizer = extloader.get_object(allocation_strategy)

        new_target_counts = allocation_optimizer(self.data_manager.hamsm_model)

        return new_target_counts

    @staticmethod
    def default_bin_optimizer(model):
        """Example bin optimization function, which assigns microstates to WE bins."""

        n_active_bins = np.count_nonzero(westpa.rc.we_driver.bin_target_counts)

        westpa.rc.pstatus(
            "\tUsing default k-means MFPT optimization (optimization.get_clustered_mfpt_bins) "
            "for bin optimization"
        )

        discrepancy, variance = optimization.solve_discrepancy(
            tmatrix=model.Tmatrix, pi=model.pSS, B=model.indTargets
        )

        microstate_assignments = optimization.get_clustered_mfpt_bins(
            variance, discrepancy, model.pSS, n_active_bins
        )

        return microstate_assignments

    def compute_optimized_bins(self):
        """
        Computes discrepancy and variance, and returns the resulting optimized bin mapper.

        If `plugin.binning_strategy` is None or not provided, :code:`optimization.get_clustered_mfpt_bins()` is used.

        Otherwise, the constructed haMSM is passed to an arbitrary function that returns an array-like with the WE bin
        index of all MSM microbins excluding the basis/target (model.indBasis and model.indTargets).

        Returns
        -------
        An OptimizedBinMapper
        """

        model = self.data_manager.hamsm_model
        binning_strategy = self.plugin_config.get("binning_strategy", None)

        n_active_bins = np.count_nonzero(self.we_driver.bin_target_counts)

        if binning_strategy is None:
            bin_optimizer = self.default_bin_optimizer

        else:
            westpa.rc.pstatus(f"\tUsing {binning_strategy} for bin optimization")
            bin_optimizer = extloader.get_object(binning_strategy)

        microstate_assignments = bin_optimizer(model)

        microstate_assignments = np.concatenate(
            [microstate_assignments, [n_active_bins - 2, n_active_bins - 1]]
        )

        westpa.rc.pstatus(f"\tMicrostate assignments are {microstate_assignments}")

        # 3. Update binning
        base_mapper = model.clusters.bin_mapper

        n_pcoord_dims = self.original_pcoord_dim

        we_bin_mapper = optimization.OptimizedBinMapper(
            n_active_bins,
            # In case the pcoord is extended, this is the original pcoord dimensionality
            n_pcoord_dims,
            # If the pcoord was extended, pcoord boundaries are in the original pcoord space
            model.basis_pcoord_bounds,
            model.target_pcoord_bounds,
            # The original, non-Optimized BinMapper that WESTPA was run with.
            #   Used for stratified clustering
            base_mapper,
            microstate_assignments,
            model.clusters,
            cluster_on_pcoord = self.cluster_on_pcoord
        )

        return we_bin_mapper

    def compute_new_pcoord_map(self):
        """
        SynD specific: Compute a new progress coordinate mapping.

        Returns
        -------
        A dictionary of {state indices : extended progress coordinates}
        """

        model = self.data_manager.hamsm_model
        processCoordinates = self.data_manager.processCoordinates

        new_pcoord_map = {}

        # TODO: is this robust if you don't already have a ray cluster started?
        n_actors = int(ray.available_resources().get("CPU", 1))
        model_actor = GlobalModelActor.remote(
            model, processCoordinates, self.synd_model, self.original_pcoord_dim
        )
        pcoord_calculators = [
            PcoordCalculator.remote(model_actor, processCoordinates)
            for i in range(n_actors)
        ]

        msm_we.msm_we.modelWE.check_connect_ray()
        ids = []

        with Progress() as progress:
            submit_task = progress.add_task(
                "Submitting structures for pcoord calculation",
                total=len(self.coord_map),
            )
            retrieve_task = progress.add_task(
                f"Retrieving structure pcoords from {n_actors} workers",
                total=len(self.coord_map),
            )

            for state_index, structure in self.coord_map.items():
                _id = pcoord_calculators[
                    state_index % n_actors
                ].compute_new_structure_pcoord.remote(structure, state_index)
                ids.append(_id)

                progress.advance(submit_task)

            while ids:
                finished, ids = ray.wait(ids, num_returns=min(50, len(ids)), timeout=5)
                results = ray.get(finished)

                for pcoord, state_index in results:
                    new_pcoord_map[state_index] = pcoord
                    progress.advance(retrieve_task)

        return new_pcoord_map

    def update_westpa_pcoord(self, new_pcoord_map):
        """
        Changing a progress coordinate during a WE run requires a number of changes in WESTPA's internal state.
        This handles making those, so you can call w_run and continue with the new, changed pcoord

        Parameters
        ----------
        new_pcoord_map: A dictionary mapping discrete states to the new, extended pcoord
        """

        # TODO: Replace this with propagator.get_pcoord
        self.propagator.synd_model._backmappers["default"] = new_pcoord_map.get
        new_pcoord_dim = new_pcoord_map.get(0).shape[0]

        westpa.rc.pstatus(f"New pcoord dimensionality is {new_pcoord_dim}")

        system = westpa.rc.get_system_driver()
        data_manager = westpa.rc.get_data_manager()
        sim_manager = westpa.rc.get_sim_manager()

        # This function causes problems when extending pcoords, and isn't actually used for anything, so just skip it
        data_manager.get_new_weight_data = lambda x: None

        #  # Update system driver
        system.pcoord_ndim = new_pcoord_dim

        #  # Update the pcoord dataset in west.h5
        data_manager.open_backing()

        iter_group = data_manager.get_iter_group(sim_manager.n_iter)
        n_iter = sim_manager.n_iter
        segments = data_manager.get_segments(
            n_iter=sim_manager.n_iter, load_pcoords=True
        )

        del data_manager.we_h5file[
            "/iterations/iter_{:0{prec}d}/pcoord".format(
                int(n_iter), prec=data_manager.iter_prec
            )
        ]
        data_manager.flush_backing()

        pcoord_opts = data_manager.dataset_options.get(
            "pcoord",
            {"name": "pcoord", "h5path": "pcoord", "compression": False},
        )

        #  # Update the currently held segments
        westpa.rc.pstatus(f"Attempting to fetch segments for iter {sim_manager.n_iter}")
        for segment in segments:

            # TODO: This is SynD specific, but should be easy to port over to something generic.
            #       Use propagator.get_pcoord directly
            parent_state_index = get_segment_parent_index(segment)

            segment.pcoord = np.concatenate(
                [
                    [new_pcoord_map[parent_state_index]],
                    np.zeros(shape=(system.pcoord_len - 1, system.pcoord_ndim)),
                ]
            )

        create_dataset_from_dsopts(
            iter_group,
            pcoord_opts,
            data=np.array([segment.pcoord for segment in segments]),
            shape=(
                len(sim_manager.segments),
                system.pcoord_len,
                system.pcoord_ndim,
            ),
            dtype=system.pcoord_dtype,
        )

        data_manager.update_segments(sim_manager.n_iter, segments)

        # The initial states that were computed for the next iteration are using the old-style pcoord,
        #   so we can't bin them correctly with the OptimizedBinMapper unless we update them.
        # Although the old istates won't be used if you have gen_istate enabled, we still have to
        #   recreate the dataset. Otherwise, it'll be the wrong shape for the new istates.
        initial_states = data_manager.get_initial_states(sim_manager.n_iter)
        n_initial_states = len(initial_states)

        del data_manager.we_h5file[
            "/iterations/iter_{:0{prec}d}/ibstates/istate_pcoord".format(
                int(n_iter), prec=data_manager.iter_prec
            )
        ]
        del data_manager.we_h5file[
            "/iterations/iter_{:0{prec}d}/ibstates/istate_index".format(
                int(n_iter), prec=data_manager.iter_prec
            )
        ]

        new_istates = data_manager.create_initial_states(
            n_states=n_initial_states, n_iter=n_iter
        )

        for old_istate, new_istate in zip(initial_states, new_istates):
            new_istate = old_istate
            bstate_id = old_istate.basis_state_id
            parent_state_index = int(sim_manager.next_iter_bstates[bstate_id].auxref)
            new_istate.pcoord = new_pcoord_map[parent_state_index]

        data_manager.flush_backing()

        for segment in sim_manager.segments.values():
            parent_state = get_segment_parent_index(segment)
            segment.pcoord = new_pcoord_map[parent_state]

        data_manager.flush_backing()
        data_manager.close_backing()
