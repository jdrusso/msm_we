import msm_we.msm_we
import westpa
from westpa.cli.core import w_run
from msm_we import optimization
import numpy as np
import pickle
from rich.progress import Progress
import ray


@ray.remote
class GlobalModelActor:
    def __init__(self, model, processCoordinates):
        msm_we.msm_we.modelWE.processCoordinates = processCoordinates
        self.model = model
        assert hasattr(self.model, 'processCoordinates')

    def get_model(self):
        return self.model


@ray.remote
class PcoordCalculator:
    def __init__(self, model_actor, processCoordinates):
        msm_we.msm_we.modelWE.processCoordinates = processCoordinates
        self.model_actor = model_actor
        self.model = ray.get(self.model_actor.get_model.remote())

    def compute_new_structure_pcoord(self, structure, state_index):
        reduceCoordinates = self.model.reduceCoordinates

        reduced_coords = reduceCoordinates(structure)

        return reduced_coords, state_index


class OptimizationDriver:
    """
    WESTPA plugin to automatically handle performing optimization.

    Using an haMSM, updates binning and allocation according to user-specified optimization algorithms.
    """

    def __init__(self, sim_manager, plugin_config):

        westpa.rc.pstatus("Initializing optimization plugin")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager
        self.we_driver = westpa.rc.get_we_driver

        self.plugin_config = plugin_config

        coord_map_path = plugin_config.get('full_coord_map')
        with open(coord_map_path, 'rb') as infile:
            self.coord_map = pickle.load(infile)

        # Big number is low priority -- this should run before anything else
        self.priority = plugin_config.get('priority', 3)

        sim_manager.register_callback(sim_manager.finalize_run, self.do_optimization, self.priority)

    def do_optimization(self):

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
        propagator = westpa.rc.get_propagator()
        new_pcoord_map = self.compute_new_pcoord_map()
        propagator.pcoord_map = new_pcoord_map

        # 4. Continue WE, with optimized parameters
        # No need to re-initialize/restart, just extend max iterations and continue
        remaining_iters = self.plugin_config.get('max_iters') - self.sim_manager.max_total_iterations
        if remaining_iters > 0:
            new_iters = min(remaining_iters, westpa.rc.config.get(['west', 'propagation', 'max_total_iterations']))
            self.sim_manager.max_total_iterations += new_iters

            westpa.rc.pstatus(f"\n\n=== Applying optimization and continuing for {new_iters} more iterations ===\n")
            w_run.run_simulation()

        else:
            westpa.rc.pstatus("No more iterations for optimization, completing.")

    def compute_optimized_allocation(self, strategy=None):
        """
        Compute the optimal allocation.

        Parameters
        ----------
        strategy

        Returns
        -------
        An array-like holding the optimized WE walker allocation

        TODO
        ----
        Implement actual bin optimization
        """

        new_target_counts = self.we_driver.bin_target_counts
        return new_target_counts

    def compute_optimized_bins(self, strategy=None):
        """

        Parameters
        ----------
        strategy

        Returns
        -------
        An OptimizedBinMapper

        TODO
        ----
        Add flexibility to multiple strategies
        """

        model = self.data_manager.hamsm_model

        n_active_bins = np.count_nonzero(self.we_driver.bin_target_counts)

        discrepancy, variance = optimization.solve_discrepancy(
            tmatrix=model.Tmatrix,
            pi=model.pSS,
            B=model.indTargets
        )

        microstate_assignments = optimization.get_clustered_mfpt_bins(
            variance, discrepancy,
            model.pSS,
            n_active_bins
        )

        microstate_assignments = np.concatenate(
            [microstate_assignments, [n_active_bins - 2, n_active_bins - 1]]
        )

        westpa.rc.pstatus(f"Microstate assignments are {microstate_assignments}")

        # 3. Update binning
        base_mapper = model.clusters.bin_mapper
        n_pcoord_dims = 1

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
            model.clusters
        )

        return we_bin_mapper

    def compute_new_pcoord_map(self):

        model = self.data_manager.hamsm_model
        processCoordinates = self.data_manager.processCoordinates

        new_pcoord_map = {}

        n_actors = int(ray.available_resources().get('CPU', 1))
        model_actor = GlobalModelActor.remote(model, processCoordinates)
        pcoord_calculators = [PcoordCalculator.remote(model_actor, processCoordinates) for i in range(n_actors)]

        msm_we.msm_we.modelWE.check_connect_ray()
        ids = []

        with Progress() as progress:
            submit_task = progress.add_task("Submitting structures for pcoord calculation",
                                            total=len(self.coord_map))
            retrieve_task = progress.add_task(f"Retrieving structure pcoords from {n_actors} workers",
                                              total=len(self.coord_map))

            for state_index, structure in self.coord_map.items():
                _id = pcoord_calculators[
                        state_index % n_actors
                    ].compute_new_structure_pcoord.remote(
                        structure, state_index
                    )
                ids.append(_id)

                progress.advance(submit_task)

            while ids:
                finished, ids = ray.wait(
                    ids, num_returns=min(50, len(ids)), timeout=5
                )
                results = ray.get(finished)

                for pcoord, state_index in results:
                    new_pcoord_map[state_index] = pcoord
                    progress.advance(retrieve_task)

        return new_pcoord_map
