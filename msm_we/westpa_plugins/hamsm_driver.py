"""Plugin for automated haMSM construction."""
import westpa
from westpa.core import extloader
from msm_we import msm_we
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


class HAMSMDriver:
    """
    WESTPA plugin to construct an haMSM.
    """

    def __init__(self, sim_manager, plugin_config):

        westpa.rc.pstatus("Initializing haMSM plugin")

        assert (
            _openmp_effective_n_threads() == 1
        ), "Set $OMP_NUM_THREADS=1 for proper msm-we functionality"

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager

        self.plugin_config = plugin_config

        # Big number is low priority -- this should run after augmentation, but before other things
        self.priority = plugin_config.get("priority", 2)

        sim_manager.register_callback(
            sim_manager.finalize_run, self.construct_hamsm, self.priority
        )

        self.h5file_paths = [self.data_manager.we_h5filename]

    def construct_hamsm(self):
        """
        Build an haMSM, for use with later plugins. The final constructed haMSM is stored on the data manager.
        """

        self.data_manager.hamsm_model = None

        # TODO: refPDBfile should no longer be necessary (or used anywhere) and should be safe to remove
        refPDBfile = self.plugin_config.get("ref_pdb_file")
        model_name = self.plugin_config.get("model_name")
        clusters_per_stratum = self.plugin_config.get("n_clusters")

        target_pcoord_bounds = self.plugin_config.get("target_pcoord_bounds")
        basis_pcoord_bounds = self.plugin_config.get("basis_pcoord_bounds")

        dimreduce_method = self.plugin_config.get("dimreduce_method", None)
        tau = self.plugin_config.get("tau", None)

        featurization_module = self.plugin_config.get("featurization")
        featurizer = extloader.get_object(featurization_module)
        msm_we.modelWE.processCoordinates = featurizer
        self.data_manager.processCoordinates = featurizer

        self.data_manager.close_backing()

        ray_kwargs = {"num_cpus": self.plugin_config.get("num_cpus", None)}

        model = msm_we.modelWE()
        model.build_analyze_model(
            file_paths=self.h5file_paths,
            ref_struct=refPDBfile,
            modelName=model_name,
            basis_pcoord_bounds=basis_pcoord_bounds,
            target_pcoord_bounds=target_pcoord_bounds,
            dimreduce_method=dimreduce_method,
            n_clusters=clusters_per_stratum,
            tau=tau,
            ray_kwargs=ray_kwargs,
            step_kwargs={},
            # For some reason if I don't specify fluxmatrix_iters, after the first time around
            # it'll keep using the arguments from the first time...
            # That's really alarming?
            fluxmatrix_iters=[1,-1],
            allow_validation_failure=True  # Don't fail if a validation model fails
        )

        westpa.rc.pstatus(f"Storing built haMSM on {self.data_manager}")
        self.data_manager.hamsm_model = model

        return model