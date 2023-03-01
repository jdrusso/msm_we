import h5py
import logging
import operator
import numpy as np
import hashlib

import westpa
from westpa.cli.core import w_init
from westpa.cli.core import w_run
from westpa.core.extloader import get_object
from westpa.core.segment import Segment
from westpa import analysis

import json

import os
import shutil
import pickle
from msm_we.westpa_plugins.hamsm_driver import HAMSMDriver

import tqdm

import mdtraj as md
from rich.logging import RichHandler

from matplotlib import pyplot as plt

from copy import deepcopy
import time
import re

EPS = np.finfo(np.float64).eps

log = logging.getLogger(__name__)
log.setLevel("INFO")
log.propagate = False
log.addHandler(RichHandler())

msm_we_logger = logging.getLogger("msm_we.msm_we")
msm_we_logger.setLevel("INFO")

# Map structure types to extensions.
# This tells the plugin what extension to put on generated start-state files.
STRUCT_EXTENSIONS = {
    md.formats.PDBTrajectoryFile: "pdb",
    md.formats.AmberRestartFile: "rst7",
}

EXTENSION_LOCKFILE = "doing_extension"


def check_target_reached(h5_filename):
    """
    Check if the target state was reached, given the data in a WEST H5 file.

    Parameters
    ----------
    h5_filename: string
        Path to a WESTPA HDF5 data file
    """
    with h5py.File(h5_filename, "r") as h5_file:
        # Get the key to the final iteration. Need to do -2 instead of -1 because there's an empty-ish final iteration
        #   written.
        for iteration_key in list(h5_file["iterations"].keys())[-2:0:-1]:
            endpoint_types = h5_file[f"iterations/{iteration_key}/seg_index"][
                "endpoint_type"
            ]
            if Segment.SEG_ENDPOINT_RECYCLED in endpoint_types:
                log.debug(
                    f"recycled segment found in file {h5_filename} at iteration {iteration_key}"
                )
                return True
    return False


def fix_deprecated_initialization(initialization_state):
    """
    I changed my initialization JSON schema to use underscores instead of hyphens so I can directly expand it into
    keywords arguments to w_init. This just handles any old-style JSON files I still had, so they don't choke and die.
    """

    log.debug(f"Starting processing, dict is now {initialization_state}")

    # Some of my initial files had this old-style formatting. Handle it for now, but remove eventually
    for old_key, new_key in [
        ("tstate-file", "tstate_file"),
        ("bstate-file", "bstate_file"),
        ("sstate-file", "sstate_file"),
        ("segs-per-state", "segs_per_state"),
    ]:
        if old_key in initialization_state.keys():
            log.warning(
                f"This initialization JSON file uses the deprecated  "
                f"hyphenated form for  {old_key}. Replace with underscores."
            )

            value = initialization_state.pop(old_key)
            initialization_state[new_key] = value

    log.debug(f"Finished processing, dict is now {initialization_state}")
    return initialization_state


class RestartDriver(HAMSMDriver):
    """
    WESTPA plugin to automatically handle estimating steady-state from a WE run, re-initializing a new WE run in that
    steady-state, and then running that initialized WE run.

    Data from the previous run will be stored in the restart<restart_number>/ subdirectory of $WEST_SIM_ROOT.

    This plugin depends on having the start-states implementation in the main WESTPA code, which allows initializing
    a WE run using states that are NOT later used for recycling.

    These are used so that when the new WE run is initialized, initial structure selection is chosen by w_init, using
    weights assigned to the start-states based on MSM bin weight and WE segment weight.

    Since it closes out the current WE run and starts a new one, this plugin should run LAST, after all other plugins.


    Can be used by including the following entries in your west.cfg::

        west:
            plugins:
            # - plugin: An augmentation plugin is also required, such as
            #           msm_we.westpa_plugins.augmentation_driver.MDAugmentationDriver
            - plugin: msm_we.westpa_plugins.restart_driver.RestartDriver
                  n_restarts: Number of total restarts to do
                  extension_iters: Amount of iterations to extend runs by if no runs have reached the target by the
                    first restart.
                  n_runs: Number of runs to do between restarts
                  n_restarts_to_use: Number of restarts to use. Can be a fraction, to use the last fraction amount, or
                    a negative integer, to use the last N.
                  initialization_file: restart_initialization.json
                  model_name: Name for the model
                  n_clusters: Number of clusters to place in each WE bin (see stratified clustering for more details)
                  basis_pcoord_bounds: [[pcoord dim 0 lower bound, upper bound], [pcoord dim 1 lower, upper], ...]
                  target_pcoord_bounds: [[pcoord dim 0 lower bound, upper bound], [pcoord dim 1 lower, upper], ...]
                  dim_reduce_method: A string specifying a dimensionality reduction method for
                    :meth:`msm_we.msm_we.modelWE.dimReduce`
                  featurization: A python module implementing a featurization
                    for msm_we.msm_we.modelWE.processCoordinates
                  n_cpus: Number of CPUs to use with Ray

    TODO
    ----
    The multi-simulation management functionality of this plugin should really be broken out into a separate,
    multi_sim_manager plugin.
    """

    def __init__(self, sim_manager, plugin_config):
        """
        Initialize the RestartDriver plugin.

        Pulls the data_manager and sim_manager from the WESTPA run that just completed, along with
        """

        # Use the
        super().__init__(sim_manager, plugin_config)
        westpa.rc.pstatus(self.sim_manager._callback_table)
        self.sim_manager._callback_table[sim_manager.finalize_run].remove(
            (2, "construct_hamsm", super().construct_hamsm)
        )

        westpa.rc.pstatus("Restart plugin initialized")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Reweighting not master, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager

        self.plugin_config = plugin_config

        self.restart_file = plugin_config.get("restart_file", "restart.dat")
        self.initialization_file = plugin_config.get(
            "initialization_file", "restart_initialization.json"
        )

        self.extension_iters = plugin_config.get("extension_iters", 0)
        self.max_total_iterations = westpa.rc.config.get(
            ["west", "propagation", "max_total_iterations"], default=None
        )
        self.base_total_iterations = self.max_total_iterations

        self.n_restarts = plugin_config.get("n_restarts", -1)
        self.n_runs = plugin_config.get("n_runs", 1)

        # May want to be able to disable this, if it causes issues with recalculating new pcoords
        #   (i.e. during optimization)
        self.cache_pcoords = plugin_config.get("cache_pcoords", True)

        # .get() might return this as a bool anyways, but be safe
        self.debug = bool(plugin_config.get("debug", False))
        if self.debug:
            log.setLevel("DEBUG")
            msm_we_logger.setLevel("DEBUG")

        # Default to using all restarts
        self.restarts_to_use = plugin_config.get("n_restarts_to_use", self.n_restarts)
        assert (
            self.restarts_to_use > 0 or self.restarts_to_use == -1
        ), "Invalid number of restarts to use"
        if self.restarts_to_use >= 1:
            assert (
                self.restarts_to_use == self.restarts_to_use // 1
            ), "If choosing a decimal restarts_to_use, must be between 0 and 1."

        struct_filetype = plugin_config.get(
            "struct_filetype", "mdtraj.formats.PDBTrajectoryFile"
        )
        self.struct_filetype = get_object(struct_filetype)

        # This should be low priority, because it closes the H5 file and starts a new WE run. So it should run LAST
        #   after any other plugins.
        self.priority = plugin_config.get(
            "priority", 100
        )  # I think a big number is lower priority...

        sim_manager.register_callback(
            sim_manager.finalize_run, self.prepare_new_we, self.priority
        )

        # If it's being used with SynD, then we need to reference discrete state IDs instead of structure files in our
        #   start-state definitions.
        # TODO: Find a more efficient way to do this than explicitly constructing the inverse dictionary.
        self.synd_full_coord_map_path = plugin_config.get(
            "synd_full_coord_map_path", None
        )
        if self.synd_full_coord_map_path is not None:
            with open(self.synd_full_coord_map_path, "rb") as infile:
                self.synd_full_coord_map = pickle.load(infile)

        self.pcoord_cache = None
        self.model = None

        # This is the base path to the west.h5 files read in the analysis
        # By explicitly setting this, we can be flexible to having Ray workers on other filesystems,
        #   as long as this path can be matched.
        # We don't assume that each Ray worker is already in the correct working directory (which they
        #   won't be, if they're running on a remote)
        self.base_path = plugin_config.get(
            "base_path",
            os.getcwd(),
        )
        log.info(f"Setting base path to {self.base_path}")

    def get_original_bins(self):
        """
        Obtains the WE bins and their probabilities at the end of the previous iteration.

        Returns
        -------
        bins : np.ndarray
            Array of WE bins

        binprobs: np.ndarray
            WE bin weights
        """

        we_driver = self.sim_manager.we_driver
        bins = we_driver.next_iter_binning
        n_bins = len(bins)
        binprobs = np.fromiter(
            map(operator.attrgetter("weight"), bins), dtype=np.float64, count=n_bins
        )

        return bins, binprobs

    @property
    def cur_iter(self):
        """
        Get the current WE iteration.

        Returns
        -------
        int: The current iteration. Subtract one, because in finalize_run the iter has been incremented
        """
        return self.sim_manager.n_iter - 1

    @property
    def is_last_iteration(self):
        """
        Get whether this is, or is past, the last iteration in this WE run.

        Returns
        -------
        bool: Whether the current iteration is the final iteration
        """

        final_iter = self.sim_manager.max_total_iterations

        return self.cur_iter >= final_iter

    def prepare_extension_run(self, run_number, restart_state, first_extension=False):
        """
        Copy the necessary files for an extension run  (versus initializing a fresh run)

        Parameters
        ----------
        run_number: int
            The index of this run (should be 1-indexed!)

        restart_state: dict
            Dictionary holding the current state of the restarting procedure

        first_extension: bool
            True if this is the first run of an extension set. If True, then back up west.cfg, and write the extended
            west.cfg file.
        """

        log.debug(f"Linking run files from restart0/run{run_number}")

        # Copy traj_segs, seg_logs, and west.h5 for restart0/runXX back into ./
        #       Later: (May only need to copy the latest iteration traj_segs, to avoid tons of back and forth)
        try:
            shutil.rmtree("traj_segs")
            shutil.rmtree("seg_logs")
        except OSError as e:
            if str(e) == "Cannot call rmtree on a symbolic link":
                os.unlink("traj_segs")
                os.unlink("seg_logs")

        os.remove(self.data_manager.we_h5filename)

        os.symlink(f"restart0/run{run_number}/traj_segs", "traj_segs")
        os.symlink(f"restart0/run{run_number}/seg_logs", "seg_logs")

        if first_extension:

            # Get lines to make a new west.cfg by extending west.propagation.max_total_iterations
            with open("west.cfg", "r") as west_config:
                lines = west_config.readlines()
                for i, line in enumerate(lines):
                    # Parse out the number of maximum iterations
                    if "max_total_iterations" in line:
                        max_iters = [
                            int(i)
                            for i in line.replace(":", " ").replace("\n", " ").split()
                            if i.isdigit()
                        ]
                        new_max_iters = max_iters[0] + self.extension_iters
                        new_line = f"{line.split(':')[0]}: {new_max_iters}\n"
                        lines[i] = new_line
                        break

        with open(self.restart_file, "w") as fp:
            json.dump(restart_state, fp)

        log.info("First WE extension run ready!")
        westpa.rc.pstatus(
            f"\n\n===== Restart {restart_state['restarts_completed']}, "
            + f"Run {restart_state['runs_completed'] + 1} extension running =====\n"
        )

        # TODO: I can't just go straight into a w_run here. w_run expects some things to be set I think, that aren't.
        #   I can do w_init, and do a new simulation just fine...
        #   I can do this on ONE run repeatedly just fine
        #   But if I try to just copy files and continue like this, there's something screwy in state somewhere that
        #       causes it to fail.
        #   The error has to do with offsets in the HDF5 file?
        #   Need to figure out what state would be cleared by w_init

        # Frankly, this is a really sketchy way of doing this, but it seems to work...
        #   I remain skeptical there's not something weird under the hood that isn't being addressed correctly with
        #   regard to state, but if it works, it's good enough for now..
        westpa.rc.sim_manager.segments = None
        shutil.copy(
            f"restart0/run{run_number}/west.h5", self.data_manager.we_h5filename
        )
        self.data_manager.open_backing()

        log.debug(f"Sim manager thought n_iter was {westpa.rc.sim_manager.n_iter}")
        log.debug(
            f"Data manager thought current_iteration was {self.data_manager.current_iteration}"
        )
        log.debug(f"{self.sim_manager} vs {westpa.rc.sim_manager}")

        if run_number == 1:
            westpa.rc.sim_manager.max_total_iterations += self.extension_iters

        w_run.run_simulation()
        return

    def generate_plots(self, restart_directory):

        model = self.model

        log.info(
            "Producing flux-profile, pseudocommittor, and target flux comparison plots."
        )
        flux_pcoord_fig, flux_pcoord_ax = plt.subplots()
        model.plot_flux(ax=flux_pcoord_ax, suppress_validation=True)
        flux_pcoord_fig.text(
            x=0.5,
            y=-0.15,
            ha="center",
            va="center",
            s="This flux profile should become flatter after restarting",
            fontsize=12,
            transform=flux_pcoord_ax.transAxes,
        )
        flux_pcoord_ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
        flux_pcoord_fig.savefig(
            f"{restart_directory}/flux_plot.pdf", bbox_inches="tight"
        )

        flux_pseudocomm_fig, flux_pseudocomm_ax = plt.subplots()
        model.plot_flux_committor(ax=flux_pseudocomm_ax, suppress_validation=True)
        flux_pseudocomm_fig.text(
            x=0.5,
            y=-0.15,
            ha="center",
            va="center",
            s="This flux profile should become flatter after restarting."
            '\nThe x-axis is a "pseudo"committor, since it may be '
            "calculated from WE trajectories in the one-way ensemble.",
            fontsize=12,
            transform=flux_pseudocomm_ax.transAxes,
        )
        flux_pseudocomm_ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
        flux_pseudocomm_fig.savefig(
            f"{restart_directory}/pseudocomm-flux_plot.pdf", bbox_inches="tight"
        )

        flux_comparison_fig, flux_comparison_ax = plt.subplots(figsize=(7, 3))
        # Get haMSM flux estimates
        models = [model]
        models.extend(model.validation_models)
        n_validation_models = len(model.validation_models)

        flux_estimates = []
        for _model in models:
            flux_estimates.append(_model.JtargetSS)

        hamsm_flux_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        direct_flux_colors = iter(
            plt.cm.cool(np.linspace(0.2, 0.8, len(model.fileList)))
        )

        # Get WE direct flux estimate
        for _file in model.fileList:

            run = analysis.Run(_file)
            last_iter = run.num_iterations
            recycled = list(run.iteration(last_iter - 1).recycled_walkers)
            target_flux = sum(walker.weight for walker in recycled) / model.tau

            # TODO: Correct for time!
            if len(_file) >= 15:
                short_filename = f"....{_file[-12:]}"
            else:
                short_filename = _file

            if target_flux == 0:
                continue

            flux_comparison_ax.axhline(
                target_flux,
                color=next(direct_flux_colors),
                label=f"Last iter WE direct {target_flux:.2e}"
                f"\n  ({short_filename})",
                linestyle="--",
            )

        flux_comparison_ax.axhline(
            flux_estimates[0],
            label=f"Main model estimate\n  {flux_estimates[0]:.2e}",
            color=next(hamsm_flux_colors),
        )
        for i in range(1, n_validation_models + 1):
            flux_comparison_ax.axhline(
                flux_estimates[i],
                label=f"Validation model {i - 1} estimate\n  {flux_estimates[i]:.2e}",
                color=next(hamsm_flux_colors),
            )

        flux_comparison_ax.legend(bbox_to_anchor=(1.01, 0.9), loc="upper left")
        flux_comparison_ax.set_yscale("log")
        flux_comparison_ax.set_ylabel("Flux")
        flux_comparison_ax.set_xticks([])
        flux_comparison_fig.tight_layout()
        flux_comparison_fig.savefig(
            f"{restart_directory}/hamsm_vs_direct_flux_comparison_plot.pdf",
            bbox_inches="tight",
        )

        coarse_flux_fig, ax = model.plot_coarse_flux_profile()
        coarse_flux_fig.savefig(
            f"{restart_directory}/coarse_flux_profile.pdf",
            bbox_inches="tight",
        )

    def init_we(self, initialization_state, pcoord_cache):

        start_time = time.perf_counter()

        original_get_pcoord = None

        if pcoord_cache is not None:

            log.info("Enabling pcoord cache for new WE run initialization")
            propagator = westpa.rc.propagator
            original_get_pcoord = propagator.get_pcoord

            def get_cached_pcoord(state):
                """
                For the cached pcoords, I'll be getting a bunch of istate/bstate/sstates, and I need to
                    map them to some cached pcoord values.

                At this point, my start states are just basis states (not initial states)... Because of that,
                in order to determine if it's a start state or an "actual" basis state, we can check if the label\
                matches the bX_sY format we use below.

                This is a little janky and fragile.

                This function is defined inline because it needs to take only the state argument, and have access to
                the cache.
                TODO: That could also be done by just adding those as attributes to the state.
                """

                # If it IS a start-state, then retrieve the pcoord from the cache
                label = state.label

                template = re.compile(r"^b(\d+)_s(\d+)$")
                is_start_state = template.match(label)

                if is_start_state:

                    # This is NOT the "segment index" as WESTPA describes it -- it's the index of this structure
                    #   among structures in this cluster.
                    cluster_idx, cluster_seg_idx = re.findall(r"\d+", state.label)
                    cluster_idx = int(cluster_idx)
                    cluster_seg_idx = int(cluster_seg_idx)

                    state.pcoord = pcoord_cache[int(cluster_idx)][int(cluster_seg_idx)]

                # If it's not a start state, then apply the normal pcoord calculation
                else:
                    log.debug(f"Not using cache for state {state}")
                    original_get_pcoord(state)

            propagator.get_pcoord = get_cached_pcoord

        w_init.initialize(
            **initialization_state,
            shotgun=False,
        )

        if pcoord_cache is not None:
            propagator.get_pcoord = original_get_pcoord

        end_time = time.perf_counter()
        log.debug(f"Runtime of w_init was {end_time - start_time:.2f} seconds")

    def prepare_new_we(self):
        """
        This function prepares a new WESTPA simulation using haMSM analysis to accelerate convergence.

        The marathon functionality does re-implement some of the functionality of w_multi_west.
        However, w_multi_west merges independent WE simulations, which may or may not be desirable.
        I think for the purposes of this, it's good to keep the runs completely independent until haMSM model building.
        Either that, or I'm just justifying not having known about w_multi_west when I wrote this. TBD.

        The algorithm is as follows:

            1. Check to see if we've just completed the final iteration

            2. Handle launching multiple runs, if desired

            3. Build haMSM

            4. Obtain structures for each haMSM bin

            5. Make each structure a start-state, with probability set by (MSM-bin SS prob / # structures in bin)

            6. Potentially some renormalization?

            7. Start new WE simulation

        TODO
        ----
        Replace all manual path-building with pathlib

        """

        # Do nothing if it's not the final iteration
        if not self.is_last_iteration:
            print(self.cur_iter)
            return

        log.debug("Final iteration, preparing restart")

        restart_state = {"restarts_completed": 0, "runs_completed": 0}

        # Check for the existence of the extension lockfile here
        doing_extension = os.path.exists(EXTENSION_LOCKFILE)

        # Look for a restart.dat file to get the current state (how many restarts have been performed already)
        if os.path.exists(self.restart_file):
            with open(self.restart_file, "r") as fp:
                restart_state = json.load(fp)

        # This is the final iteration of a run, so mark this run as completed
        restart_state["runs_completed"] += 1

        # Make the folder to store data for this marathon
        restart_directory = f"restart{restart_state['restarts_completed']}"
        run_directory = f"{restart_directory}/run{restart_state['runs_completed']}"
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        # Write coordinates to h5
        # prepare_coordinates(self.plugin_config, self.data_manager.we_h5file, self.data_manager.we_h5filename)

        for data_folder in ["traj_segs", "seg_logs"]:
            old_path = data_folder

            # If you're doing an extension, this will be a symlink. So no need to copy, just unlink it and move on
            # TODO: This overwrite is really obnoxious if you're restarting after a crash during haMSM building.
            #   At that point, traj_segs will be empty, restartXX/runYY/traj_segs will be populated, and restart.dat
            #   will indicate the restart is not yet completed. So when you try to start it after the crash, it
            #   clobbers restartXX/runYY/traj_segs with the empty traj_segs.
            if doing_extension and os.path.islink(old_path):
                log.debug("Unlinking symlink")
                os.unlink(old_path)
                os.mkdir(old_path)
                continue

            new_path = f"{run_directory}/{old_path}"

            log.debug(f"Moving {old_path} to {new_path}")

            if os.path.exists(new_path):
                log.info(f"{new_path} already exists. Removing and overwriting.")
                shutil.rmtree(new_path)

            try:
                os.rename(old_path, new_path)
            except FileNotFoundError:
                log.warning(
                    f"Folder {old_path} was not found."
                    "This may be normal, but check your configuration."
                )
            else:
                # Make a new data folder for the next run
                os.mkdir(old_path)

        last_run = restart_state["runs_completed"] >= self.n_runs
        last_restart = restart_state["restarts_completed"] >= self.n_restarts

        # We've just finished a run. Let's check if we have to do any more runs in this marathon before doing a restart.
        #   In the case of n_runs == 1, then we're just doing a single run and restarting it every so often.
        #   Otherwise, a marathon consists of multiple runs,  and restarts are performed between marathons.
        if last_run:
            log.info(f"All {self.n_runs} runs in this marathon completed.")

            if last_restart:
                log.info("All restarts completed! Performing final analysis.")

            else:
                log.info("Proceeding to prepare a restart.")

                # Duplicating this is  gross, but given the structure here, my options are either put it above these ifs
                #   entirely, meaning it'll be unnecessarily run at the end of the final restart, or duplicate it below.
                log.info("Preparing coordinates for this run.")

                # Now, continue on to haMSM calculation below.

        # If we have more runs left to do in this marathon, prepare them
        elif not last_run:

            log.info(f"Run {restart_state['runs_completed']}/{self.n_runs} completed.")

            # TODO: Initialize a new run, from the same configuration as this run was
            #   On the 1st run, I can write bstates/tstates/sstates into restart files, and use those for spawning
            #   subsequent runs in the marathon. That way, I don't make unnecessary copies of all those.
            # Basis and target states are unchanged. Can I get the original parameters passed to w_init?
            # Ideally, I should be able to call w_init with the exact same parameters that went to it the first time
            initialization_state = {
                "tstate_file": None,
                "bstate_file": None,
                "sstate_file": None,
                "tstates": None,
                "bstates": None,
                "sstates": None,
                "segs_per_state": None,
            }

            # TODO: Implement this, and get rid of the initialization_file usage right below. Placeholder for now.
            if restart_state["runs_completed"] == 1:

                # Get and write basis, target, start states and segs per state for this marathon to disk
                pass

            # Save the WESTPA h5 data from this run
            self.data_manager.finalize_run()
            shutil.copyfile("west.h5", f"{run_directory}/west.h5")

            # If this is a regular, fresh run (not an extension)
            if not doing_extension:
                if os.path.exists(self.initialization_file):
                    with open(self.initialization_file, "r") as fp:
                        initialization_dict = json.load(fp)
                        initialization_dict = fix_deprecated_initialization(
                            initialization_dict
                        )
                        initialization_state.update(initialization_dict)
                else:
                    raise Exception(
                        "No initialization JSON file provided -- "
                        "I don't know how to start new runs in this marathon."
                    )

                westpa.rc.pstatus(
                    f"\n\n===== Restart {restart_state['restarts_completed']}, "
                    + f"Run {restart_state['runs_completed']+1} initializing =====\n"
                )

                westpa.rc.pstatus(initialization_state)

                westpa.rc.pstatus(
                    f"\nRun: \n\t w_init --tstate-file {initialization_state['tstate_file']} "
                    + f"--bstate-file {initialization_state['bstate_file']} "
                    f"--sstate-file {initialization_state['sstate_file']} "
                    f"--segs-per-state {initialization_state['segs_per_state']}\n"
                )

                self.init_we(initialization_state, self.pcoord_cache)

                with open(self.restart_file, "w") as fp:
                    json.dump(restart_state, fp)

                log.info("New WE run ready!")
                westpa.rc.pstatus(
                    f"\n\n===== Restart {restart_state['restarts_completed']}, "
                    + f"Run {restart_state['runs_completed']+1} running =====\n"
                )

                w_run.run_simulation()
                return

            # If we're doing an extension set
            #   Instead of w_initting a new iteration, copy the files from restart0/runXX back into ./
            elif doing_extension:

                self.prepare_extension_run(
                    run_number=restart_state["runs_completed"] + 1,
                    restart_state=restart_state,
                )
                return

        log.debug(
            f"{restart_state['restarts_completed']}/{self.n_restarts} restarts completed"
        )

        # Build the haMSM
        log.debug("Initializing haMSM")

        # Need to write the h5 file and close it out, but I need to get the current bstates first.
        original_bstates = self.sim_manager.current_iter_bstates
        if original_bstates is None:
            original_bstates = self.data_manager.get_basis_states(
                self.sim_manager.n_iter - 1
            )

        assert original_bstates is not None, "Bstates are none in the current iteration"

        original_tstates = self.data_manager.get_target_states(self.cur_iter)

        # Flush h5 file writes and copy it to the run directory
        self.data_manager.finalize_run()
        shutil.copyfile(self.data_manager.we_h5filename, f"{run_directory}/west.h5")

        # Use all files in all restarts
        # Restarts index at 0, because there's  a 0th restart before you've... restarted anything.
        # Runs index at 1, because Run 1 is the first run.
        # TODO: Let the user pick last half or something in the plugin config.
        marathon_west_files = []
        # When doing the first restart, restarts_completed is 0 (because the first restart isn't complete yet) and
        #   the data generated during this restart is in /restart0.
        # So when doing the Nth restart, restarts_completed is N-1

        # If set to -1, use all restarts
        if self.restarts_to_use == -1:
            last_N_restarts = 1 + restart_state["restarts_completed"]
        # If this is an integer, use the last N restarts
        elif self.restarts_to_use >= 1:
            last_N_restarts = self.restarts_to_use
        # If it's a decimal between 0 and 1, use it as a fraction
        # At restart 1, and a fraction of 0.5, this should just use restart 1
        elif 0 < self.restarts_to_use < 1:
            last_N_restarts = int(
                self.restarts_to_use * (1 + restart_state["restarts_completed"])
            )

            # If this fraction is <1, use all until it's not
            if last_N_restarts < 1:
                last_N_restarts = 1 + restart_state["restarts_completed"]

        log.debug(f"Last N is {last_N_restarts}")
        first_restart = max(
            1 + restart_state["restarts_completed"] - last_N_restarts, 0
        )
        usable_restarts = range(first_restart, 1 + restart_state["restarts_completed"])

        log.info(
            f"At restart {restart_state['restarts_completed']}, building haMSM using data from restarts {list(usable_restarts)}"
        )
        for restart_number in usable_restarts:
            for run_number in range(1, 1 + restart_state["runs_completed"]):

                west_file_path = (
                    f"{self.base_path}/restart{restart_number}/run{run_number}/west.h5"
                )
                marathon_west_files.append(west_file_path)

        log.debug(f"WESTPA datafile for analysis are {marathon_west_files}")

        #
        # If this is the first restart, check to see if you got any target state flux
        if restart_state["restarts_completed"] == 0:
            pass

            # Check to see if you got any target flux in ANY runs
            target_reached = False
            for west_file_path in marathon_west_files:
                if check_target_reached(west_file_path):
                    target_reached = True
                    break

            # If you reached the target, clean up from the extensions and then continue as normal
            # If  extension_iters is set to 0, then don't do extensions.
            if target_reached or self.extension_iters == 0:

                log.info("All runs reached target!")

                # Do some cleanup from the extension run
                if doing_extension and not self.extension_iters == 0:

                    # Remove the doing_extensions.lck lockfile
                    os.remove(EXTENSION_LOCKFILE)

                    westpa.rc.sim_manager.max_total_iterations = (
                        self.base_total_iterations
                    )

                # Otherwise, just continue as normal
                pass

            # If no runs reached the target, then we need to extend them
            elif not target_reached:

                log.info("Target not reached. Preparing for extensions.")

                # Create the doing_extensions.lck "lockfile" to indicate we're in extend mode (or keep if exists)
                #   and write the initial number of iterations to it.
                if not os.path.exists(EXTENSION_LOCKFILE):
                    with open(EXTENSION_LOCKFILE, "w") as lockfile:
                        lockfile.write(str(self.max_total_iterations))

                # Reset runs_completed to 0, and rewrite restart.dat accordingly
                restart_state["runs_completed"] = 0

                self.prepare_extension_run(
                    run_number=1, restart_state=restart_state, first_extension=True
                )
                return

        log.debug("Building haMSM and computing steady-state")
        log.debug(f"Cur iter is {self.cur_iter}")
        self.h5file_paths = marathon_west_files

        # Wipe out the old pcoord cache
        self.pcoord_cache = None

        self.model = self.construct_hamsm()
        model = self.model
        westpa.rc.pstatus(f"Getting built haMSM from {self.data_manager}")

        # ss_dist, ss_flux, model = msmwe_compute_ss(self.plugin_config, marathon_west_files)
        ss_dist = model.pSS
        ss_flux = model.JtargetSS
        # model = self.data_manager.model

        log.debug(f"Steady-state distribution: {ss_dist}")
        log.info(f"Target steady-state flux is {ss_flux}")

        # Obtain cluster-structures
        log.debug("Obtaining cluster-structures")
        model.update_cluster_structures(build_pcoord_cache=self.cache_pcoords)
        self.pcoord_cache = deepcopy(model.pcoord_cache)

        # TODO: Do this with pathlib
        struct_directory = f"{restart_directory}/structs"
        if not os.path.exists(struct_directory):
            os.makedirs(struct_directory)

        flux_filename = f"{restart_directory}/JtargetSS.txt"
        with open(flux_filename, "w") as fp:

            log.debug(f"Writing flux to {flux_filename}")
            fp.write(str(model.JtargetSS))
            fp.close()

        ss_filename = f"{restart_directory}/pSS.txt"
        with open(ss_filename, "w") as fp:

            log.debug(f"Writing pSS to {ss_filename}")
            np.savetxt(fp, model.pSS)
            fp.close()

        # If this is the last run of the last restart, do nothing and exit.
        # if restart_state['runs_completed'] >= self.n_runs and restart_state['restarts_completed'] >= self.n_restarts:
        #     log.info("All restarts completed!")
        #     return

        # Construct start-state file with all structures and their weights
        # TODO: Don't explicitly write EVERY structure to disk, or this will be a nightmare for large runs.
        # However, for now, it's fine...
        log.debug("Writing structures")
        # TODO: Include start states from previous runs (this is done implicitly if using west.h5 from those runs)
        sstates_filename = f"{restart_directory}/startstates.txt"
        with open(sstates_filename, "w") as fp:

            # Track the total number of segments iterated over
            seg_idx = 0

            log.info(
                f"Obtaining potential start structures ({len(model.cluster_structures.items())} bins avail,"
                f"{len(model.cluster_structure_weights)} weights avail,"
                f"cluster mapping length is {len(model.cluster_mapping)},"
                f"ss len is {len(ss_dist)} )"
                f"model n clusters is {model.n_clusters} )"
            )

            # unique_states = set()
            # for dtraj in model.dtrajs:
            #     unique_states.update(np.unique(model.dtrajs))
            # log.info(len(unique_states))

            log.info(model.cluster_mapping)

            # Can use these for sanity checks
            total_weight = 0.0
            total_bin_weights = []

            # Loop over each set of (bin index, all the structures in that bin)
            for (msm_bin_idx, structures) in tqdm.tqdm(
                model.cluster_structures.items()
            ):

                total_bin_weights.append(0)

                # Don't put structures in the basis or target
                if msm_bin_idx in [model.n_clusters, model.n_clusters + 1]:
                    continue

                # The per-segment bin probability.
                # Map a cluster number onto a cluster INDEX, because after cleaning the cluster numbers may no longer
                # be consecutive.
                bin_prob = ss_dist[
                    # model.cluster_mapping[msm_bin_idx]
                    msm_bin_idx
                ]  # / len(structures)

                if bin_prob == 0:
                    log.info(
                        f"MSM-Bin {msm_bin_idx}  has probability 0, so not saving any structs from it."
                    )
                    continue

                # The total amount of WE weight in this MSM microbin
                msm_bin_we_weight = sum(model.cluster_structure_weights[msm_bin_idx])

                # Write each structure to disk. Loop over each structure within a bin.
                msm_bin_we_weight_tracker = 0

                if self.synd_full_coord_map_path is not None:

                    # Here, we have a bunch of structures that, if we're using SynD, we need to be able to map back to
                    #   discrete states.
                    # Maybe that's another feature to add to SynD, but in the meantime, we can do the following...
                    # We can't make a dictionary mapping structures to discrete states, because lists and arrays aren't
                    #   hashable. But, we can explicitly hash the structures, then use that.
                    # There's probably a better way of uniquely representing structures, but this will do for now.

                    self.reverse_coord_map = {}
                    for state_id, structure in self.synd_full_coord_map.items():
                        # Explicitly cash this to a float32, because dtype mismatch will give different hashes
                        _hash = hashlib.md5(structure.astype(np.float32)).hexdigest()
                        self.reverse_coord_map[_hash] = state_id

                for struct_idx, structure in enumerate(structures):

                    # One structure per segment
                    seg_we_weight = model.cluster_structure_weights[msm_bin_idx][
                        struct_idx
                    ]
                    msm_bin_we_weight_tracker += seg_we_weight

                    # Structure weights are set according to Algorithm 5.3 in
                    # Aristoff, D. & Zuckerman, D. M. Optimizing Weighted Ensemble Sampling of Steady States.
                    # Multiscale Model Sim 18, 646–673 (2020).
                    structure_weight = seg_we_weight * (bin_prob / msm_bin_we_weight)

                    # If we're using synthetic dynamics, structures need to be integer state IDs
                    if self.synd_full_coord_map_path is not None:
                        _hash = hashlib.md5(structure.astype(np.float32)).hexdigest()
                        structure_index = self.reverse_coord_map[_hash]
                        structure_filename = f"{structure_index}"

                    # If we're using the HDF5 framework, we can just link segments to their structures in that
                    elif self.data_manager.store_h5:

                        iteration, seg_id, h5_file = model.structure_iteration_segments[
                            msm_bin_idx
                        ][struct_idx]
                        structure_filename = f"hdf:{h5_file}:{iteration}:{seg_id}"

                    # Otherwise, we have to actually write structures to disk
                    else:

                        topology = model.reference_structure.topology

                        try:
                            angles = model.reference_structure.unitcell_angles[0]
                            lengths = model.reference_structure.unitcell_lengths[0] * 10
                        # This throws typeerror if reference_structure.unitcell_angles is None, or AttributeError
                        #   if reference_structure.unitcell_angles doesn't exist.
                        except (TypeError, AttributeError):
                            angles, lengths = None, None

                        coords = structure * 10  # Correct units

                        structure_filename = (
                            f"{struct_directory}/bin{msm_bin_idx}_"
                            f"struct{struct_idx}.{STRUCT_EXTENSIONS[self.struct_filetype]}"
                        )

                        with self.struct_filetype(
                            structure_filename, "w"
                        ) as struct_file:

                            # Write the structure file
                            if self.struct_filetype is md.formats.PDBTrajectoryFile:
                                struct_file.write(
                                    coords,
                                    topology,
                                    modelIndex=1,
                                    unitcell_angles=angles,
                                    unitcell_lengths=lengths,
                                )

                            elif self.struct_filetype is md.formats.AmberRestartFile:
                                # AmberRestartFile takes slightly differently named keyword args
                                struct_file.write(
                                    coords,
                                    time=None,
                                    cell_angles=angles,
                                    cell_lengths=lengths,
                                )

                            else:
                                # Otherwise, YOLO just hope all the positional arguments are in the right place
                                log.warning(
                                    f"This output filetype ({self.struct_filetype}) is probably supported, "
                                    f"but not explicitly handled."
                                    " You should ensure that it takes argument as (coords, topology)"
                                )
                                struct_file.write(coords, topology)
                                raise Exception(
                                    "Don't know what extension to use for this filetype"
                                )

                    total_bin_weights[-1] += structure_weight
                    total_weight += structure_weight

                    # Add this start-state to the start-states file
                    # This path is relative to WEST_SIM_ROOT
                    fp.write(
                        f"b{msm_bin_idx}_s{struct_idx} {structure_weight} {structure_filename}\n"
                    )
                    seg_idx += 1

                # log.info(f"WE weight ({msm_bin_we_weight_tracker:.5e} / {msm_bin_we_weight:.5e})")

            # TODO: Fix this check. It's never quite worked right, nor has it ever caught an actual problem, so just
            #   disable for now.
            # In equilibrium, all probabilities count, but in steady-state the last 2 are the target/basis
            # Subtract off the probabilities of the basis and target states, since those don't have structures
            #   assigned to them.
            # assert np.isclose(total_weight, 1 - sum(model.pSS[model.n_clusters :])), (
            #     f"Total steady-state structure weights not normalized! (Total: {total_weight}) "
            #     f"\n\t pSS: {model.pSS}"
            #     f"\n\t Total bin weights {total_bin_weights}"
            #     f"\n\t pSS sum: {sum(model.pSS)}"
            #     f"\n\t pSS -2 sum: {sum(model.pSS[:-2])}"
            #     f"\n\t pSS (+target, no basis) sum: {sum(model.pSS[:-2]) + model.pSS[-1]}"
            # )

        # ## Start the new simulation

        bstates_str = ""
        for original_bstate in original_bstates:
            # We crush the original basis state probabilities here -- they'll be represented in the start-states
            #   anyway, we mostly just need to provide them for recycling.
            # As long as their relative weights (within the set of basis states) is unaffected, recycling will work
            #   the same after this rescaling.
            # By doing this, we ensure that start-states will dominate probabilities during initialization.
            # TODO: This is a little gross though, because if we hit the probability too much, we'll get issuers down
            #   the line from floating point arithmetic. Maybe there's a better way to deprioritize basis-states
            #   relative to start-states during initialization.
            orig_bstate_prob = original_bstate.probability * 1e-10
            orig_bstate_label = original_bstate.label
            orig_bstate_aux = original_bstate.auxref

            bstate_str = f"{orig_bstate_label} {orig_bstate_prob} {orig_bstate_aux}\n"

            bstates_str += bstate_str

        bstates_filename = f"{restart_directory}/basisstates.txt"
        with open(bstates_filename, "w") as fp:
            fp.write(bstates_str)

        tstates_str = ""
        for original_tstate in original_tstates:
            orig_tstate_label = original_tstate.label
            # TODO: Handle multidimensional pcoords
            orig_tstate_pcoord = original_tstate.pcoord[0]

            tstate_str = f"{orig_tstate_label} {orig_tstate_pcoord}\n"
            tstates_str += tstate_str
        tstates_filename = f"{restart_directory}/targetstates.txt"
        with open(tstates_filename, "w") as fp:
            fp.write(tstates_str)

        # Pickle the model
        objFile = f"{restart_directory}/hamsm.obj"
        with open(objFile, "wb") as objFileHandler:
            log.debug("Pickling model")
            pickle.dump(model, objFileHandler, protocol=4)
            objFileHandler.close()

        # Before finishing this restart, make a plot of the flux profile.
        #   This is made so the user can see whether

        self.generate_plots(restart_directory)

        # At this point, the restart is completed, and the data for the next one is ready (though still need to make the
        #   initialization file and such).

        if last_restart:
            log.info("All restarts completed! Finished.")
            return

        # Update restart_file file
        restart_state["restarts_completed"] += 1
        # If we're doing a restart, then reset the number of completed runs to 0 for the next marathon.
        restart_state["runs_completed"] = 0
        with open(self.restart_file, "w") as fp:
            json.dump(restart_state, fp)

        log.info("Initializing new run")

        # TODO: Read this from config if available
        segs_per_state = 1

        old_initialization_path = self.initialization_file
        new_initialization_path = f"{restart_directory}/{self.initialization_file}"
        log.debug(
            f"Moving initialization file from {old_initialization_path} to {new_initialization_path}."
        )
        shutil.move(old_initialization_path, new_initialization_path)

        initialization_state = {
            "tstate_file": tstates_filename,
            "bstate_file": bstates_filename,
            "sstate_file": sstates_filename,
            "tstates": None,
            "bstates": None,
            "sstates": None,
            "segs_per_state": segs_per_state,
        }

        with open(self.initialization_file, "w") as fp:
            json.dump(initialization_state, fp)

        westpa.rc.pstatus(
            f"\n\n"
            f"===== Restart {restart_state['restarts_completed']}, "
            + f"Run {restart_state['runs_completed']+1} initializing =====\n"
        )

        westpa.rc.pstatus(
            f"\nRun: \n\t w_init --tstate-file {tstates_filename} "
            + f"--bstate-file {bstates_filename} --sstate-file {sstates_filename} --segs-per-state {segs_per_state}\n"
        )
        # log.critical(f"Calling init_we with model {model}")

        self.init_we(initialization_state, self.pcoord_cache)

        log.info("New WE run ready!")
        westpa.rc.pstatus(
            f"\n\n===== Restart {restart_state['restarts_completed']} running =====\n"
        )

        w_run.run_simulation()
