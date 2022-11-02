import numpy as np
import westpa
import mdtraj as md
import os
import h5py


class H5AugmentationDriver:
    """
    WESTPA plugin for augmentation when using the HDF5 framework.

    After each iteration, links iter_XXX/auxdata/coord to the appropriate coordinate dataset in iter_XXX.h5.

    Paths to the per-iteration H5 files are stored relative to the west.h5 file in $WEST_SIM_ROOT, so it should
    be possible to move west.h5 and the supporting iteration h5 files without breaking anything.

    Can be used by including the following entries in your west.cfg::

        west:
            plugins:
            - plugin: msm_we.westpa_plugins.augmentation_driver.MDAugmentationDriver

    TODO
    ----
    Make a generic AugmentationDriver that loads the correct H5AugmentationDriver/MDAugmentationDriver
    """

    def __init__(self, sim_manager, plugin_config):
        westpa.rc.pstatus("Initializing coordinate augmentation plugin")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager

        self.plugin_config = plugin_config

        # Big number is low priority -- this should run before anything else
        self.priority = plugin_config.get("priority", 1)

        sim_manager.register_callback(
            sim_manager.post_propagation, self.augment_coordinates, self.priority
        )

        # Explicitly check that the H5 framework is actually enabled
        assert (
            self.data_manager.store_h5
        ), "H5 framework not enabled! Can't use the H5 framework augmentation driver."

    def augment_coordinates(self):
        """
        After propagation completes in a WE iteration, this populates auxdata/coord with a link to the coordinates.
        """

        westpa.rc.pstatus("Performing augmentation")

        iter_group_name = self.data_manager.get_iter_group(self.sim_manager.n_iter).name

        auxcoord_dataset = f"{iter_group_name}/auxdata/coord"

        # We construct the path to the per-iteration H5 file, using the link already existing in iter_XX/trajectories.
        # This is a relative path, to make it more robust to moving the data.
        self.data_manager.we_h5file[auxcoord_dataset] = h5py.SoftLink(
            path=f"{iter_group_name}/trajectories/sorted_segment_trajectories"
        )


class MDAugmentationDriver:
    """
    WESTPA plugin to augment west.h5 with auxiliary coordinates.

    After each iteration, appends coordinates to iter_XXX/auxdata/coord, for later usage with haMSM construction.

    Can be used by including the following entries in your west.cfg::

        west:
            plugins:
            - plugin: msm_we.westpa_plugins.augmentation_driver.MDAugmentationDriver
                  topology_file: path/to/topology.pdb
                  child_traj_filename: name of segment trajectory file in traj_segs/<iter>/<seg>/
                  parent_traj_filename: name of parent trajectory file in traj_segs/<iter>/<seg>/
    """

    def __init__(self, sim_manager, plugin_config):
        westpa.rc.pstatus("Initializing coordinate augmentation plugin")

        if not sim_manager.work_manager.is_master:
            westpa.rc.pstatus("Not running on the master process, skipping")
            return

        self.data_manager = sim_manager.data_manager
        self.sim_manager = sim_manager

        self.plugin_config = plugin_config

        self.topology_file = plugin_config["topology_file"]
        self.topology = md.load(self.topology_file).topology

        self.parent_traj_filename = plugin_config["parent_traj_filename"]
        self.child_traj_filename = plugin_config["child_traj_filename"]

        # Big number is low priority -- this should run before anything else
        self.priority = plugin_config.get("priority", 1)

        sim_manager.register_callback(
            sim_manager.post_propagation, self.augment_coordinates, self.priority
        )

    def augment_coordinates(self):
        """
        After propagation completes in a WE iteration, this populates auxdata/coord with the coordinates.

        Looks for parent/child trajectory files in the segment data_ref path defined in west.cfg,
        named according to parent_traj_filename and child_traj_filename in the plugin config.

        Todo
        ----
        This all assumes segments are being written to files. Write an alternate version of this that's compatible with
        the HDF5 framework.
        """

        westpa.rc.pstatus("Performing augmentation")

        iter_group_name = self.data_manager.get_iter_group(self.sim_manager.n_iter).name
        segments = self.data_manager.get_segments(
            n_iter=self.sim_manager.n_iter, load_pcoords=True
        )
        n_walkers = len(segments)

        data_ref_formatter = os.path.expandvars(
            westpa.rc.config["west"]["data"]["data_refs"]["segment"]
        )

        for i, segment in enumerate(segments):

            # Note: This uses whatever formatter is in west.cfg -- including any environment variables!
            seg_path = data_ref_formatter.format(segment=segment)
            parent_traj_path = f"{seg_path}/{self.parent_traj_filename}"
            child_traj_path = f"{seg_path}/{self.child_traj_filename}"

            if os.path.exists(parent_traj_path):
                parent_coords = np.squeeze(
                    md.load(parent_traj_path, top=self.topology).xyz
                )
            else:
                # If the parent was an i/bstate rather than a segment,
                #   then load appropriate structure from the bstate/istate here

                try:
                    segment_istate = self.data_manager.get_segment_initial_states(
                        [segment]
                    )[0]
                except IndexError:
                    # If you can't get the initial state, this may be a continuation from a previous run
                    # TODO: Check and explicitly verify that augmentation has already been performed, before assuming
                    #   it has been
                    continue

                bstate_id = segment_istate.basis_state_id

                segment_bstate = self.data_manager.get_basis_states(
                    self.sim_manager.n_iter
                )[bstate_id]
                bstate_path = segment_bstate.auxref

                parent_coords = np.squeeze(md.load(bstate_path, top=self.topology).xyz)

            child_coords = np.squeeze(md.load(child_traj_path, top=self.topology).xyz)

            # Create auxdata/coord for the current iteration if it doesn't exist
            auxcoord_dataset = self.data_manager.we_h5file.require_dataset(
                name=f"{iter_group_name}/auxdata/coord",
                shape=(n_walkers, 2, *child_coords.shape),
                dtype=child_coords.dtype,
            )

            auxcoord_dataset[segment.seg_id, 0] = parent_coords
            auxcoord_dataset[segment.seg_id, 1] = child_coords
