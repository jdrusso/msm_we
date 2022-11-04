import h5py
import numpy as np
import sys
import mdtraj as md
from msm_we._logging import log, ProgressBar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msm_we import modelWE


class DataMixin:

    n_iter = None

    fileList = None
    """list of str: List of all filenames with data"""
    n_data_files = None
    """int: Number of files in :code:`fileList`
    **TODO**: Deprecate this, this could just be a property"""

    westList = None
    numSegments = None
    maxIter = None

    # TODO: Describe segindList better.
    segindList = None
    """list: List of segment indices(?)"""

    weightList = None
    """array-like: List of segment weights in an iteration"""

    nSeg = None
    pcoord0List = None
    pcoord1List = None
    seg_weights = {}

    coordPairList = None
    transitionWeights = None
    departureWeights = None

    n_hist = None
    """int: Number of steps of history information to use when building transitions."""

    coordsExist = None

    def get_transition_data(self: "modelWE", n_lag):
        """
        This function analyzes pairs of coordinates at the current iteration, set by :code:`self.n_iter`, and at some
            lag in the past, :code:`self.n_iter - n_lag`.

        Segments where a walker was warped (recycled) use the basis coords as the lagged coords.

        Parameters
        ----------
        n_lag : int
            Number of lags to use for transitions.

        Returns
        -------
        None
        """

        log.warning(
            "Getting transition data at arbitrary lags > 0 is not yet supported! Use at your own risk."
        )

        # get segment history data at lag time n_lag from current iter
        if n_lag > self.n_iter:
            sys.stdout.write(
                "too much lag for iter... n_lag reduced to: " + str(self.n_iter) + "\n"
            )
            n_lag = self.n_iter
        if n_lag >= self.n_hist:
            sys.stdout.write("too much lag for stored history... recalculating...\n")
            self.get_seg_histories(n_lag)

        self.n_lag = n_lag
        segindList_lagged = self.seg_histories[:, n_lag]

        # TODO: What exactly is this a list of?
        # seg_histories is a list of indices of the segments
        warpList = self.seg_histories[:, 0:n_lag]  # check for warps
        warpList = np.sum(warpList < 0, 1)

        # Get the weights for the lagged and current iterations
        # If something was split/merged between n_iter and n_iter-n_lag , then the weights may have changed, so
        #   check a particular segment's weight.
        # TODO: Does this effectively get the parent weight if it was split from something else? Then weight_histories
        #    would need to be tracking parents/children
        weightList_lagged = self.weight_histories[:, n_lag]
        # TODO: Does this copy need to be made?
        weightList = self.weightList

        # This will become a list of (lagged iter coord, current iter coord)
        coordPairList = np.zeros((self.nSeg, self.nAtoms, self.coord_ndim, 2))

        prewarpedStructures = np.zeros((self.nSeg, self.nAtoms, self.coord_ndim))
        nWarped = 0

        # Go through each segment, and get pairs of coordinates at current iter (n_iter) and
        # lagged iter (n_iter-n_lag)
        for seg_idx in range(self.nSeg):

            # FIXME: Try statements should encompass the smallest amount of code
            #  possible - anything could be tripping this
            # try:
            if seg_idx == 0:
                westFile = self.fileList[self.westList[seg_idx]]
                dataIn = h5py.File(westFile, "r")
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_current = dset[:]
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter - n_lag),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_lagged = dset[:]
            elif self.westList[seg_idx] != self.westList[seg_idx - 1]:
                # FIXME: I think you can just move this close to an if statement in the beginning, and then remove
                #   this whole if/elif. Everything after that close() seems to be duplicated.
                dataIn.close()
                westFile = self.fileList[self.westList[seg_idx]]
                dataIn = h5py.File(westFile, "r")

                # Load the data for the current iteration
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_current = dset[:]

                # Load the lagged data for (iteration - n_lag)
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter - n_lag),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coords_lagged = dset[:]

            coordPairList[seg_idx, :, :, 1] = coords_current[
                self.segindList[seg_idx], 1, :, :
            ]

            # If this segment has no warps, then add the lagged coordinates
            if warpList[seg_idx] == 0:
                # Try to set the previous coord in the transition pair to the segment's lagged coordinates
                try:
                    lagged_seg_index = segindList_lagged[seg_idx]
                    coordPairList[seg_idx, :, :, 0] = coords_lagged[
                        lagged_seg_index, 0, :, :
                    ]

                # If this fails, then there were no lagged coordinates for this structure.
                except IndexError as e:
                    log.critical(
                        f"Lagged coordinates do not exist for the structure in segment {seg_idx}"
                    )
                    raise e
                    weightList_lagged[seg_idx] = 0.0
                    weightList[
                        seg_idx
                    ] = 0.0  # set transitions without structures to zero weight

            # If something was recycled during this segment, then instead of using the lagged cooordinates,
            #   just use the basis coords.
            # But, also save the original structure before the warp!
            elif warpList[seg_idx] > 0:

                # St
                prewarpedStructures[nWarped, :, :] = coords_lagged[
                    segindList_lagged[seg_idx], 0, :, :
                ]

                assert self.basis_coords is not None

                coordPairList[seg_idx, :, :, 0] = self.basis_coords
                nWarped = nWarped + 1

            # # TODO: What triggers this? When that critical log hits, come update this comment and the main docstring.
            # # This triggers on index out of bounds... But that's a huge try statement!
            # except Exception as e:
            #     log.critical("Document whatever's causing this exception!")
            #     log.warning(e)
            #     raise e
            #     weightList_lagged[seg_idx] = 0.0
            #     weightList[
            #         seg_idx
            #     ] = 0.0  # set transitions without structures to zero weight

        # Squeeze removes axes of length 1 -- this helps with np.where returning nested lists of indices
        # FIXME: is any of this necessary? This kinda seems like it could be replaced with
        #  something like indWarped = np.squeeze(np.where(warpList > 0)).astype(int)
        indWarped = np.squeeze(np.where(warpList > 0))
        indWarpedArray = np.empty(nWarped)
        indWarpedArray[0:nWarped] = indWarped
        indWarped = indWarpedArray.astype(int)

        # Get the current and lagged weights
        # This returns (the wrong? none at all?) weights for segments with warps, which is corrected below
        transitionWeights = weightList.copy()
        departureWeights = weightList_lagged.copy()

        # Get correct weights for segments that warped
        for iW in range(nWarped):

            # The coord pair here is (pre-warped structure, reference topology) instead of
            #   (lagged struture, current structure)
            coordPair = np.zeros((1, self.nAtoms, self.coord_ndim, 2))
            coordPair[0, :, :, 0] = prewarpedStructures[iW, :, :]
            coordPair[0, :, :, 1] = self.reference_coord
            coordPairList = np.append(coordPairList, coordPair, axis=0)

            # TODO: iterWarped appears to be the iteration the warp happened at
            iterWarped = np.squeeze(np.where(self.seg_histories[indWarped[iW], :] < 0))
            try:
                nW = np.shape(iterWarped)
                iterWarped = iterWarped[0]
                sys.stdout.write(
                    "    segment "
                    + str(indWarped[iW])
                    + " warped "
                    + str(nW)
                    + " times\n"
                )
            # TODO: What exception is this handling? Why would this be tripped?
            except Exception as e:
                log.critical(f"Exception was {e}")
                log.critical("UPDATE THIS EXCEPTION HANDLER!")

                sys.stdout.write(
                    "    segment " + str(indWarped[iW]) + " warped 1 time\n"
                )

            # The "current" weights are taken at the warp iteration, so the final pair of weights are
            #   (the lagged weights, the weights at the warp)
            transitionWeights = np.append(
                transitionWeights, self.weight_histories[indWarped[iW], iterWarped]
            )
            departureWeights = np.append(
                departureWeights, self.weight_histories[indWarped[iW], n_lag]
            )

        self.coordPairList = coordPairList
        self.transitionWeights = transitionWeights
        self.departureWeights = departureWeights

    def get_transition_data_lag0(self: "modelWE"):
        """
        Get coordinate pairs at the beginning and end of this iteration.

        Updates:
            - self.coordPairList, a list of  parent/child coordinate pairs
            - self.transitionWeights, a copy of self.weightList
            - self.departureWeights, a copy of self.weightList
        """

        weightList = self.weightList
        coordPairList = np.zeros((self.nSeg, self.nAtoms, self.coord_ndim, 2))

        log.debug(
            f"Getting transition data for {self.nSeg} segs in iteration {self.n_iter}, at a lag of 0"
        )

        # the segments in this iteration may be split across a number of different files
        # "Traditionally", we store a reference for  each segment  of which WEST file it's in
        # But what if we flip that, and for each west file, get which segments are in it?

        seg_west_files = self.westList[range(self.nSeg)]
        west_file_idxs = np.unique(seg_west_files)
        west_files = [self.fileList[idx] for idx in west_file_idxs]

        for idx, west_file in enumerate(west_files):

            segs_contained = np.where(seg_west_files == idx)[0]

            with h5py.File(west_file, "r") as data_file:

                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )

                try:
                    dset = data_file[dsetName]
                except KeyError as e:
                    raise e

                coords = dset

                coordPairList[segs_contained, :, :, 0] = coords[:, 0, :, :]
                coordPairList[segs_contained, :, :, 1] = coords[
                    :, self.pcoord_len - 1, :, :
                ]

                # Check for NaNs in segments
                nan_segments = segs_contained[
                    np.argwhere(
                        np.isnan(coordPairList[segs_contained]).any(axis=(1, 2, 3))
                    )
                ]

                if nan_segments.shape[0] > 0:
                    log.warning(
                        f"Bad coordinates for segments {nan_segments}, setting weights to 0"
                    )
                    weightList[nan_segments] = 0.0

        transitionWeights = weightList.copy()
        departureWeights = weightList.copy()

        self.coordPairList = coordPairList
        self.transitionWeights = transitionWeights
        self.departureWeights = departureWeights

    def get_seg_histories(self: "modelWE", n_hist):
        """
        **TODO: What does this do exactly?**

        Updates:
            - self.seg_histories
            - self.weight_histories
            - self.n_hist

        Parameters
        ----------
        n_hist : int
            Number of steps of history information to include.

        Returns
        -------
        None

        """

        log.debug(f"Getting seg histories for {self.nSeg} segments")

        if n_hist > self.n_iter:
            sys.stdout.write(
                "we have too much history... n_hist reduced to: "
                + str(self.n_iter)
                + "\n"
            )
            n_hist = self.n_iter

        # FIXME: The only other place this is used is in get_transition_data(). May need to see how it's handled there
        #   i.e. make sure it's initialized
        self.n_hist = n_hist

        seg_histories = np.zeros((self.nSeg, self.n_hist + 1))
        weight_histories = np.zeros((self.nSeg, self.n_hist))

        # Loop over all segments
        for iS in range(self.nSeg):

            # Print a message every 100 segments
            # if iS % 100 == 0:
            #     sys.stdout.write(
            #         "        getting history for iteration "
            #         + str(self.n_iter)
            #         + " segment "
            #         + str(iS)
            #         + "...\n"
            #     )

            # Open the relevant datafile for reading
            if iS == 0:
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")
            elif self.westList[iS] != self.westList[iS - 1]:
                # If you're past the 0th segment, then close the previous file.
                # FIXME: Close the file after finished working with it, at the *end* of the loop. As-is, the last loop
                #       leaves the file open.
                dataIn.close()
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")

            seg_histories[iS, 0] = self.segindList[iS]
            warped = 0

            # Iterate over history lengths from 1 to n_hist
            for iH in range(1, self.n_hist + 1):

                indCurrentSeg = seg_histories[iS, iH - 1]

                if indCurrentSeg < 0 and warped == 0:
                    sys.stdout.write(
                        "Segment "
                        + str(iS)
                        + " warped last iter: History must end NOW!\n"
                    )
                    warped = 1

                elif indCurrentSeg >= 0 and warped == 0:
                    dsetName = "/iterations/iter_%08d/seg_index" % int(
                        self.n_iter - iH + 1
                    )

                    dset = dataIn[dsetName]
                    seg_histories[iS, iH] = dset[indCurrentSeg][1]
                    weight_histories[iS, iH - 1] = dset[indCurrentSeg][0]

                    if seg_histories[iS, iH] < 0:
                        sys.stdout.write(
                            "            segment "
                            + str(iS)
                            + " warped "
                            + str(iH)
                            + " iters ago\n"
                        )
        self.seg_histories = seg_histories[:, :-1].astype(int)
        self.weight_histories = weight_histories
        # FIXME:
        #   weight histories and segment histories go in reverse order,
        #   so final current iter is first of 0 index

    def collect_iter_coordinates(
        self: "modelWE",
    ):  # grab coordinates from WE traj_segs folder
        """
        Goes through the generated trajectory segments, and adds data from the segments to an H5 file.

        This should be implemented by the user, and this implementation assumes a really specific configuration.
        This is left in mostly as an example.

        Returns
        -------
        None

        Todo
        ----
        Generalize to different filetypes. This appears to be AMBER specific and relies on loading rst7 files
        """

        log.critical(
            "If you're calling this function, be absolutely sure it does what you want -- this is "
            "very specifically written for the output of certain simulations."
        )

        nS = self.nSeg
        westFile = self.fileList[self.westList[0]]
        dataIn = h5py.File(westFile, "a")
        coords = np.zeros((0, 2, self.nAtoms, self.coord_ndim))
        for iS in range(self.nSeg):
            # FIXME: Replace strings with Pathlib paths
            westFile = self.fileList[self.westList[iS]]
            WEfolder = westFile.replace("west.h5", "")
            trajpath = WEfolder + "traj_segs/%06d/%06d" % (self.n_iter, iS)
            coord0 = np.squeeze(
                md.load(
                    trajpath + "/parent.rst7", top=self.reference_structure.topology
                )._xyz
            )
            coord1 = np.squeeze(
                md.load(
                    trajpath + "/seg.rst7", top=self.reference_structure.topology
                )._xyz
            )
            coordT = np.array([coord0, coord1])
            coordT = coordT[np.newaxis, :, :, :]
            coords = np.append(coords, coordT, axis=0)
            try:
                if iS > 0:
                    if self.westList[iS] != self.westList[iS - 1] and iS < nS - 1:
                        dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                            int(self.n_iter),
                            self.auxpath,
                        )
                        try:
                            # TODO: Why exclude the last point?
                            dset = dataIn.create_dataset(
                                dsetName, np.shape(coords[:-1, :, :, :])
                            )
                            dset[:] = coords[:-1, :, :, :]
                        except (RuntimeError, ValueError):
                            del dataIn[dsetName]
                            dset = dataIn.create_dataset(
                                dsetName, np.shape(coords[:-1, :, :, :])
                            )
                            dset[:] = coords[:-1, :, :, :]
                            log.warning(
                                "coords exist for iteration "
                                + str(self.n_iter)
                                + " overwritten\n"
                            )

                        dataIn.close()
                        coords = np.zeros((0, 2, self.nAtoms, self.coord_ndim))
                        coords = np.append(coords, coordT, axis=0)
                        dataIn = h5py.File(westFile, "a")

                    # If it's the last segment, don't exclude the last point (why?)
                    elif iS == nS - 1:
                        dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                            int(self.n_iter),
                            self.auxpath,
                        )
                        try:
                            dset = dataIn.create_dataset(dsetName, np.shape(coords))
                            dset[:] = coords
                        except (RuntimeError, ValueError):
                            del dataIn[dsetName]
                            dset = dataIn.create_dataset(dsetName, np.shape(coords))
                            dset[:] = coords
                            log.warning(
                                "coords exist for iteration "
                                + str(self.n_iter)
                                + " overwritten\n"
                            )

                        dataIn.close()

            except Exception as e:
                log.error(
                    "error collecting coordinates from "
                    + WEfolder
                    + " , iter "
                    + str(self.n_iter)
                    + " segment "
                    + str(self.segindList[iS])
                    + "\n"
                )
                log.error(e)

    def get_iter_coordinates(self: "modelWE", iteration):
        """
        Return the valid coordinates for a certain iteration

        Parameters
        ----------
        iteration: int
            The iteration to return coordinates for

        Returns
        -------
        Array of coordinates for all atoms at the current iteration
        """

        self.load_iter_data(iteration)
        self.load_iter_coordinates()
        # indGood = np.squeeze(np.where(np.sum(np.sum(self.cur_iter_coords, 2), 1) != 0))

        bad_coords = np.isnan(self.cur_iter_coords).any(axis=(1, 2))
        good_coords = ~bad_coords

        # iter_coords = self.cur_iter_coords[indGood]
        iter_coords = self.cur_iter_coords[good_coords]

        return iter_coords

    def load_iter_coordinates(self: "modelWE"):
        """

        Returns
        -------

        """

        cur_iter_coords = np.full(
            (self.nSeg, self.nAtoms, self.coord_ndim), fill_value=np.nan
        )

        log.debug(
            f"Getting coordinates for {self.nSeg} segs in iteration {self.n_iter}, at a lag of 0"
        )

        # the segments in this iteration may be split across a number of different files
        # "Traditionally", we store a reference for  each segment  of which WEST file it's in
        # But flip that, and for each west file, get which segments are in it

        seg_west_files = self.westList[range(self.nSeg)]
        west_file_idxs = np.unique(seg_west_files)
        west_files = [self.fileList[idx] for idx in west_file_idxs]

        for idx, west_file in enumerate(west_files):

            segs_contained = np.where(seg_west_files == idx)[0]

            with h5py.File(west_file, "r") as data_file:

                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )

                try:
                    dset = data_file[dsetName]
                except (RuntimeError, ValueError, KeyError):
                    log.error(
                        f"Error getting coordinates from {west_file}, in iteration {self.n_iter}"
                    )

                    cur_iter_coords[segs_contained, :, :] = np.full(
                        (self.nAtoms, self.coord_ndim), fill_value=np.nan
                    )
                    self.coordsExist = False

                    # raise e

                else:
                    coords = dset
                    log.debug(
                        f"Attempting to obtain coordinates from west_file {west_file}, iteration {self.n_iter}"
                    )
                    # TODO: This should probably generically be -1, not 1, to deal with variable-length augmentation.
                    assert coords.shape[1] > 1, (
                        "Augmented coords only have 1 point in them -- "
                        "need at least start & end for transitions"
                    )
                    cur_iter_coords[segs_contained, :, :] = coords[:, -1, :, :]

        self.cur_iter_coords = cur_iter_coords

    def load_iter_coordinates0(self: "modelWE"):  # get iteration initial coordinates
        coordList = np.full(
            (self.nSeg, self.nAtoms, self.coord_ndim), fill_value=np.nan
        )
        for iS in range(self.nSeg):
            if iS == 0:
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coord = dset[:]
            elif self.westList[iS] != self.westList[iS - 1]:
                dataIn.close()
                westFile = self.fileList[self.westList[iS]]
                dataIn = h5py.File(westFile, "r")
                dsetName = "/iterations/iter_%08d/auxdata/%s" % (
                    int(self.n_iter),
                    self.auxpath,
                )
                dset = dataIn[dsetName]
                coord = dset[:]
            coordList[iS, :, :] = coord[self.segindList[iS], 0, :, :]
        self.cur_iter_coords = coordList

    def get_coordinates(self: "modelWE", first_iter, last_iter):
        """
        Unused

        Parameters
        ----------
        first_iter
        last_iter

        Returns
        -------
        """

        log.warning("This function is not tested or supported, use at your own risk!")

        self.first_iter = first_iter
        self.last_iter = last_iter
        iters = range(self.first_iter, self.last_iter + 1)
        coordSet = np.zeros((0, self.nAtoms, self.coord_ndim))
        for iter in iters:
            if iter % 50 == 0:
                sys.stdout.write(
                    "    gathering structures from iteration " + str(iter) + "...\n"
                )
            self.load_iter_data(iter)
            self.load_iter_coordinates()
            if self.coordsExist:
                coordSet = np.append(coordSet, self.cur_iter_coords, axis=0)
        self.all_coords = coordSet

    def get_coordSet(self: "modelWE", last_iter, streaming=None, progress_bar=None):
        """
        Loads all coordinates and progress coordinates into memory for later usage.

        If streaming, then this only loads pcoords

        Todo
        ----
        I want to avoid loading full coordinates into memory as much as possible.
        That means trying to replace usage of all_coords here.

        Parameters
        ----------
        last_iter

        Returns
        -------

        """

        if streaming is None and False:  # and self.dimReduceMethod == "vamp":
            streaming = False
        elif streaming is None:
            streaming = True

        log.debug(f"CoordSet loading up to {last_iter}")
        total_segments = int(sum(self.numSegments[:last_iter]))

        if not streaming:
            coordSet = np.full(
                (total_segments, self.nAtoms, self.coord_ndim), fill_value=np.nan
            )
        pcoordSet = np.full((total_segments, self.pcoord_ndim), fill_value=np.nan)

        last_seg_idx = total_segments

        # Update iterations N+1 -> 1

        with ProgressBar(progress_bar) as progress_bar:
            task_id = progress_bar.add_task(
                total=last_iter, completed=0, description="Getting coordSet"
            )

            for i in range(last_iter, 0, -1):
                self.load_iter_data(i)
                self.load_iter_coordinates()

                first_seg_idx = last_seg_idx - len(self.segindList)
                assert first_seg_idx >= 0, "Referencing a segment that doesn't exist"

                # Get the indices of all "good" coordinates, where a  valid coordinate has been obtained
                # Boolean mask for coords that have been successfully initialized
                bad_coords = np.isnan(self.cur_iter_coords).any(axis=(1, 2))
                good_coords = ~bad_coords

                if bad_coords.any():
                    log.warning(
                        f"Walker {np.argwhere(bad_coords).squeeze()} has bad coords in iteration(s) {self.n_iter}"
                    )

                if not streaming:
                    coordSet[first_seg_idx:last_seg_idx][
                        good_coords
                    ] = self.cur_iter_coords[good_coords, :, :]

                pcoordSet[first_seg_idx:last_seg_idx][good_coords] = self.pcoord1List[
                    good_coords, :
                ]

                last_seg_idx = first_seg_idx
                progress_bar.update(task_id, advance=1)

        # Set the coords, and pcoords
        if not streaming:
            self.all_coords = coordSet
        else:
            pass

        self.pcoordSet = pcoordSet

        first_iter_cluster = i
        self.first_iter = first_iter_cluster
        self.last_iter = last_iter

    def get_traj_coordinates(self: "modelWE", from_iter, traj_length):
        if traj_length > from_iter:
            traj_length = from_iter - 1
            sys.stdout.write(
                "trajectory length too long: set to " + str(traj_length) + "\n"
            )
        self.load_iter_data(from_iter)
        self.get_seg_histories(traj_length)
        traj_iters = np.zeros((traj_length, self.nSeg, self.nAtoms, self.coord_ndim))
        ic = traj_length - 1
        iH = 0
        nS = self.nSeg
        westList = self.westList.copy()
        for i in range(from_iter, from_iter - traj_length, -1):
            sys.stdout.write(
                "    gathering structures from iteration " + str(i) + "...\n"
            )
            self.load_iter_data(i)
            self.load_iter_coordinates()
            seg_history = self.seg_histories[:, iH]
            seg_history_index = np.zeros(nS).astype(int)
            last_index = np.zeros(nS).astype(
                int
            )  # list of final iterations for warped particles
            for iS in range(nS):
                if seg_history[iS] > 0:
                    seg_history_index[iS] = np.where(
                        np.logical_and(
                            self.segindList == seg_history[iS],
                            self.westList == westList[iS],
                        )
                    )[0][
                        0
                    ]  # segment indexes are local to the westfile, not global for the analysis set
                    last_index[iS] = 0
                elif seg_history[iS] < 0:
                    last_index[iS] = i - (from_iter - traj_length) - 1
            traj_iters[ic, :, :, :] = self.cur_iter_coords[seg_history_index, :, :]
            ic = ic - 1
            iH = iH + 1
        self.load_iter_data(from_iter)
        traj_iters = np.swapaxes(traj_iters, 0, 1)
        self.trajSet = [None] * self.nSeg
        for iS in range(self.nSeg):
            self.trajSet[iS] = traj_iters[iS, last_index[iS] :, :, :]

    def load_iter_data(self: "modelWE", n_iter: int):
        """
        Update state with the data (including pcoord but not including coords) corresponding to an iteration.

        Object fields updated with the information from the selected iteration:
            - `self.westList`
            - `self.segindList`
            - `self.weightList`
            - `self.n_segs`
            - `self.pcoord0List`
            - `self.pcoord1List`

        Parameters
        ----------
        n_iter : int
            Iteration to get data for.

        Returns
        -------
        None

        Todo
        ----
        May want to rework the logic here, depending on how this is used.
        Seems like some of this iteration can be removed/optimized.
        """

        # log.debug("Getting iteration data")

        self.n_iter = n_iter
        westList = np.array([])
        segindList = np.array([])
        weightList = np.array([])
        pcoord0List = np.empty((0, self.pcoord_ndim))
        pcoord1List = np.empty((0, self.pcoord_ndim))

        seg_weights = np.array([])

        n_segs = 0

        # Iterate through each file index, trying to find files that contains the iteration of interest
        # TODO: Can replace this with `for if, fileName in enumerate(self.\\)`
        for file_idx in range(self.n_data_files):
            fileName = self.fileList[file_idx]
            try:
                # Try to find the h5 data file associated with this iteration
                dataIn = h5py.File(fileName, "r")
                dsetName = "/iterations/iter_%08d/seg_index" % int(n_iter)

                # Check if the dataset
                dataset_exists = dsetName in dataIn

                # Check to make sure this isn't the last iteration -- last iterations have incomplete data
                is_not_last_iteration = (
                    "/iterations/iter_%08d/seg_index" % int(n_iter + 1) in dataIn
                )

                log.debug(f"From file {fileName}, loading iteration {n_iter}")

                if dataset_exists and is_not_last_iteration:

                    dset = dataIn[dsetName]
                    newSet = dset[:]
                    n_segs_in_file = np.shape(newSet)
                    n_segs_in_file = n_segs_in_file[0]
                    dsetNameP = "/iterations/iter_%08d/pcoord" % int(n_iter)
                    dsetP = dataIn[dsetNameP]
                    pcoord = dsetP[:]
                    weights = dset["weight"]
                    seg_weights = np.append(seg_weights, weights)

                    if (
                        not pcoord.shape[2] == self.pcoord_ndim
                        and not self.pcoord_shape_warned
                    ):
                        log.warning(
                            f"Dimensions of pcoord in {fileName} ({pcoord.shape[2]}) do not match specified "
                            f"pcoord dimensionality self.pcoord_ndim ({self.pcoord_ndim}). "
                            f"MSM-WE will only load up to dimension {self.pcoord_ndim}. "
                            f"This pcoord is just used for stratification, and this is expected behavior if "
                            f"you're extending your pcoord (i.e. in an optimization flow). "
                        )
                        self.pcoord_shape_warned = True

                    # Iterate over segments in this dataset
                    # TODO: This loop is probably really inefficient?
                    for seg_idx in range(n_segs_in_file):
                        # if np.sum(pcoord[seg_idx,self.pcoord_len-1,:])==0.0:
                        # # intentionally using this to write in dummy pcoords,
                        # # this is a good thing to have for post-analysis though!
                        #    raise ValueError('Sum pcoord is 0, probably middle of WE iteration, not using iteration') f
                        westList = np.append(westList, file_idx)
                        segindList = np.append(segindList, seg_idx)
                        weightList = np.append(weightList, newSet[seg_idx][0])
                        pcoord0List = np.append(
                            pcoord0List,
                            np.expand_dims(pcoord[seg_idx, 0, : self.pcoord_ndim], 0),
                            axis=0,
                        )
                        pcoord1List = np.append(
                            pcoord1List,
                            np.expand_dims(
                                pcoord[
                                    seg_idx, self.pcoord_len - 1, : self.pcoord_ndim
                                ],
                                0,
                            ),
                            axis=0,
                        )
                        n_segs = n_segs + 1
                dataIn.close()
            except Exception as dataset_exists:
                sys.stdout.write("error in " + fileName + str(sys.exc_info()[0]) + "\n")
                raise dataset_exists

        # log.debug(f"Found {n_segs} segments in iteration {n_iter}")

        self.westList = westList.astype(int)

        # This is a list of the segment indices
        self.segindList = segindList.astype(int)
        self.seg_weights[n_iter] = seg_weights
        self.weightList = weightList
        self.nSeg = n_segs
        self.pcoord0List = pcoord0List
        self.pcoord1List = pcoord1List

    def get_iterations(self: "modelWE"):
        """
        Updates internal state with the maximum number of iterations, and the number of segments in each section.

        Note
        ----
        This updates :code:`numSegments` -- :code:`numSegments` is actually a *list* of the number of segments in each iteration.

        Returns
        -------
        None
        """

        log.debug("Getting number of iterations and segments")

        numSegments = np.array([])
        nSeg = 1
        n_iter = 1

        # Loop over nSegs
        # TODO: Not sure I understand the logic in this loop
        while nSeg > 0:
            nSeg = 0

            # Iterate through each filename in fileList, and see if it contains the iteration we're looking for
            # TODO: This loop is pretty common, this should be refactored into a find_iteration() or something
            for file_index in range(self.n_data_files):
                fileName = self.fileList[file_index]
                try:
                    dataIn = h5py.File(fileName, "r")
                    dsetName = "/iterations/iter_%08d/seg_index" % int(n_iter)
                    dataset_exists = dsetName in dataIn

                    is_not_last_iteration = (
                        "/iterations/iter_%08d/seg_index" % int(n_iter + 1) in dataIn
                    )

                    if dataset_exists and is_not_last_iteration:
                        # If this file does contain the iteration of interest
                        # if dataset_exists:
                        dset = dataIn[dsetName]
                        newSet = dset[:]
                        nS = np.shape(newSet)
                        nSeg = nS[0] + nSeg
                    dataIn.close()
                except Exception as e:
                    log.error(e)
                    log.error(f"No segments in {fileName} {str(sys.exc_info()[0])}")

            if nSeg > 0:
                numSegments = np.append(numSegments, nSeg)
                log.debug(
                    "Iteration " + str(n_iter) + " has " + str(nSeg) + " segments...\n"
                )

            n_iter = n_iter + 1

        # Warning: These are not defined until this is run for the first time
        self.numSegments = numSegments
        self.maxIter = numSegments.size

    def get_iterations_iters(self: "modelWE", first_iter: int, last_iter: int):
        """
        Updates internal state with the maximum number of iterations, and the number of segments in each section.

        Parameters
        ----------
        first_iter : int
        last_iter : int

        Returns
        -------
        None

        Warning
        ----
        This is potentially deprecated or unnecessary. Currently unused.

        """

        numSegments = np.array([])

        for n_iter in range(first_iter, last_iter + 1):
            nSeg = 0
            for iF in range(self.n_data_files):
                fileName = self.fileList[iF]
                try:
                    dataIn = h5py.File(fileName, "r")
                    dsetName = "/iterations/iter_%08d/seg_index" % int(n_iter)
                    dataset_exists = dsetName in dataIn
                    if dataset_exists:
                        dset = dataIn[dsetName]
                        newSet = dset[:]
                        nS = np.shape(newSet)
                        nSeg = nS[0] + nSeg
                    dataIn.close()
                except Exception as e:
                    log.error(e)
                    log.error(f"No segments in {fileName} {str(sys.exc_info()[0])}")

            if nSeg > 0:
                numSegments = np.append(numSegments, nSeg)
                sys.stdout.write(
                    "Iteration " + str(n_iter) + " has " + str(nSeg) + " segments...\n"
                )
        self.numSegments = numSegments
        self.maxIter = last_iter
