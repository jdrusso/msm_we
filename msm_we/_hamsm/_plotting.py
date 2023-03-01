from matplotlib import pyplot as plt
import numpy as np
from msm_we._logging import log

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msm_we import modelWE

# Using the tkinter backend makes matplotlib run better on a cluster, maybe?
# import matplotlib
# matplotlib.use("TkAgg")


class PlottingMixin:
    def plot_flux_committor_pcoordcolor(
        self: "modelWE",
        nwin=1,
        ax=None,
        pcoord_to_use=0,
        **_plot_args,
    ):

        _models = [self]
        _model_labels = ["main_model"]

        plot_args = {
            "linewidth": 2,
            "s": 50,
            "marker": ">",
            "cmap": plt.cm.rainbow.reversed(),
            "alpha": 0.7,
        }

        plot_args.update(_plot_args)

        if ax is None:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

        for i, (_model, _label) in enumerate(zip(_models[::-1], _model_labels[::-1])):

            if _model is None:
                continue

            if not hasattr(_model, "q"):
                log.info(
                    f"Committors have not yet been generated for {_label}, generating now."
                )
                _model.get_committor()

            if not hasattr(_model, "Jq"):
                log.info(
                    f"Committor-fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux_committor()

            n_bins = _model.targetRMSD_centers.shape[0]
            Jq_avg = _model.Jq.copy()
            Jq_std = np.zeros_like(Jq_avg)

            q_avg = np.zeros_like(Jq_avg)

            indq = np.argsort(np.squeeze(1.0 - _model.q))
            for _i in range(n_bins - 1, nwin - 1, -1):
                iav = _i - nwin
                ind = range(_i - nwin, _i)
                Jq_avg[iav] = np.mean(_model.Jq[ind])
                Jq_std[iav] = np.std(_model.Jq[ind])
                q_avg[iav] = np.mean(_model.q[indq[ind]])

            indPlus = np.where(Jq_avg > 0.0)

            lines = ax.scatter(
                q_avg[indPlus],
                np.squeeze(Jq_avg[indPlus]),
                c=_model.targetRMSD_centers[indPlus, pcoord_to_use],
                label=f"{_label} flux toward target",
                **plot_args,
            )

        print("Plotting committor")
        ax.figure.colorbar(lines, label=f"Progress Coordinate {pcoord_to_use}")

        ax.set_xlim([-0.1, 1.1])

        ax.set_title("Full-data model")
        ax.set_yscale("log")
        ax.set_xlabel("Pseudocommittor")
        ax.set_ylabel("Flux (weight/second)")
        self.print_pseudocommittor_warning()

        return ax, lines

    def plot_flux_committor(
        self: "modelWE",
        nwin=1,
        ax=None,
        save=False,
        suppress_validation=False,
        _from_colors=None,
        _to_colors=None,
        **_plot_args,
    ):

        _models = [self]
        _models.extend(self.validation_models)

        _model_labels = ["main_model"]
        _model_labels.extend(
            [f"validation_model_{n}" for n in range(len(self.validation_models))]
        )

        cross_validation_done = (
            self.validation_models is not None and len(self.validation_models) > 1
        )
        if not cross_validation_done:
            log.critical(
                "No cross-validation models have been generated! Do this before making plots."
            )
            if not suppress_validation:
                raise Exception("Perform cross-validation before plotting results.")

        own_ax = False
        if ax is None:
            own_ax = True
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

        plot_args = {"linewidth": 2, "markersize": 10}
        plot_args.update(_plot_args)

        n_models = len(_models)
        from_colors = [
            plt.get_cmap("cool")(0.25 + (0.75 * i / n_models)) for i in range(n_models)
        ]
        to_colors = [
            plt.get_cmap("hot")(0.25 + (0.5 * i / n_models)) for i in range(n_models)
        ]

        for i, (_model, _label) in enumerate(zip(_models, _model_labels)):

            if _model is None:
                continue

            if not hasattr(_model, "q"):
                log.info(
                    f"Committors have not yet been generated for {_label}, generating now."
                )
                _model.get_committor()

            if not hasattr(_model, "Jq"):
                log.info(
                    f"Committor-fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux_committor()

            n_bins = _model.targetRMSD_centers.shape[0]
            Jq_avg = _model.Jq.copy()
            Jq_std = np.zeros_like(Jq_avg)

            q_avg = np.zeros_like(Jq_avg)

            indq = np.argsort(np.squeeze(1.0 - _model.q))
            for _i in range(n_bins - 1, nwin - 1, -1):
                iav = _i - nwin
                ind = range(_i - nwin, _i)
                Jq_avg[iav] = np.mean(_model.Jq[ind])
                Jq_std[iav] = np.std(_model.Jq[ind])
                q_avg[iav] = np.mean(_model.q[indq[ind]])

            indPlus = np.where(Jq_avg > 0.0)
            indMinus = np.where(Jq_avg < 0.0)

            if _from_colors is not None:
                plot_args["color"] = _from_colors[i]
            else:
                plot_args["color"] = from_colors[i]

            ax.plot(
                q_avg[indMinus],
                -np.squeeze(Jq_avg[indMinus]),
                "<",
                # color=to_target_colors[i%len(to_target_colors)],
                label=f"{_label} flux toward source/basis",
                **plot_args,
            )

            if _to_colors is not None:
                plot_args["color"] = _to_colors[i]
            else:
                plot_args["color"] = to_colors[i]

            ax.plot(
                q_avg[indPlus],
                np.squeeze(Jq_avg[indPlus]),
                ">",
                # color=from_target_colors[i%len(from_target_colors)],
                label=f"{_label} flux toward target",
                **plot_args,
            )

        ax.set_yscale("log")
        ax.set_xscale("linear")
        ax.set_xlim([-0.1, 1.1])
        ax.set_xlabel("Pseudocommittor")
        ax.set_ylabel("Flux (weight/second)")
        self.print_pseudocommittor_warning()

        # Plot linear fit
        if not hasattr(self, "fit_parameters"):
            log.info(
                "This appears to be a model from before flux profile curve-fits were implemented. Doing that "
                "curve fit and recalculating the flux profile now."
            )
            self.get_flux()

        slope = self.fit_parameters["slope"]
        intercept = self.fit_parameters["intercept"]
        r_value = self.fit_parameters["r_value"]
        # Omit the first and last, because those bin centers may be weird for bins reaching to infinity
        q_sort = np.argsort(self.q)[1:-1]
        ax.plot(
            self.q[q_sort],
            slope * self.all_centers[q_sort] + intercept,
            color="gray",
            label=f"Linear fit to flux profile\nm={slope:.1e}, b={intercept:.1e}\nr^2={r_value ** 2:.1e}\n",
        )

        self.check_display_overcorrection_warning(ax)

        if own_ax:
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            fig.tight_layout()

        if save:
            plot_filename = f"{self.modelName}_flux_committor.pdf"
            log.info(f"Saving flux-committor plot to {plot_filename}")
            plt.savefig(plot_filename)

        return ax

    def plot_flux(
        self: "modelWE",
        custom_name=None,
        ax=None,
        save=False,
        suppress_validation=False,
        _from_colors=None,
        _to_colors=None,
        pcoord_to_use=0,
        **_plot_args,
    ):
        """
        Make, and save, a plot of the fluxes along the RMSD.  get_flux() must be run before this.

        Parameters
        ----------
        custom_name : str (optional)
            The name for the saved plot. Defaults to flux_s<first iter>_e<last iter>.png

        Returns
        -------

        """

        _models = [self]
        _models.extend(self.validation_models)

        _model_labels = ["main_model"]
        _model_labels.extend(
            [f"validation_model_{n}" for n in range(len(self.validation_models))]
        )

        cross_validation_done = (
            self.validation_models is not None and len(self.validation_models) > 1
        )
        if not cross_validation_done:
            log.critical(
                "No cross-validation models have been generated! Do this before making plots."
            )
            if not suppress_validation:
                raise Exception("Perform cross-validation before plotting results.")

        own_ax = False
        if ax is None:
            own_ax = True
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)

        plot_args = {"linewidth": 2, "markersize": 10}
        plot_args.update(_plot_args)

        n_models = len(_models)
        from_colors = [
            plt.get_cmap("cool")(0.25 + (0.75 * i / n_models)) for i in range(n_models)
        ]
        to_colors = [
            plt.get_cmap("hot")(0.25 + (0.5 * i / n_models)) for i in range(n_models)
        ]

        # Draw the basis/target boundaries in this pcoord
        self.draw_basis_target_boundaries(ax, pcoord_to_use)

        for i, (_model, _label) in enumerate(zip(_models, _model_labels)):

            if _model is None:
                continue

            if not hasattr(_model, "J"):
                log.info(
                    f"Fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux()

            J = _model.J / _model.tau

            binCenters = _model.targetRMSD_centers[:, pcoord_to_use]
            binCenters[_model.indTargets] = _model.target_bin_centers
            binCenters[_model.indBasis] = _model.basis_bin_centers

            indPlus = np.where(J > 0.0)
            indMinus = np.where(J < 0.0)

            if _to_colors is not None:
                plot_args["color"] = _to_colors[i]
            else:
                plot_args["color"] = to_colors[i]

            ax.plot(
                binCenters[indPlus],
                np.squeeze(J[indPlus]),
                ">",
                label=f"{_label} flux toward target",
                **plot_args,
            )

            if _from_colors is not None:
                plot_args["color"] = _from_colors[i]
            else:
                plot_args["color"] = from_colors[i]

            ax.plot(
                binCenters[indMinus],
                -np.squeeze(J[indMinus]),
                "<",
                label=f"{_label} flux toward source/basis",
                **plot_args,
            )

        # Plot linear fit
        if not hasattr(self, "fit_parameters"):
            log.info(
                "This appears to be a model from before flux profile curve-fits were implemented. Doing that "
                "curve fit and recalculating the flux profile now."
            )
            self.get_flux()

        slope = self.fit_parameters["slope"]
        intercept = self.fit_parameters["intercept"]
        r_value = self.fit_parameters["r_value"]
        # Don't plot first and last points -- for bins spanning to infinity, these might be weird.
        log.debug(
            f"Doing linear fit from {self.sorted_centers[:10]} to {self.sorted_centers[-10:]}"
        )
        ax.plot(
            self.all_centers[self.sorted_centers],
            slope * self.all_centers[self.sorted_centers] + intercept,
            color="gray",
            label=f"Linear fit (m={slope:.1e}, b={intercept:.1e}, r^2={r_value ** 2:.1e})",
        )

        self.check_display_overcorrection_warning(ax)

        ax.set_yscale("log")
        ax.set_xlabel(f"Pcoord {pcoord_to_use}")
        ax.set_ylabel("Flux (weight/second)")

        if own_ax:
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            fig.tight_layout()

        if save:
            if custom_name is not None:
                plot_filename = custom_name
            else:
                plot_filename = f"{self.modelName}_flux.pdf"
            log.info(f"Saving flux plot to {plot_filename}")
            plt.savefig(plot_filename)

        return ax

    def draw_basis_target_boundaries(self: "modelWE", ax, pcoord_to_use=0):
        [
            ax.axvline(
                bound, color="r", linestyle="--", label=["", "Target boundary"][i == 0]
            )
            for i, bound in enumerate(self.target_pcoord_bounds[pcoord_to_use, :])
        ]
        [
            ax.axvline(
                bound,
                color="b",
                linestyle="--",
                label=["", "Basis/Source boundary"][i == 0],
            )
            for i, bound in enumerate(self.basis_pcoord_bounds[pcoord_to_use, :])
        ]

    def check_display_overcorrection_warning(self: "modelWE", ax):

        if not self.slope_overcorrected:
            return

        log.warning(
            "Flux profile appears to be overcorrected! In other words, the flux profile appears higher near the "
            "target than the basis. "
            "This suggests restarting may have driven the system past its true steady-state. "
            "This WE run should be continued without restarting, and allowed to relax. "
        )
        ax.text(
            0.5,
            -0.25,
            "WARNING: Possible flux overcorrection! WE should be continued without restarting now.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            weight="bold",
        )

    def plot_committor(self: "modelWE"):
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(self.targetRMSD_centers[:, 0], self.q, s=15, c="black")
        plt.yscale("log")
        plt.ylabel("Pseudocommittor to target", fontsize=12)
        plt.xlabel("Average microstate pcoord", fontsize=12)

        self.print_pseudocommittor_warning()

        plt.pause(1)
        fig.savefig(
            self.modelName
            + "_s"
            + str(self.first_iter)
            + "_e"
            + str(self.last_iter)
            + "committor.png"
        )

    def get_coarse_flux_profile(self: "modelWE", min_coarse_bins=10):
        """
        Computes a coarse-grained flux profile.
        Specifically, this downsamples the bin in pcoord-space, and coarse-grains into those.
        This is a more meaningful approach than grouping N consecutive bins, because bins may not be distributed
        uniformly through pcoord-space.

        Parameters
        ----------
        min_coarse_bins: int, Lower bound on number of coarse-bins

        Returns
        -------
        (flux profile, coarse bin boundaries)

        Notes
        -------
        This implements a more efficient (but equivalent) flux profile calculation than the one in get_flux().
        """

        binCenters = self.all_centers

        # Downsample by a factor of 10, but to no fewer than 10 bins
        n_coarse_bins = max(min_coarse_bins, self.n_clusters // 10)

        bin_boundaries = np.concatenate(
            [
                np.linspace(
                    binCenters[:-2].min() - 0.1,
                    binCenters[:-2].max() + 0.1,
                    n_coarse_bins,
                ),
            ]
        )

        coarse_bin_assignments = np.digitize(
            binCenters[:-2], bin_boundaries, right=True
        )

        flux_matrix = self.fluxMatrix.copy()

        new_net_fluxes = np.full(n_coarse_bins, fill_value=np.nan)
        for coarse_bin_idx in range(n_coarse_bins):
            # Naming conventions in this assume flux goes from left to right.
            # The boundary flux is crossing in this picture is therefore the left hand edge of each state.

            # All microstates
            microstates_forward = np.where(
                coarse_bin_assignments <= coarse_bin_idx
            )  # .squeeze()
            microstates_backward = np.where(
                coarse_bin_assignments > coarse_bin_idx
            )  # .squeeze()

            flux_forward = np.sum(
                flux_matrix[microstates_backward][:, microstates_forward]
            )
            flux_backward = np.sum(
                flux_matrix[microstates_forward][:, microstates_backward]
            )

            net_flux = flux_forward - flux_backward

            new_net_fluxes[coarse_bin_idx] = net_flux

        return new_net_fluxes, bin_boundaries

    def plot_coarse_flux_profile(self: "modelWE", pcoord_to_use=0):
        # TODO: Standardize this with the other plotting functions

        binCenters = self.all_centers

        intercept = self.fit_parameters["intercept"]
        slope = self.fit_parameters["slope"]
        r_value = self.fit_parameters["r_value"]

        new_net_fluxes, bin_boundaries = self.get_coarse_flux_profile()

        is_backwards = np.argwhere(new_net_fluxes < 0)
        is_forward = np.argwhere(new_net_fluxes >= 0)

        fig, ax = plt.subplots()

        ax.scatter(
            bin_boundaries[is_backwards],
            abs(new_net_fluxes[is_backwards] / self.tau),
            color="b",
            marker=">",
            s=20,
            label="Flux toward source/basis",
        )

        ax.plot(
            bin_boundaries[is_forward],
            new_net_fluxes[is_forward] / self.tau,
            "r<",
            alpha=1.0,
            linestyle="-",
            linewidth=1,
            label="Flux toward target",
        )

        self.check_display_overcorrection_warning(ax)
        self.draw_basis_target_boundaries(ax, pcoord_to_use)

        sorted_centers = np.argsort(binCenters)
        ax.plot(
            binCenters[sorted_centers],
            slope * binCenters[sorted_centers] + intercept,
            color="gray",
            label=f"Linear fit (m={slope:.1e}, b={intercept:.1e}, r^2={r_value ** 2:.1e})",
        )

        ax.set_ylabel("Flux (weight/second)")
        ax.set_xlabel(f"Pcoord {pcoord_to_use}")
        ax.set_yscale("log")

        ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")

        return fig, ax
