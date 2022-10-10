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
                log.warning(
                    f"Committors have not yet been generated for {_label}, generating now."
                )
                _model.get_committor()

            if not hasattr(_model, "Jq"):
                log.warning(
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
                log.warning(
                    f"Committors have not yet been generated for {_label}, generating now."
                )
                _model.get_committor()

            if not hasattr(_model, "Jq"):
                log.warning(
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

        for i, (_model, _label) in enumerate(zip(_models, _model_labels)):

            if _model is None:
                continue

            if not hasattr(_model, "J"):
                log.warning(
                    f"Fluxes have not yet been generated for {_label}, generating now."
                )
                _model.get_flux()

            J = _model.J / _model.tau

            binCenters = _model.targetRMSD_centers[:, pcoord_to_use]
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
