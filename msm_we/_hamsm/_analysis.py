import numpy as np
from scipy import sparse
from scipy.stats import linregress
import sys
import tqdm.auto as tqdm
from msm_we._logging import log
from msm_we.utils import inverse_iteration, is_connected

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msm_we import modelWE


class AnalysisMixin:

    Tmatrix = None
    pSS = None
    lagtime = None
    JtargetSS = None
    fit_parameters = {}

    def get_Tmatrix(self: "modelWE"):
        """
        Compute the transition matrix from the flux matrix.
        Corrects the "target" states to be true sink states.

        More specifically:
            - row-normalizes the flux matrix,
            - sets any states with 0 flux ot (i.e. sinks) to have 1.0
            - sets target bins to uniformly recycle into basis bins

        Updates:
            - `self.Tmatrix`

        Returns
        -------
        None
        """

        log.debug("Computing transition matrix")

        # Get a copy of the flux matrix
        fluxmatrix = self.fluxMatrix.copy()
        # Get the dimension of the flux matrix
        fluxmatrix_shape = np.shape(fluxmatrix)
        # Add up the total flux on each row, i.e. from each state
        fluxes_out = np.sum(fluxmatrix, 1)

        # For each state
        for state_idx in range(fluxmatrix_shape[0]):
            # For positive definite flux, set the matrix elements based on normalized fluxes
            if fluxes_out[state_idx] > 0:
                fluxmatrix[state_idx, :] = (
                    fluxmatrix[state_idx, :] / fluxes_out[state_idx]
                )

            # If the flux is zero, then consider it all self-transition
            # FIXME: this if can be an elif
            if fluxes_out[state_idx] == 0.0:
                fluxmatrix[state_idx, state_idx] = 1.0

        # Make the transition matrix a steady-state matrix
        # Identify the bins corresponding to target states.
        sinkBins = self.indTargets  # np.where(avBinPnoColor==0.0)

        # Get the number of sink bins
        n_sink_bins = np.shape(sinkBins)
        n_sink_bins = n_sink_bins[0]

        # TODO: The goal here is to correct for the transition probabilities out of the sink state. Not
        #   sure I fully understand this implementation, but that's the goal.
        # Correction: I think this sets all the sink bins to recycle to the basis
        sinkRates = np.zeros((1, self.nBins))
        sinkRates[0, self.indBasis] = 1.0 / self.indBasis.size
        tmatrix = fluxmatrix.copy()
        tmatrix[sinkBins, :] = np.tile(sinkRates, (n_sink_bins, 1))

        self.Tmatrix = tmatrix

    def get_eqTmatrix(self: "modelWE"):
        Mt = self.fluxMatrix.copy()
        n = np.shape(Mt)[0]
        indSpace = np.arange(n).astype(int)
        indSpace = np.setdiff1d(indSpace, np.append(self.indTargets, self.indBasis))
        Mt = Mt[indSpace, :]
        Mt = Mt[:, indSpace]
        nR = np.shape(Mt)
        sM = np.sum(Mt, 1)
        for iR in range(nR[0]):
            if sM[iR] > 0:
                Mt[iR, :] = Mt[iR, :] / sM[iR]
            if sM[iR] == 0.0:
                Mt[iR, iR] = 1.0
        self.Tmatrix = Mt

    def get_steady_state(
        self: "modelWE", flux_fractional_convergence=1e-4, max_iters=10
    ):
        """ "
        Get the steady-state distribution for the transition matrix.
        Uses scipy eigensolver to obtain an initial guess, then refines that using inverse iteration.

        Parameters
        ----------
        flux_fractional_convergence: (optional, default=1e-4) float
            Convergence of the

        max_iters: (optional, default=100) int

        Notes
        -----
        Transition matrices generated from WE data may have entries spanning many orders of magnitude, and may have
        extremely high condition numbers.
        Furthermore, the smallest entries may be those near the target state, which are also the most important
        for computing target fluxes, meaning values near machine precision can't just be truncated.
        All this means that floating-point error may substantially affect the results of eigensolvers, and may produce
        bad/negative/inaccurate values for small probability bins.

        In order to obtain better estimates, sparse matrices are used to reduce the number of floating point operations
        being performed.
        A stationary distribution is first estimated using scipy's eigensolver for sparse matrices.
        This is then used as an initial guess for the inverse iteration method, to further refine it.
        Convergence of the inverse iteration is determined using change in the flux estimate.
        """

        # Cast the matrix to a sparse matrix, to reduce floating point operations
        sparse_mat = sparse.csr_matrix(self.Tmatrix)

        algebraic_pss = self.get_steady_state_algebraic(
            max_iters=10, check_negative=False, set=False
        )

        # Get an initial flux estimate using the numpy algebraic solver
        # Call with _set=False so you don't actually update self.JtargetSS
        last_flux = self.get_steady_state_target_flux(pSS=algebraic_pss, _set=False)

        # ## Next, use that as an initial guess  for inverse iteration
        last_pSS = algebraic_pss

        log.debug(f"Initial flux: {last_flux}\n")
        flux_warned = False

        for N in range(max_iters):

            iterated = inverse_iteration(matrix=sparse_mat, guess=last_pSS)

            # Compute change in pSS
            pSS_change = np.sqrt(np.mean(np.power(iterated - last_pSS, 2)))
            log.debug(f"\t Change in SS: {pSS_change:.2e}")
            last_pSS = iterated

            # Compute change in target flux with the new pSS
            new_flux = self.get_steady_state_target_flux(pSS=last_pSS, _set=False)

            flux_change = new_flux - last_flux
            log.debug(
                f"\t Change in flux estimate: {flux_change:.2e} \t ({new_flux:.2e} raw)"
            )
            last_flux = new_flux

            # Set the convergence criterion after the iteration, in case the original result is really junk
            # Do it after so the first time this is calculated, it's using the result of the first iteration.
            # If the initial numpy estimate was really bad, it'll never be used.
            flux_convergence_criterion = last_flux * flux_fractional_convergence
            log.debug(f"\t Flux convergence criterion is {flux_convergence_criterion}")

            if N > 0:
                if last_flux == 0 and not flux_warned:
                    log.warning(
                        "Flux is 0, so steady-state solver will only converge after max iterations (using flux as "
                        "convergence criterion is not meaningful if it's 0!). "
                        "If you're looking for equilibrium, this is probably OK."
                        " Otherwise, take a look at why you have 0 flux."
                    )
                    flux_warned = True

            if abs(flux_change) < flux_convergence_criterion:
                log.info(
                    f"\nFlux converged to {last_flux:.4e} after {N + 1} iterations of inverse iteration."
                )
                break

            elif N == max_iters - 1 and not last_flux == 0:
                log.warning("Flux is nonzero and did not converge!")

        assert (last_pSS >= 0).all(), "Negative elements in pSS"
        assert last_flux >= 0, "Negative flux estimate from this pSS"

        log.debug("Done with steady-state estimation.")
        self.pSS = last_pSS

    def get_steady_state_algebraic(
        self: "modelWE", max_iters=1000, check_negative=True, set=True
    ):
        """
        Compute the steady-state distribution as the eigenvectors of the transition matrix.

        Updates:
            - `self.pSS`

        Parameters
        ----------
        max_iters: int, optional
            Defaults to 1000. Number of power-method iterations to run if the numpy eigensolver returns negative elements.

        check_negative: bool, optional
            Defaults to True. If True, then raise an Exception if there are negative elements in the normalized pSS.

        Returns
        -------
        None
        """

        # log.warning(
        #     "get_steady_state_algebraic() will be deprecated soon. Use get_steady_state() instead, which has"
        #     " a more robust eigensolver."
        # )

        log.debug("Computing steady-state from eigenvectors")

        eigenvalues, eigenvectors = np.linalg.eig(np.transpose(self.Tmatrix))

        pSS = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])

        pSS = pSS.squeeze()

        assert not np.isclose(np.sum(pSS), 0), "Steady-state distribution sums to 0!"
        pSS = pSS / np.sum(pSS)

        # The numpy eigensolver is iterative, and approximate. Given that data from WE often spans many orders of
        #   magnitude, we'll sometimes run into situations where our populations span more than machine precision.
        #   This causes hiccups in the eigensolver. However, we can't just zero these out (as attempted above),
        #   because these values are often important.
        # So, if there are any negative elements, try to correct the NP eigensolver result using the matrix method
        if sum(pSS < 0) > 0 and max_iters > 0:
            log.info(
                "Negative elements in pSS after normalization, attempting to correct with matrix power method."
            )
            pSS_last = pSS
            _tmatrix = self.Tmatrix.copy()

            for N in range(max_iters):

                pSS_new = _tmatrix.T @ pSS_last
                num_negative_elements = sum(pSS_new < 0)
                if num_negative_elements == 0:
                    log.info(f"Corrected to semidefinite pSS in {N} iterations")
                    break

                pSS_last = pSS_new
                _tmatrix = np.matmul(self.Tmatrix, _tmatrix)

            if N == max_iters - 1:
                log.warning(
                    "Power method did NOT obtain semidefinite pSS. Some negative values remain. This is weird, and you"
                    " should try to figure out why this is happening."
                )
            else:
                pSS = pSS_new

        elif max_iters == 0:
            log.debug(
                "Negative elements in pSS after normalization, "
                "but max_iters=0 so not attempting to correct with power method."
            )

        if not np.all(pSS >= 0):

            if check_negative:
                assert np.all(
                    pSS >= 0
                ), f"Some negative elements in steady-state distribution: {pSS}"
            else:
                log.warning(
                    "Some negative  elements in pSS... Ignoring, and setting model.pSS anyways."
                )

        if set:
            self.pSS = pSS
        else:
            return pSS

    def get_steady_state_matrixpowers(self: "modelWE", conv):
        """
        Compute the steady-state distribution using the matrix power method.

        Updates:
            - `self.pSS`

        Parameters
        ----------
        conv: numeric
            Convergence criterion for iteration.

        Returns
        -------
        None
        """
        max_iters = 10000
        Mt = self.Tmatrix.copy()
        dconv = 1.0e100
        N = 1
        pSS = np.mean(Mt, 0)
        pSSp = np.ones_like(pSS)
        while dconv > conv and N < max_iters:
            Mt = np.matmul(self.Tmatrix, Mt)
            N = N + 1
            if N % 10 == 0:
                pSS = np.mean(Mt, 0)
                pSS = pSS / np.sum(pSS)
                dconv = np.sum(np.abs(pSS - pSSp))
                pSSp = pSS.copy()
                sys.stdout.write("N=" + str(N) + " dconv: " + str(dconv) + "\n")
                self.pSS = pSS.copy()

    def get_steady_state_target_flux(self: "modelWE", pSS=None, _set=True):
        """
        Get the total flux into the target state(s).

        Updates:
            - `self.lagtime`
            - `self.JtargetSS`

        Parameters
        ----------
        pSS: (optional) array-like
            Steady-state distribution. If nothing provided, then use self.pSS

        _set: (optional) boolean
            If True, then update self.JtargetSS and self.lagtime. If False, then just return the value of JtargetSS.

        Returns
        -------
        None
        """

        # If this matrix isn't connected from source to target, then it's not possible to get an MFPT directly out of it
        sparse_mat = sparse.csr_matrix(self.Tmatrix)
        connected = is_connected(
            sparse_mat, self.indBasis, self.indTargets, directed=True
        )
        if not connected:
            log.critical(
                "There is no path in this matrix from the basis to the target, so no MFPT can be calculated."
            )
            return -1

        Mss = self.Tmatrix

        # If no pSS was provided, then pull from self
        if pSS is None:
            pSS = np.squeeze(np.array(self.pSS))

        lagtime = self.tau * (self.n_lag + 1)

        # Get a list of all the states that AREN'T targets, since we want to sum up
        nTargets = self.indTargets.size
        indNotTargets = np.setdiff1d(range(self.nBins), self.indTargets)
        # log.debug(f"Non-target states are those with index {indNotTargets}")

        Jt = 0.0
        # Add up the total flux into each of the targets
        for j in range(nTargets):
            log.debug(
                f"Processing flux into target state with index {self.indTargets[j]}"
            )

            jj = self.indTargets[j]

            Jt = Jt + np.sum(
                np.multiply(
                    pSS[indNotTargets],
                    np.squeeze(
                        np.array(Mss[indNotTargets, jj * np.ones_like(indNotTargets)])
                    ),
                )
            )

        if _set:
            self.lagtime = lagtime
            self.JtargetSS = Jt / self.lagtime
        else:
            return Jt / lagtime

    def get_flux(self: "modelWE"):
        """
        Get the measured flux (i.e. from the flux matrix) into the target.
        """

        # J = np.zeros_like(self.binCenters)
        nBins = np.shape(self.targetRMSD_centers)[0]
        J = np.zeros(nBins)
        fluxMatrix = self.fluxMatrix.copy()

        centers = self.targetRMSD_centers.copy()
        centers[self.indBasis] = self.basis_bin_centers
        centers[self.indTargets] = self.target_bin_centers

        # TODO: This is an assumption! You may want to sort by other pcoords, or other dimensions of the pcoord.
        pcoord_to_sort = 0
        log.warning(
            "Assuming fluxes should be sorted by pcoord dimension 0 -- this is currently built in to the code."
        )
        sorted_centers = np.argsort(centers[:, pcoord_to_sort])

        ordered_flux_matrix = fluxMatrix[sorted_centers][:, sorted_centers].copy()

        for i in range(0, nBins - 1):
            indBack = range(i + 1)
            indForward = range(i + 1, nBins)
            JR = 0.0
            JF = 0.0
            for j in indBack:
                JR = JR + np.sum(
                    ordered_flux_matrix[indForward, j * np.ones_like(indForward)]
                )
            for j in indForward:
                JF = JF + np.sum(
                    ordered_flux_matrix[indBack, j * np.ones_like(indBack)]
                )
            J[sorted_centers[i]] = JR - JF
            self.J = J

        # Now check if the flux profile is "tilted the wrong way".
        #   It's been observed that with poor quality haMSMs, continued restarting can actually drive the system to
        #   a *higher* flux than the steady-state. In other words, can drive the system to an incorrect, too-fast
        #   steady-state.
        #   If this happens, letting it run without restarting will allow it to relax back to the real steady-state.

        if not hasattr(self, "all_centers") or self.all_centers is None:
            log.info(
                "This model was built before the self.all_centers attribute was added. Populating it now..."
            )
            self.update_sorted_cluster_centers()

        slope, intercept, r_value, p_value, std_err = linregress(
            self.all_centers, self.J / self.tau
        )

        self.fit_parameters = {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }

        # Check if slope is overcorrected
        target_before_basis = any(self.target_bin_centers < self.basis_bin_centers)

        # Full disclosure: I originally wrote this as `slope_overcorrected = slope * (-1 * !target_before_basis) < 0`
        #   and I want it to be on record that I didn't

        if target_before_basis:
            self.slope_overcorrected = slope < 0
        else:
            self.slope_overcorrected = slope > 0

        if self.slope_overcorrected:
            log.warning(
                "Flux profile appears to be overcorrected! In other words, the flux profile appears higher near the "
                "target than the basis. "
                "This suggests restarting may have driven the system past its true steady-state. "
                "This WE run should be continued without restarting, and allowed to relax. "
            )

    def get_flux_committor(self: "modelWE"):
        """
        Get the flux binned according to committors

        Returns
        -------

        """

        # J = np.zeros_like(self.targetRMSD_centers)
        # nBins = np.shape(self.binCenters)[0]
        nBins = np.shape(self.targetRMSD_centers)[0]
        J = np.zeros(nBins)
        fluxMatrix = self.fluxMatrix.copy()
        indq = np.argsort(np.squeeze(1.0 - self.q))
        fluxMatrix = fluxMatrix[indq, :]
        fluxMatrix = fluxMatrix[:, indq]

        for i in tqdm.tqdm(
            range(0, nBins - 1), desc="Obtaining pseudocommittor-fluxes"
        ):
            indBack = range(i + 1)
            indForward = range(i + 1, nBins)
            JR = 0.0
            JF = 0.0
            for j in indBack:
                JR = JR + np.sum(fluxMatrix[indForward, j * np.ones_like(indForward)])
            for j in indForward:
                JF = JF + np.sum(fluxMatrix[indBack, j * np.ones_like(indBack)])

            J[indq[i]] = JR - JF

            self.Jq = J.squeeze() / self.tau
            # sys.stdout.write("%s " % i)

    def evolve_target_flux(self: "modelWE"):
        Mss = self.Tmatrix
        probTransient = self.probTransient
        nT = np.shape(probTransient)[0]
        Jtarget = np.zeros(nT)
        self.lagtime = self.tau * (self.n_lag + 1)
        nTargets = self.indTargets.size
        indNotTargets = np.setdiff1d(range(self.nBins), self.indTargets)
        JtargetTimes = np.zeros(nT)
        for iT in range(nT):
            Jt = 0.0
            for j in range(nTargets):
                jj = self.indTargets[j]
                Jt = Jt + np.sum(
                    np.multiply(
                        probTransient[iT, indNotTargets],
                        Mss[indNotTargets, jj * np.ones_like(indNotTargets)],
                    )
                )
            Jtarget[iT] = Jt
            JtargetTimes[iT] = iT * self.nStore * self.lagtime
        self.Jtarget = Jtarget / self.lagtime
        self.JtargetTimes = JtargetTimes

    def get_committor(self: "modelWE", conv=1e-5):
        """
        Iteratively obtain an estimate of the committor.

        1. Take the flux matrix, and normalize it into a transition matrix.

        2. Apply two-sided absorbing boundary conditions by setting self-transition probabilities for the basis and
            target states to 1.0, and all transitions out to 0.0.

        3. Starting with an initial committor "guess" of all 1s, iteratively multiply the guess by the transition matrix
            until convergence is below conv.

        Updates:
            - self.q

        Parameters
        ----------
        conv: numerical
            Convergence criteria for committor calculation. Calculation stops when the total difference between q_p and q
            is less than this.

        Returns
        -------

        """

        self.print_pseudocommittor_warning()

        _fluxMatrix = self.fluxMatrix.copy()

        # Number of bins/states in the fluxmatrix

        num_bins = np.shape(_fluxMatrix)
        flux_out = np.sum(_fluxMatrix, 1)

        for bin_idx in range(num_bins[0]):

            # If the flux out is positive semidefinite, then normalize that row to get transition probabilities
            if flux_out[bin_idx] > 0:
                _fluxMatrix[bin_idx, :] = _fluxMatrix[bin_idx, :] / flux_out[bin_idx]

            # If the flux out is zero, then just set the self-transition to 1.0
            # (This probably keeps a calculation later clean?)
            if flux_out[bin_idx] == 0.0:
                _fluxMatrix[bin_idx, bin_idx] = 1.0

        #  The basis states are set to have 0 flux out, and only self-transition = 1.0.
        #   These are the two-sided absorbing BCs for the committor.
        sinkBins = self.indBasis
        for ii in sinkBins:
            _fluxMatrix[ii, :] = np.zeros((1, self.nBins))
            _fluxMatrix[ii, ii] = 1.0

        q = np.zeros((self.nBins, 1))

        # Committor to the target state is 1 by definition
        q[self.indTargets, 0] = 1.0

        dconv = 100.0
        progress = 0
        qp = np.ones_like(q)

        # Iteratively update the committor estimate until it converges to stationarity
        # (The committor is the stationary distribution for two-sided absorbing boundary conditions)
        with tqdm.tqdm(total=-np.log10(conv)) as pbar:
            while dconv > conv:
                q[self.indTargets, 0] = 1.0
                q[self.indBasis, 0] = 0.0
                q = np.matmul(_fluxMatrix, q)
                dconv = np.sum(np.abs(qp - q))

                # Update with progress since last iter
                log.debug("convergence: " + str(dconv) + "\n")
                pbar.update(max(0, -np.log10(dconv)) - progress)
                progress = max(0, -np.log10(dconv))

                qp = q.copy()
                self.q = q

        self.q = q.squeeze()

    # TODO: This should probably just be a call to get_committor, followed by self.q = 1 - self.q
    def get_backwards_committor(self: "modelWE", conv):
        Mt = self.fluxMatrix.copy()
        nR = np.shape(Mt)
        sM = np.sum(Mt, 1)
        for iR in range(nR[0]):
            if sM[iR] > 0:
                Mt[iR, :] = Mt[iR, :] / sM[iR]
            if sM[iR] == 0.0:
                Mt[iR, iR] = 1.0
        sinkBins = self.indTargets  # np.where(avBinPnoColor==0.0)
        nsB = np.shape(sinkBins)
        nsB = nsB[0]
        for ii in sinkBins:
            Mt[ii, :] = np.zeros((1, self.nBins))
            Mt[ii, ii] = 1.0
        Mt = np.transpose(Mt)  # time reversal
        q = np.zeros((self.nBins, 1))
        q[self.indBasis, 0] = 1.0
        dconv = 100.0
        qp = np.ones_like(q)
        while dconv > conv:
            q[self.indBasis, 0] = 1.0
            q[self.indTargets, 0] = 0.0
            q = np.matmul(Mt, q)
            dconv = np.sum(np.abs(qp - q))
            sys.stdout.write("convergence: " + str(dconv) + "\n")
            qp = q.copy()
            self.qm = q
        self.q = q.copy()

    @staticmethod
    def print_pseudocommittor_warning():
        log.info(
            "Note that, if steady-state weighted ensemble data is being analyzed, this is a 'pseudocommittor' "
            "and not a true committor as a result of being constructed from a one-way ensemble."
        )
