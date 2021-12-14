"""
Adapted from the original NMpathAnalysis package,
https://github.com/ZuckermanLab/NMpathAnalysis
"""
import numpy as np
from copy import deepcopy
import tqdm.auto as tqdm

import msm_we.utils as utils
from msm_we.utils import Interval


class DirectFPT:
    @classmethod
    def mean_fpts(
        cls,
        trajectories,
        stateA=None,
        stateB=None,
        discrete=True,
        n_variables=None,
        lag_time=1,
    ):
        """Empirical mean first passage times (MFPTs) calculation (no model
        involved) by tracing the trajectories. Notice the difference between
        notation between FPTs and MFPTs.

        Parameters:
        -----------
        trajectories:   List of trajectories [traj1, traj2, traj4], each trajectory
                        can be a one dimensional array, e.g.,
                            [[1,2,1, ...], [0,1,1, ...], ... ]
                        or a multidimensional array (matrix) where each column
                        represents the evolution of a variable.

                        Important: If a single trajectory is given as argument it
                        also has to be inside a list (e.g. [traj1])

        stateA, stateB: List of integers
                        If the trajectories are discrete (discrete = True), both
                        states are a list of indexes. However, if the trajectories
                        are not discrete, the states are "interval" objects
                        (see Interval class).

        lag_time:       integer
                        Lag time used, the trajectory is "observed" every lag_time
                        time steps

        discrete:       boolean
                        False when the trajectories are are not discrete. In that
                        case the macrostates stateA and stateB are considered
                        interval objects.

        n_variables:    integer
                        If the trajectory is space continuous,the number of
                        variables/dimensions is needed. In this case every
                        trajectory inside "trajectories" should have the same
                        number of dimensions.

        Returns
        -------
        A dictionary with the keys: 'mfptAB', 'std_err_mfptAB', 'mfptBA',
        'std_err_mfptBA' and the corresponding values. Those values are already
        multiplied by the lag_time used (not the physical units).
        """

        passage_timesAB, passage_timesBA, tb_values = cls.fpts(
            trajectories, stateA, stateB, discrete, n_variables, lag_time
        )
        n_AB = len(passage_timesAB)
        n_BA = len(passage_timesBA)

        if sum(passage_timesAB):
            mfptAB = float(sum(passage_timesAB)) / n_AB
            std_err_mfptAB = np.std(passage_timesAB) / np.sqrt(n_AB)
        else:
            print("WARNING: No A->B events observed")
            mfptAB = "NaN"
            std_err_mfptAB = "NaN"

        if sum(passage_timesBA):
            mfptBA = float(sum(passage_timesBA)) / n_BA
            std_err_mfptBA = np.std(passage_timesBA) / np.sqrt(n_BA)
        else:
            print("WARNING: No B->A events observed")
            mfptBA = "NaN"
            std_err_mfptBA = "NaN"

        kinetics = {
            "mfptAB": mfptAB,
            "std_err_mfptAB": std_err_mfptAB,
            "mfptBA": mfptBA,
            "std_err_mfptBA": std_err_mfptBA,
        }

        print("Number of A->B/B->A  events: {}/{}".format(n_AB, n_BA))

        return kinetics

    @classmethod
    def fpts(
        cls,
        trajectories,
        stateA=None,
        stateB=None,
        discrete=True,
        n_variables=None,
        lag_time=1,
    ):
        """Empirical first passage times (FPTs) calculation (no model involved)
        by tracing the trajectories. IMPORTANT: Notice the difference in notation
        between FPTs and MFPTs.

        Parameters:
        -----------
        trajectories:   List of trajectories [traj1, traj2, traj4], each trajectory
                        can be a one dimensional array, e.g.,
                            [[1,2,1, ...], [0,1,1, ...], ... ]
                        or a mutidimensional array (matrix) where each column
                        represents the evolution of a variable.

                        Important: If a single trajectory is given as argument it
                        also has to be inside a list (e.g. [traj1])

        stateA, stateB: List of integers
                        If the trajectories are discrete (discrete = True), both
                        states are a list of indexes. However, if the trajectories
                        are not discrete, the states are "interval" objects
                        (see Interval class).

        lag_time:       integer
                        Lag time used, the trajectory is "observed" every lag_time
                        time steps

        discrete:       boolean
                        False when the trajectories are are not discrete. In that
                        case the macrostates stateA and stateB are considered
                        interval objects.

        n_variables:    integer
                        If the trajectory is space continuous,the number of
                        variables/dimensions is needed. In this case every
                        trajectory inside "trajectories" should have the same
                        number of dimensions.

        Returns
        -------
        A tuple of two 1D-ndarray (array1, array2), the first one contains the
        observed first passage times A->B and the second one the FPTs B->A. Those
        values are already multiplied by the lag_time used (not the physical units)
        """

        if (stateA is None) or (stateB is None):
            raise Exception(
                "The final and initial states have " "to be defined to compute the MFPT"
            )

        if not discrete:
            """
            The states are considered/transformed-to intervals if the Ensemble
            is a set of continuous trajectories
            """
            if n_variables is None:
                raise Exception(
                    "In continuous trajectories the number of " "variables is needed"
                )

            stateA = Interval(stateA, n_variables)
            stateB = Interval(stateB, n_variables)

        passage_timesAB = []
        passage_timesBA = []
        tb_values = []

        for traj in trajectories:
            previous_color = "Unknown"
            tb_counter = 0  # event duration counter
            fpt_counter = 0  # first passage time counter
            for i in range(0, len(traj), lag_time):
                snapshot = traj[i]
                tb_counter += 1
                # state and color determination
                if snapshot in stateA:
                    color = "A"
                elif snapshot in stateB:
                    color = "B"
                else:
                    color = previous_color
                    tb_counter += 1

                # passage times
                if (color == "A") or (color == "B"):
                    fpt_counter += 1

                if previous_color == "A" and color == "B":
                    tb_values.append(tb_counter)
                    passage_timesAB.append(fpt_counter)
                    fpt_counter = 0
                elif previous_color == "B" and color == "A":
                    tb_values.append(tb_counter)
                    passage_timesBA.append(fpt_counter)
                    fpt_counter = 0
                elif previous_color == "Unknown" and (color == "A" or color == "B"):
                    fpt_counter = 0

                if (snapshot in stateA) or (snapshot in stateB):
                    tb_counter = 0

                previous_color = color

        passage_timesAB = np.array(passage_timesAB) * lag_time
        passage_timesBA = np.array(passage_timesBA) * lag_time

        return passage_timesAB, passage_timesBA, tb_values


class MatrixFPT:
    """Define a base class for calculating FPTs using transition matrix"""

    @classmethod
    def mean_fpts(cls, tmatrix, stateA, stateB, lag_time=1):
        """Calculting mean-first passave time for at transition matrix

        Derived class will implement this function differently.
        """
        pass

    @classmethod
    def directional_mfpt(
        cls, transition_matrix, stateA, stateB, ini_probs=None, lag_time=1
    ):
        """Computes the mean-first passage time in a single direction using a recursive procedure

        This method is useful when there is no B->A ensemble but only A->B transitions,
        for instance when B is absorbing.

        Parameters:
        -----------
        transition_matrix:  Numpy 2D array

        stateA, stateB:     List of integers
                            Both states are a list of indexes.

        ini_probs:          List of float, default is None
                            initial probabilities in stateA

        lag_time:           integer
                            Lag time used, the trajectory is "observed" every lag_time
                            time steps

        Returns
        -------
        mean-first passage time from A->B


        """
        lenA = len(stateA)
        lenB = len(stateB)

        if ini_probs is None:
            ini_probs = [1.0 / lenA for i in range(lenA)]

        t_matrix = deepcopy(transition_matrix)

        ini_state = list(stateA)
        f_state = sorted(list(stateB))

        assert lenA == len(ini_probs)

        for i in range(lenB - 1, -1, -1):
            t_matrix = np.delete(t_matrix, f_state[i], axis=1)
            t_matrix = np.delete(t_matrix, f_state[i], axis=0)
            for j in range(lenA):
                if f_state[i] < ini_state[j]:
                    ini_state[j] = ini_state[j] - 1

        new_size = len(t_matrix)

        mfptAB = 0.0

        m = np.zeros(new_size)
        idty = np.identity(new_size)
        c = np.array([1.0 for i in range(new_size)])

        m = np.dot(np.linalg.inv(idty - t_matrix), c)

        for i in range(len(ini_state)):
            k = ini_state[i]
            mfptAB += ini_probs[i] * m[k]
        mfptAB = mfptAB / sum(ini_probs)

        return mfptAB * lag_time

    @classmethod
    def mfpts_to_target_microstate(cls, transition_matrix, target, lag_time=1):
        """Computes all the mean-first passage to a target microstate (k)

        Returns a list where the i-element is mfpt(i->k). This function is
        useful to compute the mfpt matrix.

        Parameters:
        -----------
        transition_matrix:  Numpy 2D array

        target:             Integer number that specifies the index of the state. The indexes
                            should be consistent with the transition matrix and python
                            (i.e. starting from 0)

        lag_time:           Integer
                            Lag time used, the trajectory is "observed" every lag_time
                            time steps

        Returns
        -------
        a list where the i-element is mfpt(i->k). This function is
        useful to compute the mfpt matrix.

        """

        t_matrix = deepcopy(transition_matrix)

        t_matrix = np.delete(t_matrix, target, axis=1)
        t_matrix = np.delete(t_matrix, target, axis=0)

        new_size = len(t_matrix)

        m = np.zeros(new_size)
        idty = np.identity(new_size)
        c = np.array([1.0 for i in range(new_size)])

        m = np.dot(np.linalg.inv(idty - t_matrix), c)
        m = np.insert(m, target, 0.0)

        return m * lag_time

    @classmethod
    def mfpts_matrix(cls, transition_matrix, lag_time=1):
        """Calculate MFPT matrix, i.e., the matrix where the ij-element is MFPT(i->j)

        Parameters:
        -----------
        transition_matrix:  Numpy 2D array

        lag_time:           Integer
                            Lag time used, the trajectory is "observed" every lag_time
                            time steps

        Returns
        -------
        mean-first passage time matrix with ij-element of MFPT(i->j)
        """

        size = len(transition_matrix)
        temp_values = []

        for i in range(size):
            temp_values.append(
                cls.mfpts_to_target_microstate(transition_matrix, i, lag_time)
            )

        mfpt_m = np.array(temp_values).T  # to nummpy array and transposed
        return mfpt_m

    @classmethod
    def min_commute_time(cls, matrix_of_mfpts):
        """Calculate minimum commuting time (round trip time) between all pairs

        of microstates from the matrix of mfpts. It also returns the indexes
        of the pair of microstates involved.

        Parameters:
        -----------
        matrix_of_mfpts:    Numpy 2D array
                            matrix of MFPTs with ij-element of MFPT(i->j)

        Returns
        -------
        Minimum commuting time (round trip time) between all pairs
        of microstates and the indexes of the pair of microstates involved.

        """

        matrix_of_mfpts = np.array(matrix_of_mfpts)

        n_states = len(matrix_of_mfpts)
        assert n_states == len(matrix_of_mfpts[0]) and n_states >= 2

        # Initial values, arbitrary choice
        index_i = 0
        index_j = 1

        commute_times = matrix_of_mfpts + matrix_of_mfpts.T
        min_ct = commute_times[index_i, index_j]

        for i in range(n_states):
            for j in range(i + 1, n_states):
                if commute_times[i, j] < min_ct:
                    min_ct = commute_times[i, j]
                    index_i = i
                    index_j = j

        return min_ct, index_i, index_j

    @classmethod
    def max_commute_time(cls, matrix_of_mfpts):
        """Calculate maximum commuting time (round trip time) between all pairs

        of microstates from the matrix of mfpts. It also returns the indexes
        of the pair of microstates involved.

        Parameters:
        -----------
        matrix_of_mfpts:    Numpy 2D array
                            matrix of MFPTs with ij-element of MFPT(i->j)

        Returns
        -------
        Maximum commuting time (round trip time) between all pairs
        of microstates and the indexes of the pair of microstates involved.

        """

        matrix_of_mfpts = np.array(matrix_of_mfpts)

        n_states = len(matrix_of_mfpts)
        assert n_states == len(matrix_of_mfpts[0]) and n_states >= 2

        # Initial values, arbitrary choice
        index_i = 0
        index_j = 1

        commute_times = matrix_of_mfpts + matrix_of_mfpts.T
        max_ct = commute_times[index_i, index_j]

        for i in range(n_states):
            for j in range(i + 1, n_states):
                if commute_times[i, j] > max_ct:
                    max_ct = commute_times[i, j]
                    index_i = i
                    index_j = j

        return max_ct, index_i, index_j

    @classmethod
    def fpt_distribution(
        cls,
        t_matrix,
        initial_state,
        final_state,
        initial_distrib,
        min_power=1,
        max_power=12,
        max_n_lags=100,
        lag_time=1,
        dt=1.0,
        clean_recycling=False,
        logscale=False,
    ):
        """Calculated distribution of first passage times from transition matrix

        Parameters:
        -----------
        t_matrix:           Numpy 2D array

        initial_state,
        final_states:       List of integer numbers
                            Specifies the indexes of initial and final states.

        ini_probs:          List of float, default is None
                            initial probabilities for initial states

        min_power,
        max_power:          Integer
                            The minimum and maximum power when the FPT distribution is
                            shown in logscale such as (10^min_power, 10^max_power)*lag_time*dt.

        max_n_lags:         Integer
                            maximum number of lags when the FPT distribution is shown in linear
                            scale such as (0, max_n_logs)*lag_time*dt. When in logscale, this is number
                            of points to shown in the range of (10^min_power, 10^max_power)*lag_time*dt.

        lag_time:           Integer
                            Lag time used, the trajectory is "observed" every lag_time
                            time steps

        dt:                 Float
                            Time step

        clean_recycling:    Bool
                            Cleaning the recycling of steady state simulation if True

        logscale:           Bool
                            Option to use logscale for FPT time in the distribution


        Returns
        -------
        Distributions of first passage times

        """

        # copy everything since they are going to be modified
        tmatrix = np.copy(t_matrix)

        ini_state = list(initial_state)
        f_state = sorted(list(final_state))

        assert len(ini_state) == len(initial_distrib)
        # Designate target state 0 as the only target state,
        # and adding all the fluxes into the other target states into state 0.
        # Namely reassign any flux into any target state into target state 0.
        tmatrix[:, f_state[0]] = np.sum(tmatrix[:, f_state], axis=1)

        # Remove all other target states
        for i in range(len(f_state) - 1, 0, -1):
            tmatrix = np.delete(tmatrix, f_state[i], axis=1)
            tmatrix = np.delete(tmatrix, f_state[i], axis=0)

            # For each initial state with a greater index than the target state we're cleaning.
            # Decrement the index by 1 to account for the removed, cleaned state.
            for j in range(len(ini_state)):
                if f_state[i] < ini_state[j]:
                    ini_state[j] = ini_state[j] - 1

        # Clean the recycling if necessary.
        # Get rid of recycling boundary conditions, otherwise  we're getting a  CDF
        if clean_recycling:
            tmatrix[f_state, :] = 0.0
            tmatrix[f_state, f_state] = 0.0

        # The new target state is the single state since all other target states have been reassigned to.
        f_state = f_state[0]
        new_n_states = len(tmatrix)
        list_of_pdfs = np.empty((len(ini_state), max_n_lags), dtype=np.float64)
        prevFmatrix = np.empty_like(tmatrix)

        # Option to set the list of lag time in logscale since FPT can be a wide distribution in several orders
        if logscale:
            lag_list = np.logspace(min_power, max_power, max_n_lags, dtype=int)
        else:
            lag_list = np.arange(0, max_n_lags, dtype=int)

        # for each ini_state calculate the FPT distribution from transition matrix
        for istateIndex in range(len(ini_state)):
            prevFmatrix = tmatrix.copy()
            Fmatrix = np.zeros((new_n_states, new_n_states))
            list_of_pdfs[istateIndex, 0] = tmatrix[ini_state[istateIndex], f_state]

            cls.calc_fmatrix(
                Fmatrix,
                tmatrix,
                prevFmatrix,
                list_of_pdfs,
                lag_list,
                ini_state,
                istateIndex,
                f_state,
            )

        # Nomalize the FPT distribution and output
        sum_ = np.sum(initial_distrib)
        initial_distrib = np.array(initial_distrib)

        density = np.sum(initial_distrib[:, None] * list_of_pdfs, axis=0) / sum_

        dt2 = lag_time * dt
        if logscale:
            # For logscale the dts at different t are different, we need to let FPT(t)
            # absorb them. Otherwise we have to use dt in variable size to calculate mean
            # value such as integration of t*dt*FPT(t).
            dens_list = [[0, 0]] + [[lag_list[0] * dt2, density[0] * lag_list[0] / dt2]]
            for i in range(1, len(lag_list)):
                dens_list += [
                    [
                        lag_list[i] * dt2,
                        density[i] * (lag_list[i] - lag_list[i - 1]) / dt2,
                    ]
                ]
            density_vs_t = np.array(dens_list)
        else:
            density_vs_t = np.array(
                [[0, 0]]
                + [[(i + 1) * dt2, dens / dt2] for i, dens in zip(lag_list, density)]
            )
        # normalized to 1
        density_vs_t[:, 1] /= sum(density_vs_t[:, 1])
        return density_vs_t

    @staticmethod
    def adaptive_fpt_distribution(
        Tmatrix,
        initial_states,
        initial_state_probs,
        target_states,
        tau=1,
        increment=5,
        fine_increment=1.2,
        relevant_thresh=1e-4,
        max_steps=int(1e6),
        max_time=np.inf,
        explicit_renormalization=False,
        verbose=False,
    ):
        """
        Adaptively computes a first-passage time distribution.

        Starting at t=tau, compute the probability flowing into the target at t.
        Then, increment t by multiplying it by the coarse increment.
        When relevant_thresh probability has entered the target state, step back to the previous coarse state, and
        swap over to incrementing with the fine increment.
        This allows you to efficiently sweep log-space.

        Procedurally, this starts probability in specified `initial_states` according to `initial_state_probs`, and then
        propagates that probability through the transition matrix.
        The FPT distribution is measured by tracking new probability entering the target state at each time.

        Note that absorbing boundary conditions are stripped from the transition matrix -- if this is not done, then
        the result is like a probability CDF, not a probability distribution.

        Parameters
        ----------
        Tmatrix: array-like
            Transition matrix

        initial_states: array-like of ints
            List of initial states to start probability in

        initial_state_probs: array-like
            Probability distribution across the initial states.

        target_states: array-like
            Target states for MFPT.

        tau
        increment: float
            Multiplicative increment for coarse steps
        fine_increment: float
            Multiplicative increment for fine steps, once the minimum probability in the target has been reached.
        relevant_thresh: float
            Amount of probability that must be in the target before switching to fine increments.
        max_steps: int
            Maximum number of steps to run
        max_time: float
            Maximum time to run to
        explicit_renormalization: bool
            Whether to explicitly renormalize the transition matrix. This should not be necessary -- if it is, there's
            probably some numerical instability you should be careful of.
        verbose: bool
            Produce verbose text output.

        Returns
        -------
        FPT distribution,
        probability distribution at each time,
        last step index,
        times at which FPT distribution was evaluated
        """

        n_states = len(Tmatrix)

        all_probabilities = np.full(shape=(max_steps + 1, n_states), fill_value=np.nan)

        # The initial probability vector is zero except in the origin states,
        #    which have their relative probabilities
        initial_probability = np.zeros(n_states)
        initial_probability[initial_states] = initial_state_probs
        initial_probability /= sum(initial_probability)

        all_probabilities[0] = initial_probability

        # Make the target states absorbing
        non_recycling_matrix = Tmatrix.copy()
        non_recycling_matrix[target_states, :] = 0.0
        for target in target_states:
            non_recycling_matrix[target, target] = 1.0

        # Track the probability that flowed into the target at each time
        probs = np.zeros(shape=max_steps)
        probs[0] = 0.0

        # At each one of our timesteps, track the amount of flux that entered the target
        last_step = 1

        get_next_step = lambda x: x * increment
        in_relevant_region = False

        steps = [1]

        with tqdm.tqdm(total=1) as pbar:

            for i in range(max_steps - 1):

                this_step = int(get_next_step(last_step))
                if this_step <= last_step:
                    this_step = int(last_step + 1)

                matrix_next = np.linalg.matrix_power(non_recycling_matrix, this_step)

                if explicit_renormalization:
                    matrix_next = matrix_next / np.sum(matrix_next, axis=1)

                probability = initial_probability @ matrix_next

                if explicit_renormalization:
                    probability /= sum(probability)

                # Check if we're just starting to get any probability
                if (
                    i > 0
                    and not in_relevant_region
                    and (sum(probability[target_states]) - sum(probs[: i + 1]))
                    > relevant_thresh
                ):
                    if verbose:
                        print(
                            f"*** Entered relevant region  at step {this_step}. "
                            f"Swapping to fine-grained, and taking a step back to {this_step / increment}."
                        )
                    # If so, then change our increment to finer resolution
                    # TODO: Would be cool to do something like as the probability increases,
                    #    continue scaling down to some minimum increment
                    in_relevant_region = True
                    this_step /= increment

                    steps.append(this_step)
                    all_probabilities[i + 1] = all_probabilities[i]
                    probs[i + 1] = probs[i]

                    get_next_step = lambda x: x * fine_increment

                    if verbose:
                        print(
                            f"Current time is {this_step}, time step will be {get_next_step(this_step)}"
                        )

                    continue

                steps.append(this_step)

                all_probabilities[i + 1] = probability

                # The amount that flowed INTO the target is the probability that's flowed in since the last t
                if i == 0:
                    # In the first iteration,  all the probability into the target just got there
                    probs[i + 1] = sum(probability[target_states])
                else:
                    # After the first, it's the amount that's there now minus the total amount that entered up until now
                    probs[i + 1] = sum(probability[target_states]) - sum(probs[: i + 1])

                pbar.update(probs[i + 1])

                # Check if we're done (i.e., all our probability has flowed  into  the target, none left.)
                if np.isclose(sum(probs), 1):
                    #         if np.isclose(probs[i+1], 1):
                    print(
                        f"*** All probability  reached the target at time {this_step}"
                    )
                    break

                if this_step > max_time:
                    print(
                        "*** Max steps reached, before all probability flowed into target."
                    )
                    break

                last_step = this_step

        print(f"Finished in {i} steps")
        print(
            f"By the last time, {sum(probs[:i])} probability has reached the target. (This should be 1!)"
        )

        times = np.array(steps, dtype=float) * float(tau)
        return probs[: i + 2], all_probabilities[: i + 2], i, times

    @classmethod
    def calc_fmatrix(
        cls,
        Fmatrix,
        tmatrix,
        prevFmatrix,
        list_of_pdfs,
        lag_list,
        ini_state,
        istateIndex,
        f_state,
    ):
        # Calculate FPT distribution from a the recursive formula, Eq. 3 in the paper below:
        # E. Suarez, A. J. Pratt, L. T. Chong, D. M. Zuckerman, Protein Science 26, 67-78 (2016).
        for time_index, time in enumerate(lag_list):
            # obtain the new transition matrix from time_index-1 to time_index
            if time_index == 0:
                tmatrix_new = np.linalg.matrix_power(tmatrix, time)
            else:
                tmatrix_new = np.linalg.matrix_power(
                    tmatrix, time - lag_list[time_index - 1]
                )
            Fmatrix = np.dot(tmatrix_new, prevFmatrix - np.diag(np.diag(prevFmatrix)))
            list_of_pdfs[istateIndex, time_index] = Fmatrix[
                ini_state[istateIndex], f_state
            ]
            prevFmatrix = Fmatrix


class MarkovFPT(MatrixFPT):
    """Derived a class for calculating FPTs using Markovian transition matrix"""

    @classmethod
    def mean_fpts(cls, markov_tmatrix, stateA, stateB, lag_time=1):
        """Computes mean first passage times using Markovian transition matrix

        in both directions A->B and B->A from a markov model. The MFPTs computed
        in this way are directly comparable with the values obtained by a long back
        and forth simulation between the target states.

        Parameters:
        -----------
        markov_matrix:      Numpy 2D array
                            Markovian transition matrix

        stateA, stateB:     List of integers
                            Both states are a list of indexes.

        lag_time:           integer
                            Lag time used, the trajectory is "observed" every lag_time
                            time steps

        Returns
        -------
        mean-first passage times from A->B and B->A

        """
        auxiliar_matrix = utils.pseudo_nm_tmatrix(markov_tmatrix, stateA, stateB)

        # Is going to return a Markovian mfpt since the auxiliar
        # matrix was build from a pure Markovian matrix
        return NonMarkovFPT.mean_fpts(auxiliar_matrix, stateA, stateB, lag_time)

    @classmethod
    def markov_commute_time(cls, transition_matrix, stateA, stateB, lag_time=1):
        """Computing commute time for Markovian Model

        Parameters:
        -----------
        transition_matrix:  Numpy 2D array
                            Markovian transition matrix

        stateA, stateB:     List of integers
                            Both states are a list of indexes.

        lag_time:           integer
                            Lag time used, the trajectory is "observed" every lag_time
                            time steps

        Returns
        -------
        Commute time from mean-first passage times
        """
        mfpts = cls.mean_fpts(transition_matrix, stateA, stateB, lag_time)
        return mfpts["mfptAB"] + mfpts["mfptBA"]


class NonMarkovFPT(MatrixFPT):
    """Derived a class for calculating FPTs using Non Markov transition matrix"""

    @classmethod
    def mean_fpts(cls, nm_transition_matrix, stateA, stateB, lag_time=1):
        """Computes the mean first passage times from a non-markovian model

        in both directions of A->B and B->A. The shape of the transition matrix
        should be (2*n_states, 2*n_states).
        Parameters:
        -----------
        nm_transition_matrix:   Numpy 2D array
                                Non-Markovian transition matrix

        stateA, stateB:         List of integers
                                Both states are a list of indexes.

        lag_time:               integer
                                Lag time used, the trajectory is "observed" every lag_time
                                time steps

        Returns
        -------
        mean-first passage times from A->B and B->A

        """
        utils.check_tmatrix(nm_transition_matrix)

        labeled_pops = utils.pops_from_tmatrix(nm_transition_matrix)

        n_states = len(labeled_pops) // 2

        fluxAB = 0
        fluxBA = 0

        for i in range(0, 2 * n_states, 2):
            for j in range(2 * n_states):
                if int(j / 2) in stateB:
                    fluxAB += labeled_pops[i] * nm_transition_matrix[i, j]

        for i in range(1, 2 * n_states + 1, 2):
            for j in range(2 * n_states):
                if int(j / 2) in stateA:
                    fluxBA += labeled_pops[i] * nm_transition_matrix[i, j]

        pop_colorA = 0.0
        pop_colorB = 0.0

        for i in range(0, 2 * n_states, 2):
            pop_colorA += labeled_pops[i]

        for i in range(1, 2 * n_states + 1, 2):
            pop_colorB += labeled_pops[i]

        if fluxAB == 0:
            mfptAB = float("inf")
        else:
            mfptAB = pop_colorA / fluxAB

        if fluxBA == 0:
            mfptBA = float("inf")
        else:
            mfptBA = pop_colorB / fluxBA

        mfptAB *= lag_time
        mfptBA *= lag_time

        return dict(mfptAB=mfptAB, mfptBA=mfptBA)
