"""
Adapted from the original NMpathAnalysis package,
https://github.com/ZuckermanLab/NMpathAnalysis
"""
import numpy as np

from msm_we.fpt import DirectFPT, MarkovFPT, NonMarkovFPT
from msm_we.ensembles import DiscreteEnsemble, DiscretePathEnsemble
from msm_we.utils import map_to_integers, normalize_markov_matrix
from msm_we.utils import pops_from_nm_tmatrix, pops_from_tmatrix
from msm_we.utils import pseudo_nm_tmatrix, weighted_choice


class NonMarkovModel(DiscreteEnsemble):
    """Define a class for analyzing MD trajectories using Markovian or non-Markovian Model


    from a list of 1D trajectories of integers representing macrostates

    For example:

    trajectories = [ [1 , 2, 0, ...], [2, 2, 1, ...], [3, 1, 2, ...], ...]

    If only one sequence is given in trajectories, the format is the same:

    trajectories = [ [1 , 2, 0, ...] ]

    Parameters
    ----------
    lag_time (integer, default: 1)
        Lag time of the model.

    sliding_window (boolean)
        Use a sliding window of length lag_time to compute the count matrix

    stateA, stateB (python lists)
        Define the initial and final macrostates in form of python lists
        for example: stateA=[0,2,5], stateB = [1]

    Attributes
    ----------
    n_states : int

    nm_cmatrix: array, with shape (2 n_states, 2 n_states)
        Stores the number of transitions between states, the i,j element cij
        stores the number of transitions observed from i to j.

    populations: array, shape (n_states,)
        Equilibrium population, the steady state solution of of the
        transition matrix
    """

    def __init__(
        self,
        trajectories,
        stateA,
        stateB,
        lag_time=1,
        clean_traj=False,
        sliding_window=True,
        reversible=True,
        markovian=False,
        coarse_macrostates=False,
        **kwargs
    ):
        """Initialize an object for Non Markovian Model Class"""

        if coarse_macrostates:
            for traj in trajectories:
                for i, _ in enumerate(traj):
                    if traj[i] in stateA:
                        traj[i] = stateA[0]
                    elif traj[i] in stateB:
                        traj[i] = stateB[0]
            stateA = [stateA[0]]
            stateB = [stateB[0]]

        self._lag_time = lag_time
        self.trajectories = trajectories
        self.stateA = stateA
        self.stateB = stateB
        self.sliding_window = sliding_window
        self.reversible = reversible
        self.markovian = markovian

        self.n_variables = 1  # by construction
        self.discrete = True  # by construction

        if (self._lag_time < 1) or (int(self._lag_time) != int(self._lag_time)):
            raise ValueError(
                "The lag time should be an integer \
            greater than 1"
            )

        if clean_traj:
            self.n_states = max([max(traj) for traj in self.trajectories]) + 1
        else:
            self._map_trajectories_to_integers()

        self.fit()

    def _map_trajectories_to_integers(self):
        # Clean the sequences
        seq_map = {}
        new_trajs = []
        for seq in self.trajectories:
            newseq, m_dict = map_to_integers(seq, seq_map)
            new_trajs.append(newseq)
        self.stateA = [seq_map[i] for i in self.stateA]
        self.stateB = [seq_map[i] for i in self.stateB]
        self.n_states = len(seq_map)
        self.trajectories = new_trajs
        self.seq_map = seq_map

    def fit(self):
        """Fits the non-Markovian model from a list of sequences"""
        # Non-Markovian count matrix
        nm_cmatrix = np.zeros((2 * self.n_states, 2 * self.n_states))

        # Markovian count matrix
        markov_cmatrix = np.zeros((self.n_states, self.n_states))

        lag = self._lag_time

        if not self.sliding_window:
            step = lag
        else:
            step = 1

        for traj in self.trajectories:
            for start in range(lag, 2 * lag, step):
                prev_color = None

                for i in range(start, len(traj), lag):

                    # Color determination
                    if traj[i] in self.stateA:
                        color = "A"
                    elif traj[i] in self.stateB:
                        color = "B"
                    else:
                        color = prev_color

                    # Count matrix for the given lag time
                    if prev_color == "A" and color == "B":
                        nm_cmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += 1.0
                    elif prev_color == "B" and color == "A":
                        nm_cmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] += 1.0
                    elif prev_color == "A" and color == "A":
                        nm_cmatrix[2 * traj[i - lag], 2 * traj[i]] += 1.0
                    elif prev_color == "B" and color == "B":
                        nm_cmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] += 1.0

                    prev_color = color

                    markov_cmatrix[traj[i - lag], traj[i]] += 1.0

        nm_tmatrix = normalize_markov_matrix(nm_cmatrix)
        markov_tmatrix = normalize_markov_matrix(markov_cmatrix, reversible=True)

        self.nm_tmatrix = nm_tmatrix

        self.nm_cmatrix = nm_cmatrix
        self.markov_cmatrix = markov_cmatrix
        self.markov_tmatrix = markov_tmatrix

    @classmethod
    def from_nm_tmatrix(cls, transition_matrix, stateA, stateB, sim_length=None, initial_state=0):
        """Generates a discrete ensemble from the transition matrix"""
        if sim_length is None:
            raise Exception("The simulation length must be given")

        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)

        n_states = len(transition_matrix)
        assert n_states == len(transition_matrix[0])

        current_state = initial_state
        discrete_traj = [initial_state // 2]

        for i in range(sim_length):
            next_state = weighted_choice([k for k in range(n_states)], transition_matrix[current_state, :])
            discrete_traj.append(next_state // 2)
            current_state = next_state

        return cls([np.array(discrete_traj)], stateA, stateB, clean_traj=True)

    @property
    def lag_time(self):
        return self._lag_time

    @lag_time.setter
    def lag_time(self, lag_time):
        self._lag_time = lag_time
        self.fit()

    def mfpts(self):
        if self.markovian:
            return MarkovFPT.mean_fpts(self.markov_tmatrix, self.stateA, self.stateB, lag_time=self._lag_time)
        else:
            return NonMarkovFPT.mean_fpts(self.nm_tmatrix, self.stateA, self.stateB, lag_time=self._lag_time)

    def empirical_mfpts(self):
        return DirectFPT.mean_fpts(self.trajectories, self.stateA, self.stateB, lag_time=self._lag_time)

    def empirical_fpts(self):
        return DirectFPT.fpts(self.trajectories, self.stateA, self.stateB, lag_time=self._lag_time)

    def populations(self):
        # In this case the results are going to be the same
        if self.markovian:
            return pops_from_tmatrix(self.markov_tmatrix)
        else:
            return pops_from_nm_tmatrix(self.nm_tmatrix)

    @property
    def popA(self):
        pop_A = 0
        pops = self.populations()
        for i, p in enumerate(pops):
            if i in self.stateA:
                pop_A += p
        return pop_A

    @property
    def popB(self):
        pop_B = 0
        pops = self.populations()
        for i, p in enumerate(pops):
            if i in self.stateB:
                pop_B += p
        return pop_B

    def tmatrixAB(self):
        if self.markovian:
            return self.markov_tmatrix

        matrixAB = []
        for i in range(0, 2 * self.n_states, 2):
            for j in range(0, 2 * self.n_states, 2):
                if (i // 2 in self.stateB) and not (j // 2 in self.stateB):
                    matrixAB.append(0.0)
                elif (i // 2 in self.stateB) and (j // 2 in self.stateB):
                    if i // 2 == j // 2:
                        matrixAB.append(1.0)
                    else:
                        matrixAB.append(0.0)
                elif not (i // 2 in self.stateB) and (j // 2 in self.stateB):
                    matrixAB.append(self.nm_tmatrix[i, j + 1])
                else:
                    matrixAB.append(self.nm_tmatrix[i, j])
        matrixAB = np.array(matrixAB)
        matrixAB = matrixAB.reshape((self.n_states, self.n_states))
        return matrixAB

    def tmatrixBA(self):
        if self.markovian:
            return self.markov_tmatrix

        matrixBA = []
        for i in range(1, 2 * self.n_states + 1, 2):
            for j in range(1, 2 * self.n_states + 1, 2):
                if (i // 2 in self.stateA) and not (j // 2 in self.stateA):
                    matrixBA.append(0.0)
                elif (i // 2 in self.stateA) and (j // 2 in self.stateA):
                    if i // 2 == j // 2:
                        matrixBA.append(1.0)
                    else:
                        matrixBA.append(0.0)
                elif not (i // 2 in self.stateA) and (j // 2 in self.stateA):
                    matrixBA.append(self.nm_tmatrix[i, j - 1])
                else:
                    matrixBA.append(self.nm_tmatrix[i, j])
        matrixBA = np.array(matrixBA)
        matrixBA = matrixBA.reshape((self.n_states, self.n_states))
        return matrixBA

    def fluxAB_distribution_on_B(self):
        if self.markovian:
            t_matrix = pseudo_nm_tmatrix(self.markov_tmatrix, self.stateA, self.stateB)
        else:
            t_matrix = self.nm_tmatrix

        distrib_on_B = np.zeros(len(self.stateB))
        labeled_pops = pops_from_tmatrix(t_matrix)
        for i in range(0, 2 * self.n_states, 2):
            for j in range(2 * self.n_states):
                if j // 2 in self.stateB:
                    distrib_on_B[self.stateB.index(j // 2)] += labeled_pops[i] * t_matrix[i, j]
        return distrib_on_B

    def fluxBA_distribution_on_A(self):
        if self.markovian:
            t_matrix = pseudo_nm_tmatrix(self.markov_tmatrix, self.stateA, self.stateB)
        else:
            t_matrix = self.nm_tmatrix

        distrib_on_A = np.zeros(len(self.stateA))
        labeled_pops = pops_from_tmatrix(t_matrix)
        for i in range(1, 2 * self.n_states + 1, 2):
            for j in range(2 * self.n_states):
                if j // 2 in self.stateA:
                    distrib_on_A[self.stateA.index(j // 2)] += labeled_pops[i] * t_matrix[i, j]
        return distrib_on_A

    def fpt_distrib_AB(self, max_x=1000, dt=1):
        return MarkovFPT.fpt_distribution(
            self.tmatrixAB(),
            self.stateA,
            self.stateB,
            self.fluxBA_distribution_on_A(),
            max_n_lags=max_x,
            lag_time=self._lag_time,
            dt=dt,
        )

    def fpt_distrib_BA(self, max_x=1000, dt=1):
        return MarkovFPT.fpt_distribution(
            self.tmatrixBA(),
            self.stateB,
            self.stateA,
            self.fluxAB_distribution_on_B(),
            max_n_lags=max_x,
            lag_time=self._lag_time,
            dt=dt,
        )

    def corr_function(self, times):
        """Compute the correlation function for a set of times.

        Parameters
        ----------
        times (list of integers):
            List of dt values used to compute the correlation function.

        Returns
        -------
        List of floats with the correlation values for the dt given in times
        """

        pAA = []
        pAB = []
        pBA = []
        pBB = []

        t_matrix = self.markov_tmatrix if self.markovian else self.nm_tmatrix
        tot_n_states = self.n_states if self.markovian else (2 * self.n_states)

        for dt in times:
            if dt % self.lag_time != 0:
                raise ValueError("The times given should be " "multiple of the lag time")
            n = int(dt / self.lag_time)
            pops_eq = self.populations()

            t_matrixT_to_n = np.linalg.matrix_power(t_matrix.T, n)

            popsA_to_propagate = np.zeros(tot_n_states)
            popsB_to_propagate = np.zeros(tot_n_states)

            if self.markovian:
                for index in self.stateA:
                    popsA_to_propagate[index] = pops_eq[index]

                for index in self.stateB:
                    popsB_to_propagate[index] = pops_eq[index]

                final_dist_from_A = np.dot(t_matrixT_to_n, popsA_to_propagate)
                final_dist_from_B = np.dot(t_matrixT_to_n, popsB_to_propagate)

                pAA.append(sum([final_dist_from_A[i] for i in self.stateA]))
                pBB.append(sum([final_dist_from_B[i] for i in self.stateB]))

                pAB.append(sum([final_dist_from_B[i] for i in self.stateA]))
                pBA.append(sum([final_dist_from_A[i] for i in self.stateB]))

            else:

                for index in self.stateA:
                    popsA_to_propagate[2 * index] = pops_eq[index]

                for index in self.stateB:
                    popsB_to_propagate[2 * index + 1] = pops_eq[index]

                final_dist_from_A = np.dot(t_matrixT_to_n, popsA_to_propagate)
                final_dist_from_B = np.dot(t_matrixT_to_n, popsB_to_propagate)

                pAA.append(sum([final_dist_from_A[2 * i] for i in self.stateA]))
                pBB.append(sum([final_dist_from_B[2 * i + 1] for i in self.stateB]))

                pAB.append(sum([final_dist_from_B[2 * i] for i in self.stateA]))
                pBA.append(sum([final_dist_from_A[2 * i + 1] for i in self.stateB]))

        return pAA, pAB, pBA, pBB

    def empirical_weighted_FS(self, tmatrix_for_classification=None, symmetric=True):
        if tmatrix_for_classification is None:
            tmatrix_for_classification = self.markov_tmatrix

        ens = DiscretePathEnsemble.from_ensemble(self, self.stateA, self.stateB)

        return ens.weighted_fundamental_sequences(tmatrix_for_classification, symmetric)

    def weighted_FS(self, tmatrix_for_classification=None, n_paths=1000, symmetric=True):
        if tmatrix_for_classification is None:
            tmatrix_for_classification = self.markov_tmatrix

        if self.markovian:
            tmatrix_to_generate_paths = self.markov_tmatrix
        else:
            tmatrix_to_generate_paths = self.tmatrixAB()

        ens = DiscretePathEnsemble.from_transition_matrix(tmatrix_to_generate_paths, self.stateA, self.stateB, n_paths)

        return ens.weighted_fundamental_sequences(tmatrix_for_classification, symmetric)


class MarkovPlusColorModel(NonMarkovModel):
    """Define a class for analyzing MD trajectories using Markovian Plus Color Model"""

    def __init__(self, trajectories, stateA, stateB, lag_time=1, clean_traj=False, sliding_window=True, hist_length=0, **kwargs):
        self.hist_length = hist_length
        super().__init__(trajectories, stateA, stateB, lag_time, clean_traj, sliding_window, **kwargs)

    def fit(self):
        """Fits the markov plus color model from a list of sequences"""

        # Non-Markovian count matrix
        nm_tmatrix = np.zeros((2 * self.n_states, 2 * self.n_states))

        # Markovian transition matrix
        markov_tmatrix = np.zeros((self.n_states, self.n_states))

        start = self._lag_time
        step = 1

        lag = self._lag_time
        hlength = self.hist_length

        if not self.sliding_window:
            step = lag

        # Markov first
        for traj in self.trajectories:
            for i in range(start, len(traj), step):
                markov_tmatrix[traj[i - lag], traj[i]] += 1.0  # counting
        markov_tmatrix = markov_tmatrix + markov_tmatrix.T
        markov_tmatrix = normalize_markov_matrix(markov_tmatrix)

        p_nm_tmatrix = pseudo_nm_tmatrix(markov_tmatrix, self.stateA, self.stateB)
        pops = pops_from_tmatrix(p_nm_tmatrix)

        # Pseudo-Markov Flux matrix
        fmatrix = p_nm_tmatrix
        for i, _ in enumerate(fmatrix):
            fmatrix[i] *= pops[i]

        for traj in self.trajectories:
            for i in range(start, len(traj), step):

                # Previous color determination (index i - lag)
                prev_color = "U"
                for k in range(i - lag, max(i - lag - hlength, 0) - 1, -1):
                    if traj[k] in self.stateA:
                        prev_color = "A"
                        break
                    elif traj[k] in self.stateB:
                        prev_color = "B"
                        break

                # Current Color (in index i)
                if traj[i] in self.stateA:
                    color = "A"
                elif traj[i] in self.stateB:
                    color = "B"
                else:
                    color = prev_color

                if prev_color == "A" and color == "B":
                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += 1.0
                elif prev_color == "B" and color == "A":
                    nm_tmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] += 1.0
                elif prev_color == "A" and color == "A":
                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i]] += 1.0
                elif prev_color == "B" and color == "B":
                    nm_tmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] += 1.0
                elif prev_color == "U" and color == "B":
                    temp_sum = fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] + fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1]
                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] += fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] / temp_sum
                elif prev_color == "U" and color == "A":
                    temp_sum = fmatrix[2 * traj[i - lag], 2 * traj[i]] + fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]]
                    nm_tmatrix[2 * traj[i - lag]][2 * traj[i]] += fmatrix[2 * traj[i - lag], 2 * traj[i]] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1][2 * traj[i]] += fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] / temp_sum

                elif prev_color == "U" and color == "U":
                    temp_sum = (
                        fmatrix[2 * traj[i - lag], 2 * traj[i] + 1]
                        + fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1]
                        + fmatrix[2 * traj[i - lag], 2 * traj[i]]
                        + fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]]
                    )

                    nm_tmatrix[2 * traj[i - lag], 2 * traj[i] + 1] += fmatrix[2 * traj[i - lag], 2 * traj[i] + 1] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1][2 * traj[i] + 1] += fmatrix[2 * traj[i - lag] + 1, 2 * traj[i] + 1] / temp_sum
                    nm_tmatrix[2 * traj[i - lag]][2 * traj[i]] += fmatrix[2 * traj[i - lag], 2 * traj[i]] / temp_sum
                    nm_tmatrix[2 * traj[i - lag] + 1][2 * traj[i]] += fmatrix[2 * traj[i - lag] + 1, 2 * traj[i]] / temp_sum

        self.nm_cmatrix = nm_tmatrix  # not normalized, it is like count matrix

        nm_tmatrix = normalize_markov_matrix(nm_tmatrix)
        self.nm_tmatrix = nm_tmatrix
        self.markov_tmatrix = markov_tmatrix

    def populations(self):
        return NotImplementedError(
            "You should use a regular Markov model or " "a non-Markovian model for estimating " "populations"
        )
