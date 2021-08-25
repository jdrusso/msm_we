"""
Adapted from the original NMpathAnalysis package,
https://github.com/ZuckermanLab/NMpathAnalysis
"""
import numpy as np
from copy import deepcopy
import networkx as nx
from math import log

from msm_we.utils import Interval
from msm_we.utils import get_shape, weighted_choice
from msm_we.utils import reverse_sort_lists
from msm_we.fpt import DirectFPT, NonMarkovFPT


class Ensemble:
    """ Define a base class to store a list of space-continuous trajectories

     e.g.: [ trajectory1, trajectory2,...], each trajectory is just a matrix
    where each row is a snapshot, and each column represents the evolution
    of the corresponding variable.
    """

    def __init__(self, trajectories=None, verbose=False, dtype="float32", discrete=False, lag_time=1, **kwargs):
        """Initialize an object of Ensemble class"""
        super().__init__(**kwargs)
        self.dtype = dtype
        self.discrete = discrete
        self.verbose = verbose
        self._lag_time = lag_time

        if (trajectories is None) or (trajectories == []):
            self.trajectories = []
            self.n_variables = 0
            if verbose:
                print("\nEmpty ensemble generated")

        else:
            # we have a list/array of trajectories
            _n_snapshots, _n_variables = get_shape(trajectories[0])

            traj_length = 0.0  # Average trajectory length
            for element in trajectories:
                traj_length += len(element)

                n_snapshots, n_variables = get_shape(element)

                if n_variables != _n_variables:
                    raise Exception("Error: All the trajectories must have the same " "number of variables")

            self.n_variables = _n_variables
            self.trajectories = trajectories
            n_trajs = len(trajectories)

            traj_length /= n_trajs  # Average

            if verbose:
                print("Read {} ({}-dimensional) trajectories of " "average length {}.".format(n_trajs, n_variables, traj_length))

    def add_trajectory(self, trajectory):
        """Add a single trajectory to the ensemble"""
        if not isinstance(trajectory, np.ndarray):
            trajectory = np.array(trajectory, dtype=self.dtype)

        _n_snapshots, _n_variables = get_shape(trajectory)

        if self.n_variables == 0:  # Empty ensemble
            self.trajectories = [trajectory]
            self.n_variables = _n_variables
            if self.verbose:
                print(self)
        else:
            if self.n_variables != _n_variables:
                raise Exception("All the trajectories in the same ensemble must " "have the same number of variables")
            else:
                self.trajectories.append(trajectory)
                if self.verbose:
                    print(self)

    def __len__(self):
        # returns the number of trajectories in the ensemble
        return len(self.trajectories)

    def __str__(self):
        if self.discrete:
            feature = "Discrete, "
        else:
            feature = "Continuous, "
        return (
            "\n"
            + feature
            + "{} with {} ({}-dimensional) trajectories".format(self.__class__.__name__, self.__len__(), self.n_variables)
            + "\nTotal number of snapshots: {}".format(sum([len(traj) for traj in self]))
        )

    def __add__(self, other):
        ensemble_sum = deepcopy(self)

        for traj in other.trajectories:
            ensemble_sum.add_trajectory(traj)

        return ensemble_sum

    def __iadd__(self, other):
        return self.__add__(other)

    def __iter__(self):
        return iter(self.trajectories)

    def __getitem__(self, arg):
        return self.trajectories[arg]

    def empirical_mfpts(self, stateA, stateB):
        """Calculate mean first passage time using direct counting method"""
        return DirectFPT.mean_fpts(
            self.trajectories, stateA, stateB, discrete=self.discrete, n_variables=self.n_variables, lag_time=self._lag_time,
        )

    def _count_matrix(self, n_states=None, map_function=None):
        """Built the count matrix"""
        if (map_function is None) or (n_states is None):
            raise Exception("The number of states and a map function " "have to be given as argument")

        count_matrix = np.zeros((n_states, n_states))

        for traj in self.trajectories:
            previous_state = "Unknown"
            for snapshot in traj:
                current_state = map_function(snapshot)
                if previous_state != "Unknown":
                    count_matrix[previous_state, current_state] += 1.0
                previous_state = current_state

        return count_matrix

    def _mle_transition_matrix(self, n_states, map_function):
        """Generate transition matrix from count matrix"""
        count_matrix = self._count_matrix(n_states, map_function)
        transition_matrix = count_matrix.copy()

        # Transforming the count matrix to a transition matrix
        for i in range(n_states):
            row_sum = sum(transition_matrix[i, :])
            if row_sum != 0.0:
                transition_matrix[i, :] = transition_matrix[i, :] / row_sum

        return transition_matrix

    def empirical_corr_function(self, stateA, stateB, times, symmetric=True):
        """Calculate correlation function for trajectories"""
        n_dim = self.n_variables

        stateA = Interval(stateA, n_dim) if not self.discrete else stateA
        stateB = Interval(stateB, n_dim) if not self.discrete else stateB

        corr_values = []

        for delay in times:
            assert type(delay) == int
            assert delay >= 1
            sum_ = 0
            counts = 0

            for traj in self.trajectories:
                for i in range(len(traj) - delay):
                    sum_ += (traj[i] in stateA) * (traj[i + delay] in stateB)
                    counts += 1

                    if symmetric:
                        sum_ += (traj[i] in stateB) * (traj[i + delay] in stateA)
                        counts += 1
            corr_values.append(sum_ / counts)

        return corr_values


class PathEnsemble(Ensemble):
    """ Derive a class to store trajectories for path analysis"""

    def __init__(
        self, trajectories=None, verbose=False, dtype="float32", discrete=False, lag_time=1, stateA=None, stateB=None, **kwargs
    ):
        super().__init__(trajectories, verbose, dtype, discrete, lag_time, **kwargs)

        if (stateA is None) or (stateB is None):
            raise Exception(
                "The initial state (stateA) and final state (stateB) \
                 have to be specified"
            )
        self.stateA = stateA
        self.stateB = stateB

    @classmethod
    def from_ensemble(
        cls, ensemble, stateA=None, stateB=None, map_function=None, discrete=False, dtype="float32",
    ):
        """Build a PathEnsemble from an ensemble object or a set of trajectories"""

        list_of_pathsAB = []

        if np.size(ensemble[0][0]):
            n_variables = np.size(ensemble[0][0])
        else:
            n_variables = 1

        if (stateA is None) or (stateB is None):
            raise Exception(
                "The initial state (stateA) and final state (stateB) \
                have to be specified"
            )

        for traj in ensemble.trajectories:
            previous_color = "Unknown"
            pathAB = []
            for _snapshot in traj:

                if map_function is not None:
                    snapshot = map_function(_snapshot)
                else:
                    snapshot = _snapshot

                # color determination
                if not discrete:
                    if snapshot in Interval(stateA, n_variables):
                        color = "A"
                    elif snapshot in Interval(stateB, n_variables):
                        color = "B"
                    else:
                        color = previous_color
                else:
                    if snapshot in stateA:
                        color = "A"
                    elif snapshot in stateB:
                        color = "B"
                    else:
                        color = previous_color

                if color == "A":
                    pathAB.append(snapshot)
                elif (color == "B") and (previous_color == "A"):
                    pathAB.append(snapshot)
                    list_of_pathsAB.append(np.array(pathAB, dtype=dtype))
                    pathAB = []

                previous_color = color

        return cls(list_of_pathsAB, stateA=stateA, stateB=stateB, dtype=dtype, discrete=discrete,)

    def cluster(self, distance_metric, n_cluster=10, method="K-means"):
        raise NotImplementedError("Not implemented yet")


class DiscreteEnsemble(Ensemble):
    """Define a base class to store a list of space-discrete trajectories"""

    def __init__(self, trajectories=None, verbose=False, dtype="int32", discrete=True, lag_time=1, **kwargs):
        """Initialize an object of Discrete Ensemble"""
        super().__init__(trajectories, verbose, dtype, discrete, lag_time, **kwargs)
        if (self.n_variables != 1) and (self.n_variables != 0):
            raise Exception(
                "A discrete trajectory must have a one-dimensional \
                index/variable unless is empty"
            )
        self.n_variables = 1  # by definition

    @classmethod
    def from_ensemble(cls, ens, map_function=None, dtype="int32"):
        """Build a DiscreteEnsemble from an ensemble object or a set of trajectories"""
        # TODO: Check the case when ens is a list of trajs
        if map_function is None:
            raise ValueError("A map function has to be given as argument")

        discrete_trajs_list = []

        if isinstance(ens, Ensemble):
            # it is an Ensemble object
            for traj in ens.trajectories:
                d_traj = np.array([], dtype=dtype)
                for snapshot in traj:
                    d_traj = np.append(d_traj, np.array([map_function(snapshot)]), axis=0)
                discrete_trajs_list.append(d_traj)
            return cls(discrete_trajs_list)
        else:
            # it is a list of trajectories
            d_traj = []
            for traj in ens:
                d_traj += [map_function(snapshot)]
            d_traj = np.array(d_traj, dtype=dtype)

            return cls([d_traj])

    @classmethod
    def from_transition_matrix(cls, transition_matrix, sim_length=None, initial_state=0):
        """Generates a DiscreteEnsemble from the transition matrix"""
        if sim_length is None:
            raise Exception("The simulation length must be given")

        if not isinstance(transition_matrix, np.ndarray):
            transition_matrix = np.array(transition_matrix)

        n_states = len(transition_matrix)
        assert n_states == len(transition_matrix[0])

        current_state = initial_state
        discrete_traj = [initial_state]

        for i in range(sim_length):
            next_state = weighted_choice([k for k in range(n_states)], transition_matrix[current_state, :])
            discrete_traj.append(next_state)
            current_state = next_state

        return cls([np.array(discrete_traj)])


class DiscretePathEnsemble(PathEnsemble, DiscreteEnsemble):
    """Derive a class to store a list of discrete trajectories for path analysis"""

    def __init__(
        self, trajectories=None, verbose=False, dtype="int32", discrete=True, lag_time=1, stateA=None, stateB=None, **kwargs
    ):

        super().__init__(trajectories, verbose, dtype, discrete, lag_time, stateA, stateB, **kwargs)

    @classmethod
    def from_transition_matrix(
        cls, transition_matrix, stateA=None, stateB=None, n_paths=1000, ini_pops=None, max_iters=1000000000,
    ):
        """
        Construct a path ensemble from a transition matrix

        stateA:     list, intitial state

        stateB:     list, final state

        n_paths:    integer, with default value of 1000
                    number of paths to generate

        ini_pops:   list or label, probability distribution over the
                    initial state used to generate the path

                    possible values:
                    ----------------
                    a) None
                    Use a uniform distribution over the states in stateA
                    c) list
                    A list with the explicit values of the populations in stateA
                    that should be used to generate the ensemble
        max_iters:  integer, with default value of 1000000000
                    maximum of iterations for generating path trajectories
        """

        if ini_pops is None:
            ini_pops = [1 / float(len(stateA)) for i in range(len(stateA))]
        elif ini_pops == "ss":
            raise NotImplementedError("Sorry: not yet implemented")

        n_states = len(transition_matrix)
        assert n_states == len(transition_matrix[0])

        d_trajectories = []

        for i in range(n_paths):
            current_state = weighted_choice(stateA, ini_pops)  # Initial state
            path = [current_state]

            for j in range(max_iters):
                next_state = weighted_choice([k for k in range(n_states)], transition_matrix[current_state, :])
                path += [next_state]
                current_state = next_state
                if j + 1 == max_iters:
                    print("\nWARNING: max iteration reached when generating " "the path ensemble, consider to increase max_iters")
                if current_state in stateB:
                    break

            path = np.array(path)
            d_trajectories.append(path)

        return cls(d_trajectories, stateA=stateA, stateB=stateB)

    @classmethod
    def from_ensemble(cls, ensemble, stateA, stateB, map_function=None):
        """Build a DiscreteEnsemble from an ensemble object or a set of trajectories"""
        ens = PathEnsemble.from_ensemble(ensemble, stateA, stateB, map_function, discrete=True, dtype="int32")
        return cls(ens.trajectories, stateA=stateA, stateB=stateB)

    def nm_mfpt(self, ini_probs=None, n_states=None):
        """Compute the mean-first passage time from the transition matrix"""
        t_matrix = self._mle_transition_matrix(n_states)
        ini_state = list(self.stateA)
        final_state = sorted(list(self.stateB))

        return NonMarkovFPT.directional_mfpt(t_matrix, ini_state, final_state, ini_probs)

    def _fundamental_sequences(self, transition_matrix, symmetric=True):
        """Divide/classify the path ensemble into fundamental sequences"""

        fundamental_seqs = []

        for path in self.trajectories:
            if symmetric:
                cmatrix = self._connectivity_matrix(path, transition_matrix * transition_matrix.T)
            else:
                cmatrix = self._connectivity_matrix(path, transition_matrix)

            path_graph = self._graph_from_matrix(cmatrix)
            shortest_path = nx.dijkstra_path(path_graph, path[0], path[-1], "distance")

            fundamental_seqs.append(shortest_path)

        return fundamental_seqs

    def weighted_fundamental_sequences(self, transition_matrix=None, symmetric=True):
        """Generate sorted fundamental sequences with weights"""
        fs_list = self._fundamental_sequences(transition_matrix, symmetric)
        element_count = {}
        tot_count = 0
        for element in fs_list:
            pseudo_index = tuple(element)
            tot_count += 1
            if pseudo_index not in element_count:
                element_count[pseudo_index] = 1
            else:
                element_count[pseudo_index] += 1

        weights = []
        new_fs_list = []
        for key, value in element_count.items():
            new_fs_list.append(key)
            weights.append(value / float(tot_count))

        reversed_sorted_weights, reversed_sorted_new_fs_list = reverse_sort_lists(weights, new_fs_list)

        return reversed_sorted_new_fs_list, reversed_sorted_weights, tot_count

    @staticmethod
    def _graph_from_matrix(matrix):
        """Builds a directed Graph from a matrix like a transition matrix"""

        size = len(matrix)
        assert size == len(matrix[0])
        matrix = np.array(matrix)

        G = nx.DiGraph()

        for node in range(size):
            G.add_node(node)

        for i in range(size):
            for j in range(size):
                if (i != j) and (matrix[i, j] != 0.0):
                    G.add_edge(i, j, distance=-log(matrix[i, j]))
        return G

    @staticmethod
    def _connectivity_matrix(path, matrix):
        """From a given path and a matrix construct a connectivity matrix

        whose elements ij are zero if the transition i->j
        is not observed in the path or (i=j), while keep the rest of the
        elements in the input matrix.

        This way, from the connectivity matrix we could later create a graph
        that represents the path, being the "distance" between nodes equal
        to -log(Tij)

        Tij --> i,j element in the transition matrix

        the path must be 1D array of indexes
        """
        matrix = np.array(matrix)
        path = np.array(path, dtype="int32")

        n_states = len(matrix)
        assert n_states == len(matrix[0])

        c_matrix = np.zeros((n_states, n_states))

        for i in range(len(path) - 1):
            c_matrix[path[i], path[i + 1]] = matrix[path[i], path[i + 1]]

        return c_matrix
