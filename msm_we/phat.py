import abc
import collections
import itertools
import networkx as nx
import numpy as np
import types

from typing import Callable


class PathwayClassifier(abc.ABC):
    """Base class for parametrized pathway classifiers."""

    @abc.abstractmethod
    def classify(self, trajectory):
        """Return the pathway class of a trajectory.

        Parameters
        ----------
        trajectory : object
            An object (e.g., a time series) representing a sample path
            of a stochastic process.

        Returns
        -------
        hashable object
            The pathway class of the trajectory.

        """
        ...

    def histogram(self, trajectories, weights=None):
        """Return a pathway histogram filled with the given trajectory data.

        Parameters
        ----------
        trajectories : iterable
            Trajectories to be classified.
        weights : iterable of float, optional
            The weight of each trajectory. By default, each trajectory
            has weight 1. Weights are paired with trajectories in the
            order they are iterated.

        Returns
        -------
        PathwayHistogram
            A pathway histogram of the given data.

        """
        return pathway_histogram(self, trajectories, weights)

    def __call__(self, trajectory):
        return self.classify(trajectory)


class FundamentalSequenceClassifier(PathwayClassifier):
    """Fundamental sequence (FS) classifier.

    Parameters
    ----------
    transition_matrix : (N, N) array_like
        Transition matrix of a discrete time Markov chain. Must be
        a stochastic matrix with row sums equal to one.
    states : (N,) array_like, optional
        State labels of the Markov chain. Values must be hashable.
        Default is ``range(N)``.
    symmetrized : bool, default True
        If True, classify paths by symmetrized FS.

    """

    def __init__(self, transition_matrix, states=None, symmetrized=True):
        self._graph = surprisal_graph(transition_matrix, states=states, symmetrized=symmetrized)

    @property
    def graph(self):
        """nx.Graph: Surprisal graph of the underlying Markov chain."""
        return self._graph

    def classify(self, discrete_trajectory):
        """Return the pathway class of a discrete trajectory.

        Parameters
        ----------
        discrete_trajectory : sequence
            A node path in :attr:`graph`.

        Returns
        -------
        tuple
            The fundamental sequence of `discrete_trajectory`.

        """
        return fundamental_sequence(self.graph, discrete_trajectory, weight_attr='surprisal')


class VoronoiPathwayClassifier(PathwayClassifier):
    """A classifier based on a Voronoi partition of trajectory space.

    Parameters
    ----------
    centers : iterable
        Trajectories to be used as Voronoi centers.
    distance : callable, optional
        A function that takes two trajectories and returns their distance.
        The default distance between two trajectories `s` and `q` is the
        symmetric difference cardinality ``len(set(s) ^ set(q))``.

    """

    def __init__(self, centers, distance=None):
        if distance is None:
            distance = symmetric_difference_cardinality
        elif not isinstance(distance, Callable):
            raise TypeError('distance function must be callable')

        self._centers = tuple(centers)
        self._distance = distance

    @property
    def centers(self):
        """sequence: Voronoi centers."""
        return self._centers

    @property
    def distance(self):
        """callable: Distance function."""
        return self._distance

    def classify(self, trajectory):
        """Return the pathway class of a trajectory.

        Parameters
        ----------
        trajectory : object
            An object (e.g., a time series) representing a sample path
            of a stochastic process.

        Returns
        -------
        int
            The zero-based index of the closest Voronoi center. Ties are
            broken according to the ordering of :attr:`centers`.

        """
        distances = np.fromiter((self._distance(trajectory, center) for center in self._centers), dtype=float)
        # TODO(Jeff): Consider using a metric tree instead of brute force.
        return np.argmin(distances)


class WeightedSample:
    """A sequence of (observation, weight) pairs.

    Parameters
    ----------
    observations : iterable, default ()
        A sequence of observations.
    weights : iterable of floats, optional
        The weight of each observation. By default, each observation
        has weight 1. Weights are paired with observations in the
        order they are iterated.

    """

    def __init__(self, observations=None, weights=None):
        observations = [] if observations is None else list(observations)

        if weights is None:
            weights = [1] * len(observations)
        else:
            weights = list(weights)
            if len(weights) != len(observations):
                raise ValueError('number of weights must match number of observations')
            if any(w < 0 for w in weights):
                raise ValueError('weights must be non-negative')

        self._observations = observations
        self._weights = weights

    def append(self, observation, weight=1):
        if weight < 0:
            raise ValueError('weight must be non-negative')
        self._observations.append(observation)
        self._weights.append(weight)

    @property
    def observations(self):
        """iterator: Iterator over the observations."""
        return iter(self._observations)

    @property
    def weights(self):
        """iterator: Iterator over the weights."""
        return iter(self._weights)

    @property
    def total_weight(self):
        """float: Sum of the weights."""
        return sum(self._weights)

    def __len__(self):
        return len(self._observations)

    def __iter__(self):
        return zip(self._observations, self._weights)

    def __getitem__(self, key):
        if type(key) is int:
            return self._observations[key], self._weights[key]
        return WeightedSample(self._observations[key], self._weights[key])

    def __bool__(self):
        return len(self) > 0

    def __add__(self, other):
        return WeightedSample(itertools.chain(self.observations, other.observations), itertools.chain(self.weights, other.weights))

    def __concat__(self, other):
        return self + other

    def __rmul__(self, coeff):
        coeff = float(coeff)
        if coeff < 0:
            raise ValueError('scale factor must be non-negative')
        return WeightedSample(self.observations, (coeff * w for w in self.weights))

    def __imul__(self, coeff):
        return coeff * self

    def __repr__(self):
        return f'<{self.__class__.__name__} {hex(id(self))}, ' + f'total_weight={self.total_weight}>'


class PathwayHistogram:
    """A pathway histogram.

    Parameters
    ----------
    classifier : callable
        A function that takes a single trajectory as input and returns a
        hashable value representing the pathway class of the trajectory.

    """

    def __init__(self, classifier):
        if not isinstance(classifier, Callable):
            raise TypeError('classifier must be callable')
        self._classifier = classifier
        self._data = collections.defaultdict(WeightedSample)

    @property
    def classifier(self):
        """callable: Mapping from trajectories to pathway classes."""
        return self._classifier

    @property
    def data(self):
        """types.MappingProxyType: Read-only view of histogram data."""
        return types.MappingProxyType(self._data)

    def classes(self):
        """Iterable[Hashable]: Pathway classes (block labels).

        Alias self.data.keys().

        """
        return self._data.keys()

    def blocks(self):
        """Iterable[WeightedSample]: Data belonging to each class.

        Alias self.data.values().

        """
        return self._data.values()

    def add(self, trajectory, weight=1):
        """Add a trajectory to the histogram.

        Parameters
        ----------
        trajectory : object
            Trajectory to be classified.
        weight : float, default 1.0
            The weight of the trajectory.

        """
        pathway_class = self.classifier(trajectory)
        self._data[pathway_class].append(trajectory, weight)

    def fill(self, trajectories, weights=None, accumulate=True):
        """Fill the histogram with data.

        Parameters
        ----------
        trajectories : iterable
            Trajectories to be classified.
        weights : iterable of float, optional
            The weight of each trajectory. By default, each trajectory
            has weight 1. Weights are paired with trajectories in the
            order they are iterated.
        accumulate : bool, default True
            If True, retain any existing data in the histogram. If False,
            clear the histogram before filling with the given data.

        """
        if not accumulate:
            self.clear()

        if weights is None:
            for trajectory in trajectories:
                self.add(trajectory)
        else:
            for trajectory, weight in zip(trajectories, weights):
                self.add(trajectory, weight)

    def clear(self):
        """Remove all data from the histogram."""
        self._data.clear()


def pathway_histogram(classifier, trajectories, weights=None):
    """Construct a pathway histogram.

    Parameters
    ----------
    classifier : PathwayClassifier
        A function that maps a trajectory to its pathway class.
    trajectories : iterable
        Trajectories to be classified.
    weights : iterable of float, optional
        The weight of each trajectory. By default, each trajectory
        has weight 1. Weights are paired with trajectories in the
        order they are iterated.

    Returns
    -------
    PathwayHistogram
        A pathway histogram of the given data.

    """
    return PathwayHistogram(classifier).fill(trajectories, weights=weights)


def surprisal_graph(transition_matrix, states=None, symmetrized=True, stationary_distribution=None):
    """Return the surprisal graph of a discrete-time Markov chain.

    Parameters
    ----------
    transition_matrix : (N, N) array_like
        Transition matrix of a discrete-time Markov chain. Must be
         a stochastic matrix with row sums equal to one.
    states : (N,) array_like, optional
        State labels of the Markov chain. Values must be unique and
        hashable. Default is ``range(N)``.
    symmetrized : bool, default True
        If True, edge weights are round-trip distances. If False, weights
        are one-way distances. See Notes for details.
    stationary_distribution : (N,) array_like, optional
        Stationary distribution of `transition_matrix`. Ignored when
        `symmetrized` is False. If `symmetrized` is True and the
        stationary distribution has already been computed, this parameter
        can be passed to avoid redundant computation.

    Returns
    -------
    graph : nx.DiGraph
        The (symmetrized) surprisal graph of the Markov chain. The weight
        of edge ``(x, y)`` is given by ``graph[x][y]['surprisal']``.

    Notes
    -----
    The *surprisal graph* of a Markov chain with state space
    :math:`V` and transition matrix :math:`T \\colon V \times V \to [0, 1]`
    is the weighted directed graph :math:`(V, E, \\delta)` with edges

    .. math:: E = \\{ (x, y) : T(x, y) > 0 \text{ and } x \ne y \\}

    and weights :math:`\\delta \\colon E \to [0, \\infty)` given by

    .. math:: \\delta(x, y) = -\\log T(x, y).

    The edge weight :math:`\\delta(x, y)` is the
    `information content <https://en.wikipedia.org/wiki/Information_content>`_,
    or *surprisal*, of a one-step transition from :math:`x` to :math:`y`.

    For a reversible Markov chain, the *symmetrized* surprisal graph is
    the graph :math:`(V, E, \\delta^*)` with edge weights

    .. math:: \\delta^*(x, y) = \\delta(x, y) + \\delta(y, x).

    """
    matrix = np.array(transition_matrix, dtype=float)

    n_states = matrix.shape[0]
    if states is None:
        states = range(n_states)
    else:
        if len(states) != n_states:
            raise ValueError('number of state labels must match number of states')
        if len(set(states)) < n_states:
            raise ValueError('state labels must be unique')

    if symmetrized:
        if not _is_reversible(matrix, stationary_distribution):
            raise ValueError('transition matrix must be reversible when ' 'symmetrized is True')
        matrix *= matrix.T
    elif not _is_stochastic_matrix(matrix):
        raise ValueError('transition matrix must be row stochastic')

    np.fill_diagonal(matrix, 0)

    graph = nx.DiGraph()
    for i, j in np.argwhere(matrix > 0):
        x, y = states[i], states[j]
        graph.add_edge(x, y, surprisal=-np.log(matrix[i, j]))

    return graph


def fundamental_sequence(graph, path, weight_attr='surprisal'):
    """Return the fundamental sequence of a path (walk).

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph.
    path : sequence
        A node path in `graph`.
    weight_attr : str or None, default 'surprisal'
        Name of the edge attribute to use as a weight/distance/cost.
        If None, all edges have equal weight.

    Returns
    -------
    tuple
        The fundamental sequence of `path`.

    """
    edges = {(x, y) for x, y in nx.utils.pairwise(path) if x != y}

    # We explicitly check that all the edges have attribute `weight_attr`.
    # This is to avoid NetworkX silently assigning a weight of 1.0 in
    # cases where the attribute is not found.
    if weight_attr is not None:
        for x, y in edges:
            if weight_attr not in graph[x][y]:
                raise ValueError(f"attribute '{weight_attr}' not found for " f"edge ({x}, {y})")

    subgraph = graph.edge_subgraph(edges)
    fs = nx.shortest_path(subgraph, source=path[0], target=path[-1], weight=weight_attr)

    return tuple(fs)


def loop_erasure(path):
    """Return the loop-erasure of a discrete sample path.

    Parameters
    ----------
    path : sequence
        A sequence of hashable values.

    Returns
    -------
    tuple
        The loop erasure of `path`.

    """
    if len(path) == 0:
        return ()
    last_index = {x: k for k, x in enumerate(path)}
    sel = [0]
    while path[sel[-1]] != path[-1]:
        sel.append(last_index[path[sel[-1]]] + 1)
    return tuple(path[k] for k in sel)


def symmetric_difference_cardinality(s, q):
    """Return the cardinality of the symmetric difference of two sets.

    Parameters
    ----------
    s : iterable
        Elements of the first set. Values must be hashable.
    q : iterable
        Elements of the second set. Values must be hashable.

    Returns
    -------
    int
        ``len(set(s) ^ set(q))``.

    """
    return len(set(s) ^ set(q))


def _is_stochastic_matrix(a, axis=1, tol=1e-10):
    if axis not in {0, 1}:
        raise ValueError('axis must be 0 or 1')
    tol = abs(tol)
    a = np.asarray(a, dtype=float)

    if a.ndim != 2:
        return False
    if a.shape[0] != a.shape[1]:
        return False
    if a.min() < -tol:
        return False
    return np.allclose(a.sum(axis=axis), np.ones(len(a)), atol=tol)


def _is_reversible(transition_matrix, stationary_distribution=None):
    if stationary_distribution is None:
        stationary_distribution = _stationary_distribution(transition_matrix)
    else:
        if not _is_stochastic_matrix(transition_matrix):
            raise ValueError('transition matrix must be row stochastic')
        stationary_distribution = np.asarray(stationary_distribution, dtype=float)
        if stationary_distribution.ndim != 1:
            raise ValueError('stationary distribution must be a 1-D array')

    flow = stationary_distribution[:, np.newaxis] * transition_matrix
    return np.allclose(flow, flow.T)


def _stationary_distribution(transition_matrix):
    if not _is_stochastic_matrix(transition_matrix):
        raise ValueError('transition matrix must be row stochastic')

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    indices = np.flatnonzero(np.isclose(eigenvalues, 1))
    p = eigenvectors[:, indices[0]]  # eigenvectors are columns
    return p / p.sum()
