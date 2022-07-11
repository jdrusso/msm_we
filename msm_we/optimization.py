import numpy as np

import logging
from rich.logging import RichHandler

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(RichHandler())
log.propagate = False


def solve_discrepancy(tmatrix, pi, B):
    """
    Given a transition matrix, solves for the discrepancy function.

    The Poisson equation for the discrepancy function is
    .. math::
        (I - K)h = 1_B - \pi(B), \:\: h \cdot \pi = 0

    however, since :math:`I-K` is singular, we instead solve
    .. math::
        (I - K + \pi \pi^T / || \pi ||^2_2)h = 1_B - \pi(B), \:\: h \cdot \pi = 0
    where :math:`h` is a volumn vector, `1_B` is an indicator function which is 1 in B and 0 everywhere
    else, :math:`\pi` is the steady-state solution of :math:`K`, and `\pi(B)` is a column vector with
    the steady-state value of :math:`\pi(B)` in every element.

    Parameters
    ----------
    tmatrix, 2D array-like: Transition matrix
    pi, array-like: Steady-state distribution for the input transition matrix
    B, array-like: Indices of target states B

    Returns
    --------
    (discrepancy, variance)
    """

    log.info("Computing pi matrix")
    norm = np.dot(pi, pi.T)
    pi_matrix = pi @ pi.T / norm

    b_indicator = np.zeros_like(pi)
    b_indicator[B] = 1.0

    pi_b = np.ones_like(pi)
    pi_b[:] = sum(pi[B])

    discrepancy = np.linalg.solve(np.identity(tmatrix.shape[0]) - tmatrix + pi_matrix,
                                  b_indicator - pi_b)

    variance = np.sqrt(
        np.dot(tmatrix, discrepancy**2) -
        np.dot(tmatrix, discrepancy)**2
    )

    return discrepancy, variance
