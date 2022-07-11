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

    This follows TODO [ what does this follow? The screenshot in Harry's mathematica ]

    :param tmatrix: Transition matrix
    :param pi: Steady-state distribution for the input transition matrix
    :param B: Indices of target states B
    :return:
    """

    log.info("Computing pi matrix")
    norm = np.dot(pi, pi.T)
    pi_matrix = pi @ pi.T / norm

    b_indicator = np.zeros_like(pi)
    b_indicator[B] = 1.0

    pi_b = np.ones_like(pi)
    pi_b[:] = pi[B]

    discrepancy = np.linalg.solve(np.identity(tmatrix.shape[0]) - tmatrix + pi_matrix,
                                  b_indicator - pi_b)

    variance = np.sqrt(
        np.dot(tmatrix, discrepancy**2) -
        np.dot(tmatrix, discrepancy)**2
    )

    return discrepancy, variance
