import numpy as np
from msm_we import optimization


def test_solve_discrepancy(completed_model, discrepancy, variance):

    calc_discrepancy, calc_variance = optimization.solve_discrepancy(
        tmatrix=completed_model.Tmatrix,
        pi=completed_model.pSS,
        B=completed_model.indTargets,
    )

    assert np.allclose(calc_discrepancy, discrepancy)
    assert np.allclose(calc_variance, variance)


def test_get_optimized_clustered_bins(
    completed_model, discrepancy, variance, ref_clustered_optimization, RANDOM_SEED
):
    """
    Test computing optimized bins.
    """

    steady_state = completed_model.pSS

    n_desired_we_bins = 11

    optimized_bins = optimization.get_clustered_mfpt_bins(
        variance[:-2],
        discrepancy[:-2],
        steady_state[:-2],
        n_desired_we_bins,
        seed=RANDOM_SEED,
    )

    # We should have an optimized WE bin assignment for every cluster, excluding the target/basis
    assert len(optimized_bins) == completed_model.n_clusters

    assert np.allclose(optimized_bins, ref_clustered_optimization)
