import numpy as np
import unittest

from msm_we.nmm import MarkovPlusColorModel


class TestNonMarkovModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(192348)
        trajectories = [np.random.randint(0, 3, 100000)]
        self.color_model = MarkovPlusColorModel(
            trajectories, stateA=[0], stateB=[2], clean_traj=False, sliding_window=True, lag_time=1
        )

    def testTransitionMatrix(self):
        nmm_tmatrix = np.array(
            [
                [0.33355320, 0.00000000, 0.33412287, 0.00000000, 0.00000000, 0.33232393],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.33502155, 0.00000000, 0.32667089, 0.00000000, 0.00000000, 0.33830756],
                [0.33502155, 0.00000000, 0.00000000, 0.32667089, 0.00000000, 0.33830756],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.33203883, 0.00000000, 0.00000000, 0.33430919, 0.00000000, 0.33365198],
            ]
        )

        self.assertTrue(np.allclose(nmm_tmatrix, self.color_model.nm_tmatrix))

    def testMFPTS(self):
        mean_fpts = self.color_model.mfpts()
        ref_mfpts = {'mfptAB': 2.991249925379562, 'mfptBA': 3.0027466707370256}

        self.assertTrue(np.isclose(mean_fpts['mfptAB'], ref_mfpts['mfptAB']))
        self.assertTrue(np.isclose(mean_fpts['mfptBA'], ref_mfpts['mfptBA']))

    def testEmpiricalMFPTs(self):
        empirical_mfpts = {
            'mfptAB': 2.9808209356515505,
            'std_err_mfptAB': 0.018764714746890707,
            'mfptBA': 2.9939057178705863,
            'std_err_mfptBA': 0.01875220558579723,
        }
        self.assertDictEqual(empirical_mfpts, self.color_model.empirical_mfpts())
