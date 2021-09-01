import numpy as np
import unittest

from msm_we.nmm import NonMarkovModel


class TestNonMarkovModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(192348)
        trajectories = [np.random.randint(0, 3, 100000)]
        self.nmm_model = NonMarkovModel(trajectories, stateA=[0], stateB=[2], clean_traj=False, sliding_window=True,
                                        lag_time=100)

    def testTransitionMatrix(self):
        nmm_tmatrix = np.array(
            [
                [0.33380383, 0.00000000, 0.33455463, 0.00000000, 0.00000000, 0.33164154],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.33983051, 0.00000000, 0.32717918, 0.00000000, 0.00000000, 0.33299031],
                [0.32879530, 0.00000000, 0.00000000, 0.33194167, 0.00000000, 0.33926302],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.33247538, 0.00000000, 0.00000000, 0.33109867, 0.00000000, 0.33642594],
            ]
        )

        self.assertTrue(np.allclose(nmm_tmatrix, self.nmm_model.nm_tmatrix))

    def testMFPTS(self):
        mean_fpts = self.nmm_model.mfpts()

        ref_mfpts = {'mfptAB': 301.1236654158612, 'mfptBA': 301.88142788292384}

        self.assertTrue(np.isclose(mean_fpts['mfptAB'], ref_mfpts['mfptAB']))
        self.assertTrue(np.isclose(mean_fpts['mfptBA'], ref_mfpts['mfptBA']))

    def testEmpiricalMFPTs(self):
        empirical_mfpts = {
            'mfptAB': 312.34567901234567,
            'std_err_mfptAB': 19.4177836447053,
            'mfptBA': 306.2111801242236,
            'std_err_mfptBA': 19.75290907085678,
        }
        self.assertDictEqual(empirical_mfpts, self.nmm_model.empirical_mfpts())

    def testPopulations(self):
        populations = np.array([0.33352775, 0.33174116, 0.3347311])
        self.assertTrue(np.allclose(populations, self.nmm_model.populations()))
