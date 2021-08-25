import numpy as np
import unittest

from msm_we.utils import Interval
from msm_we.ensembles import Ensemble, PathEnsemble, DiscretePathEnsemble


def mc_simulation(numsteps):
    x = 5
    int_1d = Interval([0, 100], 1)
    mc_traj = []

    for i in range(numsteps):
        dx = np.random.uniform(-10, 10)
        if (x + dx) in int_1d:
            x = x + dx
        mc_traj.append(x)
    return np.array(mc_traj)


def simple_mapping(x):
    return int(x / 10)


class TestEnsembles(unittest.TestCase):
    def setUp(self):
        np.random.seed(192348)
        self.test_trajectory0 = mc_simulation(10000)
        self.test_trajectory1 = mc_simulation(10000)
        self.test_trajectory2 = mc_simulation(10000)
        self.stateA = [0, 10]
        self.stateB = [90, 100]
        self.ensemble0 = Ensemble([self.test_trajectory0], verbose=True)

        ensemble2 = Ensemble([self.test_trajectory1], verbose=True)
        ensemble2.add_trajectory(self.test_trajectory2)
        self.ensemble_tot = self.ensemble0 + ensemble2
        self.K = self.ensemble_tot._mle_transition_matrix(n_states=10, map_function=simple_mapping)

    def testEnsemble(self):

        self.assertDictEqual(
            self.ensemble0.empirical_mfpts(self.stateA, self.stateB),
            {
                'mfptAB': 220.38095238095238,
                'std_err_mfptAB': 32.381265838152025,
                'mfptBA': 255.57142857142858,
                'std_err_mfptBA': 36.77686417987655,
            },
        )

        self.assertEqual(len(self.ensemble_tot), 3)
        self.assertEquals(np.array(self.ensemble_tot.trajectories).shape, (3, 10000))
        self.assertDictEqual(
            self.ensemble_tot.empirical_mfpts(self.stateA, self.stateB),
            {
                'mfptAB': 197.28125,
                'std_err_mfptAB': 18.417726005823464,
                'mfptBA': 271.5079365079365,
                'std_err_mfptBA': 26.119151297618238,
            },
        )

    def testPathEnsemble(self):
        pathE = PathEnsemble.from_ensemble(self.ensemble_tot, self.stateA, self.stateB)
        self.assertDictEqual(
            pathE.empirical_mfpts(self.stateA, self.stateB),
            {'mfptAB': 197.28125, 'std_err_mfptAB': 18.417726005823464, 'mfptBA': 'NaN', 'std_err_mfptBA': 'NaN'},
        )

    def testDiscretePathEnsemble(self):
        stateA = [0]
        stateB = [9]
        dpathEnsemble = DiscretePathEnsemble.from_transition_matrix(self.K, stateA=stateA, stateB=stateB, n_paths=5, ini_pops=[1])
        self.assertDictEqual(
            dpathEnsemble.empirical_mfpts(stateA, stateB),
            {'mfptAB': 158.0, 'std_err_mfptAB': 41.22814572594794, 'mfptBA': 'NaN', 'std_err_mfptBA': 'NaN'},
        )
        dpathEnsemble = DiscretePathEnsemble.from_ensemble(self.ensemble_tot, stateA, stateB, map_function=simple_mapping)
        self.assertDictEqual(
            dpathEnsemble.empirical_mfpts(stateA, stateB),
            {'mfptAB': 197.28125, 'std_err_mfptAB': 18.417726005823464, 'mfptBA': 'NaN', 'std_err_mfptBA': 'NaN'},
        )
