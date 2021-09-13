import numpy as np
import unittest

import msm_we.utils as utils
from  msm_we.fpt import MatrixFPT, MarkovFPT


class TestMFPT(unittest.TestCase):
    def setUp(self):
        n_states = 5
        self.T = utils.random_markov_matrix(n_states, seed=1)

    def testMarkovFPTMean(self):
        markov_mfpts = MarkovFPT.mean_fpts(self.T, [0], [4])
        self.assertTrue(np.isclose(markov_mfpts['mfptAB'], 6.420918178038423))
        self.assertTrue(np.isclose(markov_mfpts['mfptBA'], 4.920174169581114))

    def testMatrixFPTDirectional(self):
        directional_mfpt = MatrixFPT.directional_mfpt(self.T, [0], [4], [1])
        self.assertTrue(np.isclose(directional_mfpt, 6.420918178038424))

    def testMatrixFPT2TargetMicrostate(self):
        mfpts_to_micro = MatrixFPT.mfpts_to_target_microstate(self.T, 4)
        result = np.array([6.42091818, 5.35994556, 7.24671735, 6.81752892, 0.0])
        self.assertTrue(np.allclose(mfpts_to_micro, result))

    def testMatrixFPTMatrix(self):
        mfpts_matrix = MatrixFPT.mfpts_matrix(self.T)
        result = np.array(
            [
                [0.0, 2.62899481, 7.65074814, 4.37254081, 6.42091818],
                [5.58481382, 0.0, 6.21851058, 4.00702426, 5.35994556],
                [4.98545579, 3.15239528, 0.0, 3.43921253, 7.24671735],
                [4.37802054, 3.33697554, 6.03713191, 0.0, 6.81752892],
                [4.92017417, 3.26320798, 7.20000135, 4.36442835, 0.0],
            ]
        )
        self.assertTrue(np.allclose(mfpts_matrix, result))

    def testMatrixFPTMinCommuteTime(self):
        mfpts_matrix = MatrixFPT.mfpts_matrix(self.T)
        min_comm_time, a, b = MatrixFPT.min_commute_time(mfpts_matrix)
        self.assertTrue(np.isclose(min_comm_time, 7.343999799826479))
        self.assertEqual(a, 1)
        self.assertEqual(b, 3)

    def testMatrixFPTMaxCommuteTime(self):
        mfpts_matrix = MatrixFPT.mfpts_matrix(self.T)
        max_comm_time, a, b = MatrixFPT.max_commute_time(mfpts_matrix)
        self.assertTrue(np.isclose(max_comm_time, 14.446718700939037))
        self.assertEqual(a, 2)
        self.assertEqual(b, 4)

    def testMatrixFPTDistribution(self):
        fpt_distribution = MatrixFPT.fpt_distribution(self.T, [0], [4], [0.5], max_n_lags=10)
        result = [[ 0.,          0.        ],
                  [ 1.,          0.11289507],
                  [ 2.,          0.24431041],
                  [ 3.,          0.13232006],
                  [ 4.,          0.11478687],
                  [ 5.,          0.0968178 ],
                  [ 6.,          0.08185738],
                  [ 7.,          0.06886433],
                  [ 8.,          0.05804509],
                  [ 9.,          0.04890027],
                  [10.,          0.04120272]]
        self.assertTrue(np.allclose(fpt_distribution, result))

    def testMatrixFPTDistributionLog(self):
        fpt_distribution = MatrixFPT.fpt_distribution(
            self.T, [0], [4], [0.5], max_n_lags=10, clean_recycling=True, logscale=True)
        result = [[0.00000000e+000, 0.00000000e+000],
                  [1.00000000e+001, 1.00000000e+000],
                  [1.66000000e+002, 3.85697122e-011],
                  [2.78200000e+003, 1.48701736e-204],
                  [4.64150000e+004, 0.00000000e+000],
                  [7.74263000e+005, 0.00000000e+000],
                  [1.29154960e+007, 0.00000000e+000],
                  [2.15443469e+008, 0.00000000e+000],
                  [3.59381366e+009, 0.00000000e+000],
                  [5.99484250e+010, 0.00000000e+000],
                  [1.00000000e+012,  0.00000000e+000]]
        self.assertTrue(np.allclose(fpt_distribution, result))

