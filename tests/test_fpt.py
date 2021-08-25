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
        self.assertDictEqual(markov_mfpts, {'mfptAB': 6.420918178038423, 'mfptBA': 4.920174169581114})

    def tesMatrixFPTDirect(self):
        directional_mfpts = MatrixFPT.directional_mfpt(self.T, [0], [4], [1])
        self.assertAlmostEqual(directional_mfpts, 6.420918178038424)

    def testMatrixFPT2TargetMicrostate(self):
        mpfts_to_micro = MatrixFPT.mfpts_to_target_microstate(self.T, 4)
        self.assertTrue(np.allclose(mpfts_to_micro, np.array([6.42091818, 5.35994556, 7.24671735, 6.81752892, 0.0])))

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
        min_comm_time = MatrixFPT.min_commute_time(MatrixFPT.mfpts_matrix(self.T))
        self.assertAlmostEqual(min_comm_time, (7.343999799826479, 1, 3))

    def testMatrixFPTMaxCommuteTime(self):
        max_comm_time = MatrixFPT.max_commute_time(MatrixFPT.mfpts_matrix(self.T))
        self.assertAlmostEqual(max_comm_time, (14.446718700939037, 2, 4))
