import unittest
import msm_we.phat as pathway_histograms

import numpy as np


class TestFundamentalSequenceClassifier(unittest.TestCase):
    def test_uniform(self):
        states = ['A', 'I1', 'I2', 'B']
        transition_matrix = np.ones((4, 4)) / 4

        fs = pathway_histograms.FundamentalSequenceClassifier(transition_matrix, states=states, symmetrized=True)

        path = ('A', 'I2', 'I1', 'I2', 'B')
        self.assertEqual(fs(path), ('A', 'I2', 'B'))

        path = ('A', 'I2', 'I1', 'A', 'I1', 'B')
        self.assertEqual(fs(path), ('A', 'I1', 'B'))
