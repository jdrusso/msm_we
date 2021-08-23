import unittest
from msm_we.utils import Interval


class TestInterval(unittest.TestCase):
    def setUp(self):
        # 1D single interval
        self.interval_1 = Interval([1, 2], 1)
        self.interval_2 = Interval([[1, 2]], 1)

        # 1D union of intervals
        self.interval_3 = Interval([[1, 2], [3, 4]], 1)

        # 3D single interval
        self.interval_4 = Interval([[1, 2], [0, 1], [0.1, 0.2]], 3)

        # 3D union of intervals
        self.interval_5 = Interval([[[1, 2], [0, 1], [0.1, 0.2]], [[2, 3], [1, 2], [0.1, 0.2]]], 3)

    def test_interval_1(self):
        self.assertTrue(0.5 not in self.interval_1)
        self.assertTrue(1.3 in self.interval_1)

    def test_interval_2(self):
        self.assertTrue(0.5 not in self.interval_2)
        self.assertTrue(1.3 in self.interval_2)

    def test_interval_3(self):
        self.assertTrue(0.5 not in self.interval_3)
        self.assertTrue(1.9 in self.interval_3)
        self.assertTrue(3.3 in self.interval_3)

    def test_interval_4(self):
        self.assertTrue([1.5, 1.5, 0.15] not in self.interval_4)
        self.assertTrue([1.5, 0.5, 0.15] in self.interval_4)

    def test_interval_5(self):
        self.assertTrue([1.5, 1.5, 0.15] not in self.interval_5)
        self.assertTrue([1.5, 0.5, 0.15] in self.interval_5)
        self.assertTrue([2.5, 1.5, 0.15] in self.interval_5)
        self.assertTrue([2.5, 1.5, 0.3] not in self.interval_5)
