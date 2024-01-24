import unittest

import numpy as np
import pandas as pd

from fengi.utils_feat.quantization import hard_quantize


class TestHardQuantize(unittest.TestCase):
    def setUp(self):
        self.bins = [0, 0.5, 1]

    def normal_test(self):
        x = pd.Series(np.linspace(0, 5, 100)).rank(pct=True, method="first")
        expected = np.where(x < 0.5, 0, np.where(x < 1, 0.5, 1))
        result = hard_quantize(x, self.bins)
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_value(self):
        x = pd.Series([1, 1, 1]).rank(pct=True, method="first")
        expected = np.array([0.0, 0.5, 1.0])
        result = hard_quantize(x, self.bins)
        np.testing.assert_array_equal(result, expected)

    def test_nan_values(self):
        x = pd.Series([np.nan, np.nan, np.nan]).rank(pct=True, method="first")
        expected = np.array([np.nan, np.nan, np.nan])
        result = hard_quantize(x, self.bins)
        np.testing.assert_array_equal(result, expected)

    def test_hard_quantize_partial_nan_series(self):
        partial_nan_series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = hard_quantize(partial_nan_series, self.bins)
        expected = np.array([0.0, 0.5, np.nan, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
