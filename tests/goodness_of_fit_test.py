import unittest

import numpy as np
import statsmodels.api as sm

import penelope.common.goodness_of_fit as gof
import penelope.utility as utility

logger = utility.setup_logger()


class test_goodness_of_fit(unittest.TestCase):
    def setUp(self):
        pass

    def test_ols_when_x_equals_y_k_equals_one(self):
        # y = 1 * x + 0
        # Add intercept (i.e. constant k when x = 0)
        xs = sm.add_constant([1, 2, 3, 4])
        ys = [1, 2, 3, 4]
        m, k, _, _, _ = gof.fit_ordinary_least_square(ys, xs)
        self.assertAlmostEqual(0.0, m)
        self.assertAlmostEqual(1.0, k)

    def test_ols_when_x_equals_y_k_equals_expected(self):
        # y = 3 * x + 4
        #
        xs = sm.add_constant(np.array([1, 2, 3, 4]))
        ys = [7, 10, 13, 16]
        m, k, _, (_, _), (_, _) = gof.fit_ordinary_least_square(ys, xs)
        self.assertAlmostEqual(4.0, m)
        self.assertAlmostEqual(3.0, k)

    def test_gof_by_l2_norm(self):

        m = np.array(
            [
                [0.10, 0.11, 0.10, 0.09, 0.09, 0.11, 0.10, 0.10, 0.12, 0.08],
                [0.10, 0.10, 0.10, 0.08, 0.12, 0.12, 0.09, 0.09, 0.12, 0.08],
                [0.03, 0.02, 0.61, 0.02, 0.03, 0.07, 0.06, 0.05, 0.06, 0.05],
            ]
        )

        # The following will yield 0.0028, 0.0051, and 0.4529 for the rows:

        expected = [0.0028, 0.0051, 0.4529]

        result = gof.gof_by_l2_norm(m, axis=1)

        print(result)

        self.assertTrue(np.allclose(expected, result.round(4)))

    def test_fit_ordinary_least_square(self):
        Y = [1, 3, 4, 5, 2, 3, 4]
        X = sm.add_constant(range(1, 8))
        m, k, _, (_, _), (_, _) = gof.fit_ordinary_least_square(Y, X)
        assert round(k, 6) == round(0.25, 6)
        assert round(m, 6) == round(2.14285714, 6)

    def test_fit_ordinary_least_square_to_horizontal_line(self):
        Y = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        X = sm.add_constant(range(1, 8))
        m, k, _, (_, _), (_, _) = gof.fit_ordinary_least_square(Y, X)
        assert round(k, 6) == round(0.0, 6)
        assert round(m, 6) == round(2.0, 6)

    def test_fit_ordinary_least_square_to_3_14_x_plus_4(self):

        kp = 3.14
        mp = 4.0

        X = sm.add_constant(range(1, 8))
        Y = [kp * x + mp for x in range(1, 8)]

        m, k, _, (_, _), (_, _) = gof.fit_ordinary_least_square(Y, X)

        assert round(kp, 6) == round(k, 6)
        assert round(mp, 6) == round(m, 6)
