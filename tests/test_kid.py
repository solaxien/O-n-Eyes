import unittest

import numpy as np

from p3.kid import kernel_inception_distance


class KidTests(unittest.TestCase):
    def test_matching_distributions_are_near_zero(self):
        rng = np.random.default_rng(1)
        mean, _ = kernel_inception_distance(rng.normal(size=(300, 32)), rng.normal(size=(300, 32)),
                                            subsets=10, subset_size=150)
        self.assertLess(abs(mean), 0.02)

    def test_shifted_distribution_scores_higher(self):
        rng = np.random.default_rng(2)
        base = rng.normal(size=(300, 32))
        near, _ = kernel_inception_distance(base, rng.normal(size=(300, 32)), subsets=10, subset_size=150)
        far, _ = kernel_inception_distance(base, rng.normal(0.5, size=(300, 32)), subsets=10, subset_size=150)
        self.assertGreater(far, near + 0.1)

    def test_rejects_tiny_sets(self):
        with self.assertRaises(ValueError):
            kernel_inception_distance(np.zeros((1, 4)), np.zeros((5, 4)))


if __name__ == "__main__":
    unittest.main()
