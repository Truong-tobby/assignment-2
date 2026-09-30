"""Run: python -m unittest discover -s 'assignments/Assignment 2/solution' -v"""
import itertools
import unittest

import numpy as np
import pandas as pd

from bandits import (Bandit, Config, assign_slate, bootstrap_means, evaluate,
                     exposures, gini, metric_values, replay)
from diagnostics import bh_adjust, deviance, fairness


def toy_data(repeat=10):
    rows = []
    for t in range(repeat):
        for a, p in itertools.product(range(4), range(3)):
            row = {"action": a, "slot": p, "click": int(a == p), "propensity_score": .25,
                   "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=t)}
            row.update({f"user_feature_{j}": str(t % 2) for j in range(4)})
            rows.append(row)
    return pd.DataFrame(rows)


class AssignmentTests(unittest.TestCase):
    def test_assignment_matches_bruteforce_optimum(self):
        scores = np.random.default_rng(6).normal(size=(5, 3))
        slate = assign_slate(scores)
        expected = max(sum(scores[a, p] for p, a in enumerate(s))
                       for s in itertools.permutations(range(5), 3))
        self.assertEqual(len(set(slate)), 3)
        self.assertAlmostEqual(scores[slate, np.arange(3)].sum(), expected)

    def test_valid_slate_marginals_and_exposure_floor(self):
        frame = toy_data()
        for structure in ["pooled", "position", "segment"]:
            for family in ["TS", "UCB"]:
                model = Bandit(frame, Config(family=family, structure=structure, epsilon=.2), 4)
                probs = model.probabilities(np.random.default_rng(1), 20)
                np.testing.assert_allclose(probs.sum(axis=1), 1)
                self.assertTrue((probs.sum(axis=2) <= 1 + 1e-10).all())
                self.assertTrue((probs >= .2 / 4 - 1e-10).all())

    def test_ips_dr_against_known_full_factorial_truth(self):
        frame = toy_data()
        probs = np.zeros((1, 4, 3))
        probs[0, np.arange(3), np.arange(3)] = 1
        for q in [np.zeros((4, 3)), np.full((4, 3), .37)]:
            m = metric_values(evaluate(frame, probs, np.zeros(len(frame), int), q, .25))
            for key in ["IPS", "SNIPS", "DR"]:
                self.assertAlmostEqual(m[key], 1)
            self.assertAlmostEqual(m["mean_weight"], 1)
            self.assertAlmostEqual(m["ESS"], len(frame) / 4)

    def test_original_propensity_sensitivity_and_snips(self):
        frame = toy_data()
        frame["propensity_score"] = .2
        probs = np.full((1, 4, 3), .25)
        m = metric_values(evaluate(frame, probs, np.zeros(len(frame), int), np.zeros((4, 3)), .25))
        self.assertAlmostEqual(m["IPS_original_pscore"], 1.25 * m["IPS"])
        self.assertAlmostEqual(m["SNIPS"], frame.click.mean())

    def test_unknown_groups_fall_back_to_global(self):
        frame = toy_data()
        model = Bandit(frame, Config(structure="segment"), 4)
        unseen = frame.copy()
        unseen["user_feature_0"] = "unseen"
        self.assertTrue((model.groups(unseen) == 0).all())
        self.assertEqual(model.n[0].sum(), len(frame))
        self.assertEqual(model.n[1:].sum(), len(frame))

    def test_seed_reproducibility_and_batch_remainder(self):
        frame = toy_data()
        one, trace_one = replay(frame, frame, Config(), 4, 17, 42)
        two, trace_two = replay(frame, frame, Config(), 4, 17, 42)
        self.assertEqual(one, two)
        np.testing.assert_array_equal(trace_one, trace_two)
        self.assertEqual(trace_one[-1, 0], len(frame))
        self.assertGreater(one["accepted"], 0)
        self.assertLess(one["accepted"], len(frame))

    def test_first_batch_does_not_use_future_rewards(self):
        frame = toy_data()
        changed = frame.copy()
        changed.loc[24:, "click"] = 1 - changed.loc[24:, "click"]
        _, before = replay(frame, frame, Config(), 4, 24, 1)
        _, after = replay(frame, changed, Config(), 4, 24, 1)
        np.testing.assert_array_equal(before[0], after[0])

    def test_paired_bootstrap_preserves_identical_policy_difference(self):
        frame = toy_data()
        y = frame.click.to_numpy(float)
        result = bootstrap_means(frame, [y, y], repetitions=100)
        np.testing.assert_array_equal(result[:, 0] - result[:, 1], 0)

    def test_exposure_and_gini_include_all_items(self):
        frame = toy_data()
        share = exposures(frame, np.full((1, 4, 3), .25), np.zeros(len(frame), int))
        np.testing.assert_allclose(share, .25)
        self.assertAlmostEqual(gini(share), 0)
        self.assertAlmostEqual(gini([0, 0, 0, 1]), .75)

    def test_deviance_and_multiple_testing(self):
        self.assertAlmostEqual(float(deviance(np.array([[1, 1]]), np.array([[10, 10]]))), 0)
        self.assertGreater(float(deviance(np.array([[0, 10]]), np.array([[10, 10]]))), 0)
        np.testing.assert_allclose(bh_adjust([.01, .04, .03]), [.03, .04, .04])


if __name__ == "__main__":
    unittest.main()
