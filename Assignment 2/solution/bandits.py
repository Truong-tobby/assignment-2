"""Position-aware bandits and auditable off-policy evaluation.

One observation = one displayed item at one position, not a full page view.
All fitted policies return marginal probabilities from valid distinct-item slates.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


FEATURES = [f"user_feature_{j}" for j in range(4)]


@dataclass(frozen=True)
class Config:
    family: str = "TS"
    structure: str = "position"
    feature: str = "user_feature_0"
    strength: float = 100.0
    c: float = 0.01
    epsilon: float = 0.05


def assign_slate(scores):
    """Maximize sum of scores over a slate; an item cannot occupy two slots."""
    rows, cols = linear_sum_assignment(-scores.T)
    slate = np.empty(scores.shape[1], dtype=int)
    slate[rows] = cols
    return slate


class Bandit:
    def __init__(self, frame, config, n_items):
        self.config = config
        self.k = n_items
        self.levels = (sorted(frame[config.feature].unique())
                       if config.structure == "segment" else [])
        self.mapping = {v: i + 1 for i, v in enumerate(self.levels)}
        self.g = len(self.levels) + 1  # group 0 is global fallback
        self.p = 1 if config.structure == "pooled" else 3
        self.n = np.zeros((self.g, self.k, self.p))
        self.s = np.zeros_like(self.n)
        self.prior_mean = float((frame.click.sum() + 1) / (len(frame) + 2))
        self.add(frame)
        # Empirical Bayes center fitted ONLY on the warm-start data and then fixed.
        if config.structure == "segment":
            self.center = (self.s[0] + config.strength * self.prior_mean) / (
                self.n[0] + config.strength)
        else:
            self.center = np.full((self.k, self.p), self.prior_mean)

    def groups(self, frame):
        if self.config.structure != "segment":
            return np.zeros(len(frame), dtype=int)
        return frame[self.config.feature].map(self.mapping).fillna(0).to_numpy(int)

    def add(self, frame):
        if len(frame) == 0:
            return
        a = frame.action.to_numpy(int)
        p = np.zeros(len(frame), dtype=int) if self.p == 1 else frame.slot.to_numpy(int)
        y = frame.click.to_numpy(float)
        np.add.at(self.n[0], (a, p), 1)
        np.add.at(self.s[0], (a, p), y)
        if self.config.structure == "segment":
            g = self.groups(frame)
            known = g > 0
            np.add.at(self.n, (g[known], a[known], p[known]), 1)
            np.add.at(self.s, (g[known], a[known], p[known]), y[known])

    def posterior(self):
        strength = self.config.strength
        center = np.broadcast_to(self.center, self.n.shape).copy()
        # Do not reuse group-0 outcomes in its own empirical prior.
        center[0] = self.prior_mean
        return self.s + strength * center, self.n - self.s + strength * (1 - center)

    def probabilities(self, rng, draws=256):
        alpha, beta = self.posterior()
        result = np.zeros((self.g, self.k, 3))
        reps = draws if self.config.family == "TS" else 1
        for _ in range(reps):
            if self.config.family == "TS":
                scores = rng.beta(alpha, beta)
            elif self.config.family == "UCB":
                means = alpha / (alpha + beta)
                log_t = np.log(2 + self.n.sum(axis=(1, 2)))[:, None, None]
                scores = means + self.config.c * np.sqrt(log_t / (self.n + 1))
                # Exact ties broken with independently seeded tiny noise.
                scores += rng.uniform(0, 1e-12, scores.shape)
            else:
                raise ValueError("family must be TS or UCB")
            if self.p == 1:
                scores = np.repeat(scores, 3, axis=2)
            for g in range(self.g):
                result[g, assign_slate(scores[g]), np.arange(3)] += 1 / reps
        eps = self.config.epsilon
        # Mixture: choose a uniform DISTINCT-item slate with probability eps.
        # Its marginal probability is 1/K at every slot.
        return (1 - eps) * result + eps / self.k


def frozen_policy(train, target, config, k, seed=1, draws=256):
    model = Bandit(train, config, k)
    probs = model.probabilities(np.random.default_rng(seed), draws)
    return model, probs, model.groups(target)


def reward_model(train, target, k):
    """Fixed smoothed item-position reward model for DR; never fit on target."""
    model = Bandit(train, Config(strength=100), k)
    alpha, beta = model.posterior()
    return (alpha / (alpha + beta))[0]


def evaluate(frame, probs, groups, q, conditional_pscore):
    """Return row contributions, enabling paired comparisons on identical rows."""
    a, p = frame.action.to_numpy(int), frame.slot.to_numpy(int)
    y = frame.click.to_numpy(float)
    pa = probs[groups, a, p]
    w = pa / conditional_pscore
    # Compute expected reward for each group and slot, avoiding an N x K tensor.
    dm_lookup = (probs * q[None, :, :]).sum(axis=1)
    dm = dm_lookup[groups, p]
    dr = dm + w * (y - q[a, p])
    return {"w": w, "wy": w * y, "dr": dr,
            "ips_raw": pa / frame.propensity_score.to_numpy(float) * y}


def metric_values(v):
    w = v["w"]
    return {"IPS": float(v["wy"].mean()),
            "SNIPS": float(v["wy"].sum() / w.sum()),
            "DR": float(v["dr"].mean()),
            "IPS_original_pscore": float(v["ips_raw"].mean()),
            "ESS": float(w.sum() ** 2 / np.square(w).sum()),
            "mean_weight": float(w.mean()), "max_weight": float(w.max())}


def bootstrap_means(frame, values, seed=1, repetitions=1000):
    """Paired non-overlapping hourly block bootstrap of fixed-policy DR scores.

    Conditional on fitted policy and reward model; not an online-learning CI.
    """
    codes, _ = pd.factorize(frame.timestamp.dt.floor("h"), sort=True)
    counts = np.bincount(codes)
    sums = np.column_stack([np.bincount(codes, weights=v) for v in values])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(counts), size=(repetitions, len(counts)))
    return sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]


def replay(train, stream, config, k, batch_size, seed, discount=1.0):
    """Delayed-update replay on a uniform log over the retained item universe.

    B is the number of LOGGED SLOT ROWS between updates, NOT accepted events.
    Only matched feedback is used. Acceptance probability = target P(logged arm).
    Under uniform logging, marginal acceptance is 1/K, preserving contexts.
    Discount applies per B logged rows, NOT per second or deployed page view.
    """
    model = Bandit(train, config, k)
    policy_rng, match_rng = [np.random.default_rng(s) for s in
                             np.random.SeedSequence(seed).spawn(2)]
    accepted = clicks = 0
    ips_sum = 0.0
    trace = []
    for start in range(0, len(stream), batch_size):
        chunk = stream.iloc[start:start + batch_size]
        probs = model.probabilities(policy_rng, draws=1)
        g = model.groups(chunk)
        pa = probs[g, chunk.action.to_numpy(int), chunk.slot.to_numpy(int)]
        # Actions/probabilities are computed before observing this batch's rewards.
        selected = match_rng.random(len(chunk)) < pa
        matches = chunk.iloc[np.flatnonzero(selected)]
        ips_sum += float(np.dot(k * pa, chunk.click.to_numpy(float)))
        accepted += len(matches)
        clicks += int(matches.click.sum())
        model.n *= discount
        model.s *= discount
        model.add(matches)
        trace.append((start + len(chunk), accepted, clicks, ips_sum / (start + len(chunk))))
    return {"IPS": ips_sum / len(stream), "accepted": accepted,
            "accepted_clicks": clicks, "replay_CTR": clicks / accepted if accepted else np.nan,
            "acceptance_rate": accepted / len(stream)}, np.asarray(trace)


def exposures(frame, probs, groups):
    """Expected exposure shares, counting all K items including zero exposure."""
    gp = groups * 3 + frame.slot.to_numpy(int)
    counts = np.bincount(gp, minlength=len(probs) * 3).reshape(len(probs), 3)
    return np.einsum("gkp,gp->k", probs, counts) / len(frame)


def gini(shares):
    x = np.sort(np.asarray(shares))
    return float(2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum())
                 - (len(x) + 1) / len(x))
