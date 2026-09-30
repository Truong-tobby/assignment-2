"""Data loading, fairness auditing, and stationarity diagnostics."""
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2_contingency

from bandits import FEATURES, exposures, gini, metric_values


def load_data(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist()
                       if n.endswith(".csv") and not n.startswith("__MACOSX/")]
            if len(members) != 1:
                raise ValueError("ZIP must contain exactly one data CSV (excluding __MACOSX).")
            with archive.open(members[0]) as stream:
                frame = pd.read_csv(stream)
    else:
        frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")].copy()
    required = {"timestamp", "item_id", "position", "click", "propensity_score", *FEATURES}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing required values; investigate rather than silently drop.")
    if not frame.click.isin([0, 1]).all() or not frame.position.isin([1, 2, 3]).all():
        raise ValueError("Invalid rewards or positions.")
    if not frame.propensity_score.between(0, 1, inclusive="right").all():
        raise ValueError("Invalid propensity score.")
    if frame.propensity_score.nunique() != 1:
        raise ValueError("This replay implementation requires a uniform log; do not use BTS logs.")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, format="mixed")
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    items = np.sort(frame.item_id.unique())
    frame["action"] = pd.Categorical(frame.item_id, categories=items).codes.astype(int)
    frame["slot"] = frame.position.to_numpy(int) - 1
    for col in FEATURES:
        frame[col] = frame[col].astype(str)
    frame["day"] = frame.timestamp.dt.strftime("%Y-%m-%d")
    days = sorted(frame.day.unique())
    if len(days) != 7:
        raise ValueError(f"Expected 7 calendar days; found {len(days)}.")
    facts = {
        "rows": len(frame), "items_observed": len(items), "positions": frame.position.nunique(),
        "clicks": int(frame.click.sum()), "observed_CTR": float(frame.click.mean()),
        "original_pscore": float(frame.propensity_score.iloc[0]),
        "conditional_pscore_assumed": 1 / len(items),
        "duplicate_rows_retained": duplicate_rows,
        "first_timestamp": str(frame.timestamp.min()), "last_timestamp": str(frame.timestamp.max()),
        "sha256_archive": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    }
    return frame, items, days, facts


def fairness(frame, values, probs, groups, name):
    records = []
    for feature in FEATURES:
        for level, positions in frame.groupby(feature, sort=True).indices.items():
            ix = np.asarray(positions)
            m = metric_values({key: val[ix] for key, val in values.items()})
            observed = float(frame.iloc[ix].click.mean())
            clicks = int(frame.iloc[ix].click.sum())
            # Fixed before any model comparison. Retain excluded groups in the output.
            # Same population for EVERY policy: never drop a bad-performing group
            # merely because the policy changes its realized effective sample size.
            eligible = len(ix) >= 100 * probs.shape[1] and clicks >= 10
            records.append({"policy": name, "feature": feature, "group": level,
                            "rows": len(ix), "logged_clicks": clicks,
                            "logged_CTR": observed, "DR": m["DR"], "SNIPS": m["SNIPS"],
                            "ESS": m["ESS"], "DR_uplift": m["DR"] / observed - 1 if observed else np.nan,
                            "eligible": eligible, "low_realized_ESS": m["ESS"] < 100})
    detail = pd.DataFrame(records)
    share = exposures(frame, probs, groups)
    supported = detail[detail.eligible]
    gaps = supported.groupby("feature").DR.agg(lambda x: x.max() - x.min())
    summary = {"policy": name, "DR": float(values["dr"].mean()),
               "worst_supported_uplift": float(supported.DR_uplift.min()),
               "largest_within_feature_gap": float(gaps.max()),
               "eligible_groups": len(supported), "all_groups": len(detail),
               "item_gini": gini(share),
               "item_entropy_normalized": float(-xlogy(share, share).sum() / np.log(len(share))),
               "min_item_share": float(share.min()), "max_item_share": float(share.max()),
               "item_coverage": float((share > 0).mean())}
    return summary, detail, share


def bh_adjust(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    out = np.empty(len(p))
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    out[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    return out


def deviance(successes, counts):
    """Binomial likelihood-ratio statistic: day-specific vs constant within cells."""
    mu = successes.sum(axis=-1, keepdims=True) / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    expected_success = counts * mu
    failures = counts - successes
    expected_failure = counts * (1 - mu)
    return 2 * (xlogy(successes, np.divide(successes, expected_success,
                                         out=np.ones_like(expected_success), where=expected_success > 0))
                + xlogy(failures, np.divide(failures, expected_failure,
                                          out=np.ones_like(expected_failure), where=expected_failure > 0))).sum(axis=(-2, -1))


def stationarity(frame, repetitions=500, seed=1):
    days = sorted(frame.day.unique())
    rng = np.random.default_rng(seed)
    records = []
    for title, cols in [("Unadjusted reward by day", []),
                        ("Reward given item and position", ["item_id", "position"]),
                        ("Reward given item position and feature 0", ["item_id", "position", "user_feature_0"])]:
        work = frame.assign(cell="all") if not cols else frame
        keys = cols or ["cell"]
        agg = work.groupby(keys + ["day"], observed=True).click.agg(["sum", "size"])
        n = agg["size"].unstack("day").reindex(columns=days).fillna(0).to_numpy(int)
        s = agg["sum"].unstack("day").reindex(columns=days).fillna(0).to_numpy(int)
        mu = s.sum(axis=1) / n.sum(axis=1)
        observed = float(deviance(s, n))
        exceed = 0
        for start in range(0, repetitions, 50):
            b = min(50, repetitions - start)
            sim = rng.binomial(n, mu[:, None], size=(b, *n.shape))
            exceed += int((deviance(sim, n) >= observed).sum())
        records.append({"test": title, "statistic": observed, "p_value": (exceed + 1) / (repetitions + 1),
                        "method": "parametric binomial bootstrap", "cells": len(n), "B": repetitions})
    for feature in FEATURES:
        table = pd.crosstab(frame.day, frame[feature])
        stat, p, _, expected = chi2_contingency(table, correction=False)
        records.append({"test": f"Context mix by day: {feature}", "statistic": stat,
                        "p_value": p, "method": "chi square (exploratory for sparse cells)",
                        "cells": table.size, "B": 0})
    result = pd.DataFrame(records)
    result["q_BH"] = bh_adjust(result.p_value)
    result["reject_5pct"] = result.q_BH < 0.05
    return result
