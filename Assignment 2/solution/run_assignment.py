"""Run from any directory: python path/to/solution/run_assignment.py.

Use --quick for a labelled smoke/demo run. The default uses all supplied rows.
See HUONG_DAN.md for assumptions, formulas and the mapping to questions (a)-(g).
"""
import argparse
from dataclasses import asdict, replace
import json
import platform
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import scipy

from bandits import (Bandit, Config, FEATURES, assign_slate, bootstrap_means, evaluate,
                     frozen_policy, metric_values, replay, reward_model)
from diagnostics import fairness, load_data, stationarity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COLORS = {"Logger": "#515965", "Pooled TS": "#a8a9ab", "Position TS": "#2463a6",
          "Position UCB": "#c37e15", "Fair TS": "#924b83"}


def tex_escape(value):
    text = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                     ("&", r"\&"), ("#", r"\#")]:
        text = text.replace(old, new)
    return text


class Output:
    def __init__(self, directory, show=False):
        self.path = directory
        self.show = show
        for sub in ["tables", "figures", "tex"]:
            (directory / sub).mkdir(parents=True, exist_ok=True)
        self.log = (directory / "console.txt").open("w", encoding="utf-8")

    def say(self, message):
        print(message, flush=True)
        self.log.write(str(message) + "\n")
        self.log.flush()

    def table(self, number, title, frame, columns=None):
        key = f"Table_{number:02d}"
        self.say(f"\n--- {title} ({key.replace('_', ' ')}) ---")
        self.say(frame.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        frame.to_csv(self.path / "tables" / f"{key}.csv", index=False)
        if columns:
            small = frame[columns]
            lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
                     " & ".join(map(tex_escape, columns)) + r" \\", r"\hline"]
            for row in small.itertuples(index=False, name=None):
                lines.append(" & ".join(tex_escape(f"{x:.5g}" if isinstance(x, (float, np.floating))
                                                    else x) for x in row) + r" \\")
            lines += [r"\hline", r"\end{tabular}"]
            (self.path / "tex" / f"{key}.tex").write_text("\n".join(lines), encoding="utf-8")

    def figure(self, number, title, fig):
        key = f"Figure_{number:02d}"
        fig.suptitle(f"{title} (Figure {number})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.path / "figures" / f"{key}.png", dpi=170, bbox_inches="tight")
        fig.savefig(self.path / "figures" / f"{key}.pdf", bbox_inches="tight")
        self.say(f"\n--- {title} (Figure {number}) ---\n{self.path / 'figures' / (key + '.png')}")
        if self.show:
            # Agg exports are reliable in headless runs; open --show files through the OS.
            import webbrowser
            webbrowser.open((self.path / "figures" / f"{key}.png").as_uri())
        plt.close(fig)


def assess(train, target, config, k, q, seed, draws):
    model, probs, groups = frozen_policy(train, target, config, k, seed, draws)
    values = evaluate(target, probs, groups, q, 1 / k)
    return metric_values(values), (model, probs, groups, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/assignment 2/zozo_Context_80items.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="Use 20,000 rows/day; not final report data")
    parser.add_argument("--show", action="store_true", help="Open each numbered figure after saving")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Output((args.output or HERE / ("outputs_quick" if args.quick else "outputs")).resolve(), args.show)
    output.say("Assignment 2 | seed=" + str(args.seed) + (" | QUICK DEMO" if args.quick else " | FULL DATA"))
    frame, items, days, facts = load_data(args.data)
    output.say("ASSUMPTION: retained items were selected from a uniform 80-item log without outcome/context filtering. "
               "Use conditional propensity 1/K for the retained universe; also report IPS with original propensities. "
               "Do not interpret this as verified performance on all 80 original items.")
    if args.quick:
        frame = frame.groupby("day", group_keys=False).sample(n=20000, random_state=args.seed).sort_values(
            "timestamp", kind="stable").reset_index(drop=True)
    k = len(items)
    train = frame[frame.day.isin(days[:3])].reset_index(drop=True)
    valid = frame[frame.day.isin(days[3:5])].reset_index(drop=True)
    test = frame[frame.day.isin(days[5:])].reset_index(drop=True)
    fit = frame[frame.day.isin(days[:5])].reset_index(drop=True)
    seeds = list(range(args.seed, args.seed + (2 if args.quick else 5)))
    draws = 32 if args.quick else 256
    boot_n = 100 if args.quick else 2000
    output.table(1, "Source data audit (before optional quick subsample)", pd.DataFrame(
        [{"measure": key, "value": value} for key, value in facts.items()]))
    splits = pd.DataFrame([{"split": name, "from_UTC": df.day.min(), "to_UTC": df.day.max(),
                            "rows": len(df), "clicks": df.click.sum(), "CTR": df.click.mean()}
                           for name, df in [("train", train), ("validation", valid), ("test", test)]])
    output.table(2, "Chronological splits", splits, ["split", "from_UTC", "to_UTC", "rows", "CTR"])
    q_val, q_test = reward_model(train, valid, k), reward_model(fit, test, k)
    output.say("Selecting parameters ONLY on validation days; test rewards are not used for selection.")

    tuning_rows = []
    chosen = {}
    for family, parameters in [("TS", [10.0, 100.0, 1000.0]), ("UCB", [0.001, 0.01, 0.1, 1.0])]:
        candidates = []
        for parameter in parameters:
            cfg = Config(family=family, strength=parameter) if family == "TS" else Config(family=family, c=parameter)
            estimates = [assess(train, valid, cfg, k, q_val, seed, draws)[0] for seed in seeds]
            row = {"family": family, "parameter": parameter,
                   "DR_mean": np.mean([m["DR"] for m in estimates]),
                   "DR_seed_sd": np.std([m["DR"] for m in estimates], ddof=1),
                   "IPS_mean": np.mean([m["IPS"] for m in estimates]),
                   "SNIPS_mean": np.mean([m["SNIPS"] for m in estimates])}
            tuning_rows.append(row)
            candidates.append((row["DR_mean"], cfg))
        chosen[family] = max(candidates, key=lambda x: x[0])[1]
    tuning = pd.DataFrame(tuning_rows)
    output.table(6, "Parameter sensitivity on validation (seed SD is not a confidence interval)", tuning,
                 ["family", "parameter", "DR_mean", "DR_seed_sd"])

    hetero_rows = []
    for family in ["TS", "UCB"]:
        for structure in ["pooled", "position", "segment"]:
            cfg = replace(chosen[family], structure=structure)
            estimates = [assess(train, valid, cfg, k, q_val, s, draws)[0] for s in seeds]
            hetero_rows.append({"family": family, "structure": structure,
                               "feature": cfg.feature if structure == "segment" else "none",
                               "DR_mean": np.mean([m["DR"] for m in estimates]),
                               "DR_seed_sd": np.std([m["DR"] for m in estimates], ddof=1)})
    hetero = pd.DataFrame(hetero_rows)
    output.table(7, "Aggregation and heterogeneity on validation", hetero,
                 ["family", "structure", "DR_mean", "DR_seed_sd"])

    batch_rows = []
    traces = {}
    for family in ["TS", "UCB"]:
        for batch in [500, 5000, 50000]:
            records = []
            for s in seeds:
                m, trace = replay(train, valid, chosen[family], k, batch, s)
                records.append(m)
                if s == seeds[0] and batch == 5000:
                    traces[family] = trace
            batch_rows.append({"family": family, "logged_rows_per_batch": batch,
                               "IPS_mean": np.mean([m["IPS"] for m in records]),
                               "IPS_seed_sd": np.std([m["IPS"] for m in records], ddof=1),
                               "replay_CTR_mean": np.mean([m["replay_CTR"] for m in records]),
                               "accepted_mean": np.mean([m["accepted"] for m in records])})
    batch_df = pd.DataFrame(batch_rows)
    output.table(5, "Batch sensitivity using matched-feedback replay on validation", batch_df,
                 ["family", "logged_rows_per_batch", "IPS_mean", "IPS_seed_sd", "accepted_mean"])

    # Fairness extension: segmented TS + a uniform-slate exposure floor.
    # Choose a feature/epsilon on validation. No guarantee of demographic parity.
    fair_candidates = []
    base_val, _ = assess(train, valid, chosen["TS"], k, q_val, args.seed, draws)
    for feature in FEATURES:
        for eps in [0.10, 0.25, 0.50]:
            cfg = replace(chosen["TS"], structure="segment", feature=feature, epsilon=eps)
            _, (_, probs, groups, values) = assess(train, valid, cfg, k, q_val, args.seed, draws)
            summary, _, _ = fairness(valid, values, probs, groups, feature)
            summary.update(feature=feature, epsilon=eps,
                           meets_utility_floor=summary["DR"] >= 0.95 * base_val["DR"])
            fair_candidates.append((cfg, summary))
    feasible = [(cfg, row) for cfg, row in fair_candidates if row["meets_utility_floor"]]
    # If no candidate clears the utility floor, report a best-effort candidate explicitly.
    fair_cfg, _ = max(feasible or fair_candidates, key=lambda pair: pair[1]["worst_supported_uplift"])
    fair_search = pd.DataFrame([row for _, row in fair_candidates]).drop(columns=["policy"])
    output.table(16, "Fairness extension selection on validation", fair_search,
                 ["feature", "epsilon", "DR", "worst_supported_uplift", "item_gini", "meets_utility_floor"])

    policies = {"Pooled TS": replace(chosen["TS"], structure="pooled"),
                "Position TS": chosen["TS"], "Position UCB": chosen["UCB"], "Fair TS": fair_cfg}
    config_rows = [{"policy": name, **asdict(cfg)} for name, cfg in policies.items()]
    output.table(14, "Configurations locked before test evaluation", pd.DataFrame(config_rows),
                 ["policy", "structure", "strength", "c", "epsilon"])
    output.say("Evaluating frozen policies on untouched final two days (no test feedback updates).")

    artifacts = {}
    y = test.click.to_numpy(float)
    logger_values = {"w": np.ones(len(test)), "wy": y, "dr": y, "ips_raw": y}
    logger_probs = np.full((1, k, 3), 1 / k)
    logger_groups = np.zeros(len(test), dtype=int)
    artifacts["Logger"] = (None, logger_probs, logger_groups, logger_values)
    for name, cfg in policies.items():
        # The average of these slate mixtures is itself a valid stochastic policy.
        fits = [frozen_policy(fit, test, cfg, k, seed, draws) for seed in seeds]
        model, _, groups = fits[0]
        probs = np.mean([entry[1] for entry in fits], axis=0)
        values = evaluate(test, probs, groups, q_test, 1 / k)
        artifacts[name] = (model, probs, groups, values)
    names = list(artifacts)
    resamples = bootstrap_means(test, [artifacts[name][3]["dr"] for name in names], args.seed, boot_n)
    main_rows, fairness_rows, group_tables, exposure_rows = [], [], [], []
    for j, (name, (_, probs, groups, values)) in enumerate(artifacts.items()):
        main_rows.append({"policy": name, **metric_values(values),
                          "DR_low": np.quantile(resamples[:, j], .025),
                          "DR_high": np.quantile(resamples[:, j], .975)})
        summary, detail, share = fairness(test, values, probs, groups, name)
        fairness_rows.append(summary)
        group_tables.append(detail)
        exposure_rows.extend({"policy": name, "item_id": int(item), "expected_share": float(s)}
                             for item, s in zip(items, share))
    main_df = pd.DataFrame(main_rows)
    output.table(3, "Frozen test policy values (rates, not percentages)", main_df,
                 ["policy", "DR", "DR_low", "DR_high", "IPS", "SNIPS", "ESS"])
    pair_rows = []
    for left, right in [(name, "Logger") for name in names[1:]] + [("Position TS", "Position UCB")]:
        li, ri = names.index(left), names.index(right)
        delta = resamples[:, li] - resamples[:, ri]
        point = artifacts[left][3]["dr"].mean() - artifacts[right][3]["dr"].mean()
        pair_rows.append({"comparison": left + " - " + right, "DR_difference": point,
                          "low": np.quantile(delta, .025), "high": np.quantile(delta, .975)})
    pairs = pd.DataFrame(pair_rows)
    output.table(4, "Paired hourly-block bootstrap differences on test", pairs,
                 ["comparison", "DR_difference", "low", "high"])
    fair_df = pd.DataFrame(fairness_rows)
    output.table(8, "User utility and item exposure fairness on test", fair_df,
                 ["policy", "worst_supported_uplift", "item_gini", "min_item_share", "eligible_groups"])
    groups_df = pd.concat(group_tables, ignore_index=True)
    output.table(9, "All four sensitive-feature audits, including low-support groups", groups_df)
    output.table(17, "Expected test exposure for every observed item", pd.DataFrame(exposure_rows))

    daily = frame.groupby("day").click.agg(rows="size", clicks="sum", CTR="mean").reset_index()
    # Wilson score intervals for logged daily Bernoulli rates (descriptive independence assumption).
    z = 1.96
    denom = 1 + z * z / daily.rows
    center = (daily.CTR + z * z / (2 * daily.rows)) / denom
    radius = z * np.sqrt(daily.CTR * (1 - daily.CTR) / daily.rows + z * z / (4 * daily.rows ** 2)) / denom
    daily["low"], daily["high"] = center - radius, center + radius
    output.table(10, "Daily observed CTR with descriptive Wilson intervals", daily,
                 ["day", "rows", "clicks", "CTR", "low", "high"])
    stationary = stationarity(frame, repetitions=100 if args.quick else 1000, seed=args.seed)
    output.table(11, "Stationarity tests with BH adjustment across seven tests", stationary,
                 ["test", "statistic", "p_value", "q_BH", "reject_5pct"])

    discount_rows = []
    for family in ["TS", "UCB"]:
        for discount in [1.0, 0.99, 0.95]:
            metrics = [replay(train, valid, chosen[family], k, 5000, seed, discount)[0] for seed in seeds]
            discount_rows.append({"family": family, "discount_per_5000_rows": discount,
                                  "IPS_mean": np.mean([m["IPS"] for m in metrics]),
                                  "IPS_seed_sd": np.std([m["IPS"] for m in metrics], ddof=1)})
    output.table(15, "Optional nonstationarity extension on validation", pd.DataFrame(discount_rows),
                 ["family", "discount_per_5000_rows", "IPS_mean", "IPS_seed_sd"])

    recommendations = []
    for name, (model, probs, _, _) in artifacts.items():
        if model is None:
            continue
        alpha, beta = model.posterior()
        means = alpha / (alpha + beta)
        if model.p == 1:
            means = np.repeat(means, 3, axis=2)
        for group in range(model.g):
            slate = assign_slate(means[group])
            for p, a in enumerate(slate):
                recommendations.append({"policy": name, "group": "global" if group == 0 else model.levels[group - 1],
                                        "position": p + 1, "item_id": int(items[a]),
                                        "posterior_mean": means[group, a, p],
                                        "policy_probability": probs[group, a, p]})
    output.table(12, "Posterior-mean deployment slates (illustrations, not the randomized test policy)",
                 pd.DataFrame(recommendations), ["policy", "group", "position", "item_id", "posterior_mean"])
    item_rates = fit.groupby(["item_id", "position"]).click.agg(rows="size", clicks="sum", CTR="mean").reset_index()
    output.table(13, "Historical item-position statistics (training plus validation)", item_rates)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, row in main_df.iterrows():
        ax.plot([row.DR_low, row.DR_high], [j, j], color=COLORS[row.policy], linewidth=2)
        ax.scatter(row.DR, j, color=COLORS[row.policy], s=45, zorder=3)
    ax.set_yticks(range(len(main_df)), main_df.policy)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Estimated CTR per displayed slot (DR; 95% hourly-block intervals)")
    ax.set_xlim(left=0)
    output.figure(1, "Frozen policies on the final two days", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family in ["TS", "UCB"]:
        d = batch_df[batch_df.family == family]
        ax.errorbar(d.logged_rows_per_batch, d.IPS_mean, yerr=d.IPS_seed_sd, marker="o",
                    color=COLORS["Position " + family], label=family, capsize=4)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Logged slot rows per update batch")
    ax.set_ylabel("Validation IPS CTR; bars = seed SD")
    ax.legend()
    output.figure(2, "Batch size sensitivity in replay", fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, family in zip(axs, ["TS", "UCB"]):
        d = tuning[tuning.family == family]
        ax.errorbar(d.parameter, d.DR_mean, yerr=d.DR_seed_sd, marker="o", capsize=4,
                    color=COLORS["Position " + family])
        ax.set_xscale("log")
        ax.set_xlabel("TS prior strength" if family == "TS" else "UCB exploration c")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_ylabel("Validation DR CTR; bars = seed SD")
    output.figure(3, "Parameter sensitivity", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    structures = ["pooled", "position", "segment"]
    for family, offset in [("TS", -.16), ("UCB", .16)]:
        d = hetero[hetero.family == family].set_index("structure").loc[structures]
        ax.bar(np.arange(3) + offset, d.DR_mean, width=.30, yerr=d.DR_seed_sd,
               color=COLORS["Position " + family], label=family, capsize=4)
    ax.set_xticks(range(3), ["Item only", "Item and position", "Also user feature 0"])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Validation DR CTR; bars = seed SD")
    ax.legend()
    output.figure(4, "Aggregation and heterogeneity", fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    shown = fair_df[fair_df.policy != "Pooled TS"]
    for ax, column, label in [(axs[0], "item_gini", "Exposure Gini (lower = more equal)"),
                              (axs[1], "worst_supported_uplift", "Worst supported group uplift vs own logger")]:
        ax.barh(shown.policy, shown[column], color=[COLORS[n] for n in shown.policy])
        ax.set_xlabel(label)
        ax.axvline(0, color="#555555", linewidth=.7)
        ax.invert_yaxis()
    axs[1].xaxis.set_major_formatter(PercentFormatter(1))
    output.figure(5, "Fairness point estimates on test", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(pd.to_datetime(daily.day), daily.CTR,
                yerr=np.vstack([daily.CTR - daily.low, daily.high - daily.CTR]),
                marker="o", color=COLORS["Position TS"], capsize=4)
    ax.set_xticks(pd.to_datetime(daily.day), [d[5:] for d in daily.day])
    ax.set_xlabel("Date in 2019 (UTC)")
    ax.set_ylabel("Observed CTR; 95% Wilson intervals")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    output.figure(6, "Daily CTR and stationarity evidence", fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family, trace in traces.items():
        ax.plot(trace[:, 0], trace[:, 3], color=COLORS["Position " + family], label=family)
    ax.axhline(valid.click.mean(), color=COLORS["Logger"], linestyle="--", label="Logger")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Logged validation rows processed")
    ax.set_ylabel("Cumulative IPS CTR; one seed, B = 5,000")
    ax.legend()
    output.figure(7, "Replay learning trajectories", fig)

    metadata = {"quick": args.quick, "seed": args.seed, "seeds": seeds, "draws_per_seed": draws,
                "bootstrap_repetitions": boot_n, "data_path": str(args.data.resolve()), "data": facts,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "configs": config_rows, "fairness_utility_floor_feasible": bool(feasible),
                "estimand": "Frozen-policy CTR per slot in retained 79-item universe, final two UTC days",
                "pscore_assumption": "Item-only filtering from uniform logging; conditional pscore=1/K"}
    (output.path / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_findings(output, main_df, pairs, fair_df, stationary, metadata)
    output.say("\nDONE. All numbered tables printed; all numbered figures saved as PNG and PDF. "
               "Use --show to open them. See manifest.json and report.tex.")
    output.log.close()


def write_findings(output, main_df, pairs, fair_df, stationary, metadata):
    lines = []
    def paragraph(text):
        lines.append(text + "\n")
    if metadata["quick"]:
        paragraph(r"\textbf{QUICK DEMONSTRATION ONLY. Re-run the full dataset before submission.}")
    paragraph("The supplied archive contains " + f"{metadata['data']['rows']:,}" + " slot observations and "
              + str(metadata["data"]["items_observed"]) + " observed items. The original propensity is "
              + f"{metadata['data']['original_pscore']:.4f}" + ". Results condition on the retained item universe under the item-only filtering assumption.")
    for row in main_df.itertuples():
        paragraph(tex_escape(row.policy) + f": estimated DR CTR {100 * row.DR:.3f}\\% "
                  f"(95\\% hourly-block interval [{100 * row.DR_low:.3f}, {100 * row.DR_high:.3f}]\\%), "
                  f"ESS {row.ESS:,.0f}.")
    for row in pairs.itertuples():
        conclusion = ("positive" if row.low > 0 else "negative" if row.high < 0 else "inconclusive in sign")
        paragraph(tex_escape(row.comparison) + f": DR difference {100 * row.DR_difference:.3f} percentage points; "
                  f"95\\% interval [{100 * row.low:.3f}, {100 * row.high:.3f}] percentage points, {conclusion}.")
    base = fair_df.set_index("policy").loc["Position TS"]
    fair = fair_df.set_index("policy").loc["Fair TS"]
    paragraph(f"The fairness extension changes item exposure Gini from {base.item_gini:.3f} to {fair.item_gini:.3f}. "
              f"Worst supported group uplift changes from {100 * base.worst_supported_uplift:.1f}\\% to "
              f"{100 * fair.worst_supported_uplift:.1f}\\%. These are descriptive group point estimates, "
              "not a statistical or ethical guarantee of user fairness.")
    for row in stationary.itertuples():
        conclusion = "rejects" if row.reject_5pct else "does not reject"
        paragraph(tex_escape(row.test) + f": adjusted q = {row.q_BH:.4g}; {conclusion} the tested stability null at 5\\%. "
                  "Non-rejection does not establish stationarity.")
    (output.path / "tex" / "findings.tex").write_text("\n".join(lines), encoding="utf-8")
    # A concise plain-text readout, kept separate from the authored learning guide.
    (output.path / "results_summary.txt").write_text("\n".join(lines).replace(r"\%", "%").replace(r"\_", "_"),
                                                      encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
