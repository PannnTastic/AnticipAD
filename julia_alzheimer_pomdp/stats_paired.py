#!/usr/bin/env python3
"""
Paired significance analysis for the AnticipAD POMDP benchmark.

Reads per_seed_results.json (produced by run_multiseed_perseed.jl) and computes,
for each metric and each policy pair:
  - paired Wilcoxon signed-rank test across the 10 shared seeds
  - bias-corrected bootstrap 95% CI on the mean seed-level difference
  - Cliff's delta effect size
  - Holm-Bonferroni-corrected p-values across the full family of comparisons

A ranking claim is asserted only when Holm p < 0.05 AND the 95% CI excludes 0.

Usage:
    python stats_paired.py [per_seed_results.json]
"""
import json
import sys
import numpy as np
from scipy import stats

PATH = sys.argv[1] if len(sys.argv) > 1 else "per_seed_results.json"
METRICS = ["reward", "accuracy", "mci_acc", "dem_acc", "test_cost"]
PAIRS = [("MyopicPOMDP", "SARSOP"), ("MyopicPOMDP", "Expert"), ("SARSOP", "Expert")]


def cliffs_delta(a, b):
    n = len(a)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (n * n)


def bootstrap_ci(a, b, n_boot=10000, seed=0):
    diff = a - b
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    d = json.load(open(PATH))
    arr = lambda p, m: np.array(d[p][m])

    print("=== per-seed means ===")
    for p in ["SARSOP", "MyopicPOMDP", "Expert", "Random"]:
        print(f"{p:12s} reward={arr(p,'reward').mean():7.1f} "
              f"acc={arr(p,'accuracy').mean():5.1f} mci={arr(p,'mci_acc').mean():5.1f} "
              f"dem={arr(p,'dem_acc').mean():5.1f}")

    rows = []
    for m in METRICS:
        for x, y in PAIRS:
            a, b = arr(x, m), arr(y, m)
            try:
                _, p = stats.wilcoxon(a, b)
            except ValueError:
                p = float("nan")
            lo, hi = bootstrap_ci(a, b)
            rows.append((m, f"{x}-{y}", a.mean() - b.mean(), p, lo, hi, cliffs_delta(a, b)))

    # Holm-Bonferroni across the whole family, with step-down monotonicity
    ps = [r[3] for r in rows]
    order = np.argsort(ps)
    k = len(ps)
    holm = [0.0] * k
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * ps[idx]))
        holm[idx] = running

    print(f"\n=== paired tests ({k} comparisons, Holm-corrected) ===")
    for i, (m, pair, md, p, lo, hi, dd) in enumerate(rows):
        sig = "*" if holm[i] < 0.05 and not (lo <= 0 <= hi) else " "
        print(f"{m:10s} {pair:20s} d={md:+7.2f} CI95=[{lo:+6.2f},{hi:+6.2f}] "
              f"p={p:.4f} holm={holm[i]:.4f}{sig} cliff={dd:+.2f}")


if __name__ == "__main__":
    main()
