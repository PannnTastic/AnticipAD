#!/usr/bin/env julia
# Paired significance analysis for the AnticipAD POMDP benchmark (Julia port of
# stats_paired.py). Reads per_seed_results.json (from run_multiseed_perseed.jl)
# and, for each metric and policy pair, computes:
#   - exact paired Wilcoxon signed-rank test across the 10 shared seeds
#   - bootstrap 95% CI on the mean seed-level difference
#   - Cliff's delta effect size
#   - Holm-Bonferroni-corrected p-values across the full family of comparisons
#
# Uses only deps already in Project.toml (StatsBase, Statistics, Random, JSON);
# no HypothesisTests dependency. The exact Wilcoxon matches scipy's exact p-values
# for these sample sizes; bootstrap CIs differ only by RNG and converge to Python's.
#
# Usage:  julia --project=. stats_paired.jl [per_seed_results.json]

using JSON, Statistics, Random, StatsBase

const PATH    = length(ARGS) >= 1 ? ARGS[1] : "per_seed_results.json"
const METRICS = ["reward", "accuracy", "mci_acc", "dem_acc", "test_cost"]
const PAIRS   = [("MyopicPOMDP","SARSOP"), ("MyopicPOMDP","Expert"), ("SARSOP","Expert")]

cliffs_delta(a, b) = begin
    n = length(a)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    (gt - lt) / (n * n)
end

# Exact two-sided Wilcoxon signed-rank p-value (enumerates all 2^n sign patterns).
function exact_wilcoxon_p(a, b)
    d = a .- b
    d = d[d .!= 0]                      # drop zero differences
    n = length(d)
    n == 0 && return 1.0
    r = tiedrank(abs.(d))               # average ranks for ties
    Rplus = sum(r[d .> 0])
    # null distribution of the positive-rank sum over all 2^n sign assignments
    le = 0; ge = 0
    for mask in 0:(2^n - 1)
        T = 0.0
        for i in 1:n
            if (mask >> (i-1)) & 1 == 1
                T += r[i]
            end
        end
        T <= Rplus + 1e-9 && (le += 1)
        T >= Rplus - 1e-9 && (ge += 1)
    end
    tot = 2.0^n
    return min(1.0, 2 * min(le/tot, ge/tot))
end

function bootstrap_ci(a, b; nboot=10000, seed=0)
    diff = a .- b
    rng = MersenneTwister(seed)
    means = [mean(diff[rand(rng, 1:length(diff), length(diff))]) for _ in 1:nboot]
    return quantile(means, 0.025), quantile(means, 0.975)
end

function main()
    d = JSON.parsefile(PATH)
    arr(p, m) = Float64.(d[p][m])

    println("=== per-seed means ===")
    for p in ["SARSOP","MyopicPOMDP","Expert","Random"]
        println(rpad(p,12), " reward=", round(mean(arr(p,"reward")),digits=1),
                " acc=", round(mean(arr(p,"accuracy")),digits=1),
                " mci=", round(mean(arr(p,"mci_acc")),digits=1),
                " dem=", round(mean(arr(p,"dem_acc")),digits=1))
    end

    rows = []
    for m in METRICS, (x,y) in PAIRS
        a, b = arr(x,m), arr(y,m)
        push!(rows, (m, "$x-$y", mean(a)-mean(b), exact_wilcoxon_p(a,b),
                     bootstrap_ci(a,b)..., cliffs_delta(a,b)))
    end

    # Holm-Bonferroni across the whole family, with step-down monotonicity
    ps = [r[4] for r in rows]; k = length(ps)
    order = sortperm(ps); holm = zeros(k); running = 0.0
    for (rank, idx) in enumerate(order)
        running = max(running, min(1.0, (k - rank + 1) * ps[idx]))
        holm[idx] = running
    end

    println("\n=== paired tests ($k comparisons, Holm-corrected) ===")
    for (i, (m, pair, md, p, lo, hi, dd)) in enumerate(rows)
        sig = (holm[i] < 0.05 && !(lo <= 0 <= hi)) ? "*" : " "
        println(rpad(m,10), " ", rpad(pair,20),
                " d=", lpad(round(md,digits=2),7),
                " CI95=[", lpad(round(lo,digits=2),6), ",", lpad(round(hi,digits=2),6), "]",
                " p=", round(p,digits=4), " holm=", round(holm[i],digits=4), sig,
                " cliff=", round(dd,digits=2))
    end
end

main()
