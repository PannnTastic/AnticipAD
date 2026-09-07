using JSON, Statistics, SHA

source = normpath(joinpath(@__DIR__, "..", "..", "julia_alzheimer_pomdp",
                          "per_seed_results.json"))
data = JSON.parsefile(source)
report = Dict{String,Any}(
    "source" => "julia_alzheimer_pomdp/per_seed_results.json",
    "source_sha256" => bytes2hex(sha256(read(source))),
    "aggregation" => "mean and sample standard deviation over ten seeds",
    "endpoint" => "terminal action agreement with the intake latent stage",
    "policies" => Dict{String,Any}(),
)
for policy in ["SARSOP", "MyopicPOMDP", "Expert", "Random"]
    metrics = Dict(k => Float64.(v) for (k, v) in data[policy])
    @assert all(length(v) == 10 && all(isfinite, v) for v in values(metrics))
    metrics["macro_accuracy"] =
        (metrics["cn_acc"] + metrics["mci_acc"] + metrics["dem_acc"]) / 3
    for key in ["accuracy", "cn_acc", "mci_acc", "dem_acc", "macro_accuracy"]
        @assert all(0 .<= metrics[key] .<= 100)
    end
    summary = Dict(k => Dict("mean" => mean(v), "sd" => std(v))
                   for (k, v) in metrics)
    report["policies"][policy] = summary
    println(policy, ": episode-weighted=", round(mean(metrics["accuracy"]), digits=3),
            "%; macro=", round(mean(metrics["macro_accuracy"]), digits=3),
            "%; decisions=", round(mean(metrics["steps"]), digits=5))
end
open(joinpath(@__DIR__, "camera_ready_metrics.json"), "w") do io
    JSON.print(io, report, 2)
end
println("Saved camera_ready_metrics.json. No simulation or policy files were modified.")
