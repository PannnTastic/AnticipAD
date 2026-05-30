#!/usr/bin/env julia
# Re-run: full multi-seed benchmark under the RAW observed-ADNI MMSE model.
# Uses the already-solved raw-MMSE SARSOP policy (policy_rawmmse.out).
# Compares headline numbers vs the smoothed-MMSE baseline to decide whether
# the "re-run" path preserves the paper's story (Myopic best reward, etc.).

using Pkg; Pkg.activate(".")
include("src/AlzheimerPOMDP.jl")
using .AlzheimerPOMDP; using POMDPs, POMDPTools, SARSOP, Random, Statistics, Printf

const N = 10_000
const MAX_STEPS = 20
const SEEDS = [42,123,456,789,1000,2024,314,271,100,999]

raw_mmse() = begin
    d = Dict(
        :CN       => Dict(:MMSE_Normal=>0.9969,:MMSE_Sedang=>0.0031,:MMSE_Rendah=>0.0000),
        :MCI      => Dict(:MMSE_Normal=>0.9488,:MMSE_Sedang=>0.0485,:MMSE_Rendah=>0.0027),
        :Dementia => Dict(:MMSE_Normal=>0.3757,:MMSE_Sedang=>0.4617,:MMSE_Rendah=>0.1627))
    for (s,p) in d; t=sum(values(p)); for k in keys(p); p[k]=p[k]/t end end
    d
end

function evalp(pomdp,policy,updater,b0,seed)
    rng=MersenneTwister(seed)
    truth=[rand(rng,initialstate(pomdp)) for _ in 1:N]
    r=[run_episode(pomdp,policy,updater,b0,s;max_steps=MAX_STEPS,rng=rng) for s in truth]
    (rew=mean(x.total_reward for x in r), acc=mean(x.correct for x in r)*100,
     cn=mean(x.correct for x in r if x.true_state==:CN)*100,
     mci=mean(x.correct for x in r if x.true_state==:MCI)*100,
     dem=mean(x.correct for x in r if x.true_state==:Dementia)*100,
     cost=mean(x.test_cost for x in r), steps=mean(x.steps for x in r))
end
ms(v)=(round(mean(v),digits=1),round(std(v),digits=2))

function main()
    pomdp=AlzheimerPOMDPProblem(discount=0.95); pomdp.mmse_probs=raw_mmse()
    updater=POMDPTools.DiscreteUpdater(pomdp); b0=initialize_belief(updater,initialstate(pomdp))
    println("Loading raw-MMSE SARSOP (policy_rawmmse.out)...")
    sarsop=SARSOP.load_policy(pomdp,"policy_rawmmse.out")
    println("="^92)
    println("RE-RUN BENCHMARK under RAW-MMSE model  (N=$N x $(length(SEEDS)) seeds)")
    println("="^92)
    @printf("%-12s | %-13s | %-11s | %-11s | %-11s | %-11s | %-13s | %-9s\n",
            "Policy","Reward","Acc","CN","MCI","Dem","TestCost","Steps")
    println("-"^92)
    for name in ["SARSOP","MyopicPOMDP","Expert","Random"]
        R=[];A=[];C=[];M=[];D=[];T=[];S=[]
        for seed in SEEDS
            pol = name=="SARSOP" ? sarsop :
                  name=="MyopicPOMDP" ? MyopicPOMDPPlanner(pomdp,updater) :
                  name=="Expert" ? ExpertPolicy(pomdp) :
                  AlzheimerPOMDP.RandomPolicy(pomdp,MersenneTwister(seed))
            m=evalp(pomdp,pol,updater,b0,seed)
            push!(R,m.rew);push!(A,m.acc);push!(C,m.cn);push!(M,m.mci);push!(D,m.dem);push!(T,m.cost);push!(S,m.steps)
        end
        f(v)=(x=ms(v);@sprintf("%6.1f±%-5.2f",x[1],x[2]))
        @printf("%-12s | %s | %s | %s | %s | %s | %s | %s\n",name,f(R),f(A),f(C),f(M),f(D),f(T),f(S))
    end
    println("="^92)
    println("BASELINE (smoothed MMSE) for comparison:")
    println("SARSOP rew68.7 acc78.3 mci81.2 dem68.2 | Myopic rew73.0 acc78.1 mci71.4 dem87.3")
end
main()
