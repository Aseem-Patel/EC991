# EC991 Week 7: the code guide's figures, from the package's own functions, as PNGs in
# Week-7/Tutorial/ (drawn with Plots, run/plotting.jl):
#   week07_code_guide_aggregate.png    K' against K               (lecture slide 24)
#   week07_code_guide_individual.png   a' against a               (slide 25)
#   week07_code_guide_irf.png          the MIT shock              (slide 35)
#   week07_code_guide_tax.png          the labour tax reform      (slide 38)
#   week07_code_guide_bkm.png          adding up responses        (slide 44)
# Run from the package folder:
#   julia --project=. run/make_figures.jl
using EC991Week7, Printf, LinearAlgebra, Random
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

tutdir = normpath(joinpath(@__DIR__, "..", ".."))
out(name) = joinpath(tutdir, "week07_code_guide_$name.png")

# Part A
eq = solve_krusell_smith(KSEconomy())
savefig(plot_aggregate_rule(eq), out("aggregate"))
savefig(plot_individual_rules(eq), out("individual"))

# Part B
ss = solve_steady_state(Economy())
tr = mit_shock(ss)
savefig(plot_irf(tr), out("irf"))
tr_tax = tax_reform(ss, solve_steady_state(Economy(τ = 0.2)))
savefig(plot_tax(tr_tax), out("tax"))
Ȳ = ss.K^ss.eco.α * ss.s.N^(1 - ss.eco.α)
ε = 0.01 .* randn(Random.Xoshiro(7), 200)
K = 100 .* bkm_simulate(impulse_response(tr.K, ss.K, 0.01), ε) ./ ss.K
Y = 100 .* bkm_simulate(impulse_response(tr.Y, Ȳ, 0.01), ε) ./ Ȳ
savefig(plot_bkm(K, Y), out("bkm"))
println("wrote the code guide's figures to $tutdir")
