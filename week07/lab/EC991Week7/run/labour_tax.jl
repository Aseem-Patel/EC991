# EC991 Week 7, Part B: the transition after a labour tax reform (lecture slides 37 and
# 38). The economy is in its steady state with τ = 0; at t = 1 the tax rises to 0.2 for
# ever, and the revenue goes back to households as a lump sum. Solves both steady
# states and the transition between them, and writes labour_tax.png to this folder.
# Run from the package folder:
#   julia --project=. run/labour_tax.jl          # τ = 0.2
#   julia --project=. run/labour_tax.jl 0.4      # any other new tax rate
using EC991Week7, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

τ1 = isempty(ARGS) ? 0.2 : parse(Float64, ARGS[1])
ss0 = solve_steady_state(Economy(τ = 0.0))
ss1 = solve_steady_state(Economy(τ = τ1))
println("Steady states             τ = 0      τ = $τ1")
@printf("  interest rate r       %8.5f   %8.5f\n", ss0.r, ss1.r)
@printf("  wage w                %8.5f   %8.5f\n", ss0.w, ss1.w)
@printf("  capital K             %8.5f   %8.5f   (%+.2f%%)\n", ss0.K, ss1.K, 100 * (ss1.K / ss0.K - 1))

println("\nTransition, T = 300, ω = 0.2")
println("  iter   max |K^s - K^g|")
show_step(n, d) = (n <= 5 || n % 10 == 0) && @printf("  %4d   %.3e\n", n, d)
tr = tax_reform(ss0, ss1; callback = show_step)
@printf("converged in %d iterations; last transition, K^s_T - K_T = %.1e\n", tr.iterations, tr.end_gap)

# how long does the economy take? the date by which K has covered half, and 90%, of the way
gap = (tr.K .- ss0.K) ./ (ss1.K - ss0.K)
@printf("K covers half of the way by t = %d and 90%% by t = %d\n", findfirst(>=(0.5), gap), findfirst(>=(0.9), gap))

savefig(plot_tax(tr), "labour_tax.png")
println("\nwrote labour_tax.png")
