# EC991 Week 7, Part B: an MIT shock to the week 4 Aiyagari economy (lecture slides 33
# to 36). Solves the steady state, then the transition after a 1% rise in productivity
# at t = 1 that dies out at rate ρ_Z = 0.9, printing the distance at each iteration of
# the shooting algorithm (slides 39 and 40). Writes two graphs to this folder:
#   mit_shock_irf.png          Z, K, r, w, Y, C           (slide 35)
#   mit_shock_inequality.png   Gini of wealth, constrained (slide 36)
# Run from the package folder:
#   julia --project=. run/mit_shock.jl          # dampening ω = 0.2
#   julia --project=. run/mit_shock.jl 0.5      # any other ω: try it, and watch it fail
using EC991Week7, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

ω = isempty(ARGS) ? 0.2 : parse(Float64, ARGS[1])

ss = solve_steady_state(Economy())
@printf("Steady state (week 4's economy): r = %.6f, w = %.5f, K = %.5f\n", ss.r, ss.w, ss.K)

println("\nMIT shock: ε = 0.01 at t = 1, ρ_Z = 0.9, T = 300, ω = $ω")
println("  iter   max |K^s - K^g|")
show_step(n, d) = (n <= 5 || n % 10 == 0) && @printf("  %4d   %.3e\n", n, d)
tr = mit_shock(ss; ω, callback = show_step, maxiter = 500)
@printf("converged in %d iterations; last transition, K^s_T - K_T = %.1e\n", tr.iterations, tr.end_gap)

C̄ = sum(ss.c .* ss.λ)
Ȳ = ss.K^ss.eco.α * ss.s.N^(1 - ss.eco.α)
println("\nResponses (% from steady state; r in percentage points)")
println("     t        Z        K        r        w        Y        C")
for t in (1, 2, 3, 5, 10, 15, 20, 30, 50, 100)
    @printf("  %4d  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f\n", t, 100 * (tr.Z[t] - 1), 100 * (tr.K[t] / ss.K - 1),
            100 * (tr.r[t] - ss.r), 100 * (tr.w[t] / ss.w - 1), 100 * (tr.Y[t] / Ȳ - 1), 100 * (tr.C[t] / C̄ - 1))
end
tpeak = argmax(tr.K)
@printf("capital peaks at t = %d, %.3f%% above the steady state\n", tpeak, 100 * (tr.K[tpeak] / ss.K - 1))

savefig(plot_irf(tr), "mit_shock_irf.png")
savefig(plot_inequality(tr), "mit_shock_inequality.png")
println("\nwrote mit_shock_irf.png, mit_shock_inequality.png")
