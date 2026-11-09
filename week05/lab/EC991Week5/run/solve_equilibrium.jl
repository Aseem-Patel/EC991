# EC991 Week 5: the stationary equilibrium of the Hopenhayn model, step by step.
# Prints V^e at every price the bisection tries, then the statistics.
# Run from the package folder:
#   julia --project=. run/solve_equilibrium.jl                    # VFI
#   julia --project=. run/solve_equilibrium.jl policy_iteration   # or howard
using EC991Week5, Printf

method = isempty(ARGS) ? :vfi : Symbol(ARGS[1])
ind = Industry()
println("Hopenhayn, method = $method: η = $(round(ind.η; digits = 3)), β = $(ind.β), c_e = $(ind.c_e), ",
        "c_f = $(ind.c_f), D̄ = $(ind.D̄), ρ = $(ind.ρ), σ = $(ind.σ), n_z = $(ind.n_z)")
println("   step   price p     V^e(p)")
step = Ref(0)
eq = solve_equilibrium(ind; method, callback = (p, Ve) -> @printf("   %4d   %.8f  %+.3e\n", step[] += 1, p, Ve))

st = industry_stats(eq)
println("\nEquilibrium")
@printf("  price p                     %.5f\n", st.p)
@printf("  mass of entrants M          %.5f\n", st.M)
@printf("  number of firms             %.5f\n", st.firms)
@printf("  entry rate = exit rate      %.4f\n", st.entry_rate)
@printf("  exit threshold z̃            %.4f  (first grid point that stays: %.4f)\n", st.z̃, eq.s.z[findfirst(.!eq.sol.χ)])
@printf("  average size (workers)      %.2f\n", st.avg_size)
@printf("  mean age (years)            %.2f\n", st.mean_age)
@printf("  output Y = demand D̄/p       %.4f = %.4f\n", st.Y, st.demand)
@printf("  aggregate profits           %.4f\n", st.profits)
@printf("  labour: production %.2f, fixed costs %.2f, entry %.2f, total %.2f\n", st.N_prod, st.N_fixed, st.N_entry, st.N)

println("\nSize classes (production workers): share of firms and of employment")
edges = [0, 10, 20, 50, 100, 500, Inf]
sd = size_distribution(eq, edges)
for k in 1:length(edges)-1
    @printf("  %5s to %-5s  firms %5.1f%%   employment %5.1f%%\n", string(Int(edges[k])),
            isinf(edges[k+1]) ? "" : string(Int(edges[k+1])), 100sd.firms[k], 100sd.employment[k])
end
