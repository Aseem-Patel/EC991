# EC991 Week 7, Part A: the Krusell-Smith economy of lecture slides 18 to 22, step by
# step. Prints the forecasting rule at every iteration of the algorithm (slide 15),
# then the lecture's tables: the law of motion and its accuracy (slide 23), the
# wealth distribution (slide 29) and the aggregate time series (slide 30). Writes
# three graphs to this folder:
#   ks_aggregate_rule.png     K' against K, by aggregate state       (slide 24)
#   ks_individual_rules.png   a' against a, by employment and state  (slide 25)
#   ks_simulation.png         400 quarters of K and Y
# Run from the package folder:
#   julia --project=. run/solve_krusell_smith.jl
using EC991Week7, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

eco = KSEconomy()
s = ks_setup(eco)
println("Krusell-Smith: β = $(eco.β), γ = $(eco.γ), α = $(eco.α), δ = $(eco.δ), Z = $(eco.Z), y = $(eco.y)")
@printf("labour N(Z_l) = %.4f, N(Z_h) = %.4f; representative-agent K = %.3f\n", s.N..., s.K_ra)
println("\nP(y', Z' | y, Z), states (y_l, Z_l), (y_h, Z_l), (y_l, Z_h), (y_h, Z_h):")
for row in eachrow(joint_transition(eco))
    println("   ", join([@sprintf("%8.5f", x) for x in row], ""))
end

println("\n  iter   θ_0(Z_l)  θ_1(Z_l)   θ_0(Z_h)  θ_1(Z_h)   distance")
show_step(n, θ, d) = @printf("  %4d   %8.5f  %8.5f   %8.5f  %8.5f   %.2e\n", n, θ[1, 1], θ[2, 1], θ[1, 2], θ[2, 2], d)
time = @elapsed eq = solve_krusell_smith(eco; callback = show_step)
@printf("converged in %d iterations, %.0f s\n", length(eq.history), time)

println("\nLaw of motion (slide 23)")
for (z, name) in ((2, "Z = Z_h"), (1, "Z = Z_l"))
    @printf("  %s:  log K' = %.4f + %.4f log K    R² = %.7f   σ = %.5f%%\n",
            name, eq.θ[1, z], eq.θ[2, z], eq.R2[z], 100 * eq.σ[z])
end
fe = forecast_errors(eq)
@printf("  the rule alone, iterated over the whole sample: mean error %.3f%%, max %.3f%%\n", fe.dynamic_mean, fe.dynamic_max)
@printf("  the rule alone, %d quarters (25 years) ahead:  mean error %.3f%%, max %.3f%%\n",
        fe.horizon, fe.horizon_mean, fe.horizon_max)
K_sim = eq.K[eco.T_burn+1:end]
@printf("  simulated K in [%.3f, %.3f], inside the K grid [%.3f, %.3f]\n", extrema(K_sim)..., extrema(s.Kgrid)...)

w = wealth_stats(eq)
println("\nWealth distribution (slide 29), at the end of the simulation")
println("  top 1%   top 5%  top 10%  top 20%  top 30%   at a = 0    Gini")
@printf("  %5.1f    %5.1f    %5.1f    %5.1f    %5.1f     %6.3f    %.3f\n",
        w.top1, w.top5, w.top10, w.top20, w.top30, w.at_limit, w.gini)

b = business_cycle_stats(eq)
println("\nAggregate time series (slide 30)")
println("  Mean(K)   corr(C, Y)   s.d.(I)   corr(Y, Y_-4)")
@printf("  %7.3f   %8.3f    %7.4f   %9.3f\n", b.mean_K, b.corr_CY, b.sd_I, b.corr_YY4)

savefig(plot_aggregate_rule(eq), "ks_aggregate_rule.png")
savefig(plot_individual_rules(eq), "ks_individual_rules.png")
savefig(plot_ks_simulation(eq), "ks_simulation.png")
println("\nwrote ks_aggregate_rule.png, ks_individual_rules.png, ks_simulation.png")
