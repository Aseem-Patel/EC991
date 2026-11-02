# The joint distribution of wealth and income at a fixed interest rate, computed two ways (lecture
# slides 20 to 26): Young's histogram, and a Monte Carlo panel of households under the same policy.
# Prints the time each takes, the moments side by side, and the joint table.
# Run from the package folder:  julia --project=. run/solve_distribution.jl
# or in the REPL:               include("run/solve_distribution.jl")
using EC991Week4, Printf

econ = Economy()
s = setup(econ)
hh = Household(r = 0.0377)
sol = solve_household(hh, s; method = :egm)
g = policy_on_grid(sol, s.a_dist)

t_hist = @elapsed (dist = stationary_distribution(g, s.a_dist, s.P))
t_hist = @elapsed (dist = stationary_distribution(g, s.a_dist, s.P))     # the second run: no compilation
t_mc = @elapsed (panel = simulate_panel(g, s.a_dist, s.y, s.P))
λ_a = vec(sum(dist.λ; dims = 2))
sorted = sort(panel.a)
q(p) = sorted[ceil(Int, p * length(sorted))]

@printf "r = %.4f, w = %.4f, policy by EGM on the lab grid\n\n" hh.r hh.w
@printf "%-12s %12s %12s\n" "" "histogram" "Monte Carlo"
@printf "%-12s %12.3f %12.3f\n" "time (s)" t_hist t_mc
@printf "%-12s %12.3f %12.3f\n" "mean a" sum(s.a_dist .* λ_a) sum(panel.a) / length(panel.a)
for p in (0.25, 0.50, 0.95, 0.99)
    @printf "p%-11.0f %12.3f %12.3f\n" 100p percentile(s.a_dist, λ_a, p) q(p)
end
@printf "\nthe histogram took %d iterations; the panel has 50,000 households over 1000 periods\n" dist.iterations

edges = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, Inf]
T = joint_table(dist.λ, s.a_dist, edges)
Tmc = panel_table(panel, edges, length(s.y))
println("\nthe joint distribution, percent of households; rows are wealth bins, columns are y_1 to y_7")
@printf "%-10s" "a"
foreach(v -> @printf("%8.2f", v), s.y); println()
for b in axes(T, 1)
    @printf "%-10s" "$(edges[b])+"
    foreach(v -> @printf("%8.2f", 100v), T[b, :]); println()
end
@printf "\nlargest gap between the histogram's table and Monte Carlo's: %.2f percentage points\n" 100maximum(abs.(T .- Tmc))
println("mean assets by income: ", round.([sum(s.a_dist .* dist.λ[:, j]) / sum(dist.λ[:, j]) for j in eachindex(s.y)]; digits = 2))

# With EC991_FULL=1 the comparison is also written to the Figures folder.
if get(ENV, "EC991_FULL", "0") == "1"
    mkpath(normpath(joinpath(@__DIR__, "..", "..", "..", "Figures")))
    open(joinpath(normpath(joinpath(@__DIR__, "..", "..", "..", "Figures")), "week04_hist_vs_mc.tex"), "w") do io
        println(io, "\\begin{tabular}{lrr}\n\\toprule\n & Histogram & Monte Carlo \\\\\n\\midrule")
        @printf io "Time (s) & %.2f & %.2f \\\\\n" t_hist t_mc
        @printf io "Mean assets & %.3f & %.3f \\\\\n" sum(s.a_dist .* λ_a) sum(panel.a) / length(panel.a)
        for (name, p) in (("Median", 0.50), ("95th percentile", 0.95), ("99th percentile", 0.99))
            @printf io "%s & %.3f & %.3f \\\\\n" name percentile(s.a_dist, λ_a, p) q(p)
        end
        println(io, "\\bottomrule\n\\end{tabular}")
    end
end
