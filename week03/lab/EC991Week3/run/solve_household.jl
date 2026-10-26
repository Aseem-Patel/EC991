# The household problem with income risk, solved by VFI with grid search (slides 20 to 25).
# First the 3 x 3 example of slides 22 and 23 by hand, then the lab default: r = 0.03,
# 3-state Rouwenhorst (ρ = 0.9, σ = 0.1), 300 asset points on [0, 50], growth-rate spacing.
# Plots go to output/solve_household.pdf.
# Run from the package folder:  julia --project=. run/solve_household.jl
using EC991Week3, Plots, Printf

hh = Household()
logy, P = rouwenhorst(3, 0.9, 0.1)
y = exp.(logy)
@printf "r = %.2f, w = %.4f, R = %.2f, βR = %.4f\n" hh.r hh.w hh.R hh.β * hh.R
println("y = ", round.(y; digits = 3), ";  P = ", round.(P; digits = 4))

# ---------------------------------------------------------------- the 3 x 3 example
a3 = [0.0, 1.0, 2.0]
V0 = initial_guess(hh, a3, y)
EV = V0 * P'                                    # EV[k, j] = Σ_m V0[k, m] P[j, m]
println("\n3 x 3 example: the Bellman update, starting from V0 = u(r a + w y)/(1 - β)")
for (i, j) in ((1, 1), (3, 3))                  # slide 23's state, and one where every choice is feasible
    println("  at the state (a_$i, y_$j):")
    for k in 1:3
        c = hh.R * a3[i] + hh.w * y[j] - a3[k]
        v = c > 0 ? utility(c, hh) + hh.β * EV[k, j] : -Inf
        @printf "    choose a' = a_%d = %.0f:  c = %6.3f,  u(c) + β Σ_m P_%dm V_%dm = %s\n" k a3[k] c j k (isfinite(v) ? @sprintf("%.4f", v) : "-Inf")
    end
end
sol3 = solve_vfi(hh, a3, y, P)
println("  converged in $(sol3.iterations) iterations; V = ")
show(stdout, "text/plain", round.(sol3.V; digits = 3)); println()
println("  policy a'(a_i, y_j) = "); show(stdout, "text/plain", sol3.apol); println()

# ---------------------------------------------------------------- the lab default
a = asset_grid(300, 50.0)                       # ν = 0.02
sol = solve_vfi(hh, a, y, P)
println("\nLab default: converged in $(sol.iterations) iterations, last distance $(sol.distances[end])")
d = sol.distances
@printf "  largest ratio of successive distances %.4f (at most β = %.2f)\n" maximum(d[2:end] ./ d[1:end-1]) hh.β
println("  policy increasing in a and in y: ", policy_monotone(sol))
@printf "  constraint binds for a ≤ %.3f at y_1, never at y_3\n" a[findlast(sol.apol[:, 1] .== a[1])]
@printf "  at the top of the grid, a'(50, y_j) = %s\n" join((@sprintf("%.2f", v) for v in sol.apol[end, :]), ", ")
@printf "  (at y_3 the best choice lies between the grid points %.2f and 50: compare_maximisation.jl finds it)\n" a[end-1]

c_y = [:steelblue, :gray40, :darkorange]
p1 = plot(xlabel = "a", ylabel = "a'(a, y) - a", title = "net saving", legend = :topright)
p2 = plot(xlabel = "a", ylabel = "V(a, y)", title = "value", legend = :bottomright)
p3 = plot(xlabel = "a", ylabel = "c(a, y)", title = "consumption", legend = :bottomright)
for j in 1:3
    plot!(p1, a, sol.apol[:, j] .- a; color = c_y[j], label = "y_$j")
    plot!(p2, a, sol.V[:, j]; color = c_y[j], label = "y_$j")
    plot!(p3, a, sol.cpol[:, j]; color = c_y[j], label = "y_$j")
end
hline!(p1, [0]; color = :black, linestyle = :dash, label = "")
p4 = plot(d; yscale = :log10, xlabel = "iteration n", ylabel = "max |V^{n+1} - V^n|", title = "convergence", label = "")
outdir = joinpath(@__DIR__, "..", "output"); mkpath(outdir)
savefig(plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 750)), joinpath(outdir, "solve_household.pdf"))
println("\nplots written to output/solve_household.pdf")
