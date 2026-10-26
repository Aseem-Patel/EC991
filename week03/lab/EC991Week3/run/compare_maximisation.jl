# The maximisation done two ways (slides 25 to 33), at the lab default: brute-force grid
# search, and interpolate-then-optimise with linear or cubic spline interpolation and Brent's
# method. Draws one column of V interpolated both ways, the objective φ(a') at one state with
# the two choices marked, and the three saving policies; prints the gaps between them and the
# evaluation counts of Brent and golden section. Plots go to output/compare_maximisation.pdf.
# Run from the package folder:  julia --project=. run/compare_maximisation.jl
using EC991Week3, Plots, Printf

hh = Household()
logy, P = rouwenhorst(3, 0.9, 0.1)
y = exp.(logy)
a = asset_grid(300, 50.0)
sol_grid = solve_vfi(hh, a, y, P)
sol_lin = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :linear)
sol_cub = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :cubic)
@printf "iterations: grid search %d, linear + Brent %d, cubic + Brent %d\n" sol_grid.iterations sol_lin.iterations sol_cub.iterations

# ---------------------------------------------------------------- one column of V, two interpolants
# Seven stored points of V(·, y_1) on [0, 6], read off the converged 300-point solution, which
# stands in for the truth between them (slides 28 and 29).
truth = CubicSpline(a, sol_cub.V[:, 1])
xs = collect(range(0, 6; length = 7))
vs = truth.(xs)
lin, spl = LinearInterp(xs, vs), CubicSpline(xs, vs)
xx = range(0, 6; length = 601)
@printf "\none column of V, 7 nodes on [0, 6]: largest error, linear %.4f, cubic spline %.4f\n" maximum(abs.(lin.(xx) .- truth.(xx))) maximum(abs.(spl.(xx) .- truth.(xx)))
p1 = plot(xx, truth.(xx); color = :black, linewidth = 1, label = "V(a', y_1), 300 points", xlabel = "a'", title = "one column of V", legend = :bottomright)
plot!(p1, xx, lin.(xx); color = :steelblue, label = "linear, 7 nodes")
plot!(p1, xx, spl.(xx); color = :darkorange, label = "cubic spline, 7 nodes")
scatter!(p1, xs, vs; color = :black, markersize = 4, label = "")

# ---------------------------------------------------------------- φ(a') at one state
i, j = 150, 2                                   # one state: the 150th asset point, middle income
cash = hh.R * a[i] + hh.w * y[j]
V̂ = [LinearInterp(a, sol_lin.V[:, m]) for m in 1:3]
φ(ap) = utility(cash - ap, hh) + hh.β * sum(P[j, m] * V̂[m](ap) for m in 1:3)
hi = min(a[end], cash - 1e-10)
onGrid = a[a .<= hi]
kg = argmax(φ.(onGrid))
xb, fb, nb = brent(φ, a[1], hi)
xg, fg, ng = golden_section(φ, a[1], hi)
@printf "\nat the state a = %.3f, y_%d: grid search picks a' = %.4f, Brent a' = %.4f\n" a[i] j onGrid[kg] xb
@printf "  evaluations: grid search %d, Brent %d, golden section %d (golden section finds a' = %.4f)\n" length(onGrid) nb ng xg
win = range(max(a[1], xb - 0.3), min(hi, xb + 0.3); length = 301)
near = filter(x -> first(win) <= x <= last(win), onGrid)
p2 = plot(win, φ.(win); color = :black, label = "φ(a')", xlabel = "a'", title = @sprintf("objective at a = %.2f, y_%d", a[i], j), legend = :bottom)
scatter!(p2, near, φ.(near); color = :gray60, markersize = 3, label = "grid points")
scatter!(p2, [onGrid[kg]], [φ(onGrid[kg])]; color = :steelblue, markersize = 6, label = "grid search")
scatter!(p2, [xb], [fb]; color = :darkorange, markersize = 6, marker = :diamond, label = "Brent")

# ---------------------------------------------------------------- the three policies
gl, gc, glc = compare_policies(sol_lin, sol_grid), compare_policies(sol_cub, sol_grid), compare_policies(sol_lin, sol_cub)
println("\nlargest gaps between the saving policies:")
@printf "  linear vs grid search %.4f (at a = %.2f), cubic vs grid search %.4f (at a = %.2f)\n" gl.gap gl.a gc.gap gc.a
@printf "  linear vs cubic %.4f: the two interpolated policies agree far more closely\n" glc.gap
kgap = clamp(searchsortedlast(a, gl.a), 1, length(a) - 1)
@printf "  the grid gap there is %.4f: the gaps are less than one grid gap\n" a[kgap+1] - a[kgap]
@printf "  at the top of the grid with y_3: grid search a' = %.2f, linear + Brent %.4f, cubic + Brent %.4f\n" sol_grid.apol[end, 3] sol_lin.apol[end, 3] sol_cub.apol[end, 3]
p3 = plot(xlabel = "a", ylabel = "a'(a, y_2) - a", title = "net saving at y_2", legend = :topright)
plot!(p3, a, sol_grid.apol[:, 2] .- a; color = :gray40, seriestype = :steppost, label = "grid search")
plot!(p3, a, sol_lin.apol[:, 2] .- a; color = :steelblue, label = "linear + Brent")
plot!(p3, a, sol_cub.apol[:, 2] .- a; color = :darkorange, linestyle = :dash, label = "cubic + Brent")
p4 = plot(xlabel = "a", ylabel = "gap to grid search", title = "a'(a, y_2): interpolated minus grid search", legend = :topleft)
plot!(p4, a, sol_lin.apol[:, 2] .- sol_grid.apol[:, 2]; color = :steelblue, label = "linear")
plot!(p4, a, sol_cub.apol[:, 2] .- sol_grid.apol[:, 2]; color = :darkorange, label = "cubic")
plot!(p4, a[2:end], diff(a) ./ 2; color = :gray60, linestyle = :dash, label = "± half the grid gap")
plot!(p4, a[2:end], -diff(a) ./ 2; color = :gray60, linestyle = :dash, label = "")
outdir = joinpath(@__DIR__, "..", "output"); mkpath(outdir)
savefig(plot(p1, p2, p3, p4; layout = (2, 2), size = (1100, 800)), joinpath(outdir, "compare_maximisation.pdf"))
println("\nplots written to output/compare_maximisation.pdf")
