# EC991 Week 3: the figures of the tutorial deck, drawn with the package's own functions,
# into Week-3/Figures/. The discretisation figures:
#   1. week03_asset_grid.pdf     equal vs growth-rate spacing of the asset grid
#                                (a_1 = 0, a_{n_a} = 250, n_a = 10, ν = 0.4)
#   2. week03_tauchen_bins.pdf   Tauchen: the conditional density of log y' given y_j,
#                                carved into bins around the grid points
#   3. week03_tauchen_rouwenhorst.pdf  simulated paths of the true AR(1) and of the
#                                two chains (n_y = 5, ρ = 0.9, σ = 0.1), driven by
#                                the same shocks
#   4. week03_tauchen_persistence.pdf  Tauchen at low and high persistence (n_y = 5,
#                                σ = 0.1, ρ = 0.5 vs 0.99), two rows on one common axis
#                                in units of σ
# and the interpolation figures: one column of the value table, V^n(., y_m) for a fixed
# y_m, stored at a = 0, 2, 4, 6, 8, with V(a) = c + d log(1 + a), V(2) = 10, V(4) = 14
#   5. week03_interp_question.pdf   the stored dots and the missing value at a' = 2.5
#   6. week03_interp_linear.pdf     linear interpolation: V^n(2.5) = 11
#   7. week03_interp_spline.pdf     piecewise linear (corner) vs cubic spline (smooth join)
#   8. week03_interp_chebyshev.pdf  spline pieces p_1..p_4 vs one Chebyshev polynomial p
#                                   through five Chebyshev nodes
#   9. week03_interp_nodes.pdf      11 equally spaced vs 11 Chebyshev nodes
# and, for the code guide, PNGs written next to it in Week-3/Tutorial/:
#   week03_code_guide_value.png     the value function at the lab default (linear + Brent)
#   week03_code_guide_policy.png    the saving policy near the limit, consumption on the whole grid
#
# Run from the package folder:  julia --project=. run/make_figures.jl
ENV["GKSwstype"] = "100"
using EC991Week3, Plots, Printf, Random

figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
mkpath(figdir)
default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)

c_blue, c_orange, c_grid, c_text, c_shade = "#2a78d6", "#eb6834", "#d9d8d4", "#52514e", "#e9e8e4"
c_dot = "#1a1a1a"                        # stored values in the interpolation figures
base(; kwargs...) = plot(; framestyle = :grid, grid = :y, gridlinewidth = 0.8, foreground_color_grid = c_grid,
                         gridalpha = 1.0, foreground_color_text = c_text, foreground_color_axis = c_grid,
                         foreground_color_border = c_grid, tick_direction = :none, legend = false,
                         guidefontsize = 20, tickfontsize = 16, margin = 4Plots.mm, left_margin = 6Plots.mm, bottom_margin = 9Plots.mm, kwargs...)

ϕpdf(x) = exp(-x^2 / 2) / sqrt(2π)
latexstring(x) = "\$" * x * "\$"         # GR renders \$...\$ strings as LaTeX

# ---------------------------------------------------------------- 1. asset grid
a_1, a_max, n_a, ν = 0.0, 250.0, 10, 0.4
g_equal = asset_grid(n_a, a_max; ϕ = -a_1, spacing = :equal)
g_growth = asset_grid(n_a, a_max; ϕ = -a_1, spacing = :growth, ν)

p = base(xlabel = "assets, a", xlims = (-5, 255), ylims = (0.4, 2.6), grid = false,
         yticks = ([1, 2], ["growth-rate\nspacing", "equal\nspacing"]), size = (1000, 380))
for (y, g, c) in ((2, g_equal, c_blue), (1, g_growth, c_orange))
    plot!(p, [a_1, a_max], [y, y]; color = c_grid, linewidth = 1.5)
    scatter!(p, g, fill(y, n_a); color = c, markersize = 9, markerstrokewidth = 0)
end
savefig(p, joinpath(figdir, "week03_asset_grid.pdf"))
@printf "asset grid, ν = %.2f: first gaps %.2f, %.2f; last gap %.2f\n" ν g_growth[2] g_growth[3]-g_growth[2] g_growth[end]-g_growth[end-1]

# ---------------------------------------------------------------- 2. Tauchen bins
ρ, σ, n_y = 0.9, 0.1, 7
logy, P = tauchen(n_y, ρ, σ)
d = logy[2] - logy[1]
j = 5                                     # condition on s_j = s_5
μ = ρ * logy[j]
edges = [-Inf; (logy[1:end-1] .+ logy[2:end]) ./ 2; Inf]
xgrid = range(logy[1] - d, logy[end] + d; length = 1200)
dens(x) = ϕpdf((x - μ) / σ) / σ
ymax = dens(μ)

p = base(xlabel = latexstring("\\mathrm{log\\ endowment\\ tomorrow,}\\ \\log y'"), ylabel = latexstring("\\mathrm{density\\ of}\\ \\log y'\\ \\mathrm{given}\\ y_{$j}"), size = (1000, 480),
         xlims = (first(xgrid), last(xgrid)), ylims = (-0.2ymax, 1.2ymax), yticks = false,
         xticks = (logy, [latexstring("\\log y_{$k}") for k in 1:n_y]))
for k in 1:n_y
    lo, hi = max(edges[k], first(xgrid)), min(edges[k+1], last(xgrid))
    xs = range(lo, hi; length = 200)
    plot!(p, xs, dens.(xs); fillrange = 0, fillcolor = isodd(k) ? c_blue : c_orange, fillalpha = 0.35, linewidth = 0)
    P[j, k] > 0.005 && annotate!(p, logy[k], 1.12ymax, text(@sprintf("%.2f", P[j, k]), 18, c_text, "Computer Modern"))
end
vline!(p, edges[2:end-1]; color = c_text, linewidth = 0.8, linestyle = :dash)
plot!(p, xgrid, dens.(xgrid); color = c_text, linewidth = 2)
hline!(p, [0]; color = c_text, linewidth = 0.8)
scatter!(p, logy, zeros(n_y); color = c_text, markersize = 5, markerstrokewidth = 0)
scatter!(p, [logy[j]], [0]; color = c_orange, markersize = 8, markerstrokewidth = 0)
annotate!(p, logy[j], 0.07ymax, text("today", 16, c_orange, "Computer Modern"))
annotate!(p, edges[2], 1.12ymax, text(latexstring("P_{$j k}:"), 18, :right, c_text, "Computer Modern"))
savefig(p, joinpath(figdir, "week03_tauchen_bins.pdf"))

# ---------------------------------------------------------------- 3. Tauchen vs Rouwenhorst
ρ, σ, n_y, T = 0.9, 0.1, 5, 200
logyT, PT = tauchen(n_y, ρ, σ)
logyR, PR = rouwenhorst(n_y, ρ, σ)
@assert all(isapprox.(sum(PT; dims = 2), 1)) && all(isapprox.(sum(PR; dims = 2), 1))

# Common shocks: ε drives the AR(1); u = Φ(ε) picks the next state of each chain.
Random.seed!(991)
ε = randn(T)
logy_ar1 = zeros(T)
for t in 2:T
    logy_ar1[t] = ρ * logy_ar1[t-1] + σ * ε[t]
end
_, iT = simulate_chain(logyT, PT, normal_cdf.(ε))
_, iR = simulate_chain(logyR, PR, normal_cdf.(ε))

p = base(xlabel = "period, t", ylabel = latexstring("\\mathrm{log\\ endowment,}\\ \\log y"), size = (1000, 500), xlims = (1, T),
         legend = :outertop, legendfontsize = 17, legend_column = 3,
         legend_foreground_color = :transparent, background_color_legend = :transparent)
plot!(p, 1:T, logy_ar1; color = c_text, linewidth = 1.5, label = "true AR(1)")
plot!(p, 1:T, logyT[iT]; color = c_blue, linewidth = 2, seriestype = :steppost, label = "Tauchen")
plot!(p, 1:T, logyR[iR]; color = c_orange, linewidth = 2, seriestype = :steppost, label = "Rouwenhorst")
savefig(p, joinpath(figdir, "week03_tauchen_rouwenhorst.pdf"))

# ---------------------------------------------------------------- 4. Tauchen at high persistence
# Two rows on one horizontal axis, measured in units of σ, the one-step standard deviation.
# The bell curve N(ρ log y_3, σ²) = N(0, σ²) is then literally the same drawing in both rows;
# only the grid points and the bin edges move. The axis stops at ±12σ so the ρ = 0.5 grid
# (points 1.7σ apart) stays legible; the two outer points of the ρ = 0.99 grid (±21.3σ) are
# off the axis, which the slide caption says.
σ, n_y, j = 0.1, 5, 3                       # condition on the middle point, log y_3 = 0
xlim = 12.0
xgrid = range(-xlim, xlim; length = 4000)
ymax = ϕpdf(0.0)
σticks = (-10:5:10, [latexstring("-10\\sigma"), latexstring("-5\\sigma"), "0", latexstring("5\\sigma"), latexstring("10\\sigma")])
noticks = (-10:5:10, fill("", 5))         # top row: same axis, tick labels only on the bottom row
panels = []
for (ρ, bottom) in ((0.5, false), (0.99, true))
    local logy, P = tauchen(n_y, ρ, σ)
    local z = logy ./ σ                      # grid points in units of σ
    local d = z[2] - z[1]
    local edges = [-Inf; (z[1:end-1] .+ z[2:end]) ./ 2; Inf]
    dens(x) = ϕpdf(x - ρ * z[j])          # ρ z_j = 0: the bell is centred on today's point
    local p = base(xlabel = bottom ? latexstring("\\log y',\\ \\mathrm{in\\ units\\ of}\\ \\sigma") : "",
             title = latexstring("\\rho = $ρ\\mathrm{:\\ gap\\ between\\ grid\\ points\\ } $(round(d; digits = 1))\\,\\sigma"),
             titlefontsize = 34, titlefontcolor = c_text, titlelocation = :left, guidefontsize = 34,
             xlims = (-xlim, xlim), ylims = (-0.5ymax, 1.4ymax), yticks = false, grid = false,
             xticks = bottom ? σticks : noticks, tickfontsize = 28, size = (1600, 420),
             bottom_margin = bottom ? 12Plots.mm : 1Plots.mm, top_margin = 6Plots.mm)
    for k in 1:n_y
        lo, hi = max(edges[k], -xlim), min(edges[k+1], xlim)
        lo < hi || continue               # bin entirely off the axis
        xs = range(lo, hi; length = 400)
        plot!(p, xs, dens.(xs); fillrange = 0, fillcolor = isodd(k) ? c_blue : c_orange, fillalpha = 0.35, linewidth = 0)
        abs(z[k]) < xlim || continue      # ρ = 0.99: log y_1, log y_5 at ±21σ are off the axis (said in the slide caption)
        annotate!(p, z[k], 1.17ymax, text(@sprintf("%.2f", P[j, k]), 28, c_text, "Computer Modern"))
        annotate!(p, z[k], -0.2ymax, text(latexstring("\\log y_{$k}"), 26, c_text, "Computer Modern"))
    end
    vline!(p, filter(e -> abs(e) < xlim, edges[2:end-1]); color = c_text, linewidth = 0.8, linestyle = :dash)
    plot!(p, xgrid, dens.(xgrid); color = c_text, linewidth = 2)
    hline!(p, [0]; color = c_text, linewidth = 0.8)
    scatter!(p, filter(x -> abs(x) < xlim, z), zeros(count(x -> abs(x) < xlim, z)); color = c_text, markersize = 5, markerstrokewidth = 0)
    scatter!(p, [z[j]], [0]; color = c_orange, markersize = 8, markerstrokewidth = 0)
    annotate!(p, z[j], 0.2ymax, text("today", 26, c_orange, "Computer Modern"))
    @printf "Tauchen, ρ = %.2f: bin width %.2f σ, grid ±%.1f σ, P_33 = %.3f\n" ρ d z[end] P[j, j]
    push!(panels, p)
end
savefig(plot(panels...; layout = (2, 1), size = (1600, 900)), joinpath(figdir, "week03_tauchen_persistence.pdf"))

# ================================================================ interpolation figures
d = 4 / log(5 / 3)
c = 10 - d * log(3)
V(a) = c + d * log(1 + a)
ag = [0.0, 2, 4, 6, 8]
Vg = V.(ag)
@printf "stored values: %s\n" join(round.(Vg; digits = 2), ", ")

# ---------------------------------------------------------------- interpolants
# Linear and the natural cubic spline come from the package; Chebyshev interpolation is a
# notebook exercise, not in src/, so it is written out here.
# Chebyshev interpolation on [lo, hi] with n nodes: degree n - 1.
chebnodes(n, lo, hi) = [lo + (hi - lo) * (1 - cos((2i - 1) * π / (2n))) / 2 for i in 1:n]
function chebT(x, n)                       # T_0(x), ..., T_{n-1}(x) by the recurrence
    T = zeros(n); T[1] = 1
    n > 1 && (T[2] = x)
    for l in 3:n; T[l] = 2x * T[l-1] - T[l-2]; end
    return T
end
function chebyshev(f, n, lo, hi)
    xnode(a) = 2(a - lo) / (hi - lo) - 1
    an = chebnodes(n, lo, hi)
    Tn = [chebT(xnode(a), n) for a in an]
    θ = [sum(f(an[i]) * Tn[i][l] for i in 1:n) * (l == 1 ? 1 / n : 2 / n) for l in 1:n]
    return (a -> sum(θ .* chebT(xnode(a), n))), an
end

spl = CubicSpline(ag, Vg)
@assert all(isapprox.(spl.(ag), Vg))
cheb, an = chebyshev(V, 5, 0.0, 8.0)
@assert all(isapprox.(cheb.(an), V.(an)))
@printf "linear at 2.5: %.3f, spline: %.3f, Chebyshev: %.3f, true: %.3f\n" LinearInterp(ag, Vg)(2.5) spl(2.5) cheb(2.5) V(2.5)

xx = range(0, 8; length = 801)
ylab = latexstring("V^{\\,n}(a',\\, y_m)")
xlab = latexstring("a'")
common = (xlims = (-0.4, 8.4), ylims = (0, 21), xticks = 0:2:8, yticks = 0:5:20, xlabel = xlab)
dots!(p, x, y) = scatter!(p, x, y; color = c_dot, markersize = 7, markerstrokewidth = 0)

# ---------------------------------------------------------------- 1. the missing value
p = base(; common..., ylabel = ylab, size = (800, 560))
plot!(p, [2.5, 2.5], [0, 11.3]; color = c_text, linestyle = :dash, linewidth = 1.5)
dots!(p, ag, Vg)
annotate!(p, 2.5, 12.6, text("?", 30, c_orange, "Computer Modern"))
annotate!(p, 2.65, 1.3, text(latexstring("a' = 2.5"), 16, c_text, :left, "Computer Modern"))
annotate!(p, 6.0, 12.6, text("stored values", 16, c_dot, :left, "Computer Modern"))
plot!(p, [6.0, 6.0], [13.7, 15.8]; color = c_dot, linewidth = 1, arrow = true)
savefig(p, joinpath(figdir, "week03_interp_question.pdf"))

# ---------------------------------------------------------------- 2. linear interpolation
p = base(; common..., ylabel = ylab, size = (800, 560))
plot!(p, ag, Vg; color = c_blue, linewidth = 1.5, alpha = 0.45)
plot!(p, [2, 4], [10, 14]; color = c_blue, linewidth = 3.5)
plot!(p, [2.5, 2.5], [0, 11]; color = c_text, linestyle = :dash, linewidth = 1.5)
dots!(p, ag, Vg)
scatter!(p, [2.5], [11]; color = :white, markerstrokecolor = c_blue, markerstrokewidth = 3, markersize = 8)
annotate!(p, 2.3, 12.4, text("11", 18, c_blue, :right, "Computer Modern"))
savefig(p, joinpath(figdir, "week03_interp_linear.pdf"))

# ---------------------------------------------------------------- 3. linear vs spline
piece_cols = [c_blue, c_orange, c_blue, c_orange]
function pieces!(p, f, xs; labels = true)
    for i in 1:length(xs)-1
        local x = range(xs[i], xs[i+1]; length = 200)
        plot!(p, x, f.(x); color = piece_cols[i], linewidth = 3)
        labels && annotate!(p, (xs[i] + xs[i+1]) / 2 + 0.25, f((xs[i] + xs[i+1]) / 2) - 2.2,
                            text(latexstring("p_$i"), 20, piece_cols[i], "Computer Modern"))
    end
end
ring!(p, x, y) = plot!(p, x .+ 0.45 .* cos.(range(0, 2π; length = 100)), y .+ 1.15 .* sin.(range(0, 2π; length = 100));
                       color = c_text, linewidth = 1.5)
pl = base(; common..., ylabel = ylab, left_margin = 12Plots.mm, title = "piecewise linear", titlefontsize = 22, titlefontcolor = c_text)
plot!(pl, ag, Vg; color = c_blue, linewidth = 3)
dots!(pl, ag, Vg); ring!(pl, 2, 10)
annotate!(pl, 2.6, 6.6, text("corner", 16, c_text, :left, "Computer Modern"))
pr = base(; common..., title = "cubic spline", titlefontsize = 22, titlefontcolor = c_text)
pieces!(pr, spl, ag)
dots!(pr, ag, Vg); ring!(pr, 2, 10)
annotate!(pr, 2.6, 6.6, text("smooth join", 16, c_text, :left, "Computer Modern"))
savefig(plot(pl, pr; layout = (1, 2), size = (1400, 520)), joinpath(figdir, "week03_interp_spline.pdf"))

# ---------------------------------------------------------------- 4. spline pieces vs one Chebyshev polynomial
pl = base(; common..., ylabel = ylab, left_margin = 12Plots.mm, title = "cubic spline: four pieces", titlefontsize = 22, titlefontcolor = c_text)
pieces!(pl, spl, ag)
dots!(pl, ag, Vg)
pr = base(; common..., title = "Chebyshev: one polynomial", titlefontsize = 22, titlefontcolor = c_text)
plot!(pr, xx, cheb.(xx); color = c_blue, linewidth = 3)
dots!(pr, an, V.(an))
annotate!(pr, 4.6, cheb(4.0) - 3.2, text(latexstring("p"), 20, c_blue, "Computer Modern"))
savefig(plot(pl, pr; layout = (1, 2), size = (1400, 520)), joinpath(figdir, "week03_interp_chebyshev.pdf"))

# ---------------------------------------------------------------- 5. node placement
n = 11
p = base(xlabel = latexstring("a'"), xlims = (-0.3, 8.3), ylims = (0.4, 2.6), grid = false, xticks = 0:2:8,
         yticks = ([1, 2], ["Chebyshev", "equally\nspaced"]), size = (1000, 330), left_margin = 8Plots.mm)
for (y, g, col) in ((2, collect(range(0, 8; length = n)), c_dot), (1, chebnodes(n, 0.0, 8.0), c_blue))
    plot!(p, [0, 8], [y, y]; color = c_grid, linewidth = 1.5)
    scatter!(p, g, fill(y, n); color = col, markersize = 8, markerstrokewidth = 0)
end
savefig(p, joinpath(figdir, "week03_interp_nodes.pdf"))
# ================================================================ code guide figures
# The lab default solved by interpolate-then-optimise (linear + Brent): the value function,
# the saving policy and the consumption policy, one line per income level, as PNGs next to
# the code guide (Week-3/Tutorial/). Grid search gives the same shapes with jumps where the
# grid is coarse (slide 25), which would hide them here.
guidedir = normpath(joinpath(@__DIR__, "..", ".."))
hh = Household()
a = asset_grid(300, 50.0)
logy, P = rouwenhorst(3, 0.9, 0.1)
y = exp.(logy)
sol = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :linear)
c_y = [c_blue, c_text, c_orange]
ylabels = [latexstring("y_1 = $(round(y[1]; digits = 3))"), latexstring("y_2 = 1"), latexstring("y_3 = $(round(y[3]; digits = 3))")]
gkw = (legend = :bottomright, legendfontsize = 15, legend_foreground_color = :transparent,
       background_color_legend = :transparent, size = (1000, 560), dpi = 110)
p = base(; xlabel = latexstring("\\mathrm{assets\\ today,}\\ a"), ylabel = latexstring("V(a, y)"), gkw...)
for j in 1:3
    plot!(p, a, sol.V[:, j]; color = c_y[j], linewidth = 3, label = ylabels[j])
end
savefig(p, joinpath(guidedir, "week03_code_guide_value.png"))
pl = base(; xlabel = latexstring("a"), ylabel = latexstring("a'(a, y)"), title = "saving policy, near the borrowing limit",
          titlefontsize = 18, titlefontcolor = c_text, xlims = (0, 5), ylims = (0, 5), legend = :topleft, legendfontsize = 15,
          legend_foreground_color = :transparent, background_color_legend = :transparent)
plot!(pl, [0, 5], [0, 5]; color = c_grid, linewidth = 2, linestyle = :dash, label = "45-degree line")
pr = base(; xlabel = latexstring("a"), ylabel = latexstring("c(a, y)"), title = "consumption policy, whole grid",
          titlefontsize = 18, titlefontcolor = c_text)
for j in 1:3
    plot!(pl, a, sol.apol[:, j]; color = c_y[j], linewidth = 3, label = ylabels[j])
    plot!(pr, a, sol.cpol[:, j]; color = c_y[j], linewidth = 3, label = "")
end
savefig(plot(pl, pr; layout = (1, 2), size = (1400, 560), dpi = 110), joinpath(guidedir, "week03_code_guide_policy.png"))

println("figures written to $figdir, code guide figures to $guidedir")
