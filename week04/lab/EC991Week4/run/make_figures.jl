# EC991 Week 4: the lecture's model output, drawn with the package's own
# functions, into Week-4/Figures/. From the equilibrium on the dense grid, solved by EGM:
#   1. week04_policy_functions.pdf  g_c and g_a, one line per income state (lecture slide 15)
#   2. week04_policy_zoom.pdf       g_a near the borrowing limit (lecture slide 16)
#   3. week04_distribution.pdf      density of assets at the lowest and highest income, and the
#                                   joint distribution as a heat map
#      week04_distribution_3d.pdf   the same with the joint distribution in 3D (lecture slide 26)
#   4. week04_model_stats.tex       booktabs table (lecture slide 32)
#   5. week04_mpc.pdf               MPCs at the lowest and highest income (lecture slide 33)
# and from the lab grid:
#   6. week04_supply_jumps.pdf      E a(r) near the equilibrium: grid search jumps, EGM does not
#   7. week04_speed.pdf             time by method (lecture slide 31), read from the tables that compare_household.jl
#                                   and compare_equilibrium.jl write with EC991_FULL=1
# and, for the code guide, PNGs written next to it in Week-4/Tutorial/:
#   week04_code_guide_policy.png    the saving policy near the limit, all methods
#   week04_code_guide_joint.png     the distribution figure of the lecture slide
#
# Run from the package folder:  julia --project=. run/make_figures.jl
ENV["GKSwstype"] = "100"
using EC991Week4, Plots, Printf

figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
tutdir = normpath(joinpath(@__DIR__, "..", ".."))
mkpath(figdir)
default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)

c_low, c_high, c_grid, c_text = colorant"#2a78d6", colorant"#eb6834", "#d9d8d4", "#52514e"
base(; kwargs...) = plot(; framestyle = :grid, grid = :y, gridlinewidth = 0.8, foreground_color_grid = c_grid,
                         gridalpha = 1.0, foreground_color_text = c_text, foreground_color_axis = c_grid,
                         foreground_color_border = c_grid, tick_direction = :none,
                         guidefontsize = 16, tickfontsize = 12, legendfontsize = 12, titlefontsize = 16,
                         margin = 3Plots.mm, left_margin = 6Plots.mm, bottom_margin = 7Plots.mm, kwargs...)

eq = solve_equilibrium(dense_economy(); method = :egm)
(; a, y, a_dist) = eq.s
sol, hh = eq.sol, eq.hh
n_y = length(y)
shade(j) = cgrad([c_low, c_high])[(j - 1) / (n_y - 1)]
ylabels = [@sprintf("y = %.2f", v) for v in y]

# ---------------------------------------------------------------- 1. policy functions
keep = a .<= 20
p1 = base(xlabel = "assets, a", ylabel = "consumption, c", title = "consumption policy", legend = :bottomright)
p2 = base(xlabel = "assets, a", ylabel = "saving, a'", title = "saving policy", legend = :bottomright)
for j in 1:n_y
    plot!(p1, a[keep], sol.cpol[keep, j]; color = shade(j), label = ylabels[j])
    plot!(p2, a[keep], sol.apol[keep, j]; color = shade(j), label = ylabels[j])
end
savefig(plot(p1, p2; layout = (1, 2), size = (1100, 520)), joinpath(figdir, "week04_policy_functions.pdf"))

# ---------------------------------------------------------------- 2. near the limit
keep = a .<= 0.08
p = base(xlabel = "assets, a", ylabel = "saving, a'", title = "saving policy near the limit", legend = :topleft,
         xlims = (0, 0.08), ylims = (0, 0.13), size = (820, 520))
for j in 1:n_y
    sol.apol[1, j] < 0.13 && plot!(p, a[keep], sol.apol[keep, j]; color = shade(j), label = ylabels[j])   # the rest start above the window
end
savefig(p, joinpath(figdir, "week04_policy_zoom.pdf"))

# ---------------------------------------------------------------- 3. the distribution
# Density on narrow equal-width bins, leaving out the point mass at the limit: the node-by-node
# density of conditional_density ripples when the distribution grid is finer than the policy grid.
bins = collect(0.0:0.25:50.0)
function binned_density(j)
    λ_off = copy(eq.λ); λ_off[1, :] .= 0.0
    return joint_table(λ_off, a_dist, vcat(bins, Inf))[1:end-1, j] ./ (sum(eq.λ[:, j]) .* diff(bins))
end
mid = (bins[1:end-1] .+ bins[2:end]) ./ 2
d1, dn = conditional_density(eq.λ, a_dist, 1), conditional_density(eq.λ, a_dist, n_y)
p1 = base(xlabel = "assets, a", ylabel = "density given y", title = "assets at the lowest and highest income", legend = :topright)
plot!(p1, mid, binned_density(1); color = c_low, label = @sprintf("%s (%.1f%% at the limit)", ylabels[1], 100d1.at_limit))
plot!(p1, mid, binned_density(n_y); color = c_high, label = @sprintf("%s (%.1f%% at the limit)", ylabels[n_y], 100dn.at_limit))

# joint density on equal-width asset bins, so the picture does not depend on the grid's spacing
edges = collect(0.0:1.0:50.0)
T = joint_table(eq.λ, a_dist, vcat(edges, Inf))[1:end-1, :] ./ diff(edges)
p2 = heatmap((edges[1:end-1] .+ edges[2:end]) ./ 2, 1:n_y, T'; color = cgrad([:white, c_high, :black], [0.0, 0.55, 1.0]),
             xlabel = "assets, a", ylabel = "income, y", title = "joint distribution of wealth and income",
             yticks = (1:n_y, [@sprintf("%.2f", v) for v in y]), framestyle = :box, colorbar_title = "",
             guidefontsize = 16, tickfontsize = 12, titlefontsize = 16, margin = 3Plots.mm,
             left_margin = 6Plots.mm, bottom_margin = 7Plots.mm)
savefig(plot(p1, p2; layout = (1, 2), size = (1200, 520)), joinpath(figdir, "week04_distribution.pdf"))

# The same joint distribution in three dimensions, for the lecture: one density curve per income
# state, because income takes only these values. A surface would invent values between them.
bins3 = collect(0.0:0.5:40.0)
mid3 = (bins3[1:end-1] .+ bins3[2:end]) ./ 2
λ_off = copy(eq.λ); λ_off[1, :] .= 0.0
T3 = joint_table(λ_off, a_dist, vcat(bins3, Inf))[1:end-1, :] ./ diff(bins3)
p3 = plot(; title = @sprintf("joint distribution (%.1f%% at the limit not shown)", 100sum(eq.λ[1, :])), titlefontsize = 16,
          xlabel = "assets, a", ylabel = "income, y", zlabel = "density", camera = (35, 28), legend = false,
          yticks = (1:n_y, [@sprintf("%.2f", v) for v in y]), xlims = (0, 40), ylims = (0.5, n_y + 0.5),
          zlims = (0, 1.05maximum(T3)), guidefontsize = 14, tickfontsize = 11, framestyle = :box, grid = true,
          foreground_color_grid = c_grid, gridalpha = 0.6)
for j in n_y:-1:1
    for k in eachindex(mid3)
        plot3d!(p3, [mid3[k], mid3[k]], [j, j], [0, T3[k, j]]; color = shade(j), alpha = 0.22, linewidth = 1)   # the area under the curve
    end
    plot3d!(p3, mid3, fill(j, length(mid3)), T3[:, j]; color = shade(j), linewidth = 2.2)
end
savefig(plot(p1, p3; layout = grid(1, 2; widths = [0.4, 0.6]), size = (1400, 640)), joinpath(figdir, "week04_distribution_3d.pdf"))
savefig(plot(p1, p3; layout = grid(1, 2; widths = [0.4, 0.6]), size = (1400, 640)), joinpath(tutdir, "week04_code_guide_joint.png"))

# ---------------------------------------------------------------- 4. the statistics table
m = model_stats(eq)
row(name, fmt, vals...) = name * " & " * join((Printf.format(Printf.Format(fmt), v) for v in vals), " & ") * " \\\\"
open(joinpath(figdir, "week04_model_stats.tex"), "w") do io
    println(io, "\\begin{tabular}{lr@{\\hspace{10mm}}lr}\n\\toprule\n\\multicolumn{2}{l}{\\textbf{Prices and aggregates}} & \\multicolumn{2}{l}{\\textbf{Distribution of assets}} \\\\\n\\midrule")
    left = [("Interest rate \$r\$", "%.4f", m.r), ("Wage \$w\$", "%.4f", m.w), ("Capital \$K\$", "%.3f", m.K),
            ("Asset supply \$\\E a\$", "%.3f", m.Ea), ("Labour supply \$N\$", "%.3f", m.N), ("\$K/N\$", "%.3f", m.K_N),
            ("Output \$Y\$", "%.3f", m.Y), ("Assets to output", "%.3f", m.A_Y), ("Constrained households", "%.3f", m.constrained)]
    right = [("Mean", "%.3f", m.mean_a), ("Standard deviation", "%.3f", m.std_a), ("25th percentile", "%.3f", m.p25),
             ("Median", "%.3f", m.p50), ("95th percentile", "%.3f", m.p95), ("99th percentile", "%.3f", m.p99),
             ("Gini", "%.3f", m.gini), ("Top 10\\% share", "%.3f", m.top10), ("Top 1\\% share", "%.3f", m.top1)]
    for (l, r) in zip(left, right)
        fm(x) = Printf.format(Printf.Format(x[2]), x[3])
        println(io, l[1], " & ", fm(l), " & ", r[1], " & ", fm(r), " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# ---------------------------------------------------------------- 5. MPCs
M = mpc(sol, hh)
keep = a[1:end-1] .<= 50
p = base(xlabel = "assets, a", ylabel = "MPC", title = "marginal propensity to consume", legend = :topright, size = (820, 520))
plot!(p, a[1:end-1][keep], M[keep, 1]; color = c_low, label = ylabels[1])
plot!(p, a[1:end-1][keep], M[keep, n_y]; color = c_high, label = ylabels[n_y])
savefig(p, joinpath(figdir, "week04_mpc.pdf"))

# ---------------------------------------------------------------- 6. asset supply near the equilibrium
econ = Economy()
s = setup(econ)
rs = collect(range(0.0365, 0.0382; length = 69))
function supply(method)
    prev = nothing
    out = Float64[]
    for r in rs
        prev = excess_demand(r, econ, s; method, start = prev === nothing ? nothing : prev.sol)
        push!(out, prev.Ea)
    end
    return out
end
Ks = capital_demand.(rs, s.N)
p = base(xlabel = "interest rate, r", ylabel = "assets, capital", legend = :topleft, size = (900, 500))
plot!(p, rs, Ks; color = :gray40, label = "capital demand K(r)")
plot!(p, rs, supply(:vfi_fast); color = c_low, seriestype = :steppost, label = "asset supply, grid search")
plot!(p, rs, supply(:egm); color = c_high, label = "asset supply, EGM")
savefig(p, joinpath(figdir, "week04_supply_jumps.pdf"))

# ---------------------------------------------------------------- 7. speed by method
# Rows of the timing tables: method & n_a & time & ...  A log scale, so dots rather than bars.
function read_times(file)
    rows = [strip.(split(l, "&")) for l in eachline(joinpath(figdir, file)) if startswith(l, "VFI") || startswith(l, "EGM")]
    return [(String(r[1]), parse(Int, r[2]), parse(Float64, r[3])) for r in rows]
end
label_time(t) = t < 0.01 ? @sprintf("%.3f", t) : t < 1 ? @sprintf("%.2f", t) : t < 100 ? @sprintf("%.1f", t) : @sprintf("%.0f", t)
function speed_panel(file, title; ylabels = true, kwargs...)
    rows = read_times(file)
    names = unique(first.(rows))
    pos = Dict(n => length(names) + 1 - i for (i, n) in enumerate(names))
    p = base(; title, xlabel = "time in seconds, log scale", xscale = :log10, xlims = (0.002, 3000),
             xticks = ([0.01, 0.1, 1, 10, 100, 1000], ["0.01", "0.1", "1", "10", "100", "1000"]),
             yticks = ([pos[n] for n in names], ylabels ? names : fill("", length(names))), ylims = (0.4, length(names) + 0.6), grid = :x, kwargs...)
    for (n_a, c, off) in ((300, c_low, -0.16), (1000, c_high, 0.16))
        sub = filter(r -> r[2] == n_a, rows)
        ys = [pos[r[1]] + off for r in sub]
        scatter!(p, last.(sub), ys; color = c, markersize = 8, markerstrokecolor = :white, label = "\$n_a = $n_a\$")
        for (r, yy) in zip(sub, ys)
            annotate!(p, r[3] * 1.5, yy, text(label_time(r[3]), 11, :left, c_text))
        end
    end
    return p
end
if isfile(joinpath(figdir, "week04_timing_household.tex")) && isfile(joinpath(figdir, "week04_timing_equilibrium.tex"))
    p1 = speed_panel("week04_timing_household.tex", "one household solve"; legend = :bottomright)
    p2 = speed_panel("week04_timing_equilibrium.tex", "the full equilibrium"; legend = :none, ylabels = false)
    savefig(plot(p1, p2; layout = grid(1, 2; widths = [0.6, 0.4]), size = (1250, 480)), joinpath(figdir, "week04_speed.pdf"))
else
    println("speed figure skipped: run the two compare scripts with EC991_FULL=1 first")
end

# ---------------------------------------------------------------- the code guide's policy figure
hh_lab = Household(r = eq.r)
sols = [("grid search", solve_household(hh_lab, s; method = :vfi_fast), :gray40, :steppost),
        ("interpolation", solve_household(hh_lab, s; method = :vfi_interp_howard), c_low, :path),
        ("EGM", solve_household(hh_lab, s; method = :egm), c_high, :path)]
keep = s.a .<= 2
p = base(xlabel = "assets, a", ylabel = "a'(a, y_1)", title = "lab grid, lowest income", legend = :topleft, size = (760, 460))
for (name, so, c, st) in sols
    plot!(p, s.a[keep], so.apol[keep, 1]; color = c, seriestype = st, label = name)
end
savefig(p, joinpath(tutdir, "week04_code_guide_policy.png"))

println("r = ", round(m.r; digits = 5), ", K = ", round(m.K; digits = 4), ", Φ = ", round(eq.Φ; digits = 5))
println("constrained ", round(m.constrained; digits = 4), ", at the limit ", round(m.at_limit; digits = 4), ", top of grid a' at y_max: ", round(sol.apol[end, end]; digits = 2))
println("zoom: a' = a_1 up to a = ", [round(a[something(findlast(sol.apol[:, j] .<= a[1]), 1)]; digits = 4) for j in 1:n_y])
println("figures written to ", figdir)
