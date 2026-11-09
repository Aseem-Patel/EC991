# EC991 Week 5: the lecture's model output, drawn with the package's own functions,
# into Week-5/Figures/, at the Industry() defaults:
#   1. week05_misallocation.pdf      output lost to wedges, random or taxing productive firms
#   2. week05_value_function.pdf     π, β E[V(z')|z] and V near the exit threshold
#   3. week05_free_entry.pdf         V^e(p), and the price where it is zero
#   4. week05_size_distribution.pdf  firms and employment by size
#   5. week05_stats.tex              booktabs table of the equilibrium
#   6. week05_exit_grid.pdf          exit rate against c_e: grid, lottery, fine grid
#      week05_grid.tex               the statistics as the grid is refined
#   7. week05_comparative_statics.pdf  p, z̃, exit rate and size as c_e or c_f rise
#      week05_comparative_table.tex  baseline, c_e = 60, c_f = 30
#   8. week05_age_profile.pdf        a cohort of entrants: survival, size, exit by age
# and, for the code guide, PNGs in Week-5/Tutorial/:
#   week05_code_guide_value.png, week05_code_guide_exit.png
# The speed table week05_speed.tex comes from run/compare_methods.jl with EC991_FULL=1.
#
# Run from the package folder:  julia --project=. run/make_figures.jl
ENV["GKSwstype"] = "100"
using EC991Week5, Plots, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead

figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
tutdir = normpath(joinpath(@__DIR__, "..", ".."))
mkpath(figdir)
default(fontfamily = "Computer Modern", linewidth = 2, framestyle = :box, grid = false)

c_low, c_high, c_grid, c_text = colorant"#2a78d6", colorant"#eb6834", "#d9d8d4", "#52514e"
c_dark = colorant"#3d3c3a"
base(; kwargs...) = plot(; framestyle = :grid, grid = :y, gridlinewidth = 0.8, foreground_color_grid = c_grid,
                         gridalpha = 1.0, foreground_color_text = c_text, foreground_color_axis = c_grid,
                         foreground_color_border = c_grid, tick_direction = :none,
                         guidefontsize = 16, tickfontsize = 12, legendfontsize = 12, titlefontsize = 16,
                         margin = 3Plots.mm, left_margin = 6Plots.mm, bottom_margin = 7Plots.mm, kwargs...)

ind = Industry()
eq = solve_equilibrium(ind)
st = industry_stats(eq)
(; z, logz, F, g) = eq.s

# ---------------------------------------------------------------- 1. misallocation
# z from the model's invariant distribution. Random wedges: at every z, half the firms get
# log(1 + τ) = +sd and half -sd. Correlated: log(1 + τ) = -sd × standardised log z.
ζ = (logz .- sum(g .* logz)) ./ sqrt(sum(g .* (logz .- sum(g .* logz)) .^ 2))
sds = collect(0.0:0.01:0.6)
loss_rand = [100 * (1 - misallocation(vcat(z, z), exp.(vcat(fill(v, length(z)), fill(-v, length(z)))) .- 1;
                                       ind.η, m = vcat(g, g) ./ 2)) for v in sds]
loss_corr = [100 * (1 - misallocation(z, exp.(-v .* ζ) .- 1; ind.η, m = g)) for v in sds]
p = base(xlabel = "standard deviation of log(1 + τ)", ylabel = "output lost, % (1 - Ω)", legend = :topleft, size = (820, 500))
plot!(p, sds, loss_rand; color = c_low, label = "random wedges")
plot!(p, sds, loss_corr; color = c_high, label = "wedges that tax productive firms")
savefig(p, joinpath(figdir, "week05_misallocation.pdf"))

# ---------------------------------------------------------------- 2. the value function near the threshold
keep = (z .>= 1.6) .& (z .<= 4.2)
p = base(xlabel = "productivity, z (log scale)", ylabel = "value", legend = :topleft, xscale = :log10,
         xticks = ([2, 3, 4], ["2", "3", "4"]), size = (900, 520))
vspan!(p, [z[findfirst(keep)], eq.z̃]; color = :gray90, label = "exit")
hline!(p, [0.0]; color = c_grid, linewidth = 1, label = "")
plot!(p, z[keep], eq.π[keep]; color = :gray45, linestyle = :dash, label = "profit today, π(z)")
plot!(p, z[keep], ind.β .* eq.sol.EV[keep]; color = c_low, label = "continuation, β E[V(z')|z]")
plot!(p, z[keep], eq.sol.V[keep]; color = c_high, label = "value, V(z)")
scatter!(p, z[keep], eq.sol.V[keep]; color = c_high, markersize = 3, markerstrokewidth = 0, label = "")
vline!(p, [eq.z̃]; color = c_dark, linewidth = 1, label = "")
annotate!(p, eq.z̃ * 1.02, 0.85 * maximum(eq.sol.V[keep]), text(@sprintf("z̃ = %.3f", eq.z̃), 12, :left, c_text))
savefig(p, joinpath(figdir, "week05_value_function.pdf"))
savefig(p, joinpath(tutdir, "week05_code_guide_value.png"))

# ---------------------------------------------------------------- 3. free entry
ps = collect(range(1.2, 1.8; length = 61))
Ve = [entry_value(pp, ind, eq.s).Ve for pp in ps]
p = base(xlabel = "price, p", ylabel = "value of entry", legend = :topleft, size = (820, 500), xlims = (1.2, 1.8))
plot!(p, [1.2, 1.8], [0.0, 0.0]; color = c_grid, linewidth = 1, label = "")
plot!(p, ps, Ve; color = c_high, label = "\$V^e(p) = \\beta \\sum_i g_i V(z_i; p) - c_e\$")
tried = filter(h -> 1.2 <= h[1] <= 1.8, eq.history)[1:min(6, end)]
scatter!(p, first.(tried), last.(tried); color = c_low, markersize = 5, markerstrokecolor = :white, label = "first bisection steps")
scatter!(p, [eq.p], [0.0]; color = c_dark, markersize = 6, label = @sprintf("p* = %.4f", eq.p))
savefig(p, joinpath(figdir, "week05_free_entry.pdf"))

# ---------------------------------------------------------------- 4. the size distribution
edges = [0, 10, 20, 50, 100, 500, Inf]
names = ["<10", "10-19", "20-49", "50-99", "100-499", "500+"]
sd = size_distribution(eq, edges)
x = 1:length(names)
p1 = base(xlabel = "firm size, production workers", ylabel = "share, %", legend = :topleft, xticks = (x, names),
          title = "by size class")
bar!(p1, x .- 0.2, 100sd.firms; bar_width = 0.4, color = c_low, linecolor = :white, label = "firms")
bar!(p1, x .+ 0.2, 100sd.employment; bar_width = 0.4, color = c_high, linecolor = :white, label = "employment")
# the share at every grid point, against size. On a log axis the grid is even.
stay = .!eq.sol.χ
p2 = base(xlabel = "firm size n(z), log scale", ylabel = "share per grid point, %", legend = :topleft,
          xscale = :log10, xticks = ([1, 10, 100, 1000], ["1", "10", "100", "1000"]), title = "at every grid point")
# the top grid point is left out: it holds Tauchen's whole upper tail, so its mass spikes
top = 1:length(z)-1
plot!(p2, eq.n[top], 100 .* eq.μ[top] ./ sum(eq.μ); color = c_low, label = "firms")
plot!(p2, eq.n[top], 100 .* eq.μ[top] .* eq.n[top] ./ sum(eq.μ .* eq.n); color = c_high, label = "employment")
vline!(p2, [eq.n[findfirst(stay)]]; color = c_dark, linewidth = 1, linestyle = :dash, label = "smallest firm that stays")
savefig(plot(p1, p2; layout = (1, 2), size = (1250, 500)), joinpath(figdir, "week05_size_distribution.pdf"))

# ---------------------------------------------------------------- 5. the statistics table
open(joinpath(figdir, "week05_stats.tex"), "w") do io
    println(io, "\\begin{tabular}{lr@{\\hspace{10mm}}lrr}\n\\toprule\n\\multicolumn{2}{l}{\\textbf{Equilibrium}} & \\multicolumn{3}{l}{\\textbf{Size classes (workers)}} \\\\")
    println(io, " & & & Firms & Employment \\\\\n\\midrule")
    left = [("Price \$p\$", @sprintf("%.4f", st.p)), ("Mass of entrants \$M\$", @sprintf("%.4f", st.M)),
            ("Number of firms", @sprintf("%.4f", st.firms)), ("Entry rate = exit rate", @sprintf("%.3f", st.entry_rate)),
            ("Exit threshold \$\\tilde z\$", @sprintf("%.3f", st.z̃)), ("Average size (workers)", @sprintf("%.1f", st.avg_size)),
            ("Mean age (years)", @sprintf("%.1f", st.mean_age)), ("Output \$Y = \\bar D / p\$", @sprintf("%.2f", st.Y)),
            ("Labour: production, fixed, entry", @sprintf("%.1f, %.1f, %.1f", st.N_prod, st.N_fixed, st.N_entry))]
    for (k, l) in enumerate(left)
        right = k <= length(names) ? @sprintf("%s & %.1f\\%% & %.1f\\%%", replace(names[k], "-" => "--"), 100sd.firms[k], 100sd.employment[k]) : " & & "
        println(io, l[1], " & ", l[2], " & ", right, " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# ---------------------------------------------------------------- 6. exit decisions on a grid
ces = collect(30.0:0.25:50.0)
rate(ind; kw...) = industry_stats(solve_equilibrium(ind; kw...)).entry_rate
r_grid = [rate(Industry(c_e = c)) for c in ces]
r_lot = [rate(Industry(c_e = c); exit = :lottery) for c in ces]
r_fine = [rate(Industry(c_e = c, n_z = 1001)) for c in ces]
p = base(xlabel = "entry cost, \$c_e\$", ylabel = "entry rate = exit rate", legend = :topright, size = (800, 470))
plot!(p, ces, r_fine; color = :gray60, linewidth = 4, alpha = 0.6, label = "exit on the grid, \$n_z = 1001\$")
plot!(p, ces, r_grid; color = c_low, label = "exit on the grid, \$n_z = 101\$")
plot!(p, ces, r_lot; color = c_high, label = "exit lottery, \$n_z = 101\$")
savefig(p, joinpath(figdir, "week05_exit_grid.pdf"))
savefig(p, joinpath(tutdir, "week05_code_guide_exit.png"))

open(joinpath(figdir, "week05_grid.tex"), "w") do io
    println(io, "\\begin{tabular}{rrrrr}\n\\toprule\n & & \\multicolumn{2}{c}{Entry rate = exit rate} & \\\\\n\\cmidrule(lr){3-4}")
    println(io, "\$n_z\$ & Price \$p\$ & Exit on the grid & Exit lottery & Threshold \$\\tilde z\$ \\\\\n\\midrule")
    for n_z in (51, 101, 201, 501, 1001, 2001)
        e1 = solve_equilibrium(Industry(; n_z))
        e2 = solve_equilibrium(Industry(; n_z); exit = :lottery)
        @printf(io, "%d & %.5f & %.4f & %.4f & %.4f \\\\\n", n_z, e1.p, industry_stats(e1).entry_rate,
                industry_stats(e2).entry_rate, e1.z̃)
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# ---------------------------------------------------------------- 7. comparative statics
# The fine grid with the lottery, so the lines show the economics, not the grid.
ks = collect(0.5:0.025:1.5)
sweep(f) = [industry_stats(solve_equilibrium(f(k); exit = :lottery)) for k in ks]
S_e = sweep(k -> Industry(c_e = 40k, n_z = 1001))
S_f = sweep(k -> Industry(c_f = 20k, n_z = 1001))
panel(key, ylabel; kw...) = begin
    p = base(; xlabel = "cost relative to baseline", ylabel, legend = false, kw...)
    plot!(p, ks, getfield.(S_e, key); color = c_low, label = "entry cost \$c_e\$")
    plot!(p, ks, getfield.(S_f, key); color = c_high, linestyle = :dash, label = "fixed cost \$c_f\$")
    i = findfirst(ks .== 1.0)
    scatter!(p, [1.0], [getfield(S_e[i], key)]; color = c_dark, markersize = 5, label = "baseline")
end
pp = plot(panel(:p, "price p"; legend = :topleft), panel(:z̃, "exit threshold z̃"),
          panel(:entry_rate, "entry rate = exit rate"), panel(:avg_size, "average size");
          layout = (2, 2), size = (1000, 680))
savefig(pp, joinpath(figdir, "week05_comparative_statics.pdf"))

cases = [Industry(), Industry(c_e = 60.0), Industry(c_f = 30.0)]
S = [industry_stats(solve_equilibrium(c)) for c in cases]
open(joinpath(figdir, "week05_comparative_table.tex"), "w") do io
    println(io, "\\begin{tabular}{lrrr}\n\\toprule\n & Baseline & \$c_e = 60\$ & \$c_f = 30\$ \\\\\n\\midrule")
    for (name, key, fmt) in [("Price \$p\$", :p, "%.4f"), ("Exit threshold \$\\tilde z\$", :z̃, "%.3f"),
                             ("Entry rate = exit rate", :entry_rate, "%.3f"), ("Mean age (years)", :mean_age, "%.1f"),
                             ("Number of firms", :firms, "%.3f"), ("Average size (workers)", :avg_size, "%.1f"),
                             ("Output \$Y\$", :Y, "%.2f"), ("Aggregate profits", :profits, "%.2f")]
        println(io, name, " & ", join((Printf.format(Printf.Format(fmt), getfield(s, key)) for s in S), " & "), " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# ---------------------------------------------------------------- 8. a cohort of entrants
ap = age_profile(eq, 20)
q1 = base(xlabel = "age", ylabel = "share of cohort", legend = false, ylims = (0, 1.05), title = "survival")
plot!(q1, ap.age, ap.survival; color = c_low, marker = :circle, markersize = 3, markerstrokewidth = 0)
q2 = base(xlabel = "age", ylabel = "workers per firm", legend = :bottomright, title = "average size of survivors")
plot!(q2, ap.age, ap.avg_size; color = c_high, marker = :circle, markersize = 3, markerstrokewidth = 0, label = "cohort")
plot!(q2, [0, 20], [st.avg_size, st.avg_size]; color = :gray50, linestyle = :dash, linewidth = 1.5, label = "all firms")
q3 = base(xlabel = "age", ylabel = "share of age group", legend = :topright, title = "exit rate")
plot!(q3, ap.age, ap.exit_rate; color = c_dark, marker = :circle, markersize = 3, markerstrokewidth = 0, label = "cohort")
plot!(q3, [0, 20], [st.exit_rate, st.exit_rate]; color = :gray50, linestyle = :dash, linewidth = 1.5, label = "all firms")
savefig(plot(q1, q2, q3; layout = (1, 3), size = (1500, 500), left_margin = 9Plots.mm, bottom_margin = 10Plots.mm), joinpath(figdir, "week05_age_profile.pdf"))

println("p = ", round(eq.p; digits = 6), ", M = ", round(eq.M; digits = 5), ", z̃ = ", round(eq.z̃; digits = 4))
println("cohort: survival at 1 = ", round(ap.survival[2]; digits = 3), ", size at 0 and 20 = ",
        round(ap.avg_size[1]; digits = 1), ", ", round(ap.avg_size[end]; digits = 1))
println("misallocation at sd 0.5: random ", round(loss_rand[51]; digits = 2), "%, correlated ", round(loss_corr[51]; digits = 2), "%")
println("firms staying with π < 0: ", count((eq.π .< 0) .& .!eq.sol.χ), " grid points, z from ",
        round(z[findfirst(.!eq.sol.χ)]; digits = 3), " to ", round(z[findlast((eq.π .< 0) .& .!eq.sol.χ)]; digits = 3))
println("figures written to ", figdir)
