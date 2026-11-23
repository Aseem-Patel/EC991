# EC991 Week 7: plotting helpers shared by the run scripts. No model logic here:
# everything plotted comes from the package.
ENV["GKSwstype"] = "100"          # draw without opening a window
using Plots

default(fontfamily = "Computer Modern", linewidth = 2.5, framestyle = :box, grid = false)
const c_rec, c_boom, c_grid, c_text = colorant"#2a78d6", colorant"#eb6834", "#d9d8d4", "#52514e"
const c_dark = colorant"#3d3c3a"

base(; kwargs...) = plot(; framestyle = :grid, grid = :y, gridlinewidth = 0.8, foreground_color_grid = c_grid,
                         gridalpha = 1.0, foreground_color_text = c_text, foreground_color_axis = c_grid,
                         foreground_color_border = c_grid, tick_direction = :none,
                         guidefontsize = 14, tickfontsize = 11, legendfontsize = 11, titlefontsize = 14,
                         margin = 3Plots.mm, left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, kwargs...)

# ---------------------------------------------------------------- Part A: Krusell-Smith

"""
    plot_aggregate_rule(eq)

Lecture slide 24: tomorrow's against today's aggregate capital. The two lines are
the law of motion for each Z (thin dark lines), over the range of K in the
simulation; the dots are the simulated pairs (K_t, K_{t+1}), coloured by Z_t; the
dashed line is the 45° line.
"""
function plot_aggregate_rule(eq)
    T0 = eq.eco.T_burn + 1
    K, z = eq.K[T0:end], eq.zpath[T0:end]
    lo, hi = extrema(K)
    grid = range(lo, hi; length = 100)
    p = base(; xlabel = "today's aggregate capital, K", ylabel = "tomorrow's aggregate capital, K'",
             legend = :topleft, xlims = (lo, hi), ylims = (lo, hi), size = (700, 600))
    plot!(p, [lo, hi], [lo, hi]; color = c_grid, linestyle = :dash, linewidth = 1.5, label = "45° line")
    for (zz, col) in ((2, c_boom), (1, c_rec))
        ts = findall(==(zz), z[1:end-1])
        scatter!(p, K[ts], K[ts .+ 1]; color = col, markersize = 4, markerstrokewidth = 0, alpha = 0.35,
                 label = "simulated, $(zz == 2 ? "boom" : "recession")")
        plot!(p, grid, forecast_K.(grid, zz, Ref(eq.θ)); color = c_dark, linewidth = 1,
              label = zz == 2 ? "law of motion" : "")
    end
    return p
end

"""
    plot_individual_rules(eq)

Lecture slide 25: tomorrow's against today's individual assets, for the employed
and the unemployed, in a boom (solid) and a recession (dashed), at the mean of
aggregate capital. The right panel zooms in near the borrowing limit, where the
rules bend (slide 26).
"""
function plot_individual_rules(eq)
    K̄ = sum(eq.K[eq.eco.T_burn+1:end]) / (length(eq.K) - eq.eco.T_burn)
    a = eq.s.a
    panels = []
    for (amax, title) in ((2.2K̄, "all households"), (0.15K̄, "near the borrowing limit"))
        keep = a .<= amax
        p = base(; xlabel = "today's assets, a", ylabel = "tomorrow's assets, a'", legend = :topleft, title,
                 xlims = (0, amax), ylims = (0, amax))
        plot!(p, [0, amax], [0, amax]; color = c_grid, linestyle = :dash, linewidth = 1.5, label = "45° line")
        for (j, col, who) in ((2, c_dark, "employed"), (1, c_rec, "unemployed"))
            plot!(p, a[keep], individual_rule(eq.sol, eq.s, j, 2, K̄)[keep]; color = col, label = "$who, boom")
            plot!(p, a[keep], individual_rule(eq.sol, eq.s, j, 1, K̄)[keep]; color = col, linestyle = :dash,
                  label = "$who, recession")
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (1, 2), size = (1250, 540))
end

"""
    plot_ks_simulation(eq; quarters = 400)

The first `quarters` after the burn-in: recessions shaded; aggregate capital, with
the path the rule alone predicts from the first date (dashed); and output.
"""
function plot_ks_simulation(eq; quarters = 400)
    x = aggregate_series(eq)
    fe = forecast_errors(eq)
    T0 = eq.eco.T_burn + 1
    t = 1:quarters
    rec = x.Z[t] .< 1
    # the recession spells, as [start, end, start, end, ...] for vspan!
    edges = Float64[]
    for i in t
        rec[i] && (i == 1 || !rec[i-1]) && push!(edges, i - 0.5)
        rec[i] && (i == quarters || !rec[i+1]) && push!(edges, i + 0.5)
    end
    shade!(p) = vspan!(p, edges; color = c_rec, alpha = 0.12, linealpha = 0, label = "recession")
    Kt, K̂t = x.K[t], fe.K̂[T0 .+ t .- 1]
    p1 = base(; xlabel = "quarter", ylabel = "aggregate capital", legend = :bottomright, title = "capital")
    shade!(p1)
    plot!(p1, t, Kt; color = c_dark, label = "simulated K")
    plot!(p1, t, K̂t; color = c_boom, linestyle = :dash, label = "the rule alone")
    Yt = x.Y[t]
    p2 = base(; xlabel = "quarter", ylabel = "output", legend = :bottomright, title = "output")
    shade!(p2)
    plot!(p2, t, Yt; color = c_dark, label = "Y")
    return plot(p1, p2; layout = (1, 2), size = (1250, 480))
end

# ---------------------------------------------------------------- Part B: sequence space

# a path with the steady state put in front, at t = 0
withss(x, xss) = [xss; x]

"""
    plot_irf(tr)

Lecture slide 35: the responses to the MIT shock, from t = 0 (the steady state):
Z, K, w, Y and C in % deviations, r in percentage points.
"""
function plot_irf(tr; tmax = 150)
    ss = tr.ss0
    t = 0:length(tr.K)
    pct(x, xss) = 100 .* (withss(x, xss) ./ xss .- 1)
    C̄ = sum(ss.c .* ss.λ)
    Ȳ = ss.K^ss.eco.α * ss.s.N^(1 - ss.eco.α)
    series = (("productivity Z", pct(tr.Z, ss.eco.Z), "%"), ("capital K", pct(tr.K, ss.K), "%"),
              ("interest rate r", 100 .* (withss(tr.r, ss.r) .- ss.r), "p.p."),
              ("wage w", pct(tr.w, ss.w), "%"), ("output Y", pct(tr.Y, Ȳ), "%"), ("consumption C", pct(tr.C, C̄), "%"))
    ps = [plot!(base(; title, ylabel = unit, xlabel = "t", legend = false, xlims = (0, tmax)),
                t, x; color = c_rec) for (title, x, unit) in series]
    for p in ps
        hline!(p, [0]; color = c_grid, linewidth = 1, label = "")
    end
    return plot(ps...; layout = (2, 3), size = (1300, 700))
end

"""
    plot_inequality(tr)

Lecture slide 36: the Gini of wealth for everyone, for the lowest and for the
highest endowment, and the share of households at the borrowing limit, as
changes from the steady state.
"""
function plot_inequality(tr; tmax = 150)
    ss = tr.ss0
    t = 0:length(tr.K)
    series = (("Gini of wealth, all", tr.gini, "change"), ("Gini of wealth, lowest y", tr.gini_low, "change"),
              ("Gini of wealth, highest y", tr.gini_high, "change"),
              ("share at the borrowing limit", 100 .* tr.constrained, "p.p. change"))
    ps = [plot!(base(; title, ylabel = unit, xlabel = "t", legend = false, xlims = (0, tmax)),
                t, withss(x, x[end]) .- x[end]; color = c_rec) for (title, x, unit) in series]
    for p in ps
        hline!(p, [0]; color = c_grid, linewidth = 1, label = "")
    end
    return plot(ps...; layout = (2, 2), size = (1100, 700))
end

"""
    plot_tax(tr)

Lecture slide 38: the labour tax, then K and w in % from the initial steady
state and r in percentage points, with the final steady state dashed.
"""
function plot_tax(tr; tmax = 150)
    ss0, ss1 = tr.ss0, tr.ss1
    t = 0:length(tr.K)
    pct(x, x0) = 100 .* (x ./ x0 .- 1)
    series = (("labour tax τ", withss(tr.τ, ss0.eco.τ), ss1.eco.τ, ""),
              ("capital K", pct(withss(tr.K, ss0.K), ss0.K), pct(ss1.K, ss0.K), "% from initial ss"),
              ("interest rate r", 100 .* (withss(tr.r, ss0.r) .- ss0.r), 100 * (ss1.r - ss0.r), "p.p. from initial ss"),
              ("wage w", pct(withss(tr.w, ss0.w), ss0.w), pct(ss1.w, ss0.w), "% from initial ss"))
    ps = []
    for (title, x, final, unit) in series
        p = base(; title, ylabel = unit, xlabel = "t", legend = false, xlims = (0, tmax))
        hline!(p, [final]; color = c_grid, linestyle = :dash, linewidth = 1.5)
        plot!(p, t, x; color = c_rec, linetype = title == "labour tax τ" ? :steppost : :path)
        push!(ps, p)
    end
    return plot(ps...; layout = (2, 2), size = (1100, 700))
end

"""
    plot_bkm(K, Y)

Lecture slide 44: capital and output simulated by adding up impulse responses,
in % deviations from the steady state.
"""
function plot_bkm(K, Y)
    t = 1:length(K)
    p = base(; xlabel = "year", ylabel = "% deviation from steady state", legend = :topright,
             title = "simulated by Boppart, Krusell and Mitman", size = (1100, 480))
    hline!(p, [0]; color = c_grid, linewidth = 1, label = "")
    plot!(p, t, Y; color = c_boom, linewidth = 1.8, label = "output Y")
    plot!(p, t, K; color = c_rec, label = "capital K")
    return p
end
