# EC991 Week 6: the tutorial's model output, from the package's own functions, into
# Week-6/Figures/ (prefix week06_hr_, next to the market power lecture's files). As in
# the lecture, the deck draws the graphs itself with pgfplots, in its own font (Fira
# Sans); this script writes their data:
#   1. week06_hr_inaction.dat   z, n_L(z), n_H(z) at τ = 0.1 and τ = 1   (tutorial slides 13, 14)
#   2. week06_hr_policy.dat     chosen n against inherited m at one z, τ = 0, 0.1, 1  (tutorial slide 9)
#      week06_hr_macros.tex     that z and the targets n_L, n_H, as macros for the slide
#   3. week06_hr_taxes.dat      τ from 0 to 1: misallocation and aggregates   (tutorial slide 17)
#   4. week06_hr_grid.tex       the MPL gap at τ = 0 and 0.1 as the n grid is refined
#      week06_hr_numbers.tex    numbers quoted on the slides, as comments
# and, for the code guide, PNGs in Week-6/Tutorial/ (drawn with Plots, run/plotting.jl):
#   week06_code_guide_inaction.png, week06_code_guide_taxes.png
#
# Run from the package folder (20 equilibria, two of them on a 1000-point grid):
#   julia --project=. run/make_figures.jl
using EC991Week6, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
tutdir = normpath(joinpath(@__DIR__, "..", ".."))
mkpath(figdir)

eqs = Dict(τ => solve_equilibrium(Economy(τ = τ)) for τ in (0.0, 0.1, 1.0))
eq = eqs[0.1]

# ---------------------------------------------------------------- 1. the inaction region
r1, r2 = inaction_region(eqs[0.1].sol, eqs[0.1].s), inaction_region(eqs[1.0].sol, eqs[1.0].s)
open(joinpath(figdir, "week06_hr_inaction.dat"), "w") do io
    println(io, "z nL1 nH1 nL2 nH2")                  # 1: τ = 0.1, 2: τ = 1
    for i in eachindex(r1.z)
        r1.z[i] <= 12.5 && @printf(io, "%.4f %.3f %.3f %.3f %.3f\n", r1.z[i], r1.nL[i], r1.nH[i], r2.nL[i], r2.nH[i])
    end
end
p_base = plot_inaction(eqs[0.1])
p_high = plot_inaction(eqs[1.0])
savefig(plot(plot!(deepcopy(p_base); title = "τ = 0.1"), plot!(deepcopy(p_high); title = "τ = 1.0");
             layout = (1, 2), size = (1250, 500)), joinpath(tutdir, "week06_code_guide_inaction.png"))

# ---------------------------------------------------------------- 2. the policy at one z
# n = n^d(z_i, m) on every grid point m from 100 to 700, at the z_i nearest 6
i = argmin(abs.(eq.s.z .- 6.0))
keep = (eq.s.n .>= 100) .& (eq.s.n .<= 700)
open(joinpath(figdir, "week06_hr_policy.dat"), "w") do io
    println(io, "nprev n0 n1 n2")                     # 0: τ = 0, 1: τ = 0.1, 2: τ = 1
    for k in findall(keep)
        @printf(io, "%.3f %.3f %.3f %.3f\n", eq.s.n[k], (eqs[τ].s.n[eqs[τ].sol.pol[i, k]] for τ in (0.0, 0.1, 1.0))...)
    end
end
open(joinpath(figdir, "week06_hr_macros.tex"), "w") do io
    println(io, "% From EC991Week6/run/make_figures.jl: the policy at one z (tutorial slide 9).")
    r = Dict(τ => inaction_region(eqs[τ].sol, eqs[τ].s) for τ in (0.0, 0.1, 1.0))
    @printf(io, "\\def\\hrz{%.2f}\n", eq.s.z[i])
    @printf(io, "\\def\\hrNzero{%.0f}\n", r[0.0].nL[i])
    @printf(io, "\\def\\hrNLa{%.0f}\\def\\hrNHa{%.0f}\n", r[0.1].nL[i], r[0.1].nH[i])
    @printf(io, "\\def\\hrNLb{%.0f}\\def\\hrNHb{%.0f}\n", r[1.0].nL[i], r[1.0].nH[i])
end

# ---------------------------------------------------------------- 3. τ from 0 to 1
τs = collect(0.0:0.1:1.0)
sweep = [(aggregates(e), misallocation(e)) for e in (solve_equilibrium(Economy(τ = τ)) for τ in τs)]
a0 = sweep[1][1]
p1 = base(; xlabel = "firing tax, τ", ylabel = "%", legend = :topleft, title = "misallocation")
plot!(p1, τs, [m.mpl_gap for (a, m) in sweep]; color = c_low, label = "average MPL gap")
plot!(p1, τs, [m.output_lost for (a, m) in sweep]; color = c_high, label = "output lost, 1 - Ω")
p2 = base(; xlabel = "firing tax, τ", ylabel = "% change from τ = 0", legend = :bottomleft, title = "aggregates")
plot!(p2, τs, [100 * (a.Y / a0.Y - 1) for (a, m) in sweep]; color = c_high, label = "output Y")
plot!(p2, τs, [100 * (a.Y / a.Ns / (a0.Y / a0.Ns) - 1) for (a, m) in sweep]; color = c_low, label = "output per worker Y / N")
plot!(p2, τs, [100 * (a.job_creation / a0.job_creation - 1) for (a, m) in sweep]; color = c_dark,
      linestyle = :dash, label = "job reallocation")
pt = plot(p1, p2; layout = (1, 2), size = (1250, 520), legendfontsize = 14, tickfontsize = 14, guidefontsize = 17)
savefig(pt, joinpath(tutdir, "week06_code_guide_taxes.png"))
open(joinpath(figdir, "week06_hr_taxes.dat"), "w") do io
    println(io, "tau gap lost dY dYN dJR")
    for (τ, (a, m)) in zip(τs, sweep)
        @printf(io, "%.2f %.4f %.4f %.4f %.4f %.4f\n", τ, m.mpl_gap, m.output_lost, 100 * (a.Y / a0.Y - 1),
                100 * (a.Y / a.Ns / (a0.Y / a0.Ns) - 1), 100 * (a.job_creation / a0.job_creation - 1))
    end
end

# ---------------------------------------------------------------- 4. refining the n grid
grid = [(n_n, misallocation(solve_equilibrium(Economy(τ = 0.0, n_n = n_n))).mpl_gap,
         misallocation(solve_equilibrium(Economy(τ = 0.1, n_n = n_n))).mpl_gap) for n_n in (250, 500, 1000)]
open(joinpath(figdir, "week06_hr_grid.tex"), "w") do io
    println(io, "\\begin{tabular}{rrrr}\n\\toprule\nGrid points \$n_n\$ & Step in \$n\$ & MPL gap, \$\\tau = 0\$ & MPL gap, \$\\tau = 0.1\$ \\\\\n\\midrule")
    for (n_n, g0, g1) in grid
        e = Economy(n_n = n_n)
        step = (e.n_max / e.n_min)^(1 / (n_n - 2)) - 1
        println(io, n_n, " & ", @sprintf("%.1f\\%%", 100step), " & ", @sprintf("%.3f\\%%", g0), " & ", @sprintf("%.3f\\%%", g1), " \\\\")
    end
    println(io, "\\bottomrule\n\\end{tabular}")
end

# ---------------------------------------------------------------- numbers quoted on the slides
open(joinpath(figdir, "week06_hr_numbers.tex"), "w") do io
    println(io, "% From EC991Week6/run/make_figures.jl. Numbers quoted in week06_hopenhayn_rogerson.tex.")
    for τ in (0.0, 0.1, 1.0)
        e = eqs[τ]; r = inaction_region(e.sol, e.s)
        j = argmin(abs.(e.s.z .- 6.0))
        @printf(io, "%% τ = %.1f: p = %.4f, VFI iterations at the last price %d, bisection steps %d; at z = %.2f: n_L = %.1f, n_H = %.1f\n",
                τ, e.p, e.sol.iterations, length(e.history), e.s.z[j], r.nL[j], r.nH[j])
    end
    # at τ = 1, where the exit decision depends on n: z_i at which small firms exit but large ones stay
    e = eqs[1.0]
    mixed = [k for k in eachindex(e.s.z) if e.sol.χ[k, 1] && !e.sol.χ[k, end]]
    @printf(io, "%% τ = 1: small firms exit but large firms stay at z from %.2f to %.2f\n", e.s.z[first(mixed)], e.s.z[last(mixed)])
end
println("wrote figures to $figdir and code guide PNGs to $tutdir")
