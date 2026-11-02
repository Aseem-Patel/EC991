# The full equilibrium by every method, from the same bracket: the race (lecture slides 28 to 30).
# Only Step 3 of the loop differs. Default: the lab grid. With EC991_FULL=1 also the dense grid,
# which takes some minutes for the two methods without speed-ups.
# Times are one thread, after a warm-up on a small grid, one run each; they depend on your machine.
# Run from the package folder:  julia --project=. run/compare_equilibrium.jl
# or in the REPL:               include("run/compare_equilibrium.jl")
using EC991Week4, LinearAlgebra, Printf
BLAS.set_num_threads(1)

methods = [(:vfi_grid, "VFI, grid search"), (:vfi_fast, "VFI, grid search + speed-ups"),
           (:vfi_interp, "VFI, interpolation"), (:vfi_interp_howard, "VFI, interpolation + Howard"), (:egm, "EGM")]

small = Economy(; n_a = 60, ν = 0.1)
foreach(m -> solve_equilibrium(small; method = m[1], tol_r = 1e-3), methods)   # compile first

full = get(ENV, "EC991_FULL", "0") == "1"
economies = [("lab grid: n_a = 300", Economy())]
full && push!(economies, ("dense grid: n_a = 1000, distribution on 5000 points", dense_economy()))

rows = String[]
for (label, econ) in economies
    println("\n", label)
    @printf "%-30s %9s %6s %10s %9s %9s %8s %8s\n" "method" "time (s)" "steps" "r*" "K" "Φ" "mean ε" "Gini"
    for (method, name) in methods
        t = @elapsed (eq = solve_equilibrium(econ; method))
        e = euler_errors(eq.sol, eq.hh, eq.s.P)
        m = model_stats(eq)
        @printf "%-30s %9.2f %6d %10.6f %9.4f %9.5f %8.2f %8.3f\n" name t length(eq.history) eq.r eq.K eq.Φ e.mean_log10 m.gini
        push!(rows, @sprintf("%s & %d & %.2f & %d & %.4f & %.3f & %.2f \\\\", name, econ.n_a, t, length(eq.history), eq.r, eq.K, e.mean_log10))
    end
end

# With EC991_FULL=1 the table is also written to the Figures folder.
if full
    figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
    mkpath(figdir)
    n = length(methods)
    open(joinpath(figdir, "week04_timing_equilibrium.tex"), "w") do io
        println(io, "\\begin{tabular}{lrrrrrr}\n\\toprule\nMethod & \$n_a\$ & Time (s) & Steps & \$r^*\$ & \$K\$ & Mean \$\\log_{10}|\\varepsilon|\$ \\\\\n\\midrule")
        foreach(r -> println(io, r), rows[1:n]); println(io, "\\midrule")
        foreach(r -> println(io, r), rows[n+1:end]); println(io, "\\bottomrule\n\\end{tabular}")
    end
end
