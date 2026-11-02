# One household solve at a fixed interest rate, by all five methods: the speed-accuracy table.
# Grid search, grid search with the speed-ups, interpolate-then-optimise, the same with Howard,
# and EGM (lecture slides 5 to 19). Accuracy is measured two ways: Euler errors between grid
# points, and the gap to a reference policy from EGM on a much finer grid, both for assets up to 50.
# Default: the lab grid, n_a = 300. With EC991_FULL=1 also the dense grid, n_a = 1000.
# Times are one thread, after a warm-up, the median of three runs; they depend on your machine.
# Run from the package folder:  julia --project=. run/compare_household.jl
# or in the REPL:               include("run/compare_household.jl")
using EC991Week4, LinearAlgebra, Printf
BLAS.set_num_threads(1)

hh = Household(r = 0.0377)                      # close to the equilibrium rate
logy, P = rouwenhorst(7, 0.9, 0.1)
y = exp.(logy)
n_H = 50

methods = [
    ("VFI, grid search",                (h, a) -> solve_vfi(h, a, y, P)),
    ("VFI, grid search + speed-ups",    (h, a) -> solve_vfi(h, a, y, P; step = bellman_step_fast!, n_H)),
    ("VFI, interpolation",              (h, a) -> solve_vfi(h, a, y, P; step = bellman_step_interp!)),
    ("VFI, interpolation + Howard",     (h, a) -> solve_vfi(h, a, y, P; step = bellman_step_interp!, n_H, howard_from = 0.01)),
    ("EGM",                             (h, a) -> solve_egm(h, a, y, P)),
]

reference = solve_egm(hh, asset_grid(5000, 250.0; ν = 0.002), y, P; tol = 1e-10)
small = asset_grid(30, 250.0; ν = 0.2)
foreach(m -> m[2](hh, small), methods)          # compile everything before the clock starts

median3(x) = sort(x)[2]
grids = [("lab grid, n_a = 300, ν = 0.025", asset_grid(300, 250.0; ν = 0.025))]
get(ENV, "EC991_FULL", "0") == "1" && push!(grids, ("dense grid, n_a = 1000, ν = 0.008", asset_grid(1000, 250.0; ν = 0.008)))

@printf "r = %.4f, w = %.4f, n_y = %d, Howard n_H = %d\n" hh.r hh.w length(y) n_H
rows = String[]
for (label, a) in grids
    println("\n", label)
    @printf "%-30s %10s %7s %12s %12s %12s\n" "method" "time (s)" "iters" "max log10|ε|" "mean log10|ε|" "policy gap"
    for (name, f) in methods
        t = median3([@elapsed(f(hh, a)) for _ in 1:3])
        sol = f(hh, a)
        e = euler_errors(sol, hh, P)
        gap = compare_policies(sol, reference; a_hi = 50.0).gap
        @printf "%-30s %10.3f %7d %12.2f %12.2f %12.5f\n" name t sol.iterations e.max_log10 e.mean_log10 gap
        push!(rows, @sprintf("%s & %s & %.3f & %d & %.2f & %.5f \\\\", name, length(a), t, sol.iterations, e.mean_log10, gap))
    end
end

# With EC991_FULL=1 the table is also written to the Figures folder.
if length(grids) == 2
    figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
    mkpath(figdir)
    n = length(methods)
    open(joinpath(figdir, "week04_timing_household.tex"), "w") do io
        println(io, "\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & \$n_a\$ & Time (s) & Iterations & Mean \$\\log_{10}|\\varepsilon|\$ & Policy gap \\\\\n\\midrule")
        foreach(r -> println(io, r), rows[1:n]); println(io, "\\midrule")
        foreach(r -> println(io, r), rows[n+1:end]); println(io, "\\bottomrule\n\\end{tabular}")
    end
end
