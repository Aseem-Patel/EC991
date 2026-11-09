# EC991 Week 5: VFI, Howard and policy iteration on the incumbent's problem.
#   1. one solve at a fixed price, from the same cold start: iterations, time, distance to the exact V
#   2. the full equilibrium: time and the price, on the lab grid and a fine grid
# One BLAS thread, after a warm-up run, median of 5 runs. With EC991_FULL=1 the
# equilibrium table is also written to Week-5/Figures/week05_speed.tex for the lecture.
# Run from the package folder:  julia --project=. run/compare_methods.jl
using EC991Week5, Printf, LinearAlgebra

BLAS.set_num_threads(1)
methods = [(:vfi, "VFI"), (:howard, "VFI + Howard (n_H = 20)"), (:policy_iteration, "policy iteration")]
median_time(f; n = 5) = (f(); sort([@elapsed(f()) for _ in 1:n])[cld(n, 2)])

# ---------------------------------------------------------------- 1. one solve at p = 1.4862
println("One solve of the incumbent's problem at p = 1.4862, from V = π")
println("  n_z   method                     iterations   time (ms)   max |V - V_exact| / max |V|")
for n_z in (101, 1001)
    ind = Industry(; n_z)
    s = setup(ind)
    d = static_decisions(1.4862, s.z; ind.η, ind.c_f)
    exact = solve_firm(d.π, s.F, ind.β; method = :policy_iteration).V
    for (m, name) in methods
        sol = solve_firm(d.π, s.F, ind.β; method = m)
        t = median_time(() -> solve_firm(d.π, s.F, ind.β; method = m))
        @printf("  %4d  %-25s  %10d  %10.3f   %.1e\n", n_z, name, sol.iterations, 1000t,
                maximum(abs.(sol.V .- exact)) / maximum(abs.(exact)))
    end
end

# ---------------------------------------------------------------- 2. the equilibrium
println("\nThe equilibrium, by bisection on p")
println("  n_z   method                     time (s)   price p")
rows = String[]
for n_z in (101, 1001), (m, name) in methods
    ind = Industry(; n_z)
    t = median_time(() -> solve_equilibrium(ind; method = m); n = 3)
    p = solve_equilibrium(ind; method = m).p
    @printf("  %4d  %-25s  %8.3f   %.6f\n", n_z, name, t, p)
    push!(rows, @sprintf("%s & %d & %.3f & %.5f \\\\", replace(name, "n_H = 20" => "\$n_H = 20\$"), n_z, t, p))
end

if get(ENV, "EC991_FULL", "0") == "1"
    figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
    open(joinpath(figdir, "week05_speed.tex"), "w") do io
        println(io, "\\begin{tabular}{lrrr}\n\\toprule\nMethod & \$n_z\$ & Time (s) & Price \$p\$ \\\\\n\\midrule")
        foreach(r -> println(io, r), rows[1:3]); println(io, "\\addlinespace")
        foreach(r -> println(io, r), rows[4:6]); println(io, "\\bottomrule\n\\end{tabular}")
    end
    println("table written to ", figdir)
end
