# EC991 Week 6: the stationary equilibrium of Hopenhayn-Rogerson, step by step.
# Prints V^e at every price the bisection tries, then the model's statistics in the
# order of the table in the code guide.
# Run from the package folder:
#   julia --project=. run/solve_equilibrium.jl          # τ = 0.1
#   julia --project=. run/solve_equilibrium.jl 0        # any other firing tax
using EC991Week6, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead

τ = isempty(ARGS) ? 0.1 : parse(Float64, ARGS[1])
eco = Economy(τ = τ)
println("Hopenhayn-Rogerson: τ = $(eco.τ), η = $(round(eco.η; digits = 3)), β = $(eco.β), c_e = $(eco.c_e), ",
        "c_f = $(eco.c_f), D̄ = $(eco.D̄), ρ = $(eco.ρ), σ = $(eco.σ), grid $(eco.n_z) × $(eco.n_n)")
println("   step   price p     V^e(p)")
# the bisection calls this at every price it tries: print the step, p and V^e(p)
step = Ref(0)
eq = solve_equilibrium(eco; callback = (p, Ve) -> @printf("   %4d   %.8f  %+.3e\n", step[] += 1, p, Ve))

a = aggregates(eq)
m = misallocation(eq)
println("\nModel statistics")
@printf("  price p                     %.4f\n", a.p)
@printf("  average firm size           %.2f\n", a.avg_size)
@printf("  exit rate = entry rate      %.4f = %.4f\n", a.exit_rate, a.entry_rate)
@printf("  average productivity        %.4f\n", a.avg_z)
@printf("  average misallocation (%%)   %.3f   (employment-weighted %.3f)\n", m.mpl_gap, m.mpl_gap_emp)
@printf("  output lost to it (%%)       %.3f   (Ω = %.5f)\n", m.output_lost, m.Ω)
@printf("  firms in the inaction region %.3f\n", m.share_inactive)
@printf("  output Y = demand C         %.4f = %.4f\n", a.Y, a.C)
@printf("  labour supply = demand      %.4f = %.4f\n", a.Ns, a.Nd)
@printf("  firing tax revenue T        %.4f\n", a.T)
@printf("  profits Π                   %.4f\n", a.Π)
@printf("  mass of firms B             %.4f\n", a.B)
@printf("  mass of entrants M          %.4f\n", a.M)
@printf("  job creation = destruction  %.4f = %.4f  (share of employment)\n", a.job_creation, a.job_destruction)

println("\nCumulative shares by size (workers n)")
sd = size_distribution(eq, [10, 20, 50, 100, 1000])
@printf("  size          %s\n", join([@sprintf("%6d", c) for c in sd.cutoffs], ""))
@printf("  firms         %s\n", join([@sprintf("%6.3f", x) for x in sd.firms], ""))
@printf("  employment    %s\n", join([@sprintf("%6.3f", x) for x in sd.employment], ""))
