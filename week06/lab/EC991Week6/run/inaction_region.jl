# EC991 Week 6: the inaction region, at the baseline firing tax and at ten times it
# (tutorial slides 13 and 14). Solves both equilibria, prints the hiring and firing
# targets at a few productivities, and writes the two graphs to this folder:
#   inaction_region.png            τ = 0.1
#   inaction_region_high_tax.png   τ = 1.0
# Run from the package folder:
#   julia --project=. run/inaction_region.jl
#   julia --project=. run/inaction_region.jl 0.5     # any other tax for the second graph
using EC991Week6, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

τ_high = isempty(ARGS) ? 1.0 : parse(Float64, ARGS[1])
for (τ, file, title) in ((0.1, "inaction_region.png", "Inaction region, τ = 0.1"),
                         (τ_high, "inaction_region_high_tax.png", "Inaction region, τ = $τ_high"))
    eq = solve_equilibrium(Economy(τ = τ))
    r = inaction_region(eq.sol, eq.s)
    println("τ = $τ: equilibrium price p = ", @sprintf("%.4f", eq.p))
    println("      z     n_L(z)    n_H(z)   n_H / n_L")
    for zz in (2.0, 4.0, 6.0, 8.0, 10.0)
        i = argmin(abs.(r.z .- zz))                      # the grid point nearest zz
        @printf("  %5.2f  %9.1f %9.1f   %7.3f\n", r.z[i], r.nL[i], r.nH[i], r.nH[i] / max(r.nL[i], eps()))
    end
    savefig(plot_inaction(eq; title), file)
    println("  wrote $file\n")
end
