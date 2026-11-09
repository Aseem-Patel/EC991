# EC991 Week 5: the static economy (lecture slides 7 to 22) in numbers.
#   1. two firms, z = (1, 2), η = 2/3, N = 9: the efficient split, Y = Z N^η, and an equal split
#   2. misallocation: Ω when wedges are random, and when they tax productive firms
#   3. the planner's choice of the number of firms
# Run from the package folder:  julia --project=. run/static_examples.jl
using EC991Week5, Printf

# ---------------------------------------------------------------- 1. two firms
z, η, N = [1.0, 2.0], 2 / 3, 9.0
e = efficient_allocation(z, N; η)
Z = aggregate_tfp(z; η)
println("Two firms, z = $z, η = 2/3, N = 9")
@printf("  efficient: n = (%.2f, %.2f), y = (%.2f, %.2f), Y = %.3f\n", e.n..., e.y..., e.Y)
@printf("  Z = (1^3 + 2^3)^(1/3) = %.4f, and Z N^η = %.3f\n", Z, Z * N^η)
τ = z[1] ./ z .- 1                                   # (1 + τ) z equal across firms: equal split
w = allocation_with_wedges(z, τ, N; η)
@printf("  equal split: n = (%.2f, %.2f), Y = %.3f, Ω = %.4f, output lost %.1f%%\n",
        w.n..., w.Y, misallocation(z, τ; η), 100 * (1 - w.Y / e.Y))

# ---------------------------------------------------------------- 2. misallocation
# The productivity distribution of the dynamic model: its grid and invariant weights.
s = setup(Industry())
f = s.g
ζ = (s.logz .- sum(f .* s.logz)) ./ sqrt(sum(f .* (s.logz .- sum(f .* s.logz)) .^ 2))   # standardised log z
println("\nMisallocation, η = 2/3, z from the model's invariant distribution")
println("  sd of log(1 + τ)   random ±sd   taxing productive firms")
for sd in (0.0, 0.1, 0.2, 0.3, 0.5)
    # random: half the firms at each z get +sd, half get -sd (in logs)
    zz, mm = vcat(s.z, s.z), vcat(f, f) ./ 2
    Ω_rand = misallocation(zz, exp.(vcat(fill(sd, length(f)), fill(-sd, length(f)))) .- 1; η, m = mm)
    Ω_corr = misallocation(s.z, exp.(-sd .* ζ) .- 1; η, m = f)
    @printf("  %18.1f   %10.4f   %22.4f\n", sd, Ω_rand, Ω_corr)
end

# ---------------------------------------------------------------- 3. entry
println("\nThe planner's choice of firms, η = 2/3, N = 100")
for c_e in (0.5, 1.0, 2.0)
    pe = planner_entry(100.0, c_e; η)
    @printf("  c_e = %.1f: M = %.2f firms, %.1f production workers, Y/N = %.3f\n", c_e, pe.M, pe.N_p, pe.Y / 100)
end
