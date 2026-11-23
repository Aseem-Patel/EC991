# EC991 Week 7, Part B: from one shock to many (lecture slides 42 to 45). Solves the
# MIT shock for three sizes of innovation and compares their impulse responses per
# unit of shock (are they linear?), then simulates 200 years of capital and output by
# adding up impulse responses (Boppart, Krusell and Mitman, 2018). Writes
# bkm_simulation.png to this folder.
# Run from the package folder:
#   julia --project=. run/bkm_simulation.jl
using EC991Week7, Printf, LinearAlgebra, Random
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

ss = solve_steady_state(Economy())
Ȳ = ss.K^ss.eco.α * ss.s.N^(1 - ss.eco.α)

# 1 to 3: the impulse response per unit of innovation, for three innovations
h = Dict{Float64, NamedTuple}()
for ε in (0.01, -0.01, 0.05)
    tr = mit_shock(ss; ε)
    h[ε] = (K = impulse_response(tr.K, ss.K, ε), Y = impulse_response(tr.Y, Ȳ, ε))
    @printf("ε = %+.2f: solved in %d iterations\n", ε, tr.iterations)
end
println("\nIs the response linear? Capital's response per unit of shock, relative to the steady state:")
println("  horizon s   ε = +0.01   ε = -0.01   ε = +0.05")
for s in (1, 5, 10, 20, 50)
    @printf("  %6d     %9.4f   %9.4f   %9.4f\n", s, (h[ε].K[s+1] / ss.K for ε in (0.01, -0.01, 0.05))...)
end
gap(x, y) = maximum(abs.(x .- y)) / maximum(abs.(x))
@printf("largest gap from ε = +0.01, as a share of the peak response: %.2f%% (ε = -0.01), %.2f%% (ε = +0.05)\n",
        100 * gap(h[0.01].K, h[-0.01].K), 100 * gap(h[0.01].K, h[0.05].K))

# 4: draw innovations and add up their effects
σ_ε, years = 0.01, 200
ε = σ_ε .* randn(Random.Xoshiro(7), years)
K = 100 .* bkm_simulate(h[0.01].K, ε) ./ ss.K
Y = 100 .* bkm_simulate(h[0.01].Y, ε) ./ Ȳ
sd(x) = sqrt(sum((x .- sum(x) / length(x)) .^ 2) / (length(x) - 1))
@printf("\n%d years with σ_ε = %.2f: s.d. of K %.2f%%, of Y %.2f%%\n", years, σ_ε, sd(K), sd(Y))

savefig(plot_bkm(K, Y), "bkm_simulation.png")
println("wrote bkm_simulation.png")
