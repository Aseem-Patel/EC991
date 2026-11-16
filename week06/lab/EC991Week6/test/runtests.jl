using EC991Week6, Test, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead

# ---------------------------------------------------------------- the choice of n
@testset "exhaustive search over n: a case worked by hand" begin
    # four grid points, firing tax 2 per worker
    n, h, τ = [0.0, 1.0, 2.0, 3.0], [1.0, 6.0, 5.0, 2.0], 2.0
    V, pol = zeros(4), zeros(Int, 4)
    best_employment!(V, pol, h, n, τ)
    @test V == [6.0, 6.0, 5.0, 3.0]
    @test n[pol] == [1.0, 1.0, 2.0, 2.0]
end

@testset "exhaustive search over n: ties" begin
    # inheriting m = 2: staying at 2 is worth 4, firing down to 1 is worth 6 - 2 = 4.
    # On a tie the firm fires nobody.
    n, h, τ = [0.0, 1.0, 2.0, 3.0], [1.0, 6.0, 4.0, 2.0], 2.0
    V, pol = zeros(4), zeros(Int, 4)
    best_employment!(V, pol, h, n, τ)
    @test V[3] == 4.0
    @test pol[3] == 3
    # two choices that fire nobody and are worth the same: the smaller one
    h2 = [5.0, 5.0, 1.0, 0.0]
    best_employment!(V, pol, h2, n, τ)
    @test pol[1] == 1
end

@testset "exhaustive search over n: the definition, on an uneven grid" begin
    # an arbitrary h with several peaks: the value is the max of h(n) - τ max{0, m - n}
    n = [0.0; sort(rand(40)) .* 100]
    h = sin.(n ./ 7) .* 50 .- 0.01 .* (n .- 40) .^ 2
    for τ in (0.0, 0.3, 2.0)
        V, pol = zeros(length(n)), zeros(Int, length(n))
        best_employment!(V, pol, h, n, τ)
        for k in eachindex(n)
            vals = [h[j] - firing_cost(n[j], n[k], τ) for j in eachindex(n)]
            @test V[k] == maximum(vals)
            @test vals[pol[k]] == maximum(vals)
        end
    end
end

@testset "the one-pass search gives the same answer as the exhaustive search" begin
    n = [0.0; sort(rand(60)) .* 100]
    for τ in (0.0, 0.3, 2.0), rep in 1:20
        h = sin.(n ./ 7) .* 50 .- 0.01 .* (n .- 40) .^ 2 .+ randn(length(n))
        V1, pol1 = zeros(length(n)), zeros(Int, length(n))
        V2, pol2 = zeros(length(n)), zeros(Int, length(n))
        best_employment!(V1, pol1, h, n, τ)
        best_employment_fast!(V2, pol2, h, n, τ)
        @test V1 ≈ V2
        @test pol1 == pol2
    end
end

# ---------------------------------------------------------------- the frictionless benchmark
eq0 = solve_equilibrium(Economy(τ = 0.0))
@testset "τ = 0: chosen employment does not depend on inherited employment" begin
    (; pol) = eq0.sol
    # every inherited m leads to the same n at a given z
    @test all(pol .== pol[:, 1])
    # and that n is one of the two grid points around the first-order condition η p z n^(η-1) = 1
    n_foc = (eq0.eco.η * eq0.p .* eq0.s.z) .^ (1 / (1 - eq0.eco.η))
    for i in eachindex(eq0.s.z)
        n_foc[i] > eq0.s.n[2] || continue
        k = searchsortedlast(eq0.s.n, n_foc[i])
        @test pol[i, 1] in (k, k + 1)
    end
    # so the only misallocation left is the grid's
    @test misallocation(eq0).mpl_gap < 0.5
    @test misallocation(eq0).output_lost < 0.01
end

# ---------------------------------------------------------------- the baseline
eq = solve_equilibrium(Economy())
@testset "τ = 0.1: equilibrium conditions" begin
    a = aggregates(eq)
    @test abs(eq.Ve) < 1e-6                        # free entry
    @test a.Y ≈ a.C                                # goods market
    @test a.Ns ≈ a.Nd rtol = 1e-8                  # labour market, by Walras' law
    @test a.entry_rate ≈ a.exit_rate rtol = 1e-8   # stationarity: entrants replace exits
    @test a.job_creation ≈ a.job_destruction rtol = 1e-8
    @test a.Π ≈ sum(eq.μ .* (eq.p .* output_per_firm(eq.sol, eq.s, eq.eco) .- eq.s.n[eq.sol.pol] .-
                     eq.eco.τ .* (max.(0, eq.s.n' .- eq.s.n[eq.sol.pol]) .+ eq.sol.χ .* eq.s.n[eq.sol.pol]))) - eq.M * eq.eco.c_e
    # μ is stationary under the law of motion
    μ1, tmp = similar(eq.μ), similar(eq.μ)
    law_of_motion!(μ1, eq.μ, eq.sol, eq.s, eq.M, tmp)
    @test μ1 ≈ eq.μ rtol = 1e-9
    # the employment grid is wide enough: no firm with mass chooses its top point
    @test all(eq.sol.pol[eq.μ .> 0] .< length(eq.s.n))
end

@testset "τ = 0.1: the inaction region" begin
    (; pol) = eq.sol
    r = inaction_region(eq.sol, eq.s)
    @test all(r.nL .<= r.nH)
    @test any(r.nL .< r.nH)
    # policy: hire up to n_L below it, keep m inside, fire down to n_H above it
    for i in eachindex(eq.s.z), k in eachindex(eq.s.n)
        m = eq.s.n[k]
        target = m < r.nL[i] ? r.nL[i] : m > r.nH[i] ? r.nH[i] : m
        @test eq.s.n[pol[i, k]] == target
    end
end

@testset "higher τ: wider inaction region, less reallocation, more misallocation" begin
    eq1 = solve_equilibrium(Economy(τ = 1.0))
    r, r1 = inaction_region(eq.sol, eq.s), inaction_region(eq1.sol, eq1.s)
    mid = (eq.s.z .> 3.5) .& (eq.s.z .< 10)
    @test all(r1.nH[mid] ./ r1.nL[mid] .> r.nH[mid] ./ r.nL[mid])
    a0, a, a1 = aggregates(eq0), aggregates(eq), aggregates(eq1)
    @test a0.p < a.p < a1.p
    @test a0.Y > a.Y > a1.Y
    @test a0.job_creation > a.job_creation > a1.job_creation
    m0, m, m1 = misallocation(eq0), misallocation(eq), misallocation(eq1)
    @test m0.mpl_gap < m.mpl_gap < m1.mpl_gap
    @test m0.output_lost < m.output_lost < m1.output_lost
    @test all(eq1.sol.pol[eq1.μ .> 0] .< length(eq1.s.n))
end
