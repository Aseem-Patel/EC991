using EC991Week5, Test, LinearAlgebra, Random
BLAS.set_num_threads(1)     # small matrices: threads only add overhead

# ---------------------------------------------------------------- the chain
@testset "Tauchen around a mean, entrants" begin
    logz, F = tauchen(101, 0.9, 0.2; mean = 1.0, m = 4.0)
    @test all(sum(F; dims = 2) .≈ 1) && all(F .>= 0)
    @test logz[51] ≈ 1.0 && logz[end] - logz[1] ≈ 8 * 0.2 / sqrt(1 - 0.81)
    f = stationary(F)
    @test sum(f) ≈ 1 && f' * F ≈ f'
    @test sum(f .* logz) ≈ 1.0 atol = 1e-8                    # symmetric chain: mean at the centre
    g = discretise_normal(logz, 0.5, 0.2)
    @test sum(g) ≈ 1 && all(g .>= 0)
    @test sum(g .* logz) ≈ 0.5 atol = 1e-3
end

# ---------------------------------------------------------------- the static economy
@testset "static economy: the two-firm example" begin
    # slide 12 example: z = (1, 2), η = 2/3, N = 9. n2/n1 = 2^3 = 8, so n = (1, 8)
    z, η = [1.0, 2.0], 2 / 3
    e = efficient_allocation(z, 9.0; η)
    @test e.n ≈ [1.0, 8.0] && e.Y ≈ 9.0
    # the marginal products are equal
    @test η * z[1] * e.n[1]^(η - 1) ≈ η * z[2] * e.n[2]^(η - 1)
    # Y = Z N^η with Z = 9^(1/3)
    Z = aggregate_tfp(z; η)
    @test Z ≈ 9^(1 / 3) && Z * 9^η ≈ e.Y
    # an equal split: Y = 4.5^(2/3) (1 + 2), and Ω is the ratio to the efficient Y
    @test sum(z .* 4.5 .^ η) ≈ 3 * 4.5^(2 / 3)
    τ = z[1] ./ z .- 1                                           # (1 + τ) z equal: the split is equal
    w = allocation_with_wedges(z, τ, 9.0; η)
    @test w.n ≈ [4.5, 4.5] && w.Y ≈ 3 * 4.5^(2 / 3)
    @test misallocation(z, τ; η) ≈ w.Y / e.Y
    @test w.Y / e.Y ≈ 0.9086 atol = 1e-4
end

@testset "static economy: wedges and entry" begin
    η = 0.6
    z = exp.(range(-1, 1; length = 9))
    m = collect(range(1.0, 2.0; length = 9))                    # unequal masses
    @test misallocation(z, fill(0.3, 9); η, m) ≈ 1              # a common wedge moves only the wage
    τ = 0.4 .* sin.(1:9)
    Ω = misallocation(z, τ; η, m)
    w = allocation_with_wedges(z, τ, 5.0; η, m)
    @test 0 < Ω < 1 && w.Y ≈ Ω * aggregate_tfp(z; η, m) * 5.0^η   # Y = Z Ω N^η
    # planner: brute force over M
    N, c_e = 10.0, 0.5
    pe = planner_entry(N, c_e; η)
    Ms = range(0.01, N / c_e - 0.01; length = 100_001)
    Ys = Ms .^ (1 - η) .* (N .- c_e .* Ms) .^ η
    @test Ms[argmax(Ys)] ≈ pe.M rtol = 1e-3
    @test pe.N_p ≈ η * N
    @test pe.Y ≈ η^η * (1 - η)^(1 - η) * c_e^(-(1 - η)) * N
end

# ---------------------------------------------------------------- the incumbent
ind = Industry()
s = setup(ind)

@testset "static decisions" begin
    p, η, c_f = 1.3, ind.η, ind.c_f
    d = static_decisions(p, s.z; η, c_f)
    @test η .* p .* s.z .* d.n .^ (η - 1) ≈ ones(length(s.z))     # first-order condition, w = 1
    @test d.π ≈ (1 - η) .* (p .* s.z) .^ (1 / (1 - η)) .* η^(η / (1 - η)) .- c_f   # slide 26
end

@testset "Bellman: three methods, one answer" begin
    d = static_decisions(1.48, s.z; ind.η, ind.c_f)
    sols = [solve_firm(d.π, s.F, ind.β; method) for method in (:vfi, :howard, :policy_iteration)]
    for sol in sols[2:end]
        @test sol.V ≈ sols[1].V rtol = 1e-7
        @test sol.χ == sols[1].χ
    end
    sol = sols[3]
    # a fixed point of the Bellman operator
    V1 = similar(sol.V); bellman_step!(V1, sol.V, d.π, s.F, ind.β)
    @test V1 ≈ sol.V rtol = 1e-12
    # exit is a threshold rule: exit below z̃, stay above
    k = findfirst(.!sol.χ)
    @test all(sol.χ[1:k-1]) && !any(sol.χ[k:end])
    # some firms stay with negative profits, expecting z to recover (slide 28)
    @test any((d.π .< 0) .& .!sol.χ)
    # the exit threshold lies between the last exiting and the first staying grid point
    z̃ = exit_threshold(sol, s.z).z̃
    @test s.z[k-1] < z̃ < s.z[k]
    # policy iteration converges in a handful of rounds, VFI takes many more
    @test sol.iterations <= 6 < sols[1].iterations
    # evaluating a fixed rule exactly = iterating it
    W = copy(d.π)
    for _ in 1:400; W = d.π .+ ind.β .* (1 .- sol.χ) .* (s.F * W); end
    @test evaluate_policy(sol.χ, d.π, s.F, ind.β) ≈ W rtol = 1e-10
end

@testset "free entry" begin
    Ves = [entry_value(p, ind, s).Ve for p in (1.2, 1.4, 1.6)]
    @test issorted(Ves)                                          # V^e rises with p
    e, history = solve_price(ind, s)
    @test abs(e.Ve) < 1e-6
end

# ---------------------------------------------------------------- the measure of firms
@testset "the measure of firms" begin
    e, _ = solve_price(ind, s)
    F̂ = exit_transition(s.F, e.sol.χ)
    μ_lin = stationary_measure(F̂, s.g)
    μ_it = stationary_measure(F̂, s.g; method = :iterate)
    @test μ_lin ≈ μ_it rtol = 1e-9
    @test stationary_measure(F̂, s.g; M = 3.0) ≈ 3 .* μ_lin       # linear in M
    @test vec(sum(cohorts(F̂, s.g, 400); dims = 2)) ≈ μ_lin rtol = 1e-9
    # stationarity: entrants replace exits, M = Σ χ μ
    @test sum(e.sol.χ .* μ_lin) ≈ 1.0
end

# ---------------------------------------------------------------- the equilibrium
@testset "equilibrium" begin
    eq = solve_equilibrium()
    st = industry_stats(eq)
    @test st.Y ≈ st.demand rtol = 1e-10                          # the goods market clears
    @test st.N_prod ≈ ind.η * ind.D̄                              # w N_prod = η p Y = η D̄
    @test st.entry_rate ≈ st.exit_rate rtol = 1e-6               # stationary
    # the three methods give the same equilibrium
    for method in (:howard, :policy_iteration)
        @test solve_equilibrium(; method).p ≈ eq.p rtol = 1e-8
    end
    # cohorts: survival falls, survivors grow, exit falls with age
    ap = age_profile(eq, 20)
    @test issorted(ap.survival; rev = true) && issorted(ap.avg_size) && ap.exit_rate[1] > ap.exit_rate[end]
    sd = size_distribution(eq, [0, 10, 20, 50, 100, 500, Inf])
    @test sum(sd.firms) ≈ 1 && sum(sd.employment) ≈ 1
end

@testset "the exit lottery" begin
    eq = solve_equilibrium(; exit = :lottery)
    frac = filter(x -> 0 < x < 1, eq.χ)
    @test length(frac) <= 1 && all(x -> x == 0 || x == 1 || 0 < x < 1, eq.χ)
    @test eq.p == solve_equilibrium().p                          # the price depends only on V
    fine = industry_stats(solve_equilibrium(Industry(n_z = 2001))).entry_rate
    coarse = industry_stats(solve_equilibrium()).entry_rate
    lottery = industry_stats(eq).entry_rate
    @test abs(lottery - fine) < abs(coarse - fine) / 5
end

@testset "simulated panel" begin
    eq = solve_equilibrium()
    st = industry_stats(eq)
    panel = simulate_panel(eq; T = 10, N = 20_000, rng = MersenneTwister(1))
    # one row per firm and year, years 1 to 10
    @test extrema(panel.year) == (1, 10)
    # firms never come back after exiting, and age by one year at a time
    last_year, last_age, ok = Dict{Int, Int}(), Dict{Int, Int}(), true
    for r in eachindex(panel.firm)
        f = panel.firm[r]
        if haskey(last_year, f)
            ok &= panel.year[r] == last_year[f] + 1 && panel.age[r] == last_age[f] + 1
        end
        last_year[f], last_age[f] = panel.year[r], panel.age[r]
    end
    @test ok
    # the panel's averages are close to the stationary equilibrium's
    @test sum(panel.exit) / length(panel.exit) ≈ st.exit_rate atol = 0.005
    @test sum(panel.age .== 0) / length(panel.age) ≈ st.entry_rate atol = 0.005
    @test sum(panel.n) / length(panel.n) ≈ st.avg_size rtol = 0.03
end
