using EC991Week7, Test, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead

# ---------------------------------------------------------------- shared tools
@testset "Young's lottery keeps the mean; Gini and top shares" begin
    a = asset_grid(50, 100.0)
    for a′ in (0.0, 0.3, 17.2, 99.9)
        k, p = young_weights(a′, a)
        @test p * a[k] + (1 - p) * a[k+1] ≈ a′
    end
    # everyone the same: no inequality; 1% of households hold everything: Gini 0.99
    @test abs(gini([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])) < 1e-12
    @test gini([0.0, 100.0], [0.99, 0.01]) ≈ 0.99
    # equal mass on 0, 1, ..., 99: the top 10% hold (90 + ... + 99) / (0 + ... + 99)
    @test top_share(collect(0.0:99.0), ones(100), 0.10) ≈ sum(90:99) / sum(0:99)
end

# ---------------------------------------------------------------- Part A: Krusell-Smith
eco = KSEconomy()
s = ks_setup(eco)

@testset "the 4 × 4 transition matrix (slides 18 to 22)" begin
    Π = joint_transition(eco)
    Py = employment_transitions(eco)
    # every row is a probability distribution
    @test all(Π .>= 0)
    @test all(sum(Π; dims = 2) .≈ 1)
    # the unemployed rows of slides 20 and 21
    @test Py[1, 1, 2, 2] ≈ 1 / 3     # boom to boom: spell of 1.5 quarters
    @test Py[1, 1, 1, 1] ≈ 0.6       # recession to recession: 2.5 quarters
    @test Py[1, 1, 2, 1] ≈ 0.75      # boom to recession
    @test Py[1, 1, 1, 2] ≈ 0.25      # recession to boom
    # an employed row worked by hand: boom to boom, (0.04 - 0.04 / 3) / 0.96
    @test Py[2, 1, 2, 2] ≈ (0.04 - 0.04 / 3) / 0.96
    for z in 1:2, z′ in 1:2
        # adding up (slide 6): summing over y' gives P(Z' | Z), whatever y
        for j in 1:2
            @test sum(Π[(z-1)*2+j, (z′-1)*2+1:(z′-1)*2+2]) ≈ s.PZ[z, z′]
        end
        # consistency (slide 22): unemployment moves from u(Z) to u(Z')
        πy = [eco.u[z], 1 - eco.u[z]]
        @test (πy' * Py[:, :, z, z′])' ≈ [eco.u[z′], 1 - eco.u[z′]]
    end
end

@testset "factor prices exhaust output" begin
    for z in 1:2, K in (30.0, 37.0)
        R, w = ks_prices(eco.Z[z], K, s.N[z], eco)
        Y = eco.Z[z] * K^eco.α * s.N[z]^(1 - eco.α)
        @test (R - 1 + eco.δ) * K + w * s.N[z] ≈ Y
    end
end

@testset "the histogram keeps unemployment at u(Z_t) at every date" begin
    θ = [log(s.K_ra) log(s.K_ra); 0.0 0.0]
    sol = solve_ks_household(θ, eco, s)
    zpath = simulate_Z(eco, s)[1:300]
    λ = initial_distribution(s.K_ra, zpath[1], eco, s)
    # move the distribution by hand, as simulate_capital does, and check the shares
    gd = policy_on_dist(sol, s)
    μ = similar(λ)
    for t in 1:299
        k, ωK = bracket(sum(s.a_dist .* λ), s.Kgrid)
        g = ωK .* gd[:, :, zpath[t], k] .+ (1 - ωK) .* gd[:, :, zpath[t], k+1]
        move_assets!(μ, λ, g, s.a_dist)
        λ = μ * s.Py[:, :, zpath[t], zpath[t+1]]
        @test sum(λ) ≈ 1
        @test sum(λ[:, 1]) ≈ eco.u[zpath[t+1]]
    end
    # the simulation function gives the same capital
    @test simulate_capital(sol, zpath, eco, s; λ_start = initial_distribution(s.K_ra, zpath[1], eco, s)).K[end] ≈
          sum(s.a_dist .* λ)
end

# a shorter simulation than the default, to keep the tests quick
eq = solve_krusell_smith(KSEconomy(T_sim = 4_000, T_burn = 500))
@testset "Krusell-Smith: the approximate equilibrium" begin
    # the rule reproduces itself: one more pass gives the same coefficients
    @test eq.history[end] < 1e-6
    # quasi-aggregation (slide 23): R² near one, tiny residuals, slopes just below one
    @test all(eq.R2 .> 0.99999)
    @test all(eq.σ .< 1e-4)
    @test all(0.9 .< eq.θ[2, :] .< 1)
    # booms raise tomorrow's capital at any K
    for K in s.Kgrid
        @test forecast_K(K, 2, eq.θ) > forecast_K(K, 1, eq.θ)
    end
    # simulated capital never leaves the K grid
    K = eq.K[eq.eco.T_burn+1:end]
    @test s.Kgrid[1] < minimum(K) && maximum(K) < s.Kgrid[end]
    # nobody reaches the top of the asset grid
    @test sum(eq.λ[end-20:end, :]) < 1e-10
    # forecasts 25 years ahead are off by well under 1%
    @test forecast_errors(eq).horizon_max < 0.5
end

@testset "Krusell-Smith: the household's choices" begin
    (; apol) = eq.sol
    # saving rises with assets, and the employed save more than the unemployed
    @test all(diff(apol; dims = 1) .>= 0)
    @test all(apol[:, 2, :, :] .>= apol[:, 1, :, :])
    # the borrowing limit binds for poor unemployed households, never for rich ones
    @test any(apol[:, 1, :, :] .== 0)
    @test all(apol[end÷2:end, :, :, :] .> 0)
end

# ---------------------------------------------------------------- Part B: sequence space
ss = solve_steady_state(Economy())
@testset "the steady state is week 4's" begin
    @test abs(ss.r - 0.0377) < 1e-4
    @test abs(ss.Ea - ss.K) < 1e-9
    # one more EGM step at the same prices changes nothing
    c, apol = similar(ss.c), similar(ss.c)
    R = 1 + ss.r
    egm_backward!(c, apol, ss.c, R, R, income(ss.w, 0.0, ss.s.y, ss.s.N), ss.s.a, ss.s.P, ss.eco.β, ss.eco.γ)
    @test maximum(abs.(c .- ss.c)) < 1e-10
end

@testset "no shock, no transition" begin
    tr = mit_shock(ss; ε = 0.0, T = 100)
    @test maximum(abs.(tr.K .- ss.K)) < 1e-8
    @test maximum(abs.(tr.r .- ss.r)) < 1e-10
end

tr = mit_shock(ss)
@testset "the MIT shock" begin
    (; α, δ) = ss.eco
    T = length(tr.K)
    # the market clears at every date
    @test maximum(abs.(tr.Ks[2:T-1] .- tr.K[2:T-1])) < 1e-8
    # capital is predetermined at t = 1, then rises and returns
    @test tr.K[1] ≈ ss.K
    @test maximum(tr.K) > ss.K * 1.005
    @test abs(tr.end_gap) < 1e-4 * ss.K
    # the goods market: C_t + K_{t+1} - (1 - δ) K_t = Y_t
    @test maximum(abs.(tr.C[1:T-2] .+ tr.K[2:T-1] .- (1 - δ) .* tr.K[1:T-2] .- tr.Y[1:T-2])) < 1e-6
    # on impact the wage jumps with Z and the interest rate too
    @test tr.w[1] > ss.w && tr.r[1] > ss.r
end

@testset "the labour tax reform" begin
    ss1 = solve_steady_state(Economy(τ = 0.2))
    @test ss1.K < ss.K
    tr1 = tax_reform(ss, ss1)
    # capital falls step by step from the old steady state to the new one
    @test all(diff(tr1.K[1:end-1]) .<= 1e-10)      # K_T is pinned at the new steady state
    @test abs(tr1.K[end-1] - ss1.K) < 1e-3
    @test abs(tr1.end_gap) < 1e-4 * ss1.K
end

@testset "Boppart, Krusell and Mitman: adding up impulse responses" begin
    h = impulse_response(tr.K, ss.K, 0.01)
    # one innovation at t = 1 gives back the impulse response, scaled
    ε = zeros(50); ε[1] = 0.02
    @test bkm_simulate(h, ε) ≈ 0.02 .* h[1:50]
    # two innovations add up
    ε2 = zeros(50); ε2[10] = -0.01
    @test bkm_simulate(h, ε .+ ε2) ≈ bkm_simulate(h, ε) .+ bkm_simulate(h, ε2)
    # and the model is close to linear for small shocks
    hneg = impulse_response(mit_shock(ss; ε = -0.01).K, ss.K, -0.01)
    @test maximum(abs.(h .- hneg)) < 0.05 * maximum(abs.(h))
end
