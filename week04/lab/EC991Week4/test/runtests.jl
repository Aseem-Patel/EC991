using EC991Week4, Test

# ---------------------------------------------------------------- the week 3 checks, unchanged
@testset "household and grids" begin
    # w_from_r: with K/L = (α/(r + δ))^(1/(1-α)), the firm's two first-order conditions hold
    k = (0.33 / (0.03 + 0.05))^(1 / 0.67)
    @test 0.33 * k^(0.33 - 1) - 0.05 ≈ 0.03 && w_from_r(0.03) ≈ 0.67 * k^0.33
    hh = Household()
    @test hh.R == 1 + hh.r && hh.w == w_from_r(hh.r)
    for spacing in (:equal, :growth)
        a = asset_grid(300, 50.0; ϕ = hh.ϕ, spacing)
        @test a[1] == -hh.ϕ && a[end] ≈ 50.0 && issorted(a)
    end
    g = diff(asset_grid(10, 250.0; spacing = :growth, ν = 0.4))
    @test all(g[2:end] ./ g[1:end-1] .≈ 1.4)                  # each gap 1 + ν times the last
    @test asset_grid(5, 3.0; ϕ = 1.0)[1] == -1.0
end

@testset "Rouwenhorst" begin
    for n_y in (2, 3, 5, 9), ρ in (0.4, 0.9, 0.99)
        σ = 0.1
        logy, P = rouwenhorst(n_y, ρ, σ)
        @test all(sum(P; dims = 2) .≈ 1)
        m = chain_moments(logy, P)
        @test m.sd ≈ σ / sqrt(1 - ρ^2) atol = 1e-12            # exact for every n
        @test m.autocorr ≈ ρ atol = 1e-12
    end
    # n_y = 2, ρ = 0.4 and a stationary standard deviation of 0.1: log y = ±0.1, p = 0.7
    logy, P = rouwenhorst(2, 0.4, 0.1 * sqrt(1 - 0.4^2))
    @test logy ≈ [-0.1, 0.1] && P ≈ [0.7 0.3; 0.3 0.7]
    # week 3 slide 16 matrix and slide 17 grid: n = 3, ρ = 0.9, σ = 0.1
    logy, P = rouwenhorst(3, 0.9, 0.1)
    @test P ≈ [0.9025 0.095 0.0025; 0.0475 0.905 0.0475; 0.0025 0.095 0.9025]
    @test round.(exp.(logy); digits = 3) == [0.723, 1.0, 1.383]
end

@testset "Tauchen" begin
    for n_y in (3, 5, 11), ρ in (0.5, 0.9, 0.99)
        logy, P = tauchen(n_y, ρ, 0.1)
        @test all(sum(P; dims = 2) .≈ 1) && all(P .>= 0)
    end
    # at ρ = 0.5 the standard deviation approaches the truth as n grows
    err(n_y) = abs(chain_moments(tauchen(n_y, 0.5, 0.1)...).sd - 0.1 / sqrt(0.75))
    @test err(21) < err(11) < err(5) && err(21) < 1e-3
    @test abs(chain_moments(tauchen(21, 0.5, 0.1)...).autocorr - 0.5) < 2e-3
    # at ρ = 0.99 and n_y = 5 the middle state is absorbing to seven digits (week 3 slide 15)
    logy, P = tauchen(5, 0.99, 0.1)
    @test P[3, :] ≈ [0, 0, 1, 0, 0] atol = 1e-6
end

@testset "stationary distribution and simulation" begin
    P = [0.8 0.2; 0.2 0.8]
    @test stationary(P) ≈ [0.5, 0.5]
    π = stationary(rouwenhorst(5, 0.9, 0.1)[2])                # binomial(4, 1/2)
    @test π ≈ [1, 4, 6, 4, 1] ./ 16
    # u ≤ P[j, 1] stays in state 1 of a two-state chain; u > P[j, 1] moves
    vals, idx = simulate_chain([-1.0, 1.0], P, [0.0, 0.1, 0.9, 0.5]; j0 = 1)
    @test idx == [1, 1, 2, 2] && vals == [-1.0, -1.0, 1.0, 1.0]
end

# Week 3's lab default: r = 0.03, 3-state Rouwenhorst (ρ = 0.9, σ = 0.1), 300 points on [0, 50]
hh = Household()
a = asset_grid(300, 50.0)
logy, P = rouwenhorst(3, 0.9, 0.1)
y = exp.(logy)
sol = solve_vfi(hh, a, y, P)
sol_lin = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :linear)
sol_cub = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :cubic)

@testset "VFI by grid search" begin
    d = sol.distances
    @test d[end] < 1e-7
    @test all(d[n] <= hh.β * d[n-1] + 1e-12 for n in 2:length(d))   # contraction with modulus β
    @test policy_monotone(sol) == (in_a = true, in_y = true)
    @test sol.apol[1, 1] == a[1] && sol.apol[1, end] > a[1]        # binds at low assets, low income only
    @test all(sol.cpol .> 0)
    @test all(sol_lin.apol[end, :] .< a[end])                       # the upper bound does not bind
    # the continuation values as one matrix product: EV[k, j] = Σ_m V[k, m] P[j, m]
    EV = sol.V * P'
    @test EV[7, 2] ≈ sum(sol.V[7, m] * P[2, m] for m in 1:3)
    # one more Bellman step leaves the converged V where it is
    V1, pol = similar(sol.V), similar(sol.pol)
    bellman_step!(V1, pol, sol.V, hh, a, y, P)
    @test maximum(abs.(V1 .- sol.V)) < 1e-7 && pol == sol.pol
end

@testset "interpolation" begin
    x = [0.0, 1.0, 2.5, 4.0, 7.0]
    lin, spl = LinearInterp(x, 3 .+ 2 .* x), CubicSpline(x, 3 .+ 2 .* x)
    @test all(lin(t) ≈ 3 + 2t && spl(t) ≈ 3 + 2t for t in range(0, 7; length = 50))   # exact on lines
    v = sin.(x)
    @test LinearInterp(x, v).(x) ≈ v && CubicSpline(x, v).(x) ≈ v                    # through the nodes
    # week 3 slide 28 and 29: the stored values come from V(a) = c + d log(1 + a), V(2) = 10, V(4) = 14
    xs = [0.0, 2, 4, 6, 8]
    d = 4 / log(5 / 3); c = 10 - d * log(3)
    vs = c .+ d .* log.(1 .+ xs)
    @test LinearInterp(xs, vs)(2.5) ≈ 11                                           # the worked number
    @test CubicSpline(xs, vs)(2.5) ≈ 11.3796 atol = 1e-4                           # the figure's spline
    # fourth-order accuracy where the natural end conditions are right (sin'' = 0 at 0 and π)
    err(n) = (g = collect(range(0, π; length = n)); s = CubicSpline(g, sin.(g));
              maximum(abs(s(t) - sin(t)) for t in range(0, π; length = 2001)))
    @test err(9) / err(17) > 14
end

@testset "one-dimensional optimisers" begin
    f(x) = log(x) - x                                                # maximum at x = 1
    xg, fg, ng = golden_section(f, 0.1, 5.0)
    xb, fb, nb = brent(f, 0.1, 5.0)
    @test xg ≈ 1 atol = 1e-6
    @test xb ≈ 1 atol = 1e-6
    @test nb < ng                                                    # Brent needs fewer evaluations
    @test brent(x -> -(x - 0.3)^2, 0.0, 1.0)[1] ≈ 0.3 atol = 1e-6
end

@testset "interpolate, then optimise" begin
    @test sol_lin.distances[end] < 1e-7 && sol_cub.distances[end] < 1e-7
    @test policy_monotone(sol_lin) == (in_a = true, in_y = true)
    @test policy_monotone(sol_cub) == (in_a = true, in_y = true)
    @test maximum(abs.(sol_lin.apol .- sol_cub.apol)) < 0.02          # linear and cubic agree
    # each differs from grid search by about one grid gap next to the grid choice at most;
    # slightly more is possible (at one state, 1.06 gaps) because the two methods iterate on
    # slightly different value functions, so the margin is 10%
    gapk(k) = max(k > 1 ? a[k] - a[k-1] : 0.0, k < length(a) ? a[k+1] - a[k] : 0.0)
    for s in (sol_lin, sol_cub)
        @test maximum(abs(s.apol[i, j] - sol.apol[i, j]) / gapk(sol.pol[i, j]) for i in eachindex(a), j in 1:3) <= 1.1
    end
end

# ---------------------------------------------------------------- new this week
# The lecture's worked examples: every number below is printed on a slide.
@testset "worked examples" begin
    # slide 10: c̃ = 1.5, a' = 2, w y_3 = 1, R = 1.02
    @test endogenous_assets(1.5, 2.0, 1.0, 1.02) ≈ 2.5 / 1.02
    @test round(endogenous_assets(1.5, 2.0, 1.0, 1.02); digits = 3) == 2.451
    @test inverse_marginal_utility(marginal_utility(1.5, 2.0), 2.0) ≈ 1.5
    # slide 19: log utility, β = 0.95, R = 1
    e = euler_error(2.0, [1.5, 1.8, 2.0, 2.2, 2.5], fill(0.2, 5), 0.95, 1.0, 1.0)
    @test round(e.c_EE; digits = 4) == 2.0425
    @test round(e.ε; digits = 4) == -0.0213
end

# Close to the equilibrium: 7-state Rouwenhorst and the lab grid
hh = Household(r = 0.0377)
a = asset_grid(300, 250.0; ν = 0.025)
logy, P = rouwenhorst(7, 0.9, 0.1)
y = exp.(logy)
sol_grid = solve_vfi(hh, a, y, P)
sol_fast = solve_vfi(hh, a, y, P; step = bellman_step_fast!)
sol_howard = solve_vfi(hh, a, y, P; step = bellman_step_fast!, n_H = 50)
sol_interp = solve_vfi(hh, a, y, P; step = bellman_step_interp!)
sol_interp_howard = solve_vfi(hh, a, y, P; step = bellman_step_interp!, n_H = 50, howard_from = 0.01)
sol_egm = solve_egm(hh, a, y, P)
gapk(k) = max(k > 1 ? a[k] - a[k-1] : 0.0, k < length(a) ? a[k+1] - a[k] : 0.0)

@testset "VFI speed-ups change nothing but the time" begin
    @test sol_fast.pol == sol_grid.pol                        # same choices, state by state
    @test sol_fast.iterations == sol_grid.iterations
    @test maximum(abs.(sol_howard.apol .- sol_grid.apol) ./ gapk.(sol_grid.pol)) <= 1.1
    @test sol_howard.iterations < sol_grid.iterations / 5
    @test maximum(abs.(sol_interp_howard.apol .- sol_interp.apol)) < 1e-3
    @test sol_interp_howard.iterations < sol_interp.iterations / 2
    # one Howard update with a policy on the grid is the Bellman update at that policy
    V1, pol = similar(sol_grid.V), similar(sol_grid.pol)
    bellman_step!(V1, pol, sol_grid.V, hh, a, y, P)
    W = copy(sol_grid.V)
    howard_update!(W, pol, hh, a, y, P; n_H = 1)
    @test W ≈ V1
    # the same off the grid: a Howard update at the chosen a' reproduces the Bellman value
    V2, apol = similar(sol_interp.V), similar(sol_interp.V)
    bellman_step_interp!(V2, apol, sol_interp.V, hh, a, y, P)
    W2 = copy(sol_interp.V)
    howard_update!(W2, apol, hh, a, y, P; n_H = 1)
    @test W2 ≈ V2
end

@testset "EGM" begin
    @test sol_egm.distances[end] < 1e-7 && sol_egm.V === nothing
    @test policy_monotone(sol_egm) == (in_a = true, in_y = true)
    @test all(sol_egm.apol .>= a[1]) && all(sol_egm.cpol .> 0)
    @test sol_egm.apol[1, 1] == a[1] && sol_egm.apol[1, end] > a[1]  # binds at low assets, low income only
    @test sol_egm.cpol ≈ hh.R .* a .+ hh.w .* y' .- sol_egm.apol     # the budget constraint
    # applying the step again leaves the converged policy where it is
    c1, apol = similar(sol_egm.cpol), similar(sol_egm.cpol)
    egm_step!(c1, apol, sol_egm.cpol, hh, a, y, P)
    @test maximum(abs.(c1 .- sol_egm.cpol)) < 1e-6
    # the two ways of solving the household problem agree where people live
    # (the gap grows with a only because the grid does: it stays a few percent of the local gap)
    g = compare_policies(sol_egm, sol_interp; a_hi = 50.0)
    k = searchsortedlast(a, g.a)
    @test g.gap < 0.05 * (a[k+1] - a[k])
end

@testset "Euler errors" begin
    e_grid, e_interp, e_egm = euler_errors.((sol_grid, sol_interp, sol_egm), Ref(hh), Ref(P))
    @test e_egm.mean_log10 < e_grid.mean_log10
    @test e_interp.mean_log10 < e_grid.mean_log10
    # on a finer grid the same method is more accurate
    a2 = asset_grid(1000, 250.0; ν = 0.008)
    @test euler_errors(solve_egm(hh, a2, y, P), hh, P).mean_log10 < e_egm.mean_log10
end

@testset "the lottery and Q (slide 25)" begin
    g5 = [0.0, 1.0, 2.0, 3.0, 4.0]
    k, p = young_weights(1.3, g5)
    @test k == 2 && p ≈ 0.7 && p * g5[k] + (1 - p) * g5[k+1] ≈ 1.3
    @test young_weights(-1.0, g5) == (1, 1.0) && young_weights(9.0, g5) == (4, 0.0)
    P5 = fill(0.2, 5, 5)
    g = [clamp(0.8x, 0, 4) for x in g5, _ in 1:5]
    g[3, 3] = 1.3                                             # the policy at (2, y_3)
    Q = build_Q(g, g5, P5)
    s = 2 * 5 + 3                                             # the state (2, y_3)
    @test Q[s, 1 + 1] ≈ 0.14 && Q[s, 1 + 2] ≈ 0.06             # to (1, y_1) and (2, y_1)
    @test all(sum(Q; dims = 2) .≈ 1)
    @test all(count(>(0), Q[r, :]) <= 2 * 5 for r in axes(Q, 1))
    # the forward loop finds the same λ as λ = Q'λ
    dist = stationary_distribution(g, g5, P5; tol = 1e-13)
    @test vec(dist.λ)' * Q ≈ vec(dist.λ)' atol = 1e-11
end

@testset "stationary distribution" begin
    g = policy_on_grid(sol_egm, a)
    dist = stationary_distribution(g, a, P)
    λ = dist.λ
    @test sum(λ) ≈ 1 && all(λ .>= 0)
    @test vec(sum(λ; dims = 1)) ≈ stationary(P) atol = 1e-8        # the income marginal is π
    # one step preserves mass and the mean: next period's mean is the mean of the policy
    μ = zeros(size(λ))
    for j in axes(λ, 2), i in axes(λ, 1)
        k, p = young_weights(g[i, j], a)
        μ[k, j] += p * λ[i, j]; μ[k+1, j] += (1 - p) * λ[i, j]
    end
    @test sum(μ) ≈ 1
    @test sum(a .* sum(μ; dims = 2)) ≈ sum(g .* λ)
    @test sum(a .* sum(μ * P; dims = 2)) ≈ sum(a .* sum(λ; dims = 2)) atol = 1e-6     # stationarity
    # Monte Carlo finds the same distribution
    panel = simulate_panel(g, a, y, P; n_households = 20_000, T = 600)
    @test sum(panel.a) / length(panel.a) ≈ sum(a .* sum(λ; dims = 2)) rtol = 0.02
    edges = [0.0, 1.0, 5.0, 10.0, 20.0, Inf]
    @test maximum(abs.(joint_table(λ, a, edges) .- panel_table(panel, edges, length(y)))) < 0.01
    @test sum(joint_table(λ, a, edges)) ≈ 1
    # a policy on the grid has absorbing states; the same economy off the grid has none
    abs_k = absorbing_states(sol_grid)
    @test !isempty(abs_k) && isempty(absorbing_states(sol_egm))
    @test all(sol_grid.apol[k, j] == a[k] for k in abs_k, j in axes(sol_grid.apol, 2))
    # mass that starts on one stays, so the start matters on the grid and not off it
    g_grid = policy_on_grid(sol_grid, a)
    λ_abs = zeros(size(λ)); λ_abs[abs_k[1], :] .= stationary(P)
    @test sum(stationary_distribution(g_grid, a, P; λ_start = λ_abs).λ[abs_k[1], :]) ≈ 1
    @test stationary_distribution(g, a, P; λ_start = fill(1 / length(g), size(g))).λ ≈ λ atol = 1e-6
    # the density integrates to the mass off the limit
    d = conditional_density(λ, a, 1)
    width = (a[3:end] .- a[1:end-2]) ./ 2
    @test sum(d.density .* width) + d.at_limit + λ[end, 1] / sum(λ[:, 1]) ≈ 1
end

@testset "statistics" begin
    # two equal groups, one with all the wealth: Gini of a two-point distribution
    x = wealth_inequality([0.0, 1.0], [0.5, 0.5])
    @test x.gini ≈ 0.5 && x.top10 ≈ 0.2 && x.top1 ≈ 0.02
    @test wealth_inequality([2.0, 3.0], [0.0, 1.0]).gini ≈ 0 atol = 1e-12    # everyone the same
    @test wealth_inequality([1.0, 2.0, 3.0], [0.5, 0.5, 0.0]).top10 ≈ 0.2 / 1.5  # empty nodes at the top
    @test percentile([0.0, 1.0, 2.0], [0.2, 0.4, 0.4], 0.4) ≈ 0.5
    @test percentile([0.0, 1.0, 2.0], [0.2, 0.4, 0.4], 0.1) == 0.0
    m = mpc(sol_egm, hh)
    @test m[1, 1] ≈ 1 atol = 1e-9                                 # constrained: spends any windfall
    @test all(m[end-50:end, :] .< 0.1)                            # the rich save it
end

econ = Economy()
eq = solve_equilibrium(econ; method = :egm)
stats = model_stats(eq)

@testset "the firm and the setup" begin
    s = setup(econ)
    @test s.N ≈ sum(s.y .* s.π)
    K, w = capital_demand(0.03, s.N), w_from_r(0.03)
    @test 0.33 * (K / s.N)^(0.33 - 1) - 0.05 ≈ 0.03                  # the two first-order conditions
    @test w ≈ 0.67 * (K / s.N)^0.33
    @test first(eq.history)[2] < 0 < eq.history[2][2]                # Φ changes sign on the bracket
end

@testset "the equilibrium" begin
    @test abs(eq.Φ) < 1e-3
    @test eq.hh.β * eq.hh.R < 1
    @test eq.K ≈ capital_demand(eq.r, eq.s.N) && eq.w ≈ w_from_r(eq.r)
    @test all(eq.sol.apol[searchsortedlast(a, 50.0), :] .< a[end])
    # the statistics on the lab grid
    @test stats.r ≈ 0.0377 atol = 2e-4
    @test stats.w ≈ 1.2871 atol = 2e-3
    @test stats.N ≈ 1.0266 atol = 1e-3
    @test stats.A_Y ≈ 3.766 atol = 0.02
    @test stats.at_limit ≈ 0.035 atol = 2e-3
    @test stats.p50 ≈ 5.58 atol = 0.1
    @test stats.p99 ≈ 29.1 atol = 0.3
    # the other methods reach the same equilibrium
    for method in (:vfi_fast, :vfi_interp_howard)
        @test solve_equilibrium(econ; method).r ≈ eq.r atol = 2e-3
    end
end
