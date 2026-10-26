using EC991Week3, Test

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
    # slide 16 matrix and slide 17 grid: n = 3, ρ = 0.9, σ = 0.1
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
    # at ρ = 0.99 and n_y = 5 the middle state is absorbing to seven digits (slide 15)
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

# The lab default: r = 0.03, 3-state Rouwenhorst (ρ = 0.9, σ = 0.1), 300 points on [0, 50]
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
    # slides 28 and 29: the stored values come from V(a) = c + d log(1 + a), V(2) = 10, V(4) = 14
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
