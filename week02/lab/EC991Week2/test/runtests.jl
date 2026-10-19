using EC991Week2, Test

@testset "infinite horizon" begin
    hh = Household()
    a = asset_grid(hh; amax = 3.0, na = 200)
    sol = solve_vfi(hh, a)
    @test sol.distances[end] < 1e-6
    @test issorted(sol.apol)                       # monotone savings policy
    @test all(sol.cpol .> 0)
    apath, cpath = simulate(sol, 2.5, 300)
    @test apath[end] == a[1]                       # drifts to the borrowing limit
    @test minimum(euler_residuals(cpath, hh)) > -0.05   # Euler inequality, up to grid error
    # contraction: every successive distance shrinks by at least the factor β
    d = sol.distances
    @test all(d[ℓ] <= hh.β * d[ℓ - 1] + 1e-12 for ℓ in 2:length(d))
    # a worse guess converges to the same V, geometrically
    V0 = zeros(length(a)); V1 = similar(V0); pol = zeros(Int, length(a))
    for ℓ in 1:400
        bellman_step!(V1, pol, V0, hh, a); V0, V1 = V1, V0
    end
    @test maximum(abs.(V0 .- sol.V)) < 1e-3
end

@testset "finite horizon reuses the Bellman step" begin
    hh = Household()
    a = asset_grid(hh; amax = 3.0, na = 100)
    sol = solve_vfi(hh, a)
    # one backward pass from the converged V reproduces V (a fixed point stays put)
    fsol = solve_backward(hh, a, fill(hh.y, 1); terminal = sol.V)
    @test maximum(abs.(fsol.V[:, 1] .- sol.V)) < 1e-6
end

@testset "cake eating closed form" begin
    T, a0 = 20, 1.0
    cake = Household(β = 0.95, R = 1.0, y = 0.0, ϕ = 0.0, γ = 1.0)
    errs = map((100, 400, 1600)) do na
        a = collect(range(0.0, a0; length = na))
        fsol = solve_backward(cake, a, zeros(T))
        apath, cpath = simulate_finite(fsol, a0)
        maximum(abs.(cpath .- [cake_eating_consumption(apath[t], t, T, cake.β) for t in 1:T]))
    end
    @test errs[end] < 2e-3
    @test errs[1] > errs[2] > errs[3]              # error shrinks with the grid
    ap, cp = cake_eating_path(a0, T, cake.β)
    @test sum(cp) ≈ a0                             # the cake is eaten exactly
end
