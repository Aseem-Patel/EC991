# >>> simulate_Z
"""
    simulate_Z(eco, s)

The path of the aggregate state, z_t ∈ {1, 2} for t = 1, ..., T_sim, starting in
a boom, from the seed in `eco`. It is drawn once and kept fixed while the
forecasting rule is updated, so every iteration sees the same shocks.
"""
simulate_Z(eco::KSEconomy, s) = simulate_chain(s.PZ, 2, eco.T_sim, Random.Xoshiro(eco.seed))
# <<< simulate_Z

# >>> policy_on_dist
"""
    policy_on_dist(sol, s)

The saving rule on the distribution grid, gd[d, j, z, k] at (a_dist[d], y_j, Z_z,
K_k), read off by linear interpolation in a and kept inside the grid.
"""
function policy_on_dist(sol::KSSolution, s)
    (; a, a_dist, Kgrid) = s
    gd = zeros(length(a_dist), 2, 2, length(Kgrid))
    for k in eachindex(Kgrid), z in 1:2, j in 1:2
        f = LinearInterp(a, sol.apol[:, j, z, k])
        for (d, x) in enumerate(a_dist)
            gd[d, j, z, k] = clamp(f(x), a_dist[1], a_dist[end])
        end
    end
    return gd
end
# <<< policy_on_dist

# >>> initial_distribution
"""
    initial_distribution(K0, z1, eco, s)

Where the simulation starts: every household holds K0 (split between the two
grid points around it, so that the mean is exactly K0), with unemployment at
its rate in the first aggregate state, π(y | Z_{z1}).
"""
function initial_distribution(K0, z1, eco::KSEconomy, s)
    λ = zeros(length(s.a_dist), 2)
    k, p = young_weights(K0, s.a_dist)
    shares = [eco.u[z1], 1 - eco.u[z1]]
    λ[k, :] .= p .* shares
    λ[k+1, :] .= (1 - p) .* shares
    return λ
end
# <<< initial_distribution

# >>> simulate_capital
"""
    simulate_capital(sol, zpath, eco, s; λ_start)

The slide 15 simulation, with a histogram in place of a panel: the panel of week
4's Monte Carlo with infinitely many households. At each date t:

1. aggregate capital is the mean of the distribution, K_t = Σ a λ_t;
2. households save by the rule at (Z_t, K_t), interpolated between two points of
   the K grid;
3. the distribution moves forward: assets by Young's lottery, then endowments by
   P(y' | y, Z_t, Z_{t+1}). Consistency (slide 22) keeps the unemployment rate
   at exactly π(y_l | Z_{t+1}).

Returns the capital path K and the last distribution.
"""
function simulate_capital(sol::KSSolution, zpath, eco::KSEconomy, s; λ_start)
    (; a_dist, Kgrid, Py) = s
    gd = policy_on_dist(sol, s)
    T = length(zpath)
    K = zeros(T)
    λ, μ = copy(λ_start), similar(λ_start)
    g = zeros(length(a_dist), 2)
    for t in 1:T
        z = zpath[t]
        # 1. aggregate capital: the first moment of the wealth distribution
        K[t] = sum(a_dist .* λ)
        t == T && break
        # 2. the saving rule at (Z_t, K_t)
        k, ωK = bracket(K[t], Kgrid)
        @views g .= ωK .* gd[:, :, z, k] .+ (1 - ωK) .* gd[:, :, z, k+1]
        # 3. assets move, then endowments, with the probabilities for Z_t → Z_{t+1}
        move_assets!(μ, λ, g, a_dist)
        mul!(λ, μ, view(Py, :, :, z, zpath[t+1]))
    end
    return (; K, λ)
end
# <<< simulate_capital
