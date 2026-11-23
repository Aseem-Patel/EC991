# >>> KSEconomy
"""
    KSEconomy(; β, γ, α, δ, Z, y, u, p_ZZ, stay_unemployed, n_a, a_max, ν,
                n_K, K_band, n_dist, ν_dist, T_sim, T_burn, seed)

The Krusell-Smith economy of lecture slides 18 to 22. One period is a quarter.

  preferences   β = 0.99, log utility (γ = 1)
  firms         Y = Z K^α N^(1-α), α = 0.36, δ = 0.025
  aggregate     Z ∈ (Z_l, Z_h) = (0.99, 1.01), P(Z' = Z | Z) = 0.875
  idiosyncratic y ∈ (y_l, y_h) = (0.25, 1): unemployed and employed
  unemployment  π(y_l | Z_l) = 0.10, π(y_l | Z_h) = 0.04
  stay_unemployed[z, z′] = P(y' = y_l | Z' = Z_{z′}, y = y_l, Z = Z_z), slides 20 and 21

Index 1 is always the low state: Z_l (recession) and y_l (unemployed).

The numerical choices: the asset grid has n_a points from 0 to a_max with growth
rate ν; the grid for aggregate capital has n_K points, evenly spaced over
K_band = ±15% around the representative-agent steady state; the distribution
lives on n_dist points (growth rate ν_dist); the simulation is T_sim quarters,
the first T_burn of them discarded, with the random seed `seed`.
"""
Base.@kwdef struct KSEconomy
    β::Float64 = 0.99
    γ::Float64 = 1.0
    α::Float64 = 0.36
    δ::Float64 = 0.025
    Z::Vector{Float64} = [0.99, 1.01]
    y::Vector{Float64} = [0.25, 1.0]
    u::Vector{Float64} = [0.10, 0.04]
    p_ZZ::Float64 = 0.875
    stay_unemployed::Matrix{Float64} = [0.6 0.25; 0.75 1/3]
    n_a::Int = 100
    a_max::Float64 = 500.0
    ν::Float64 = 0.05
    n_K::Int = 7
    K_band::Float64 = 0.15
    n_dist::Int = 1000
    ν_dist::Float64 = 0.006
    T_sim::Int = 11_000
    T_burn::Int = 1_000
    seed::Int = 2026
end
# <<< KSEconomy

# >>> employment_transitions
"""
    employment_transitions(eco)

The 2 × 2 matrices P(y' | y, Z, Z'), stored as Py[j, j′, z, z′] (slides 18 to 21).
The unemployed rows are given. The employed rows follow from aggregate
consistency (slide 22): the unemployed share tomorrow must be π(y_l | Z'),

    u(Z') = P(y_l | y_l, Z, Z') u(Z) + P(y_l | y_h, Z, Z') (1 - u(Z)),

which is one equation in the one unknown P(y_l | y_h, Z, Z').
"""
function employment_transitions(eco::KSEconomy)
    (; u, stay_unemployed) = eco
    Py = zeros(2, 2, 2, 2)
    for z in 1:2, z′ in 1:2
        # the unemployed: given on slides 20 and 21
        p_ll = stay_unemployed[z, z′]
        Py[1, :, z, z′] = [p_ll, 1 - p_ll]
        # the employed: the job-loss probability that keeps the unemployment rate at u(Z')
        p_hl = (u[z′] - p_ll * u[z]) / (1 - u[z])
        0 <= p_hl <= 1 || error("no job-loss probability is consistent with u for z = $z, z′ = $z′")
        Py[2, :, z, z′] = [p_hl, 1 - p_hl]
    end
    return Py
end
# <<< employment_transitions

# >>> joint_transition
"""
    joint_transition(eco)

The 4 × 4 matrix P(y', Z' | y, Z) = P(y' | y, Z, Z') P(Z' | Z) of slides 6 and 18,
with the states numbered (y_l, Z_l), (y_h, Z_l), (y_l, Z_h), (y_h, Z_h). The
code never builds it: it uses the two factors. It is here to look at and test.
"""
function joint_transition(eco::KSEconomy)
    PZ = [eco.p_ZZ 1-eco.p_ZZ; 1-eco.p_ZZ eco.p_ZZ]
    Py = employment_transitions(eco)
    Π = zeros(4, 4)
    for z in 1:2, j in 1:2, z′ in 1:2, j′ in 1:2
        Π[(z-1)*2+j, (z′-1)*2+j′] = PZ[z, z′] * Py[j, j′, z, z′]
    end
    return Π
end
# <<< joint_transition

# >>> ks_prices
"""
    ks_prices(Z, K, N, eco)

Factor prices from the firm's first-order conditions (slide 3): the interest
rate r = α Z (K/N)^(α-1) - δ, so the rental rate is r + δ, and the wage
w = (1-α) Z (K/N)^α. Returns (R, w) with R = 1 + r.
"""
function ks_prices(Z, K, N, eco::KSEconomy)
    (; α, δ) = eco
    r = α * Z * (K / N)^(α - 1) - δ
    w = (1 - α) * Z * (K / N)^α
    return 1 + r, w
end
# <<< ks_prices

# >>> ks_setup
"""
    ks_setup(eco)

The objects that stay fixed while the forecasting rule is updated: P(Z' | Z) in
PZ, P(y' | y, Z, Z') in Py, labour N(Z) = Σ_y y π(y | Z) (it depends on Z only,
slide 10), the representative-agent steady state K_ra, the grids for
individual assets `a`, for aggregate capital `Kgrid`, and for the distribution
`a_dist`.
"""
function ks_setup(eco::KSEconomy)
    (; β, α, δ, y, u, p_ZZ) = eco
    PZ = [p_ZZ 1-p_ZZ; 1-p_ZZ p_ZZ]
    Py = employment_transitions(eco)
    # labour: unemployed share u(Z) supply y_l each, the rest y_h
    N = [u[z] * y[1] + (1 - u[z]) * y[2] for z in 1:2]
    # the representative agent's steady state at average labour, Z = 1: β (1 + r) = 1
    K_ra = sum(N) / 2 * (α / (1 / β - 1 + δ))^(1 / (1 - α))
    Kgrid = collect(range((1 - eco.K_band) * K_ra, (1 + eco.K_band) * K_ra; length = eco.n_K))
    a = asset_grid(eco.n_a, eco.a_max; ν = eco.ν)
    a_dist = asset_grid(eco.n_dist, eco.a_max; ν = eco.ν_dist)
    return (; PZ, Py, N, K_ra, Kgrid, a, a_dist)
end
# <<< ks_setup
