# >>> Economy
"""
    Economy(; η, β, c_e, c_f, τ, D̄, ρ, σ, logz_mean, n_z, m, n_n, n_min, n_max)

The parameters of the Hopenhayn-Rogerson economy (tutorial slides 3, 4 and 12).

Firms produce y = z n^η and pay the fixed cost c_f in units of output every period
they produce. A firm that inherits m workers and chooses n pays the firing tax
τ max{0, m - n} in units of labour, also when it exits (it fires everyone). Entry costs c_e workers. Firms discount by β.
The household's utility is ln C - A N, so C = 1 / (A p); we write D̄ = 1/A, so
demand is C = D̄ / p as in week 5. The wage is the numeraire, w = 1.

Log productivity is an AR(1), log z_{t+1} - logz_mean = ρ (log z_t - logz_mean) + σ ε,
discretised by Tauchen with n_z points and width `m` standard deviations (the
field `m` is this width, not inherited employment); entrants draw z from its
invariant distribution. Employment lives on a grid of n_n points: 0, then n_n - 1
points spaced evenly in log n between n_min and n_max.
"""
Base.@kwdef struct Economy
    η::Float64 = 2 / 3
    β::Float64 = 0.8
    c_e::Float64 = 20.0
    c_f::Float64 = 20.0
    τ::Float64 = 0.1
    D̄::Float64 = 100.0
    ρ::Float64 = 0.9
    σ::Float64 = 0.2
    logz_mean::Float64 = 1.0
    n_z::Int = 101
    m::Float64 = 4.0
    n_n::Int = 500
    n_min::Float64 = 0.1
    n_max::Float64 = 8000.0
end
# <<< Economy

# >>> setup
"""
    setup(eco)

The objects that stay fixed while the price is updated: the productivity grid z,
the transition matrix F (F[i, j] = Pr(z_{t+1} = z_j | z_t = z_i)), the entrants'
probabilities g, and the employment grid n, with n[1] = 0 (where entrants start).
"""
function setup(eco::Economy)
    # the productivity grid and transition matrix, by Tauchen around the mean
    logz, F = tauchen(eco.n_z, eco.ρ, eco.σ; mean = eco.logz_mean, m = eco.m)
    # entrants draw z from the invariant distribution of the chain
    g = stationary(F)
    # employment: zero, then n_n - 1 points evenly spaced in log n
    n = [0.0; exp.(range(log(eco.n_min), log(eco.n_max); length = eco.n_n - 1))]
    return (; z = exp.(logz), logz, F, g, n)
end
# <<< setup

# >>> firing_cost
"""
    firing_cost(n_new, n_old, τ)

The firing tax τ max{0, m - n}: a firm that inherits m = n_old workers and chooses
n = n_new pays τ for each worker it lets go. Hiring is free.
"""
firing_cost(n_new, n_old, τ) = τ * max(0.0, n_old - n_new)
# <<< firing_cost
