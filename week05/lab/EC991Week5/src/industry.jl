# >>> Industry
"""
    Industry(; η, β, c_e, c_f, D̄, ρ, σ, logz_mean, n_z, m, entrants, μ_g, σ_g)

The parameters of the dynamic model (lecture slide 41):
technology y = z n^η; firms discount by β; entry costs c_e and the per-period
fixed cost c_f, both in units of labour; demand D(p) = D̄ / p; log productivity
an AR(1), log z' - logz_mean = ρ (log z - logz_mean) + σ ε, discretised by
Tauchen with n_z points and width m. The wage is the numeraire, w = 1.

Entrants draw z from G. `entrants = :invariant` (the default) makes G the invariant distribution of the incumbents' chain;
`entrants = :lognormal` draws log z from N(μ_g, σ_g²) instead.
"""
Base.@kwdef struct Industry
    η::Float64 = 2 / 3
    β::Float64 = 0.8
    c_e::Float64 = 40.0
    c_f::Float64 = 20.0
    D̄::Float64 = 100.0
    ρ::Float64 = 0.9
    σ::Float64 = 0.2
    logz_mean::Float64 = 1.0
    n_z::Int = 101
    m::Float64 = 4.0
    entrants::Symbol = :invariant
    μ_g::Float64 = 0.0
    σ_g::Float64 = 0.2
end
# <<< Industry

# >>> setup
"""
    setup(ind)

The objects that stay fixed as the price is updated (slide 34): the productivity
grid z, the transition matrix F (F[i, j] = Pr(z' = z_j | z = z_i)) and the
entrants' probabilities g over the same grid.
"""
function setup(ind::Industry)
    logz, F = tauchen(ind.n_z, ind.ρ, ind.σ; mean = ind.logz_mean, m = ind.m)
    g = ind.entrants === :invariant ? stationary(F) :
        ind.entrants === :lognormal ? discretise_normal(logz, ind.μ_g, ind.σ_g) :
        throw(ArgumentError("entrants must be :invariant or :lognormal"))
    return (; z = exp.(logz), logz, F, g)
end
# <<< setup

# >>> static_decisions
"""
    static_decisions(p, z; η, c_f, w = 1.0)

The incumbent's static problem at price p (slide 26): π = max_n p z n^η - w n - w c_f.
The first-order condition η p z n^(η-1) = w gives

    n = (η p z / w)^(1/(1-η)),   y = z n^η,   π = p y - w n - w c_f.

Returns vectors n, y and π over the grid z.
"""
function static_decisions(p, z; η, c_f, w = 1.0)
    # labour from the first-order condition, at every grid point
    n = (η .* p .* z ./ w) .^ (1 / (1 - η))
    # output, and profit after the wage bill and the fixed cost
    y = z .* n .^ η
    π = p .* y .- w .* n .- w * c_f
    return (; n, y, π)
end
# <<< static_decisions
