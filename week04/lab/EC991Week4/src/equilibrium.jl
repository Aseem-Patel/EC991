# >>> Economy
"""
    Economy(; β, γ, ϕ, α, δ, ρ, σ, n_y, n_a, a_max, ν, n_dist = n_a, ν_dist = ν)

Everything fixed before the interest rate loop (slide 28): preferences, the firm,
the AR(1) for log income (log y' = ρ log y + σ ε, Rouwenhorst with n_y states) and
the two asset grids. The defaults are the lab grid, 300 × 7 on [0, 250], where the
distribution uses the policy grid. `dense_economy()` is the grid behind the
lecture's figures.
"""
Base.@kwdef struct Economy
    β::Float64 = 0.96
    γ::Float64 = 2.0
    ϕ::Float64 = 0.0
    α::Float64 = 0.33
    δ::Float64 = 0.05
    ρ::Float64 = 0.9
    σ::Float64 = 0.1
    n_y::Int = 7
    n_a::Int = 300
    a_max::Float64 = 250.0
    ν::Float64 = 0.025
    n_dist::Int = n_a
    ν_dist::Float64 = ν
end
dense_economy(; kwargs...) = Economy(; n_a = 1000, ν = 0.008, n_dist = 5000, ν_dist = 0.0016, kwargs...)
# <<< Economy

# >>> setup
"""
    setup(econ)

The objects that stay fixed as r is updated (slide 28): the income states y, their
transition matrix P and stationary distribution π, labour supply N = Σ_j y_j π_j,
the policy grid `a` and the distribution grid `a_dist`.
"""
function setup(econ::Economy)
    logy, P = rouwenhorst(econ.n_y, econ.ρ, econ.σ)
    y = exp.(logy)
    π = stationary(P)
    a = asset_grid(econ.n_a, econ.a_max; ϕ = econ.ϕ, ν = econ.ν)
    a_dist = econ.n_dist == econ.n_a && econ.ν_dist == econ.ν ? a :
             asset_grid(econ.n_dist, econ.a_max; ϕ = econ.ϕ, ν = econ.ν_dist)
    return (; y, P, π, N = sum(y .* π), a, a_dist)
end
# <<< setup

# >>> capital_demand
"""
    capital_demand(r, N; α = 0.33, δ = 0.05)

Step 2 (slide 28): capital demand K(r) = (α / (r + δ))^(1/(1-α)) N from the firm's
first-order condition. The wage that goes with it is `w_from_r(r; α, δ)`.
"""
capital_demand(r, N; α = 0.33, δ = 0.05) = (α / (r + δ))^(1 / (1 - α)) * N
# <<< capital_demand

# >>> solve_household
"""
    solve_household(hh, s; method = :egm, n_H = 50, start = nothing)

Step 3 (slide 29), the one thing the methods change. Every method returns a
`Solution` holding the saving policy, which is all Step 4 needs.

  :vfi_grid           VFI by grid search, no interpolation (week 3)
  :vfi_fast           grid search with monotonicity, concavity and Howard's updates
  :vfi_interp         VFI by interpolating V and optimising with Brent (week 3)
  :vfi_interp_howard  the same with Howard's updates, once the distance is below 0.01
  :egm                the endogenous grid method

`start` is an earlier `Solution`: the new solve starts from its V or its c.
"""
function solve_household(hh::Household, s; method = :egm, n_H = 50, start = nothing)
    (; a, y, P) = s
    V_start = start === nothing ? nothing : start.V
    method === :egm && return solve_egm(hh, a, y, P; c_start = start === nothing ? nothing : start.cpol)
    method === :vfi_grid && return solve_vfi(hh, a, y, P; V_start)
    method === :vfi_fast && return solve_vfi(hh, a, y, P; step = bellman_step_fast!, n_H, V_start)
    method === :vfi_interp && return solve_vfi(hh, a, y, P; step = bellman_step_interp!, V_start)
    method === :vfi_interp_howard && return solve_vfi(hh, a, y, P; step = bellman_step_interp!, n_H, howard_from = 0.01, V_start)
    throw(ArgumentError("unknown method $method"))
end
# <<< solve_household

# >>> excess_demand
"""
    excess_demand(r, econ, s; method = :egm, start = nothing, λ_start = nothing)

Steps 2 to 5 at one interest rate (slides 28 to 30): prices, the household problem,
the stationary distribution on `a_dist`, and Φ(r) = E a(r) - K(r), asset supply
minus capital demand. Returns all of them, and whether the distribution converged.
"""
function excess_demand(r, econ::Economy, s; method = :egm, start = nothing, λ_start = nothing, n_H = 50)
    (; β, γ, ϕ, α, δ) = econ
    K = capital_demand(r, s.N; α, δ)                                # Step 2
    w = w_from_r(r; α, δ)
    # the household at these prices
    hh = Household(; β, γ, ϕ, r, w)
    sol = solve_household(hh, s; method, start, n_H)                # Step 3
    g = policy_on_grid(sol, s.a_dist)                               # Step 4
    # the stationary distribution under this policy
    dist = stationary_distribution(g, s.a_dist, s.P; λ_start)
    λ = dist.λ
    # asset supply: each asset level times the mass at that level
    Ea = sum(s.a_dist .* sum(λ; dims = 2))                          # Step 5
    return (; Φ = Ea - K, Ea, K, w, r, hh, sol, g, λ, converged = dist.converged)
end
# <<< excess_demand

# >>> Equilibrium
"""
The equilibrium found by `solve_equilibrium`: the prices and aggregates, the
household `Household`, its `Solution`, the policy `g` and distribution `λ` on the
distribution grid, the objects from `setup` in `s`, and `history`, the (r, Φ) tried
at every step, in order.
"""
struct Equilibrium
    econ::Economy
    method::Symbol
    r::Float64
    w::Float64
    K::Float64
    Ea::Float64
    Φ::Float64
    hh::Household{Float64}
    sol::Solution{Float64}
    g::Matrix{Float64}
    λ::Matrix{Float64}
    s::NamedTuple
    history::Vector{Tuple{Float64, Float64}}
end
# <<< Equilibrium

# >>> solve_equilibrium
"""
    solve_equilibrium(econ = Economy(); method = :egm, bracket, tol_Φ = 1e-3, tol_r = 1e-6,
                      warm_start = true, n_H = 50, callback = (r, Φ) -> nothing)

The equilibrium interest rate by bisection (slide 30, Step 6). The initial bracket
is the theory's (backup slide): just above -δ and just below 1/β - 1. If Φ(r) > 0,
assets exceed capital demand, so r falls; if Φ(r) < 0 it rises. Stop when
|Φ| < `tol_Φ` or the bracket is narrower than `tol_r`. The second test matters for
grid search, whose Φ(r) moves in jumps and may never come that close to zero.
`warm_start` starts each solve from the previous r's policy and, off the grid, its
distribution. A policy on the grid does not reuse the distribution, because mass
left on an absorbing state would stay there (see `absorbing_states`). Far above the
equilibrium a grid-search distribution may never settle as that mass leaks away;
Φ still has the right sign there, so the search goes on, but the final r must
have converged. `callback` is called at every r, so a script can print as it goes.
"""
function solve_equilibrium(econ::Economy = Economy(); method = :egm,
                           bracket = (-econ.δ + 0.001, 1 / econ.β - 1 - 0.001),
                           tol_Φ = 1e-3, tol_r = 1e-6, warm_start = true, n_H = 50,
                           callback = (r, Φ) -> nothing)
    s = setup(econ)
    history = Tuple{Float64, Float64}[]
    last = Ref{Any}(nothing)
    function evaluate(r)
        prev = warm_start ? last[] : nothing
        e = excess_demand(r, econ, s; method, n_H, start = prev === nothing ? nothing : prev.sol,
                          λ_start = prev === nothing || prev.sol.pol !== nothing ? nothing : prev.λ)
        last[] = e
        push!(history, (r, e.Φ))
        callback(r, e.Φ)
        return e
    end
    # the two ends of the bracket for r
    lo, hi = bracket
    # solve the whole model at both ends
    elo, ehi = evaluate(lo), evaluate(hi)
    # stop if the equilibrium is not inside the bracket
    elo.Φ < 0 < ehi.Φ || error("Φ does not change sign on the bracket: Φ($lo) = $(elo.Φ), Φ($hi) = $(ehi.Φ)")
    # keep the r with the smallest |Φ| so far
    best = abs(elo.Φ) < abs(ehi.Φ) ? elo : ehi
    # halve the bracket until either stopping rule is met
    while hi - lo > tol_r && abs(best.Φ) > tol_Φ
        # solve the whole model at the midpoint
        e = evaluate((lo + hi) / 2)
        abs(e.Φ) < abs(best.Φ) && (best = e)
        # Φ > 0: too many assets, so r must fall. Otherwise it must rise
        e.Φ > 0 ? (hi = e.r) : (lo = e.r)
    end
    best.converged || error("the distribution did not converge at the equilibrium r = $(best.r)")
    return Equilibrium(econ, method, best.r, best.w, best.K, best.Ea, best.Φ, best.hh, best.sol,
                       best.g, best.λ, s, history)
end
# <<< solve_equilibrium
