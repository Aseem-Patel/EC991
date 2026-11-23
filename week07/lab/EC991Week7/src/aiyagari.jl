# >>> Economy
"""
    Economy(; β, γ, ϕ, α, δ, ρ, σ, n_y, n_a, a_max, ν, Z, τ)

The standard Aiyagari economy of the sequence-space part (lecture slides 33 to
38): week 4's lab economy, with aggregate productivity Z in front of the
production function, Y = Z K^α N^(1-α), and a labour tax τ whose revenue is paid
back as a lump sum, T = τ w N (slide 37). The defaults are week 4's: β = 0.96,
γ = 2, ϕ = 0, α = 0.33, δ = 0.05; 7-state Rouwenhorst with ρ = 0.9, σ = 0.1;
300 asset points on [0, 250], growth rate ν = 0.025. One period is a year. In
the steady state Z = 1 and τ = 0, and the economy is exactly week 4's.
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
    Z::Float64 = 1.0
    τ::Float64 = 0.0
end
# <<< Economy

# >>> setup
"""
    setup(eco)

The objects that never change, in any steady state or along any transition: the
endowments y, their transition matrix P and stationary distribution π, labour
N = Σ_j y_j π_j, and the asset grid `a`, which the distribution also uses.
"""
function setup(eco::Economy)
    logy, P = rouwenhorst(eco.n_y, eco.ρ, eco.σ)
    y = exp.(logy)
    π = stationary(P)
    a = asset_grid(eco.n_a, eco.a_max; ϕ = eco.ϕ, ν = eco.ν)
    return (; y, P, π, N = sum(y .* π), a)
end
# <<< setup

# >>> prices
"""
    prices(Z, K, N, α, δ)

The firm's first-order conditions with productivity Z: r = α Z (K/N)^(α-1) - δ
and w = (1-α) Z (K/N)^α. Returns (R, w) with R = 1 + r.
"""
prices(Z, K, N, α, δ) = (1 + α * Z * (K / N)^(α - 1) - δ, (1 - α) * Z * (K / N)^α)
# <<< prices

# >>> income
"""
    income(w, τ, y, N)

Non-asset income at each endowment: labour income after the tax plus the lump
sum, (1 - τ) w y_j + τ w N (slide 37).
"""
income(w, τ, y, N) = w .* ((1 - τ) .* y .+ τ * N)
# <<< income

# >>> egm_backward!
"""
    egm_backward!(c, apol, c′, R′, R, inc, a, P, β, γ)

One step of EGM backwards in time. Given tomorrow's consumption policy c′ and
tomorrow's gross return R′, find today's policy when today's return is R and
today's incomes are inc[j]. For each choice a'_i and endowment y_j:

    u'(c̃) = β R′ Σ_m P[j, m] u'(c′[i, m]),    â = (c̃ + a'_i - inc[j]) / R,

then interpolate back onto the grid, with a' ≥ a_1. With R′ = R this is week 4's
`egm_step!`; along a transition R′ and R differ.
"""
function egm_backward!(c, apol, c′, R′, R, inc, a, P, β, γ)
    # read a[i] as tomorrow's assets: EMU[i, j] = Σ_m P[j, m] u'(c′[i, m])
    EMU = marginal_utility.(c′, γ) * P'
    â = similar(a)
    for j in axes(c, 2)
        # invert the Euler equation, then the budget constraint
        for i in eachindex(a)
            c̃ = inverse_marginal_utility(β * R′ * EMU[i, j], γ)
            â[i] = (c̃ + a[i] - inc[j]) / R
        end
        # back on the grid: a[i] now means today's assets
        ĝ = LinearInterp(â, a)
        for i in eachindex(a)
            apol[i, j] = max(ĝ(a[i]), a[1])
            c[i, j] = R * a[i] + inc[j] - apol[i, j]
        end
    end
    return c, apol
end
# <<< egm_backward!

# >>> stationary_distribution
"""
    stationary_distribution(apol, a, P; tol = 1e-13, maxiter = 100_000)

Week 4's histogram: iterate λ_{t+1} = Q' λ_t until it stops moving, starting
with everyone at the borrowing limit.
"""
function stationary_distribution(apol, a, P; tol = 1e-13, maxiter = 100_000)
    λ0 = zeros(size(apol))
    λ0[1, :] .= stationary(P)
    μ, λ1 = similar(λ0), similar(λ0)
    for n in 1:maxiter
        # assets move by the lottery, then income by P
        move_assets!(μ, λ0, apol, a)
        mul!(λ1, μ, P)
        maximum(abs.(λ1 .- λ0)) < tol && return λ1
        λ0, λ1 = λ1, λ0
    end
    error("the distribution did not converge in $maxiter iterations")
end
# <<< stationary_distribution

# >>> SteadyState
"""
A stationary equilibrium: the economy, the objects from `setup` in `s`, the
interest rate r, the wage w, capital demand K and asset supply Ea (equal up to
the tolerance), the consumption and saving policies, and the distribution λ.
"""
struct SteadyState
    eco::Economy
    s::NamedTuple
    r::Float64
    w::Float64
    K::Float64
    Ea::Float64
    c::Matrix{Float64}
    apol::Matrix{Float64}
    λ::Matrix{Float64}
end
# <<< SteadyState

# >>> solve_steady_state
"""
    solve_steady_state(eco = Economy(); tol = 1e-10)

Week 4's equilibrium loop with Z and τ: bisection on r between -δ and 1/β - 1.
At each r: capital demand K(r) = N (α Z / (r + δ))^(1/(1-α)), the wage, the
household's policy by EGM, the stationary distribution, and asset supply
Ea = Σ a λ. Stop when |Ea - K| < tol. The tolerances are tight because the
transitions start and end here: a steady state that is a little off would show
up as a drift in every transition.
"""
function solve_steady_state(eco::Economy = Economy(); tol = 1e-10)
    (; β, γ, α, δ, Z, τ) = eco
    s = setup(eco)
    (; y, P, N, a) = s
    c = [0.05 * ai + yj for ai in a, yj in y]          # a first guess for consumption
    c1, apol = similar(c), similar(c)
    lo, hi = -δ + 1e-3, 1 / β - 1 - 1e-5
    for step in 1:200
        r = (lo + hi) / 2
        # the firm at r, then the household at the prices r and w
        K = N * (α * Z / (r + δ))^(1 / (1 - α))
        R, w = prices(Z, K, N, α, δ)
        inc = income(w, τ, y, N)
        for n in 1:100_000
            egm_backward!(c1, apol, c, R, R, inc, a, P, β, γ)
            d = maximum(abs.(c1 .- c))
            c, c1 = c1, c
            d < 1e-12 && break
        end
        # the distribution and asset supply
        λ = stationary_distribution(apol, a, P)
        Ea = sum(a .* λ)
        if abs(Ea - K) < tol || hi - lo < 1e-14
            return SteadyState(eco, s, r, w, K, Ea, copy(c), copy(apol), λ)
        end
        # too many assets: r must fall
        Ea > K ? (hi = r) : (lo = r)
    end
    error("no steady state found")
end
# <<< solve_steady_state
