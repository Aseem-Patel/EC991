# >>> backward_pass
"""
    backward_pass(K, Z, τ, ss1)

Steps (ii) and (iii) of the algorithm (slide 39). Prices at every date
from the guessed capital path, r_t + δ = α Z_t (K_t/N)^(α-1) and
w_t = (1-α) Z_t (K_t/N)^α. Then the households, backwards: at T they follow the
final steady state's policy, and for t = T-1, ..., 1 one EGM step gives the
policy at t from the policy at t + 1, with today's prices and tomorrow's return.
Returns the policies apol[:, :, t] and c[:, :, t], and the prices.
"""
function backward_pass(K, Z, τ, ss1::SteadyState)
    (; β, γ, α, δ) = ss1.eco
    (; y, P, N, a) = ss1.s
    T = length(K)
    pr = [prices(Z[t], K[t], N, α, δ) for t in 1:T]
    R, w = first.(pr), last.(pr)
    c = zeros(length(a), length(y), T)
    apol = similar(c)
    # date T: the final steady state
    c[:, :, T] .= ss1.c
    apol[:, :, T] .= ss1.apol
    # dates T-1, ..., 1: V_T → V_{T-1} → ... → V_1, one EGM step at a time
    for t in T-1:-1:1
        @views egm_backward!(c[:, :, t], apol[:, :, t], c[:, :, t+1], R[t+1], R[t],
                             income(w[t], τ[t], y, N), a, P, β, γ)
    end
    return (; c, apol, R, w)
end
# <<< backward_pass

# >>> forward_pass
"""
    forward_pass(apol, ss0)

Steps (iv) and (v) (slide 40). Start from the initial steady state's
distribution, λ_1 = λ^{initial ss}, and move it forward with each date's policy,
λ_{t+1} = Q_t' λ_t. The capital households supply at each date is the mean of the
distribution, K^s_t = Σ_j a_j λ_{t,j}. Returns K^s and the distributions.
"""
function forward_pass(apol, ss0::SteadyState)
    (; P, a) = ss0.s
    T = size(apol, 3)
    λ = zeros(size(apol))
    λ[:, :, 1] .= ss0.λ
    μ = similar(ss0.λ)
    for t in 1:T-1
        # assets move by the lottery, then income by P
        @views move_assets!(μ, λ[:, :, t], apol[:, :, t], a)
        @views mul!(λ[:, :, t+1], μ, P)
    end
    Ks = [sum(a .* view(λ, :, :, t)) for t in 1:T]
    return Ks, λ
end
# <<< forward_pass

# >>> Transition
"""
A perfect-foresight transition from `ss0` to `ss1`: the paths of productivity Z,
the tax τ, capital K (the guess that clears the market), the capital households
supply Ks, the interest rate r, the wage w, output Y, consumption C, the Gini of
wealth for everyone and within the lowest and highest endowments, the share of
households at the borrowing constraint, the number of iterations and the
distance at each one. `end_gap` = K^s_T - K_T is slide 41's check on the last
transition: if T is long enough, the capital households supply at T is the
final steady state's.
"""
struct Transition
    ss0::SteadyState
    ss1::SteadyState
    Z::Vector{Float64}
    τ::Vector{Float64}
    K::Vector{Float64}
    Ks::Vector{Float64}
    r::Vector{Float64}
    w::Vector{Float64}
    Y::Vector{Float64}
    C::Vector{Float64}
    gini::Vector{Float64}
    gini_low::Vector{Float64}
    gini_high::Vector{Float64}
    constrained::Vector{Float64}
    iterations::Int
    history::Vector{Float64}
    end_gap::Float64
end
# <<< Transition

# >>> solve_transition
"""
    solve_transition(ss0, ss1, Z, τ; ω = 0.2, tol = 1e-8, maxiter = 2_000, K_start = nothing,
                     callback = (n, d) -> nothing)

The shooting algorithm of slides 39 and 40, for given paths of productivity Z_t
and the tax τ_t, t = 1, ..., T. Households learn the paths at t = 1 (an MIT shock
or an unexpected reform) and expect them for sure.

  (i)   the steady states ss0 and ss1 are given; T = length(Z)
  (ii)  guess {K_t}: K_1 is fixed by the initial distribution and K_T at the
        final steady state; the first guess is K_t = K_ss1 at every other date
  (iii) households backwards (`backward_pass`)
  (iv)  the distribution forwards, and (v) the capital it implies (`forward_pass`)
  (vi)  stop if max_{2 ≤ t ≤ T-1} |K^s_t - K^g_t| < tol; otherwise update with
        dampening, K^g ← ω K^s + (1 - ω) K^g, and go back to (ii)
"""
function solve_transition(ss0::SteadyState, ss1::SteadyState, Z, τ; ω = 0.2, tol = 1e-8,
                          maxiter = 2_000, K_start = nothing, callback = (n, d) -> nothing)
    (; α, δ) = ss1.eco
    (; y, N, a) = ss1.s
    T = length(Z)
    # (ii) the first guess; the two ends never change
    K = K_start === nothing ? fill(ss1.K, T) : copy(K_start)
    K[1], K[T] = ss0.Ea, ss1.K
    history = Float64[]
    for n in 1:maxiter
        # (iii) households backwards, (iv) and (v) the distribution forwards
        bw = backward_pass(K, Z, τ, ss1)
        Ks, λ = forward_pass(bw.apol, ss0)
        # (vi) compare implied with guessed capital
        d = maximum(abs.(Ks[2:T-1] .- K[2:T-1]))
        push!(history, d)
        callback(n, d)
        if d < tol
            return transition_paths(ss0, ss1, Z, τ, K, Ks, bw, λ, n, history)
        end
        K[2:T-1] .= ω .* Ks[2:T-1] .+ (1 - ω) .* K[2:T-1]
    end
    error("the transition did not converge in $maxiter iterations")
end
# <<< solve_transition

# >>> transition_paths
"""
    transition_paths(ss0, ss1, Z, τ, K, Ks, bw, λ, n, history)

Everything that is reported along a converged transition: output
Y_t = Z_t K_t^α N^(1-α), consumption C_t = Σ c_t λ_t, the Gini of wealth for
everyone, for the lowest and for the highest endowment (lecture slide 36), and
the share of households who choose a' = a_1, the borrowing limit.
"""
function transition_paths(ss0, ss1, Z, τ, K, Ks, bw, λ, n, history)
    (; α) = ss1.eco
    (; N, a) = ss1.s
    T = length(Z)
    n_y = size(λ, 2)
    Y = Z .* K .^ α .* N^(1 - α)
    C = [sum(view(bw.c, :, :, t) .* view(λ, :, :, t)) for t in 1:T]
    g_all = [gini(a, vec(sum(view(λ, :, :, t); dims = 2))) for t in 1:T]
    g_low = [gini(a, λ[:, 1, t]) for t in 1:T]
    g_high = [gini(a, λ[:, n_y, t]) for t in 1:T]
    constrained = [sum(view(λ, :, :, t)[view(bw.apol, :, :, t) .<= a[1]]) for t in 1:T]
    return Transition(ss0, ss1, collect(Z), collect(τ), K, Ks, bw.R .- 1, bw.w, Y, C,
                      g_all, g_low, g_high, constrained, n, history, Ks[T] - K[T])
end
# <<< transition_paths

# >>> mit_shock
"""
    mit_shock(ss; ε = 0.01, ρ_Z = 0.9, T = 300, kwargs...)

The MIT shock of slide 33. The economy is in its steady state at t = 0; at
t = 1 productivity jumps unexpectedly, log Z_1 = ε, and then returns,
log Z_t = ρ_Z log Z_{t-1}. Nobody expected the shock, and nobody expects
another. The transition starts and ends in the same steady state.
"""
function mit_shock(ss::SteadyState; ε = 0.01, ρ_Z = 0.9, T = 300, kwargs...)
    Z = ss.eco.Z .* exp.(ε .* ρ_Z .^ (0:T-1))
    τ = fill(ss.eco.τ, T)
    return solve_transition(ss, ss, Z, τ; kwargs...)
end
# <<< mit_shock

# >>> tax_reform
"""
    tax_reform(ss0, ss1; T = 300, kwargs...)

The reform of slides 37 and 38: the economy is in the steady state ss0 (τ = 0)
and at t = 1 the tax rises, unexpectedly and for ever, to ss1's (τ = 0.2).
"""
function tax_reform(ss0::SteadyState, ss1::SteadyState; T = 300, kwargs...)
    Z = fill(ss1.eco.Z, T)
    τ = fill(ss1.eco.τ, T)
    return solve_transition(ss0, ss1, Z, τ; kwargs...)
end
# <<< tax_reform
