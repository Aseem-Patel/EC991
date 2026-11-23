# >>> regress_law_of_motion
"""
    regress_law_of_motion(K, zpath, T_burn)

The regression of slides 15 and 16, run separately for each aggregate state:

    log K_{t+1} = θ_{0Z} + θ_{1Z} log K_t + ε_{t+1},   over the dates t with Z_t = Z,

on the simulated path after the first T_burn quarters. Ordinary least squares
with one regressor: the slope is cov(x, y) / var(x). Returns the coefficients θ
(2 × 2, column z is (θ_{0Z}, θ_{1Z})), the R² and the standard deviation of the
residuals for each Z.
"""
function regress_law_of_motion(K, zpath, T_burn)
    θ, R2, σ = zeros(2, 2), zeros(2), zeros(2)
    for z in 1:2
        # the dates in state z, and the pairs (log K_t, log K_{t+1})
        ts = [t for t in T_burn+1:length(K)-1 if zpath[t] == z]
        x, yv = log.(K[ts]), log.(K[ts .+ 1])
        x̄, ȳ = sum(x) / length(x), sum(yv) / length(yv)
        slope = sum((x .- x̄) .* (yv .- ȳ)) / sum((x .- x̄) .^ 2)
        θ[:, z] = [ȳ - slope * x̄, slope]
        resid = yv .- θ[1, z] .- slope .* x
        R2[z] = 1 - sum(resid .^ 2) / sum((yv .- ȳ) .^ 2)
        σ[z] = sqrt(sum(resid .^ 2) / (length(resid) - 2))
    end
    return θ, R2, σ
end
# <<< regress_law_of_motion

# >>> KSEquilibrium
"""
The approximate equilibrium found by `solve_krusell_smith`: the economy and the
objects from `ks_setup`, the forecasting rule θ with its R² and residual standard
deviations σ, the household's solution, the simulated shocks `zpath` and capital
`K`, the last distribution λ, and `history`, the distance at each iteration.
"""
struct KSEquilibrium
    eco::KSEconomy
    s::NamedTuple
    θ::Matrix{Float64}
    R2::Vector{Float64}
    σ::Vector{Float64}
    sol::KSSolution
    zpath::Vector{Int}
    K::Vector{Float64}
    λ::Matrix{Float64}
    history::Vector{Float64}
end
# <<< KSEquilibrium

# >>> solve_krusell_smith
"""
    solve_krusell_smith(eco = KSEconomy(); θ_start, ω = 0.3, tol = 1e-6, maxiter = 200,
                        callback = (n, θ, d) -> nothing)

The Krusell-Smith algorithm of slides 15 and 16:

1. Guess θ. The default is slide 17's: θ_{0Z} = log K_ss, θ_{1Z} = 0, with
   K_ss the representative agent's steady state.
2. Solve the household problem given the rule (`solve_ks_household`).
3. Simulate the economy (`simulate_capital`), with the same shocks every time.
4. Regress log K_{t+1} on log K_t for each Z (`regress_law_of_motion`).
5. Stop if the new coefficients are within `tol` of the guess. Otherwise update
   with dampening, θ ← ω θ_new + (1 - ω) θ, and go back to step 2.

Each household solve starts from the last one's policy. `callback` is called
after every iteration, so a script can print as it goes.
"""
function solve_krusell_smith(eco::KSEconomy = KSEconomy(); θ_start = nothing, ω = 0.3, tol = 1e-6,
                             maxiter = 200, callback = (n, θ, d) -> nothing)
    s = ks_setup(eco)
    zpath = simulate_Z(eco, s)
    # 1. the first guess: K stays at the representative agent's steady state
    θ = θ_start === nothing ? [log(s.K_ra) log(s.K_ra); 0.0 0.0] : copy(θ_start)
    # every simulation starts with all households at that steady state
    λ_start = initial_distribution(s.K_ra, zpath[1], eco, s)
    sol, history = nothing, Float64[]
    for n in 1:maxiter
        # 2. the household problem, given the rule
        sol = solve_ks_household(θ, eco, s; c_start = sol === nothing ? nothing : sol.c)
        # 3. simulate, 4. regress
        sim = simulate_capital(sol, zpath, eco, s; λ_start)
        θ_new, R2, σ = regress_law_of_motion(sim.K, zpath, eco.T_burn)
        # 5. compare, then update
        d = maximum(abs.(θ_new .- θ))
        push!(history, d)
        callback(n, θ_new, d)
        if d < tol
            return KSEquilibrium(eco, s, θ_new, R2, σ, sol, zpath, sim.K, sim.λ, history)
        end
        θ = ω .* θ_new .+ (1 - ω) .* θ
    end
    error("the forecasting rule did not converge in $maxiter iterations")
end
# <<< solve_krusell_smith

# >>> forecast_errors
"""
    forecast_errors(eq; horizon = 100)

How good is the rule? Two checks after the burn-in (slide 23, and Den Haan 2010):

  dynamic      start from the actual K once and iterate the rule forward for the
               whole sample, never looking at the actual K again. The rule only
               sees the shocks. Reports the mean and the maximum gap, in %.
  at horizon   start from the actual K at every date and iterate `horizon`
               quarters (100 quarters = 25 years); the gap at the end, in %.

The one-step-ahead errors are the regression residuals, σ.
"""
function forecast_errors(eq::KSEquilibrium; horizon = 100)
    (; K, zpath, θ) = eq
    T0, T = eq.eco.T_burn + 1, length(K)
    # dynamic: the rule iterated from one starting point
    K̂ = zeros(T)
    K̂[T0] = K[T0]
    for t in T0:T-1
        K̂[t+1] = forecast_K(K̂[t], zpath[t], θ)
    end
    dyn = 100 .* abs.(log.(K̂[T0:T]) .- log.(K[T0:T]))
    # at a fixed horizon, from every starting date
    hor = Float64[]
    for t0 in T0:T-horizon
        k̂ = K[t0]
        for t in t0:t0+horizon-1
            k̂ = forecast_K(k̂, zpath[t], θ)
        end
        push!(hor, 100 * abs(log(k̂) - log(K[t0+horizon])))
    end
    return (; dynamic_mean = sum(dyn) / length(dyn), dynamic_max = maximum(dyn),
            horizon, horizon_mean = sum(hor) / length(hor), horizon_max = maximum(hor), K̂)
end
# <<< forecast_errors
