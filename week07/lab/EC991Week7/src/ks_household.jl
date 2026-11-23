# >>> forecast_K
"""
    forecast_K(K, z, θ)

The perceived law of motion (slides 13 and 14): log K' = θ[1, z] + θ[2, z] log K,
with θ[1, z] = θ_{0Z} and θ[2, z] = θ_{1Z} for Z = Z_z.
"""
forecast_K(K, z, θ) = exp(θ[1, z] + θ[2, z] * log(K))
# <<< forecast_K

# >>> KSSolution
"""
The household's solution at one forecasting rule θ: consumption `c[i, j, z, k]`
and saving `apol[i, j, z, k]` at assets a_i, endowment y_j, aggregate state Z_z
and aggregate capital K_k, and the number of EGM iterations.
"""
struct KSSolution
    c::Array{Float64, 4}
    apol::Array{Float64, 4}
    iterations::Int
end
# <<< KSSolution

# >>> ks_egm_step!
"""
    ks_egm_step!(c1, apol, c0, θ, eco, s)

One EGM update of the consumption policy (week 4's EGM with two more states).
At every aggregate state (Z_z, K_k):

1. prices today, R and w, from Z_z and K_k;
2. tomorrow's capital from the rule, K' = forecast_K(K_k, z, θ), and tomorrow's
   return R' in each Z'; tomorrow's consumption c0(a', y', Z', K') is read off
   the K grid by linear interpolation;
3. for each y and each choice a'_i on the grid, the Euler equation (slide 10)

       u'(c) = β Σ_{y',Z'} P(y', Z' | y, Z) R' u'(c0(a'_i, y', Z', K')),

   inverted for c, then the budget constraint for today's assets â;
4. the pairs (â, a'_i) interpolated back onto the grid, with a' ≥ 0.
"""
function ks_egm_step!(c1, apol, c0, θ, eco::KSEconomy, s)
    (; β, γ, Z, y) = eco
    (; a, Kgrid, N, PZ, Py) = s
    EMU = zeros(length(a))
    â = similar(a)
    for k in eachindex(Kgrid), z in 1:2
        # 1. prices today
        R, w = ks_prices(Z[z], Kgrid[k], N[z], eco)
        # 2. tomorrow's capital, from the perceived law of motion, and where it sits on the K grid
        K′ = forecast_K(Kgrid[k], z, θ)
        kk, ωK = bracket(K′, Kgrid)
        for j in 1:2
            # 3. expected marginal utility times the return, for every choice a'_i
            fill!(EMU, 0.0)
            for z′ in 1:2
                R′, _ = ks_prices(Z[z′], K′, N[z′], eco)
                for j′ in 1:2
                    prob = PZ[z, z′] * Py[j, j′, z, z′]
                    for i in eachindex(a)
                        # tomorrow's consumption at K', between two points of the K grid
                        c′ = ωK * c0[i, j′, z′, kk] + (1 - ωK) * c0[i, j′, z′, kk+1]
                        EMU[i] += prob * R′ * marginal_utility(c′, γ)
                    end
                end
            end
            # invert the Euler equation, then the budget constraint: c + a' = R a + w y
            for i in eachindex(a)
                c̃ = inverse_marginal_utility(β * EMU[i], γ)
                â[i] = (c̃ + a[i] - w * y[j]) / R
            end
            # 4. back on the grid: a[i] now means today's assets
            ĝ = LinearInterp(â, a)
            for i in eachindex(a)
                apol[i, j, z, k] = max(ĝ(a[i]), a[1])
                c1[i, j, z, k] = R * a[i] + w * y[j] - apol[i, j, z, k]
            end
        end
    end
    return c1, apol
end
# <<< ks_egm_step!

# >>> solve_ks_household
"""
    solve_ks_household(θ, eco, s; c_start = nothing, tol = 1e-8, maxiter = 10_000)

Iterate `ks_egm_step!` until max |c^{n+1} - c^n| < tol. The first guess is
`c_start` or, failing that, consuming the interest and the wage income.
"""
function solve_ks_household(θ, eco::KSEconomy, s; c_start = nothing, tol = 1e-8, maxiter = 10_000)
    (; Z, y) = eco
    (; a, Kgrid, N) = s
    dims = (length(a), 2, 2, length(Kgrid))
    if c_start === nothing
        c0 = zeros(dims)
        for k in eachindex(Kgrid), z in 1:2, j in 1:2, i in eachindex(a)
            R, w = ks_prices(Z[z], Kgrid[k], N[z], eco)
            c0[i, j, z, k] = (R - 1) * a[i] + w * y[j]
        end
    else
        c0 = copy(c_start)
    end
    c1, apol = similar(c0), similar(c0)
    for n in 1:maxiter
        ks_egm_step!(c1, apol, c0, θ, eco, s)
        d = maximum(abs.(c1 .- c0))
        d < tol && return KSSolution(copy(c1), copy(apol), n)
        c0, c1 = c1, c0
    end
    error("the household problem did not converge in $maxiter iterations")
end
# <<< solve_ks_household

# >>> individual_rule
"""
    individual_rule(sol, s, j, z, K)

The saving rule a' = g_a(a, y_j; Z_z, K) on the asset grid, at any aggregate
capital K (slide 25): linear interpolation between the two nearest points of the
K grid.
"""
function individual_rule(sol::KSSolution, s, j, z, K)
    k, ωK = bracket(K, s.Kgrid)
    return ωK .* sol.apol[:, j, z, k] .+ (1 - ωK) .* sol.apol[:, j, z, k+1]
end
# <<< individual_rule
