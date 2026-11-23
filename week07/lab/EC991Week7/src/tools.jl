# >>> asset_grid
"""
    asset_grid(n_a, a_max; ϕ = 0.0, ν = 0.025)

The asset grid of week 4, from the borrowing limit a_1 = -ϕ to a_max, with
growth-rate spacing: each gap is (1 + ν) times the one before, so the points
bunch near the constraint, where the policy bends.
"""
function asset_grid(n_a, a_max; ϕ = 0.0, ν = 0.025)
    a_1 = -ϕ
    return [a_1 + (a_max - a_1) * ((1 + ν)^(i - 1) - 1) / ((1 + ν)^(n_a - 1) - 1) for i in 1:n_a]
end
# <<< asset_grid

# >>> LinearInterp
"""
    LinearInterp(x, v)

Linear interpolation through the points (x_k, v_k), x increasing (week 3).
Call it like a function. Outside [x_1, x_n] it extends the end segments.
"""
struct LinearInterp{T<:Real}
    x::Vector{T}
    v::Vector{T}
end
function (f::LinearInterp)(x)
    # the gap that holds x, or the end gap if x is outside the grid
    k = clamp(searchsortedlast(f.x, x), 1, length(f.x) - 1)
    return f.v[k] + (x - f.x[k]) / (f.x[k+1] - f.x[k]) * (f.v[k+1] - f.v[k])
end
# <<< LinearInterp

# >>> bracket
"""
    bracket(x, grid)

The index k and weight ω with x ≈ ω grid[k] + (1 - ω) grid[k+1]: linear
interpolation weights. x outside the grid is moved to the nearest end, so ω is
always in [0, 1].
"""
function bracket(x, grid)
    xc = clamp(x, grid[1], grid[end])
    k = clamp(searchsortedlast(grid, xc), 1, length(grid) - 1)
    return k, (grid[k+1] - xc) / (grid[k+1] - grid[k])
end
# <<< bracket

# >>> marginal_utility
"""
    marginal_utility(c, γ)  and  inverse_marginal_utility(m, γ)

CRRA marginal utility u'(c) = c^(-γ) and its inverse m^(-1/γ), as in week 4's
EGM. γ = 1 is log utility.
"""
marginal_utility(c, γ) = c^(-γ)
inverse_marginal_utility(m, γ) = m^(-1 / γ)
# <<< marginal_utility

# >>> young_weights
"""
    young_weights(a′, a)

Young's lottery from week 4: the index k with a[k] ≤ a′ ≤ a[k+1], and the
probability p = (a[k+1] - a′) / (a[k+1] - a[k]) of going to the point below, so
the mean of assets is preserved. A choice outside the grid goes to the nearest end.
"""
function young_weights(a′, a)
    a′ <= a[1] && return 1, 1.0
    a′ >= a[end] && return length(a) - 1, 0.0
    k = searchsortedlast(a, a′)
    return k, (a[k+1] - a′) / (a[k+1] - a[k])
end
# <<< young_weights

# >>> move_assets!
"""
    move_assets!(μ, λ, g, a)

The first half of λ_{t+1} = Q_t' λ_t: the mass λ[i, j] at (a_i, y_j) moves to the
two grid points around its choice g[i, j], by Young's lottery. Income has not
moved yet; the second half, μ times the income transition, is one matrix product.
"""
function move_assets!(μ, λ, g, a)
    fill!(μ, 0.0)
    for j in axes(λ, 2), i in axes(λ, 1)
        k, p = young_weights(g[i, j], a)
        # share p of the mass goes to the point below, the rest to the point above
        μ[k, j] += p * λ[i, j]
        μ[k+1, j] += (1 - p) * λ[i, j]
    end
    return μ
end
# <<< move_assets!

# >>> gini
"""
    gini(a, m)

The Gini coefficient of wealth when mass m[k] holds a[k], a increasing: one minus
twice the area under the Lorenz curve, by the trapezoid rule.
"""
function gini(a, m)
    m = m ./ sum(m)
    # cumulative population and cumulative wealth shares: the Lorenz curve
    F = cumsum(m)
    L = cumsum(a .* m) ./ sum(a .* m)
    area = sum(m .* (L .+ [0.0; L[1:end-1]]) ./ 2)
    return 1 - 2area
end
# <<< gini

# >>> top_share
"""
    top_share(a, m, x)

The share of wealth held by the richest fraction x of the population, when mass
m[k] holds a[k], a increasing. Mass is taken from the top of the grid down; the
last grid point is split if it straddles the cut.
"""
function top_share(a, m, x)
    m = m ./ sum(m)
    left, wealth = x, 0.0
    for k in length(a):-1:1
        take = min(m[k], left)
        wealth += take * a[k]
        left -= take
        left <= 0 && break
    end
    return wealth / sum(a .* m)
end
# <<< top_share
