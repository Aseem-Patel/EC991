# >>> percentile
"""
    percentile(a, λ_a, p)

The asset level below which a share p of households lie, from the marginal
distribution λ_a on the grid `a`, interpolating the cumulative distribution
between grid points. The first grid point if p is below the mass there.
"""
function percentile(a, λ_a, p)
    cdf = cumsum(λ_a)
    p <= cdf[1] && return a[1]
    k = searchsortedfirst(cdf, p) - 1
    k >= length(a) && return a[end]
    return a[k] + (p - cdf[k]) / (cdf[k+1] - cdf[k]) * (a[k+1] - a[k])
end
# <<< percentile

# >>> wealth_inequality
"""
    wealth_inequality(a, λ_a)

The Gini coefficient of wealth and the shares of it held by the top 10% and the
top 1%, on the grid `a` with masses λ_a. Top shares split the mass of the node
where the cut falls, so they do not jump with the grid.
"""
function wealth_inequality(a, λ_a)
    total = sum(a .* λ_a)
    F, S, area = 0.0, 0.0, 0.0                   # population, wealth, area under the Lorenz curve
    for i in eachindex(a)
        S1 = S + a[i] * λ_a[i] / total
        area += λ_a[i] * (S + S1) / 2
        F, S = F + λ_a[i], S1
    end
    function top(q)
        mass, wealth = 0.0, 0.0
        for i in reverse(eachindex(a))
            mass >= q && break
            take = min(λ_a[i], q - mass)
            mass += take
            wealth += take * a[i]
        end
        return wealth / total
    end
    return (gini = 1 - 2area, top10 = top(0.10), top1 = top(0.01))
end
# <<< wealth_inequality

# >>> model_stats
"""
    model_stats(eq)

The statistics of slide 32. `at_limit` is the mass at the first grid point; it
depends on the grid, because the lottery sends mass there
from just above. `constrained` is the mass of households whose policy actually
chooses a' = a_1.
"""
function model_stats(eq::Equilibrium)
    (; α) = eq.econ
    (; a_dist, N, y) = eq.s
    λ_a = vec(sum(eq.λ; dims = 2))
    Y = eq.K^α * N^(1 - α)
    mean_a = sum(a_dist .* λ_a)
    return (; r = eq.r, w = eq.w, K = eq.K, Ea = eq.Ea, N, K_N = eq.K / N, Y, A_Y = eq.Ea / Y,
            at_limit = λ_a[1], constrained = sum(eq.λ[eq.g .<= a_dist[1]]),
            mean_a, std_a = sqrt(sum(λ_a .* (a_dist .- mean_a) .^ 2)),
            p25 = percentile(a_dist, λ_a, 0.25), p50 = percentile(a_dist, λ_a, 0.50),
            p95 = percentile(a_dist, λ_a, 0.95), p99 = percentile(a_dist, λ_a, 0.99),
            wealth_inequality(a_dist, λ_a)...)
end
# <<< model_stats

# >>> mpc
"""
    mpc(sol, hh)

The marginal propensity to consume out of a windfall (slide 33), by finite
differences on the grid: MPC[i, j] = (c[i+1, j] - c[i, j]) / (R (a[i+1] - a[i])),
because a unit of extra assets today is R units of extra resources. It belongs
to the interval starting at a[i], and is one where the household consumes all of
any extra resources.
"""
function mpc(sol::Solution, hh::Household)
    return diff(sol.cpol; dims = 1) ./ (hh.R .* diff(sol.a))
end
# <<< mpc

# >>> conditional_density
"""
    conditional_density(λ, a, j)

The density of assets given income state j on the interior grid points: mass at
each node, divided by the income state's mass and by the width of the node's bin,
half the gap on each side. Dividing by the width matters on a growth-rate grid,
where the mass per node also reflects how far apart the nodes are. The first node
holds a point mass, not a density, so it is returned separately: `at_limit`, the
share of that income state at the first node.
"""
function conditional_density(λ, a, j)
    total = sum(view(λ, :, j))
    width = (a[3:end] .- a[1:end-2]) ./ 2
    return (a = a[2:end-1], density = λ[2:end-1, j] ./ (total .* width), at_limit = λ[1, j] / total)
end
# <<< conditional_density

# >>> joint_table
"""
    joint_table(λ, a, edges)

The joint distribution of wealth and income in a table: entry [b, j] is the mass
with edges[b] ≤ a < edges[b+1] and income state j. The columns sum to the stationary
distribution of income and everything sums to one.
"""
function joint_table(λ, a, edges)
    T = zeros(length(edges) - 1, size(λ, 2))
    for j in axes(λ, 2), i in eachindex(a)
        b = clamp(searchsortedlast(edges, a[i]), 1, length(edges) - 1)
        T[b, j] += λ[i, j]
    end
    return T
end
# <<< joint_table

# >>> panel_table
"""
    panel_table(panel, edges, n_y)

The same table as `joint_table`, from the households of `simulate_panel`.
"""
function panel_table(panel, edges, n_y)
    T = zeros(length(edges) - 1, n_y)
    for (x, j) in zip(panel.a, panel.j)
        T[clamp(searchsortedlast(edges, x), 1, length(edges) - 1), j] += 1 / length(panel.a)
    end
    return T
end
# <<< panel_table
