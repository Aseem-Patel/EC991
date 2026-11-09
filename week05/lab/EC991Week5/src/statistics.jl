# >>> industry_stats
"""
    industry_stats(eq)

The equilibrium's statistics (tutorial slide 12). Prices and the entry margin:
the price p, the mass of entrants M, the exit threshold z̃. The industry: the
number of firms Σμ; the entry rate M / Σμ, which equals the exit rate in a
stationary equilibrium; output Y = Σ y μ, which equals demand D̄ / p; average
firm size in production workers; mean firm age. Labour, by use: production
Σ n μ, the fixed cost c_f Σμ paid by every producing firm, and entry c_e M.
"""
function industry_stats(eq::Equilibrium)
    (; μ, n, y, π, M, p) = eq
    (; c_e, c_f) = eq.ind
    firms = sum(μ)
    F̂ = exit_transition(eq.s.F, eq.χ)
    # mean age: Σ_a a (F̂')^a g / Σ_a (F̂')^a g, both sums in closed form
    A = I - F̂'
    x = A \ eq.s.g
    mean_age = sum(A \ (F̂' * x)) / sum(x)
    N_prod = sum(n .* μ)
    return (; p, M, z̃ = eq.z̃, firms, entry_rate = M / firms, exit_rate = sum(eq.χ .* μ) / firms,
            Y = sum(y .* μ), demand = eq.ind.D̄ / p, avg_size = N_prod / firms, mean_age,
            N_prod, N_fixed = c_f * firms, N_entry = c_e * M, N = N_prod + c_f * firms + c_e * M,
            profits = sum(π .* μ))
end
# <<< industry_stats

# >>> size_distribution
"""
    size_distribution(eq, edges)

Shares of firms and of employment by size class, size meaning production
workers n(z): bin k holds firms with edges[k] ≤ n < edges[k+1]. Returns the two
vectors of shares; each sums to one.
"""
function size_distribution(eq::Equilibrium, edges)
    firms, emp = zeros(length(edges) - 1), zeros(length(edges) - 1)
    for i in eachindex(eq.μ)
        k = searchsortedlast(edges, eq.n[i])
        1 <= k < length(edges) || continue
        firms[k] += eq.μ[i]
        emp[k] += eq.μ[i] * eq.n[i]
    end
    return (; firms = firms ./ sum(eq.μ), employment = emp ./ sum(eq.μ .* eq.n))
end
# <<< size_distribution

# >>> age_profile
"""
    age_profile(eq, A)

What happens to a cohort of entrants as it ages (lecture slides 44 and 45), for a = 0, ..., A:
`survival`, the share of the cohort still producing at age a; `avg_size`, mean
production workers of the survivors; `exit_rate`, the share of age-a firms that
exit at the end of the year.
"""
function age_profile(eq::Equilibrium, A)
    F̂ = exit_transition(eq.s.F, eq.χ)
    C = cohorts(F̂, eq.s.g, A)
    mass = vec(sum(C; dims = 1))
    return (; age = 0:A, survival = mass ./ mass[1], avg_size = vec(eq.n' * C) ./ mass,
            exit_rate = vec(eq.χ' * C) ./ mass)
end
# <<< age_profile
