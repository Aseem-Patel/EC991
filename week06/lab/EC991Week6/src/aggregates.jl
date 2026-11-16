# >>> aggregates
"""
    aggregates(eq)

The equilibrium's aggregates (tutorial slide 5), with m = n_k the employment a
firm inherits and n = n^d(z_i, n_k) the employment it chooses:

    Y   = Σ μ(z, m) (z n^η - c_f)                     output, net of fixed costs
    N^d = Σ μ(z, m) n + M c_e                         labour demand: production and entry
    T   = Σ μ(z, m) τ (max{0, m - n} + χ(z, m) n)     firing taxes, rebated to the household
    Π   = p Y - N^d - T                               profits net of entry costs
    C   = D̄ / p,   N^s = D̄ - Π - T                    the household

Goods market clearing, Y = C, holds by construction of M; labour market clearing,
N^s = N^d, then follows from the household's budget. Also: the number of firms B,
entry and exit rates, average size and productivity, and job flows. Job creation
is the sum of hires; job destruction the sum of separations, by continuing firms
and by exiting firms (which fire all their n workers). In a stationary
equilibrium the two are equal.
"""
function aggregates(eq::Equilibrium)
    (; μ, M, p, s, sol, eco) = eq
    (; η, c_f, c_e, τ, D̄) = eco
    # every state (z_i, n_k) as a matrix: inherited m, chosen n, productivity z
    n_old = repeat(s.n', length(s.z))              # m = n_k, inherited employment
    n_new = s.n[sol.pol]                           # n, chosen employment
    z = repeat(s.z, 1, length(s.n))
    # the number of firms, and output net of fixed costs
    B = sum(μ)
    Y = sum(μ .* (z .* n_new .^ η .- c_f))
    # labour demand: workers in production, and workers who create the entrants
    N_prod = sum(μ .* n_new)
    N_entry = c_e * M
    Nd = N_prod + N_entry
    # firing taxes: on workers let go by continuing firms, and on all n at firms that exit
    T = τ * sum(μ .* (max.(0.0, n_old .- n_new) .+ sol.χ .* n_new))
    Π = p * Y - Nd - T
    # the household: demand, and the labour supply its budget implies
    C = D̄ / p
    Ns = D̄ - Π - T
    # job flows: hires, and separations (including every worker at an exiting firm)
    JC = sum(μ .* max.(0.0, n_new .- n_old))
    JD = sum(μ .* (max.(0.0, n_old .- n_new) .+ sol.χ .* n_new))
    return (; p, M, B, Y, C, Nd, Ns, N_prod, N_entry, T, Π,
            entry_rate = M / B, exit_rate = sum(μ .* sol.χ) / B,
            avg_size = N_prod / B, avg_z = sum(μ .* z) / B,
            job_creation = JC / N_prod, job_destruction = JD / N_prod)
end
# <<< aggregates

# >>> misallocation
"""
    misallocation(eq)

How far the allocation of labour is from the efficient one (tutorial slide 16).

Without firing costs every firm hires until p z f'(n) = 1: the marginal product
of labour equals the real wage 1/p everywhere. With them, firms inside the
inaction region keep n = m, and the marginal products spread out. Two measures:

`mpl_gap`: the average over firms of

    |z f'(n) - 1/p| / (1/p) × 100 = |p η z n^(η-1) - 1| × 100,   n = n^d(z, m),

with weights μ(z, m), over firms with n > 0 (at n = 0 the marginal product is
infinite).
`mpl_gap_emp` weights each firm by its employment instead.

`output_lost`: week 5's Ω, applied to the equilibrium. Take the same firms and the
same production labour N = Σ μ(z, m) n, reallocate the workers so that marginal
products are equal, and compare output:

    Ω = Σ μ(z, m) z n^η / (Z N^η),   Z = (Σ μ(z, m) z^{1/(1-η)})^{1-η},   output_lost = 100 (1 - Ω).

`share_inactive`: the share of firms (n > 0) that keep their inherited employment, n = m.
"""
function misallocation(eq::Equilibrium)
    (; μ, p, s, sol, eco) = eq
    η = eco.η
    n_new = s.n[sol.pol]                               # chosen n at every state
    z = repeat(s.z, 1, length(s.n))
    # the MPL gap, over firms with n > 0 (at n = 0 the marginal product is infinite)
    pos = n_new .> 0
    gap = abs.(p .* η .* z[pos] .* n_new[pos] .^ (η - 1) .- 1) .* 100
    mpl_gap = sum(μ[pos] .* gap) / sum(μ[pos])
    mpl_gap_emp = sum(μ[pos] .* n_new[pos] .* gap) / sum(μ[pos] .* n_new[pos])
    # Ω: actual output over Z N^η, the output of the same firms and workers allocated efficiently
    N = sum(μ .* n_new)
    Z = sum(μ .* z .^ (1 / (1 - η)))^(1 - η)
    Ω = sum(μ .* z .* n_new .^ η) / (Z * N^η)
    # inactive: firms whose chosen n is the n they inherited
    stays = [sol.pol[i, k] == k for i in axes(μ, 1), k in axes(μ, 2)]
    share_inactive = sum(μ[pos .& stays]) / sum(μ[pos])
    return (; mpl_gap, mpl_gap_emp, Ω, output_lost = 100 * (1 - Ω), share_inactive)
end
# <<< misallocation

# >>> size_distribution
"""
    size_distribution(eq, cutoffs)

Cumulative shares of firms and of employment in firms with at most each cutoff
of workers n, for example 10, 20, 50, 100 and 1000.
"""
function size_distribution(eq::Equilibrium, cutoffs)
    n_new = eq.s.n[eq.sol.pol]                         # chosen n at every state
    emp = eq.μ .* n_new                                # employment at every state
    # the shares of firms and of workers at states with n ≤ each cutoff
    firms = [sum(eq.μ[n_new .<= c]) / sum(eq.μ) for c in cutoffs]
    employment = [sum(emp[n_new .<= c]) / sum(emp) for c in cutoffs]
    return (; cutoffs, firms, employment)
end
# <<< size_distribution
