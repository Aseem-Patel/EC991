# The static economy of the lecture's first part: a fixed set of firms, labour
# the only input, y = z n^η with 0 < η < 1. Every function takes the firms'
# productivities z and, where it matters, their weights m (the mass of firms at
# each z, all ones for a list of firms).

# >>> efficient_allocation
"""
    efficient_allocation(z, N; η, m = ones)

The planner's split of N workers across firms (lecture slide 9): marginal
products are equal, η z_i n_i^(η-1) = μ, so n_i ∝ z_i^(1/(1-η)). Returns the
employment of each firm, its output and total output Y.
"""
function efficient_allocation(z, N; η, m = ones(length(z)))
    # each firm's weight: employment is proportional to z^(1/(1-η))
    s = z .^ (1 / (1 - η))
    # share out the N workers in proportion to the weights
    n = N .* s ./ sum(m .* s)
    # output of each firm, then total output over all firms
    y = z .* n .^ η
    return (; n, y, Y = sum(m .* y))
end
# <<< efficient_allocation

# >>> aggregate_tfp
"""
    aggregate_tfp(z; η, m = ones)

Aggregate TFP in Y = Z N^η (slide 11): Z = (Σ_i m_i z_i^(1/(1-η)))^(1-η). Firms
with high z count for more than their share, because they hire more: Z is a
power mean of productivity, scaled by the number of firms.
"""
aggregate_tfp(z; η, m = ones(length(z))) = sum(m .* z .^ (1 / (1 - η)))^(1 - η)
# <<< aggregate_tfp

# >>> allocation_with_wedges
"""
    allocation_with_wedges(z, τ, N; η, m = ones)

Firms facing wedges τ_i (slide 14) hire until η (1 + τ_i) z_i n_i^(η-1) = w.
The wage clears the labour market, so n_i ∝ ((1 + τ_i) z_i)^(1/(1-η)). Returns
employment, output and total output.
"""
function allocation_with_wedges(z, τ, N; η, m = ones(length(z)))
    # hiring as if productivity were (1 + τ) z ...
    n = efficient_allocation((1 .+ τ) .* z, N; η, m).n
    # ... but producing with the true z
    y = z .* n .^ η
    return (; n, y, Y = sum(m .* y))
end
# <<< allocation_with_wedges

# >>> misallocation
"""
    misallocation(z, τ; η, m = ones)

The misallocation term Ω in Y = Z Ω N^η (slide 15), Ω ∈ (0, 1]. With
x_i = z_i^(1/(1-η)) and t_i = 1 + τ_i:

    Ω = Σ m x t^(η/(1-η)) / [ (Σ m x t^(1/(1-η)))^η (Σ m x)^(1-η) ]

Ω = 1 when every firm faces the same wedge, which then moves only the wage.
"""
function misallocation(z, τ; η, m = ones(length(z)))
    # x_i = z_i^(1/(1-η)): how much firm i would hire without wedges
    x = z .^ (1 / (1 - η))
    # t_i = 1 + τ_i: how much the wedge scales firm i's productivity when hiring
    t = 1 .+ τ
    # numerator: output with wedges; denominator: the same labour, allocated efficiently
    return sum(m .* x .* t .^ (η / (1 - η))) / (sum(m .* x .* t .^ (1 / (1 - η)))^η * sum(m .* x)^(1 - η))
end
# <<< misallocation

# >>> planner_entry
"""
    planner_entry(N, c_e; η, Z = 1.0)

Slides 20 to 22: each firm costs c_e workers to create, so the planner chooses
the number of firms M and production workers N_p to maximise Z M^(1-η) N_p^η
subject to c_e M + N_p = N. Solution: N_p = η N and M = (1-η) N / c_e, and output
Y = Z η^η (1-η)^(1-η) c_e^(-(1-η)) N.
"""
function planner_entry(N, c_e; η, Z = 1.0)
    N_p = η * N
    M = (1 - η) * N / c_e
    return (; N_p, M, Y = Z * M^(1 - η) * N_p^η)
end
# <<< planner_entry
