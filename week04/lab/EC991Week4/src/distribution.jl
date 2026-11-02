# >>> young_weights
"""
    young_weights(a′, a)

The lottery of slide 23: the index k with a[k] ≤ a′ ≤ a[k+1], and the probability
p = (a[k+1] - a′) / (a[k+1] - a[k]) of going to the point below, so that
p a[k] + (1 - p) a[k+1] = a′ and the mean of assets is preserved. A choice
outside the grid goes to the nearest end.
"""
function young_weights(a′, a)
    # below the grid: all the mass goes to the first point
    a′ <= a[1] && return 1, 1.0
    # above the grid: all the mass goes to the last point
    a′ >= a[end] && return length(a) - 1, 0.0
    # the index of the grid point just below a′
    k = searchsortedlast(a, a′)
    # the point below, and the probability of going to it
    return k, (a[k+1] - a′) / (a[k+1] - a[k])
end
# <<< young_weights

# >>> policy_on_grid
"""
    policy_on_grid(sol, a_dist)

The saving policy on the distribution grid (slide 30, Step 4), read off by linear
interpolation and kept inside [a_dist[1], a_dist[end]]. When `a_dist` equals the
policy grid this only applies that cap.
"""
function policy_on_grid(sol::Solution, a_dist)
    g = similar(sol.apol, length(a_dist), length(sol.y))
    for j in eachindex(sol.y)
        f = LinearInterp(sol.a, sol.apol[:, j])
        for (i, x) in enumerate(a_dist)
            g[i, j] = clamp(f(x), a_dist[1], a_dist[end])
        end
    end
    return g
end
# <<< policy_on_grid

# >>> build_Q
"""
    build_Q(g, a, P)

The transition function as a matrix (slides 21 and 24), for small grids and for
learning: Q[s, s′] with the state (a_i, y_j) numbered s = (j - 1) n_a + i. Row s
has the lottery probabilities times P[j, m], so at most 2 n_y entries are non-zero.
`stationary_distribution` does the same thing without storing Q.
"""
function build_Q(g, a, P)
    n_a, n_y = size(g)
    Q = zeros(n_a * n_y, n_a * n_y)
    for j in 1:n_y, i in 1:n_a
        k, p = young_weights(g[i, j], a)
        s = (j - 1) * n_a + i
        for m in 1:n_y
            Q[s, (m - 1) * n_a + k] += p * P[j, m]
            Q[s, (m - 1) * n_a + k + 1] += (1 - p) * P[j, m]
        end
    end
    return Q
end
# <<< build_Q

# >>> stationary_distribution
"""
    stationary_distribution(g, a, P; λ_start = nothing, tol = 1e-10, maxiter = 30_000)

The invariant distribution λ (n_a × n_y) by iteration (slide 24): λ_{t+1} = Q' λ_t
until max |λ_{t+1} - λ_t| < `tol`. Q is the lottery times P, so one step is done
in those two parts without storing Q: first move the mass in assets within each
income, then apply the income chain with one matrix product. `g` is the saving
policy on the grid `a`.

The default start puts everyone at the borrowing limit, with income from the
stationary distribution of P. This is not only tidy. A policy that lives on the
grid can have absorbing states, assets where every income chooses a' = a, and any
mass that starts there stays there, so a uniform start would change the answer.
Returns λ, the number of iterations and whether it converged.
"""
function stationary_distribution(g, a, P; λ_start = nothing, tol = 1e-10, maxiter = 30_000)
    n_a, n_y = size(g)
    ks = Matrix{Int}(undef, n_a, n_y)
    ps = similar(g)
    for j in 1:n_y, i in 1:n_a
        ks[i, j], ps[i, j] = young_weights(g[i, j], a)
    end
    if λ_start === nothing
        λ0 = zeros(n_a, n_y)
        λ0[1, :] .= stationary(P)
    else
        λ0 = copy(λ_start)
    end
    μ, λ1 = similar(λ0), similar(λ0)
    d = Inf
    # each pass moves the distribution forward one period
    for n in 1:maxiter
        # μ: the mass after saving, before the income shock
        fill!(μ, 0.0)
        for j in 1:n_y, i in 1:n_a               # the savings decision
            # this state's lottery: the point below and its probability
            k, p = ks[i, j], ps[i, j]
            # share p of the mass goes to the point below
            μ[k, j] += p * λ0[i, j]
            # the rest goes to the point above
            μ[k+1, j] += (1 - p) * λ0[i, j]
        end
        mul!(λ1, μ, P)                           # the income shocks: λ1[k, m] = Σ_j μ[k, j] P[j, m]
        # the largest change between two passes
        d = maximum(abs.(λ1 .- λ0))
        d < tol && return (λ = λ1, iterations = n, converged = true)
        # the new distribution becomes the current one
        λ0, λ1 = λ1, λ0
    end
    return (λ = λ0, iterations = maxiter, converged = false)
end
# <<< stationary_distribution

# >>> absorbing_states
"""
    absorbing_states(sol)

The grid points where every income state chooses a' = a, for a solution with its
policy on the grid. A household that reaches one never leaves. They appear where
the grid gap is wider than the differences in saving across incomes, so that
everyone rounds to the same point. Returns their indices; none off the grid.
"""
function absorbing_states(sol::Solution)
    sol.pol === nothing && return Int[]
    return [k for k in eachindex(sol.a) if all(==(k), view(sol.pol, k, :))]
end
# <<< absorbing_states
