# >>> bellman_step!
"""
    bellman_step!(V1, pol, V0, hh, a, y, P)

One application of the Bellman operator by brute-force grid search, on the
asset grid `a` (n_a points) and the income grid `y` (n_y levels, not logs)
with transition matrix P:

    V1[i, j] = max_k  u(R a_i + w y_j - a_k) + β E[V0(a_k, y') | y_j]

The continuation values for every choice k and every income today j come
from one matrix product (slide 23): EV = V0 P', so EV[k, j] = Σ_m V0[k, m] P[j, m].
Then for every state (a_i, y_j) try every a_k, skip the ones that leave no
consumption (value -∞), and keep the best. `pol[i, j]` records the index k,
so the policy function is `a[pol]`. Same loop as week 2, one more state.
"""
function bellman_step!(V1, pol, V0, hh::Household, a, y, P)
    (; β, R, w) = hh
    EV = V0 * P'                             # EV[k, j] = E[V0(a_k, y') | y_j]
    n_a, n_y = length(a), length(y)
    for j in 1:n_y, i in 1:n_a               # state (a_i, y_j)
        best, kbest = -Inf, 1
        for k in 1:n_a                       # candidate a'_k
            c = R * a[i] + w * y[j] - a[k]
            c <= 0 && continue               # infeasible: value -∞, never chosen
            v = utility(c, hh) + β * EV[k, j]
            if v > best
                best, kbest = v, k
            end
        end
        V1[i, j], pol[i, j] = best, kbest
    end
    return V1, pol
end
# <<< bellman_step!
