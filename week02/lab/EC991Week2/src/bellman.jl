# >>> bellman_step!
"""
    bellman_step!(V1, pol, V0, hh, a; y = hh.y)

One application of the Bellman operator T on the grid `a`:

    V1[i] = max_k  u(y + R a[i] - a[k]) + β V0[k]        (slide 11)

For every state a[i], try every candidate a[k], skip the ones that leave
no consumption, and keep the best. `pol[i]` records the index k of the
chosen a', so the policy function is `a[pol]`.

Income `y` is a keyword so the same function serves the finite-horizon
problem, where income varies with the date. Nothing else changes.

A state with no feasible choice at all (zero income and zero assets in the
cake-eating problem) gets value -∞, so no earlier state ever chooses it.
"""
function bellman_step!(V1, pol, V0, hh::Household, a; y = hh.y)
    (; β, R) = hh
    na = length(a)
    for i in 1:na                        # state a_i
        best, kbest = -Inf, 0
        for k in 1:na                    # candidate a'_k
            c = R * a[i] + y - a[k]
            c <= 0 && continue           # infeasible: value -∞, never chosen
            v = utility(c, hh) + β * V0[k]   # the Bellman right-hand side
            if v > best
                best, kbest = v, k
            end
        end
        if kbest == 0                    # nothing feasible (e.g. a = 0 with no income)
            best, kbest = -Inf, 1        # value -∞; the index is a placeholder
        end
        V1[i], pol[i] = best, kbest
    end
    return V1, pol
end
# <<< bellman_step!
