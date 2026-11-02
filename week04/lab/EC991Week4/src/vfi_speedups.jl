# >>> bellman_step_fast!
"""
    bellman_step_fast!(V1, pol, V0, hh, a, y, P)

The grid-search Bellman step of `bellman_step!`, with the two shortcuts of
slide 6. Monotone policy: work up the asset grid at each income, and start the
search at the previous state's best choice. Concave objective: stop at the
first fall. Same arguments, same result, far fewer choices tried.
"""
function bellman_step_fast!(V1, pol, V0, hh::Household, a, y, P)
    (; β, R, w) = hh
    # EV[k, j]: the expected value of V0 at a_k, given income y_j today
    EV = V0 * P'
    n_a, n_y = length(a), length(y)
    # one income today at a time
    for j in 1:n_y
        kstart = 1                           # monotonicity: the lowest choice still possible
        # work up the asset grid at this income
        for i in 1:n_a
            # the best value so far, and the index of the choice that gave it
            best, kbest = -Inf, kstart
            # try the choices a_k, from the previous state's best upwards
            for k in kstart:n_a
                c = R * a[i] + w * y[j] - a[k]
                c <= 0 && break              # consumption only falls with k
                v = utility(c, hh) + β * EV[k, j]
                v < best && break            # concavity: the first fall is past the peak
                best, kbest = v, k
            end
            # store the new value and the index of the best choice
            V1[i, j], pol[i, j] = best, kbest
            # the next, richer state starts its search here
            kstart = kbest
        end
    end
    return V1, pol
end
# <<< bellman_step_fast!

# >>> howard_update!
"""
    howard_update!(W, pol, hh, a, y, P; n_H, interp = :linear)

Howard's evaluation step (slide 5): hold the saving policy fixed and update the
value estimate `W` in place `n_H` times, with no maximisation,

    W^{k+1}[i, j] = u(R a_i + w y_j - g[i, j]) + β Σ_m P[j, m] W^k(g[i, j], y_m).

`pol` is either the grid index of the choice (from a grid search) or a' itself
(chosen off the grid). In the second case W is interpolated at g. Interpolating
the column EV[:, j] gives the same number as interpolating each W[:, m] and
weighting by P[j, m], because both interpolants are linear in the stored values.
"""
function howard_update!(W, pol::Matrix{Int}, hh::Household, a, y, P; n_H, interp = :linear)
    (; β, R, w) = hh
    # utility under the fixed policy: it never changes, so compute it once
    U = [utility(R * a[i] + w * y[j] - a[pol[i, j]], hh) for i in eachindex(a), j in eachindex(y)]
    # empty matrix to hold the expectation
    EV = similar(W)
    # n_H cheap updates, with no maximisation
    for _ in 1:n_H
        # EV = W * P', written into the existing matrix
        mul!(EV, W, P')
        for j in eachindex(y), i in eachindex(a)
            # utility today + discounted expected value at the fixed choice
            W[i, j] = U[i, j] + β * EV[pol[i, j], j]
        end
    end
    return W
end
function howard_update!(W, apol::Matrix{<:AbstractFloat}, hh::Household, a, y, P; n_H, interp = :linear)
    (; β, R, w) = hh
    # choose the interpolant: linear or cubic spline
    build = interp === :linear ? LinearInterp : interp === :cubic ? CubicSpline :
            throw(ArgumentError("interp must be :linear or :cubic"))
    # utility under the fixed policy: it never changes, so compute it once
    U = [utility(R * a[i] + w * y[j] - apol[i, j], hh) for i in eachindex(a), j in eachindex(y)]
    # empty matrix to hold the expectation
    EV = similar(W)
    # n_H cheap updates, with no maximisation
    for _ in 1:n_H
        # EV = W * P', written into the existing matrix
        mul!(EV, W, P')
        for j in eachindex(y)
            # the expectation at this income, as a function of a'
            Ê = build(a, EV[:, j])
            for i in eachindex(a)
                # read the expectation at the off-grid choice apol[i, j]
                W[i, j] = U[i, j] + β * Ê(apol[i, j])
            end
        end
    end
    return W
end
# <<< howard_update!
