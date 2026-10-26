# >>> bellman_step_interp!
"""
    bellman_step_interp!(V1, apol, V0, hh, a, y, P; interp = :linear, routine = :brent, tol = 1e-8)

One application of the Bellman operator with a' chosen off the grid
(slides 26 and 32): construct once, evaluate many times.

1. For each income tomorrow y_m, build one interpolant V̂(·, y_m) through the
   column V0[:, m] (`interp = :linear` or `:cubic`).
2. For each state (a_i, y_j), maximise the continuous objective

       φ(a') = u(R a_i + w y_j - a') + β Σ_m P[j, m] V̂(a', y_m)

   over a_1 ≤ a' ≤ min(a_{n_a}, R a_i + w y_j - c_min) with a one-dimensional
   routine (`routine = :brent` or `:golden`), and compare with the corner
   a' = a_1, where the borrowing constraint binds.

Today's y_j enters only through the weights P[j, m]. `apol[i, j]` is the
chosen a' itself, not a grid index.
"""
function bellman_step_interp!(V1, apol, V0, hh::Household, a, y, P;
                              interp = :linear, routine = :brent, tol = 1e-8)
    (; β, R, w) = hh
    n_a, n_y = length(a), length(y)
    build = interp === :linear ? LinearInterp : interp === :cubic ? CubicSpline :
            throw(ArgumentError("interp must be :linear or :cubic"))
    search = routine === :brent ? brent : routine === :golden ? golden_section :
             throw(ArgumentError("routine must be :brent or :golden"))
    V̂ = [build(a, V0[:, m]) for m in 1:n_y]          # once per iteration, per y_m
    c_min = 1e-10
    for j in 1:n_y, i in 1:n_a
        cash = R * a[i] + w * y[j]
        φ(ap) = utility(cash - ap, hh) + β * sum(P[j, m] * V̂[m](ap) for m in 1:n_y)
        hi = min(a[end], cash - c_min)
        best_a, best_v = a[1], φ(a[1])                # the corner: constraint binds
        if hi > a[1]
            x, fx, _ = search(φ, a[1], hi; tol)
            fx > best_v && ((best_a, best_v) = (x, fx))
        end
        V1[i, j], apol[i, j] = best_v, best_a
    end
    return V1, apol
end
# <<< bellman_step_interp!
