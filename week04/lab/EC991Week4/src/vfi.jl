# >>> Solution
"""
The converged solution on the grids `a` and `y`: value function `V`
(n_a × n_y, `nothing` for EGM, which never computes one), asset policy `apol`,
consumption policy `cpol = R a + w y - apol`, the number of iterations, and the
sup-norm distance at every iteration (for a convergence plot). `pol` holds the
grid index of the choice when the maximisation is a grid search, and is
`nothing` when a' is chosen off the grid.
"""
struct Solution{T<:Real}
    a::Vector{T}
    y::Vector{T}
    V::Union{Matrix{T}, Nothing}
    pol::Union{Matrix{Int}, Nothing}
    apol::Matrix{T}
    cpol::Matrix{T}
    iterations::Int
    distances::Vector{T}
end
# <<< Solution

# >>> initial_guess
"""
    initial_guess(hh, a, y)

Step 1 of VFI: the value of consuming interest plus labour income forever,
V_0(a, y) = u(max(r, 0) a + w y) / (1 - β). Any guess converges; this one starts
close, and stays feasible when the equilibrium loop tries a negative r.
"""
initial_guess(hh::Household, a, y) = [utility(max(hh.r, 0) * ai + hh.w * yj, hh) / (1 - hh.β) for ai in a, yj in y]
# <<< initial_guess

# >>> solve_vfi
"""
    solve_vfi(hh, a, y, P; step = bellman_step!, n_H = 0, howard_from = Inf, V_start = nothing, tol = 1e-7, maxiter = 1200, kwargs...)

Steps 1 to 4 of VFI (week 3 slide 24): start from `V_start`, or from
`initial_guess` if there is none, apply `step` (Steps 2 and 3), stop when
max |V^{n+1} - V^n| < `tol` (Step 4). `step` is `bellman_step!` (grid search),
`bellman_step_fast!` (grid search with monotonicity and concavity) or
`bellman_step_interp!` (interpolate, then optimise); any keywords are passed on
to it. Between two steps, `n_H > 0` runs that many Howard updates (slide 5); the
distance is always measured across a maximisation, so `tol` means the same with
and without them. Howard's updates run only when that distance is below
`howard_from` and the lowest so far. With interpolation, the objective has kinks
and Brent assumes one peak, so far from the solution the policy can jump between
sweeps, and evaluating a poor policy many times can stop the iteration from
settling: there, let ordinary steps bring the distance down first. Returns a
`Solution`.
"""
function solve_vfi(hh::Household, a, y, P; step = bellman_step!, n_H = 0, howard_from = Inf, V_start = nothing,
                   tol = 1e-7, maxiter = 1200, kwargs...)
    V0 = V_start === nothing ? initial_guess(hh, a, y) : copy(V_start)
    V1 = similar(V0)
    gridsearch = step === bellman_step! || step === bellman_step_fast!
    pol = gridsearch ? ones(Int, size(V0)) : similar(V0)   # index k, or a' itself
    interp = get(kwargs, :interp, :linear)
    distances = eltype(V0)[]
    d_best = Inf
    for n in 1:maxiter
        step(V1, pol, V0, hh, a, y, P; kwargs...)          # V1 = T V0
        d = maximum(abs.(V1 .- V0))
        push!(distances, d)
        if d < tol                                         # Step 4: converged
            apol = gridsearch ? a[pol] : copy(pol)
            cpol = hh.R .* a .+ hh.w .* y' .- apol
            return Solution(collect(a), collect(y), copy(V1), gridsearch ? copy(pol) : nothing,
                            apol, cpol, n, distances)
        end
        # run Howard only if the distance is below howard_from and the lowest so far
        if d < d_best && d < howard_from
            d_best = d
            # evaluate the current policy n_H times, updating V1 in place
            n_H > 0 && howard_update!(V1, pol, hh, a, y, P; n_H, interp)
        end
        V0, V1 = V1, V0                                    # the update becomes the guess
    end
    error("value function iteration did not converge in $maxiter iterations")
end
# <<< solve_vfi
