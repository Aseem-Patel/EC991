# >>> Solution
"""
The converged solution on the grids `a` and `y`: value function `V`
(n_a × n_y), asset policy `apol`, consumption policy `cpol = R a + w y - apol`,
the number of iterations, and the sup-norm distance at every iteration (for a
convergence plot). `pol` holds the grid index of the choice when the
maximisation is a grid search, and is `nothing` when a' is chosen off the grid.
"""
struct Solution{T<:Real}
    a::Vector{T}
    y::Vector{T}
    V::Matrix{T}
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
V_0(a, y) = u(r a + w y) / (1 - β). Any guess converges; this one starts close.
"""
initial_guess(hh::Household, a, y) = [utility(hh.r * ai + hh.w * yj, hh) / (1 - hh.β) for ai in a, yj in y]
# <<< initial_guess

# >>> solve_vfi
"""
    solve_vfi(hh, a, y, P; step = bellman_step!, tol = 1e-7, maxiter = 1200, kwargs...)

Steps 1 to 4 of VFI (slide 24): start from `initial_guess`, apply `step`
(Steps 2 and 3), stop when max |V^{n+1} - V^n| < `tol` (Step 4). `step` is
`bellman_step!` (grid search) or `bellman_step_interp!` (interpolate, then
optimise); any keywords are passed on to it. Returns a `Solution`.
"""
function solve_vfi(hh::Household, a, y, P; step = bellman_step!, tol = 1e-7, maxiter = 1200, kwargs...)
    V0 = initial_guess(hh, a, y)
    V1 = similar(V0)
    gridsearch = step === bellman_step!
    pol = gridsearch ? ones(Int, size(V0)) : similar(V0)   # index k, or a' itself
    distances = eltype(V0)[]
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
        V0, V1 = V1, V0                                    # the update becomes the guess
    end
    error("value function iteration did not converge in $maxiter iterations")
end
# <<< solve_vfi
