# >>> Solution
"""
The converged solution of the infinite-horizon problem on the grid `a`:
value function `V`, policy index `pol`, asset policy `apol = a[pol]`,
consumption policy `cpol = y + R a - apol`, the number of iterations, and
the sup-norm distance at every iteration (for a convergence plot).
"""
struct Solution{T<:Real}
    a::Vector{T}
    V::Vector{T}
    pol::Vector{Int}
    apol::Vector{T}
    cpol::Vector{T}
    iterations::Int
    distances::Vector{T}
end
# <<< Solution

# >>> initial_guess
"""
    initial_guess(hh, a)

Step 2 of the algorithm: the value of consuming the interest plus income,
r a + y, forever:  V_0(a) = u(r a + y) / (1 - β).  Any guess converges;
this one starts close.
"""
initial_guess(hh::Household, a) = utility.((hh.R - 1) .* a .+ hh.y, hh.γ) ./ (1 - hh.β)
# <<< initial_guess

# >>> solve_vfi
"""
    solve_vfi(hh, a; tol = 1e-6, maxiter = 2000)

Steps 2 to 4 of the algorithm: start from `initial_guess`, apply
`bellman_step!` (Step 3), stop when the sup-norm change in V is below
`tol` (Step 4). Returns a `Solution`.
"""
function solve_vfi(hh::Household, a; tol = 1e-6, maxiter = 2000)
    V0 = initial_guess(hh, a)
    V1 = similar(V0)
    pol = zeros(Int, length(a))
    distances = eltype(V0)[]
    for ℓ in 1:maxiter
        bellman_step!(V1, pol, V0, hh, a)          # Step 3: V1 = T V0
        d = maximum(abs(V1[i] - V0[i]) for i in eachindex(a))
        push!(distances, d)
        if d < tol                                 # Step 4: converged
            apol = a[pol]
            cpol = hh.y .+ hh.R .* a .- apol
            return Solution(a, copy(V1), copy(pol), apol, cpol, ℓ, distances)
        end
        V0, V1 = V1, V0                            # the update becomes the guess
    end
    error("value function iteration did not converge in $maxiter iterations")
end
# <<< solve_vfi
