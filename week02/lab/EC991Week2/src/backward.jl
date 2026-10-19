# >>> FiniteSolution
"""
The solution of the T-period problem on the grid `a`: `V[i, t]` is
V_t(a_i) for t = 1, ..., T+1 (column T+1 is the terminal condition),
`pol[i, t]` the chosen index, `apol` and `cpol` the policies by date.
"""
struct FiniteSolution{T<:Real}
    a::Vector{T}
    y::Vector{T}
    V::Matrix{T}
    pol::Matrix{Int}
    apol::Matrix{T}
    cpol::Matrix{T}
end
# <<< FiniteSolution

# >>> solve_backward
"""
    solve_backward(hh, a, y; terminal = zeros(length(a)))

Finite horizon by backward induction (slides 13 and 14). `y[t]` is income
at date t = 1, ..., T. Column T+1 of V holds the terminal condition,
V_{T+1}(a) = 0 unless a bequest motive is supplied through `terminal`.

    for t = T, ..., 1:   V_t = T_t V_{t+1}

The inner step is the same `bellman_step!` as in the infinite-horizon
solver. There is no convergence check: after T passes we are done.
"""
function solve_backward(hh::Household, a, y; terminal = zeros(length(a)))
    T = length(y)
    na = length(a)
    V = zeros(eltype(a), na, T + 1)
    pol = zeros(Int, na, T)
    V[:, T + 1] .= terminal                                   # Step 1
    for t in T:-1:1                                           # Step 2
        bellman_step!(view(V, :, t), view(pol, :, t), view(V, :, t + 1), hh, a; y = y[t])
    end
    apol = a[pol]
    cpol = [y[t] + hh.R * a[i] - apol[i, t] for i in 1:na, t in 1:T]
    return FiniteSolution(collect(a), collect(y), V, pol, apol, cpol)
end
# <<< solve_backward
