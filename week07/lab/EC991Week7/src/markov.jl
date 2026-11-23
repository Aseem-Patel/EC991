# >>> rouwenhorst
"""
    rouwenhorst(n_y, ρ, σ)

Rouwenhorst's method for log y' = ρ log y + σ ε, as in week 4. The grid logy is
n_y equally spaced points between ∓√(n_y - 1) σ/√(1-ρ²). P starts from the 2 × 2
matrix with p = (1+ρ)/2 on the diagonal and is built up one state at a time.
"""
function rouwenhorst(n_y, ρ, σ)
    p = (1 + ρ) / 2
    logy_max = sqrt(n_y - 1) * σ / sqrt(1 - ρ^2)
    P = [p 1-p; 1-p p]
    for k in 3:n_y
        # four copies of the previous matrix in the four corners, weighted p, 1-p, 1-p, p
        Z = zeros(k, k)
        Z[1:k-1, 1:k-1] .+= p .* P
        Z[1:k-1, 2:k]   .+= (1 - p) .* P
        Z[2:k, 1:k-1]   .+= (1 - p) .* P
        Z[2:k, 2:k]     .+= p .* P
        # the interior rows were counted twice
        Z[2:k-1, :] ./= 2
        P = Z
    end
    return collect(range(-logy_max, logy_max; length = n_y)), P
end
# <<< rouwenhorst

# >>> stationary
"""
    stationary(P)

The stationary distribution π of the chain: π' P = π' with the entries
summing to one. Solved directly: the system (I - P') π = 0 with its last
equation replaced by sum(π) = 1.
"""
function stationary(P)
    n_y = size(P, 1)
    A = Matrix(I - P')
    A[end, :] .= 1
    b = zeros(n_y); b[end] = 1
    return A \ b
end
# <<< stationary

# >>> simulate_chain
"""
    simulate_chain(P, s1, T, rng)

A path of length T of the Markov chain with transition matrix P, starting in
state s1: at each date, draw u uniform on [0, 1) and move to the first state
whose cumulative probability exceeds u.
"""
function simulate_chain(P, s1, T, rng)
    s = zeros(Int, T)
    s[1] = s1
    for t in 1:T-1
        # the cumulative probabilities of the row of today's state
        u, cum, next = rand(rng), 0.0, size(P, 2)
        for j in axes(P, 2)
            cum += P[s[t], j]
            if u < cum
                next = j
                break
            end
        end
        s[t+1] = next
    end
    return s
end
# <<< simulate_chain
