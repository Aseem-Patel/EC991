# >>> normal_cdf
"""
    normal_cdf(x)

Standard normal CDF Φ(x), by Abramowitz and Stegun 7.1.26 (absolute error
below 1.5e-7). Base Julia has no erf, and the lab uses no packages beyond Plots.
"""
function normal_cdf(x)
    z = abs(x) / sqrt(2)
    t = 1 / (1 + 0.3275911z)
    erfz = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * exp(-z^2)
    return x >= 0 ? (1 + erfz) / 2 : (1 - erfz) / 2
end
# <<< normal_cdf

# >>> tauchen
"""
    tauchen(n_y, ρ, σ; m = 3.0)

Tauchen's method for log y' = ρ log y + σ ε, ε ~ N(0, 1). Returns the grid
logy = [log y_1, ..., log y_{n_y}], equally spaced at distance d with
-log y_1 = log y_{n_y} = m σ/√(1-ρ²), and the transition matrix P, with P[j, k]
the probability that log y' falls in the bin around log y_k given log y_j; the
two outer bins run to ±∞.
"""
function tauchen(n_y, ρ, σ; m = 3.0)
    logy_max = m * σ / sqrt(1 - ρ^2)
    logy = collect(range(-logy_max, logy_max; length = n_y))
    d = logy[2] - logy[1]
    P = zeros(n_y, n_y)
    for j in 1:n_y
        μ = ρ * logy[j]                       # mean of log y' given log y_j
        P[j, 1] = normal_cdf((logy[1] - μ + d / 2) / σ)
        P[j, n_y] = 1 - normal_cdf((logy[n_y] - μ - d / 2) / σ)
        for k in 2:n_y-1
            P[j, k] = normal_cdf((logy[k] - μ + d / 2) / σ) - normal_cdf((logy[k] - μ - d / 2) / σ)
        end
    end
    return logy, P
end
# <<< tauchen

# >>> rouwenhorst
"""
    rouwenhorst(n_y, ρ, σ)

Rouwenhorst's method for the same AR(1). The grid logy is n_y equally
spaced points between ∓√(n_y - 1) σ/√(1-ρ²). P starts from the 2 × 2 matrix
with p = (1+ρ)/2 on the diagonal and is built up one state at a time: four
copies of the previous matrix in the four corners, weighted p, 1-p, 1-p, p,
added, and the interior rows halved. The chain matches the variance and the
autocorrelation of the AR(1) exactly, for any n_y.
"""
function rouwenhorst(n_y, ρ, σ)
    p = (1 + ρ) / 2
    logy_max = sqrt(n_y - 1) * σ / sqrt(1 - ρ^2)
    P = [p 1-p; 1-p p]
    for k in 3:n_y
        Z = zeros(k, k)
        Z[1:k-1, 1:k-1] .+= p .* P
        Z[1:k-1, 2:k]   .+= (1 - p) .* P
        Z[2:k, 1:k-1]   .+= (1 - p) .* P
        Z[2:k, 2:k]     .+= p .* P
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

# >>> chain_moments
"""
    chain_moments(logy, P)

Standard deviation and first-order autocorrelation of the chain on the grid
`logy` with transition matrix P, both under its stationary distribution π.
With mean μ = Σ_j π_j log y_j:

    var  = Σ_j π_j (log y_j - μ)²
    acov = Σ_j π_j (log y_j - μ) Σ_k P_jk (log y_k - μ)

Compare with σ/√(1-ρ²) and ρ for the AR(1).
"""
function chain_moments(logy, P)
    π = stationary(P)
    μ = sum(π .* logy)
    dev = logy .- μ
    var = sum(π .* dev .^ 2)
    acov = sum(π .* dev .* (P * dev))
    return (sd = sqrt(var), autocorr = acov / var)
end
# <<< chain_moments

# >>> simulate_chain
"""
    simulate_chain(grid, P, u; j0 = cld(length(grid), 2))

A path of the chain from state j0, driven by given uniform draws u: at date t
the next state is the first k with P[j, 1] + ... + P[j, k] ≥ u[t]. Feeding the
same u = Φ(ε) to two chains, and ε to the AR(1), puts all three on common
shocks. u[1] is not used: the path starts at j0. Returns the values and the
state indices.
"""
function simulate_chain(grid, P, u; j0 = cld(length(grid), 2))
    T, n_y = length(u), size(P, 1)
    C = cumsum(P; dims = 2)
    idx = fill(j0, T)
    for t in 2:T
        idx[t] = min(searchsortedfirst(view(C, idx[t-1], :), u[t]), n_y)
    end
    return grid[idx], idx
end
# <<< simulate_chain
