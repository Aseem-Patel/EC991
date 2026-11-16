# >>> normal_cdf
"""
    normal_cdf(x)

Standard normal CDF Φ(x), by Abramowitz and Stegun 7.1.26 (absolute error
below 1.5e-7). Base Julia has no erf, and the lab uses no packages beyond Plots.
From week 3.
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
    tauchen(n_z, ρ, σ; mean = 0.0, m = 3.0)

Tauchen's method (week 3) for an AR(1) around a mean:

    log z' - mean = ρ (log z - mean) + σ ε,   ε ~ N(0, 1).

Returns the grid logz, n_z equally spaced points at distance d within
mean ± m σ/√(1-ρ²), and the transition matrix F, with F[i, j] the probability
that log z' falls in the bin around log z_j given log z_i; the two outer bins
run to ±∞. Week 3's version is the case mean = 0.
"""
function tauchen(n_z, ρ, σ; mean = 0.0, m = 3.0)
    width = m * σ / sqrt(1 - ρ^2)
    logz = collect(range(mean - width, mean + width; length = n_z))
    d = logz[2] - logz[1]
    F = zeros(n_z, n_z)
    for i in 1:n_z
        μ = mean + ρ * (logz[i] - mean)          # mean of log z' given log z_i
        F[i, 1] = normal_cdf((logz[1] - μ + d / 2) / σ)
        F[i, n_z] = 1 - normal_cdf((logz[n_z] - μ - d / 2) / σ)
        for j in 2:n_z-1
            F[i, j] = normal_cdf((logz[j] - μ + d / 2) / σ) - normal_cdf((logz[j] - μ - d / 2) / σ)
        end
    end
    return logz, F
end
# <<< tauchen

# >>> stationary
"""
    stationary(F)

The stationary distribution of the chain: f' F = f' with the entries summing to
one. Solved directly: the system (I - F') f = 0 with its last equation replaced
by sum(f) = 1. From week 3.
"""
function stationary(F)
    n = size(F, 1)
    A = Matrix(I - F')
    A[end, :] .= 1
    b = zeros(n); b[end] = 1
    return A \ b
end
# <<< stationary

# >>> discretise_normal
"""
    discretise_normal(logz, μ, σ)

The probabilities that a N(μ, σ²) draw of log z falls in the bin around each
grid point: bins end halfway between neighbouring points, and the two outer
bins run to ±∞. Used for an entrant distribution G that differs from the
incumbents' invariant distribution.
"""
function discretise_normal(logz, μ, σ)
    edges = (logz[1:end-1] .+ logz[2:end]) ./ 2
    cdf = [0.0; normal_cdf.((edges .- μ) ./ σ); 1.0]
    return diff(cdf)
end
# <<< discretise_normal
