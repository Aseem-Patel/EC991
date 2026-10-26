# >>> LinearInterp
"""
    LinearInterp(x, v)

Linear interpolation through the points (x_k, v_k), x increasing (slide 28).
Call it like a function: for x_k ≤ x ≤ x_{k+1},

    f(x) = v_k + (x - x_k) / (x_{k+1} - x_k) (v_{k+1} - v_k).

Nothing is computed in advance: each call finds the gap and mixes its two
ends. Outside [x_1, x_n] it extends the end segments.
"""
struct LinearInterp{T<:Real}
    x::Vector{T}
    v::Vector{T}
end
function (f::LinearInterp)(x)
    k = clamp(searchsortedlast(f.x, x), 1, length(f.x) - 1)
    return f.v[k] + (x - f.x[k]) / (f.x[k+1] - f.x[k]) * (f.v[k+1] - f.v[k])
end
# <<< LinearInterp

# >>> CubicSpline
"""
    CubicSpline(x, v)

Natural cubic spline through the points (x_k, v_k) (slide 29, backup slides 35 and 36).
On each gap [x_k, x_{k+1}] it is the cubic

    p_k(x) = b_k0 + b_k1 (x - x_k) + b_k2 (x - x_k)^2 + b_k3 (x - x_k)^3.

The coefficients are built once, when the spline is constructed: the
curvatures M_k = p''(x_k) solve the tridiagonal system

    h_{k-1} M_{k-1} + 2 (h_{k-1} + h_k) M_k + h_k M_{k+1} = 6 (Δ_k - Δ_{k-1}),

with h_k = x_{k+1} - x_k, Δ_k = (v_{k+1} - v_k) / h_k and M_1 = M_n = 0, and

    b_k0 = v_k,  b_k1 = Δ_k - h_k (2 M_k + M_{k+1}) / 6,  b_k2 = M_k / 2,
    b_k3 = (M_{k+1} - M_k) / (6 h_k).

Each call then only finds the gap and evaluates its cubic.
"""
struct CubicSpline{T<:Real}
    x::Vector{T}
    b::Matrix{T}            # b[k, :] = (b_k0, b_k1, b_k2, b_k3)
end
function CubicSpline(x::AbstractVector, v::AbstractVector)
    n = length(x)
    h = diff(x)
    Δ = diff(v) ./ h
    # the tridiagonal system for M_2, ..., M_{n-1}, solved by forward
    # elimination and back substitution
    M = zeros(n)
    if n > 2
        lo = h[2:n-2]                        # below the diagonal
        di = [2 * (h[k-1] + h[k]) for k in 2:n-1]
        up = h[2:n-2]                        # above the diagonal
        rhs = [6 * (Δ[k] - Δ[k-1]) for k in 2:n-1]
        for k in 2:n-2
            μ = lo[k-1] / di[k-1]
            di[k] -= μ * up[k-1]
            rhs[k] -= μ * rhs[k-1]
        end
        m = n - 2
        M[m+1] = rhs[m] / di[m]
        for k in m-1:-1:1
            M[k+1] = (rhs[k] - up[k] * M[k+2]) / di[k]
        end
    end
    b = zeros(n - 1, 4)
    for k in 1:n-1
        b[k, 1] = v[k]
        b[k, 2] = Δ[k] - h[k] * (2M[k] + M[k+1]) / 6
        b[k, 3] = M[k] / 2
        b[k, 4] = (M[k+1] - M[k]) / (6h[k])
    end
    return CubicSpline(collect(float.(x)), b)
end
function (f::CubicSpline)(x)
    k = clamp(searchsortedlast(f.x, x), 1, length(f.x) - 1)
    t = x - f.x[k]
    return f.b[k, 1] + t * (f.b[k, 2] + t * (f.b[k, 3] + t * f.b[k, 4]))
end
# <<< CubicSpline
