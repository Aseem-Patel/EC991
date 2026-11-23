# >>> impulse_response
"""
    impulse_response(x, x̄, ε)

Step 3 of slide 43: the response of a variable at each horizon, per unit of the
initial innovation, h_s = (x_{s+1} - x̄) / ε for s = 0, 1, ..., T-1. The date
of the shock, t = 1, is horizon s = 0.
"""
impulse_response(x, x̄, ε) = (x .- x̄) ./ ε
# <<< impulse_response

# >>> bkm_simulate
"""
    bkm_simulate(h, ε)

Step 4 of slide 43 (Boppart, Krusell and Mitman, 2018): add up the effects of a
sequence of innovations ε_1, ε_2, ..., each scaled by the response at its age,

    x_t - x̄ ≈ h_0 ε_t + h_1 ε_{t-1} + h_2 ε_{t-2} + ... + h_{t-1} ε_1.

Horizons beyond the impulse response, length(h), count as zero. Returns the
deviations x_t - x̄.
"""
function bkm_simulate(h, ε)
    x = zeros(length(ε))
    for t in eachindex(ε), s in 0:min(t - 1, length(h) - 1)
        x[t] += h[s+1] * ε[t-s]
    end
    return x
end
# <<< bkm_simulate
