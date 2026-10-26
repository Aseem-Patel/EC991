# >>> chain_vs_ar1
"""
    chain_vs_ar1(logy, P, ρ, σ)

The chain's standard deviation and autocorrelation next to the AR(1)'s,
σ/√(1-ρ²) and ρ (slides 14 and 15).
"""
function chain_vs_ar1(logy, P, ρ, σ)
    m = chain_moments(logy, P)
    return (sd_chain = m.sd, sd_ar1 = σ / sqrt(1 - ρ^2), autocorr_chain = m.autocorr, autocorr_ar1 = ρ)
end
# <<< chain_vs_ar1

# >>> policy_monotone
"""
    policy_monotone(sol)

Theory says the saving policy a'(a, y) increases in assets and in income.
Returns whether it does on the grid, in each direction.
"""
function policy_monotone(sol::Solution)
    in_a = all(issorted(sol.apol[:, j]) for j in axes(sol.apol, 2))
    in_y = all(issorted(sol.apol[i, :]) for i in axes(sol.apol, 1))
    return (in_a = in_a, in_y = in_y)
end
# <<< policy_monotone

# >>> compare_policies
"""
    compare_policies(sol_a, sol_b)

Largest gap between two saving policies, for each income level. The policy
of `sol_b` is read off at the asset points of `sol_a` by linear interpolation,
so the two solutions may use different grids. Returns the largest gap over
all states and the asset level where it occurs.
"""
function compare_policies(sol_a::Solution, sol_b::Solution)
    gap, where = 0.0, (sol_a.a[1], 1)
    for j in eachindex(sol_a.y)
        fb = LinearInterp(sol_b.a, sol_b.apol[:, j])
        for (i, ai) in enumerate(sol_a.a)
            g = abs(sol_a.apol[i, j] - fb(ai))
            g > gap && ((gap, where) = (g, (ai, j)))
        end
    end
    return (gap = gap, a = where[1], j = where[2])
end
# <<< compare_policies
