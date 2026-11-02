# >>> euler_error
"""
    euler_error(c, c′, probs, β, R, γ)

The error ε of slide 18 at one state. `c` is today's consumption and `c′[m]` is
tomorrow's in income state m, with probability `probs[m]`. The consumption the
Euler equation implies is c^EE = u'^(-1)(β R Σ_m probs[m] u'(c′[m])), and
ε = 1 - c^EE / c. Returns both.
"""
function euler_error(c, c′, probs, β, R, γ)
    # the consumption the Euler equation implies, given tomorrow's consumption c′
    c_EE = inverse_marginal_utility(β * R * sum(p * marginal_utility(x, γ) for (p, x) in zip(probs, c′)), γ)
    # ε = 0 means the Euler equation holds exactly
    return (c_EE = c_EE, ε = 1 - c_EE / c)
end
# <<< euler_error

# >>> euler_errors
"""
    euler_errors(sol, hh, P; n_test = 2000, a_hi = 50.0)

The Euler errors of a solution on a different grid from the one it was solved on
(slide 18): `n_test` equally spaced assets on [a_1, a_hi] at every income. The
policies are read off by linear interpolation. States where the borrowing
constraint binds are left out, because the Euler equation holds there only as an
inequality. Returns the largest and the mean of log10 |ε|, and all the errors.
"""
function euler_errors(sol::Solution, hh::Household, P; n_test = 2000, a_hi = 50.0)
    (; β, γ, R, w) = hh
    a, y = sol.a, sol.y
    gc = [LinearInterp(a, sol.cpol[:, m]) for m in eachindex(y)]
    errs = Float64[]
    # every income, at equally spaced test assets x off the solution grid
    for j in eachindex(y), x in range(a[1], a_hi; length = n_test)
        # the policy's consumption at x, by interpolation
        c = gc[j](x)
        # budget constraint: tomorrow's assets
        a′ = R * x + w * y[j] - c
        a′ <= a[1] + 1e-10 && continue           # the constraint binds
        # tomorrow's consumption at a′ in each income, with row j of P as the probabilities
        e = euler_error(c, [gc[m](a′) for m in eachindex(y)], view(P, j, :), β, R, γ)
        push!(errs, abs(e.ε))
    end
    logs = log10.(max.(errs, 1e-16))
    return (max_log10 = maximum(logs), mean_log10 = sum(logs) / length(logs), errors = errs)
end
# <<< euler_errors
