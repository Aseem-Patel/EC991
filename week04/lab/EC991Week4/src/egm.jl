# >>> marginal_utility
"""
    marginal_utility(c, γ)  and  inverse_marginal_utility(m, γ)

CRRA marginal utility u'(c) = c^(-γ) and its inverse u'^(-1)(m) = m^(-1/γ).
The inverse exists in closed form, which is what makes EGM cheap (slide 17).
"""
marginal_utility(c, γ) = c^(-γ)
inverse_marginal_utility(m, γ) = m^(-1 / γ)
# <<< marginal_utility

# >>> endogenous_assets
"""
    endogenous_assets(c̃, a′, wy, R)

Step 4 of EGM (slide 13): the assets â the household must hold today to consume
c̃ and carry a′ into tomorrow, from the budget constraint c̃ + a′ = R â + wy.
"""
endogenous_assets(c̃, a′, wy, R) = (c̃ + a′ - wy) / R
# <<< endogenous_assets

# >>> egm_step!
"""
    egm_step!(c1, apol, c0, hh, a, y, P)

One EGM update of the consumption policy c0 on the grid `a` (slides 12 to 14).
Read each grid point as tomorrow's assets a'_i. For every pair (a'_i, y_j):

1. the right-hand side of the Euler equation, RHS = β R Σ_m P[j, m] u'(c0[i, m]);
   the sums for every i and j are one matrix product, as in `bellman_step!`;
2. consumption today, c̃ = u'^(-1)(RHS);
3. the endogenous grid, â = (c̃ + a'_i - w y_j) / R.

At each income the pairs (â_i, a'_i) trace out the saving policy. Interpolate it
back onto the grid, set a' = a_1 wherever the result falls below the borrowing
limit, and recover consumption from the budget constraint.
"""
function egm_step!(c1, apol, c0, hh::Household, a, y, P)
    (; β, γ, R, w) = hh
    # read a[i] as tomorrow's assets: row i of EMU is the choice a'_i
    EMU = marginal_utility.(c0, γ) * P'          # EMU[i, j] = Σ_m P[j, m] u'(c0[i, m])
    # empty vector for the generated assets at one income
    â = similar(a)
    # one income today at a time
    for j in eachindex(y)
        # each grid point is a choice of tomorrow's assets
        for i in eachindex(a)
            # invert the Euler equation: consumption today, with no root finder
            c̃ = inverse_marginal_utility(β * R * EMU[i, j], γ)
            # budget constraint: the assets today that lead to this a'
            â[i] = endogenous_assets(c̃, a[i], w * y[j], R)
        end
        ĝ = LinearInterp(â, a)                   # a' as a function of today's assets
        # back on the grid: a[i] now means today's assets
        for i in eachindex(a)
            apol[i, j] = max(ĝ(a[i]), a[1])      # the borrowing constraint (slide 14)
            # consumption from the budget constraint
            c1[i, j] = R * a[i] + w * y[j] - apol[i, j]
        end
    end
    return c1, apol
end
# <<< egm_step!

# >>> solve_egm
"""
    solve_egm(hh, a, y, P; c_start = nothing, tol = 1e-7, maxiter = 5000)

Iterate `egm_step!` until max |c^{n+1} - c^n| < `tol` (slide 13, Step 6). The
first guess is `c_start`, or else consuming the interest and labour income, as
in `initial_guess`. Returns a `Solution` with `V = nothing`.
"""
function solve_egm(hh::Household, a, y, P; c_start = nothing, tol = 1e-7, maxiter = 5000)
    (; r, w) = hh
    c0 = c_start === nothing ? [max(r, 0) * ai + w * yj for ai in a, yj in y] : copy(c_start)
    c1, apol = similar(c0), similar(c0)
    distances = eltype(c0)[]
    for n in 1:maxiter
        egm_step!(c1, apol, c0, hh, a, y, P)
        d = maximum(abs.(c1 .- c0))
        push!(distances, d)
        if d < tol
            return Solution(collect(a), collect(y), nothing, nothing, copy(apol), copy(c1), n, distances)
        end
        c0, c1 = c1, c0
    end
    error("EGM did not converge in $maxiter iterations")
end
# <<< solve_egm
