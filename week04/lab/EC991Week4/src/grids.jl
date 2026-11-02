# >>> asset_grid
"""
    asset_grid(n_a, a_max; ϕ = 0.0, spacing = :growth, ν = 0.02)

Step 1 of the algorithm: the grid g_A = [a_1, ..., a_{n_a}] from the borrowing
limit a_1 = -ϕ to the upper bound a_{n_a} = `a_max`. Both the state a and the
choice a' live on it.

  spacing = :equal    a_i = a_1 + (a_{n_a} - a_1) (i - 1) / (n_a - 1)
  spacing = :growth   a_i = a_1 + (a_{n_a} - a_1) ((1 + ν)^(i-1) - 1) / ((1 + ν)^(n_a-1) - 1)

With growth-rate spacing each gap is (1 + ν) times the one before, so the
points bunch near the constraint, where the policy bends.
"""
function asset_grid(n_a, a_max; ϕ = 0.0, spacing = :growth, ν = 0.02)
    a_1 = -ϕ
    spacing === :equal && return collect(range(a_1, a_max; length = n_a))
    spacing === :growth || throw(ArgumentError("spacing must be :equal or :growth"))
    return [a_1 + (a_max - a_1) * ((1 + ν)^(i - 1) - 1) / ((1 + ν)^(n_a - 1) - 1) for i in 1:n_a]
end
# <<< asset_grid
