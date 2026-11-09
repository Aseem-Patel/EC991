# >>> exit_transition
"""
    exit_transition(F, χ)

F̂ (lecture slide 37): the transition matrix with the rows of exiting firms set
to zero, F̂_ij = (1 - χ_i) F_ij. Rows of firms that stay still sum to one; rows of
firms that exit sum to zero, because they leave the industry.
"""
exit_transition(F, χ) = (1 .- χ) .* F
# <<< exit_transition

# >>> stationary_measure
"""
    stationary_measure(F̂, g; M = 1.0, method = :linear, tol = 1e-12, maxiter = 100_000)

The stationary measure of firms over the grid (slide 37), from the law of motion

    μ' = F̂' μ + M g.

`:linear` solves μ = (I - F̂')⁻¹ M g directly. `:iterate` applies the law of
motion from μ = 0 until it stops changing, as for the household distribution in
week 4. Unlike week 4, μ is a measure, not a probability: it does not sum to one,
and it is proportional to M.
"""
function stationary_measure(F̂, g; M = 1.0, method = :linear, tol = 1e-12, maxiter = 100_000)
    # :linear: μ = F̂' μ + M g rearranges to (I - F̂') μ = M g, solved in one step
    method === :linear && return (I - F̂') \ (M .* g)
    method === :iterate || throw(ArgumentError("method must be :linear or :iterate"))
    μ0 = zeros(length(g))
    μ1 = similar(μ0)
    for n in 1:maxiter
        # survivors move with F̂, entrants arrive with g
        mul!(μ1, F̂', μ0)
        μ1 .+= M .* g
        # stop when the measure no longer changes
        maximum(abs.(μ1 .- μ0)) < tol * maximum(μ1) && return μ1
        μ0, μ1 = μ1, μ0
    end
    error("the measure did not converge in $maxiter iterations")
end
# <<< stationary_measure

# >>> cohorts
"""
    cohorts(F̂, g, A)

Firms by age, per unit mass of entrants: column a + 1 is the measure of firms of
age a, (F̂')^a g, for a = 0, ..., A. Age 0 is the entrants' first period of
production. Summing the columns over all ages gives (I - F̂')⁻¹ g, the stationary
measure, because each cohort is the previous one moved on by a year.
"""
function cohorts(F̂, g, A)
    C = zeros(length(g), A + 1)
    # age 0: the entrants, spread over the grid by g
    C[:, 1] .= g
    for a in 1:A
        # each year the survivors move on with F̂'; exiters drop out
        C[:, a+1] .= F̂' * C[:, a]
    end
    return C
end
# <<< cohorts
