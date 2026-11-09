# >>> mass_of_entrants
"""
    mass_of_entrants(p, y, μ1; D̄)

Step 6 (lecture slide 38). The measure is linear in M, μ = M μ1 with μ1 the
measure for one unit of entrants, so goods market clearing D(p) = Σ_i y_i μ_i
gives M directly:

    M = D(p) / Σ_i y(z_i; p) μ1_i,   D(p) = D̄ / p.

With this demand M is always positive, so the no-entry case of slide 39 does not
arise; `solve_equilibrium` still checks.
"""
mass_of_entrants(p, y, μ1; D̄) = (D̄ / p) / sum(y .* μ1)
# <<< mass_of_entrants

# >>> Equilibrium
"""
The stationary equilibrium found by `solve_equilibrium`: the price `p`, the mass
of entrants `M`, the measure of firms `μ` on the grid, the incumbent's solution
`sol` (V, exit rule on the grid), the exit shares `χ` that the measure uses (0 or 1
with `exit = :grid`; one fractional cell with `exit = :lottery`), the static
decisions `n`, `y`, `π`, the exit threshold
`z̃`, the value of entry `Ve` (zero up to the tolerance), the objects from `setup`
in `s`, and `history`, the (p, V^e) tried in order.
"""
struct Equilibrium
    ind::Industry
    method::Symbol
    exit::Symbol
    p::Float64
    M::Float64
    μ::Vector{Float64}
    sol::FirmSolution
    χ::Vector{Float64}
    n::Vector{Float64}
    y::Vector{Float64}
    π::Vector{Float64}
    z̃::Float64
    Ve::Float64
    s::NamedTuple
    history::Vector{Tuple{Float64, Float64}}
end
# <<< Equilibrium

# >>> solve_equilibrium
"""
    solve_equilibrium(ind = Industry(); method = :vfi, exit = :grid, kwargs...)

The stationary equilibrium in two blocks (slide 32): first the price from free
entry alone (Steps 1 to 4, `solve_price`), then the mass of entrants from the
goods market (Steps 5 and 6). The price block never needs M or μ: that is what
makes the model easy to solve. `method` is passed to `solve_firm`; other keywords
to `solve_price`. `exit = :lottery` splits the grid cell that straddles the exit
threshold between exit and stay (`exit_lottery`); the price is unchanged, since
it depends only on V.
"""
function solve_equilibrium(ind::Industry = Industry(); method = :vfi, exit = :grid, kwargs...)
    # the grid z, the transition matrix F and the entrants' probabilities g
    s = setup(ind)
    # Steps 1 to 4: the price at which free entry holds
    e, history = solve_price(ind, s; method, kwargs...)
    # the exit rule the measure uses: 0 or 1 on the grid, or with the lottery cell
    χ = exit === :grid ? Float64.(e.sol.χ) : exit === :lottery ? exit_lottery(e.sol, s.z) :
        throw(ArgumentError("exit must be :grid or :lottery"))
    # Step 5: remove the exiters' rows from F, then the measure for one unit of entrants
    F̂ = exit_transition(s.F, χ)
    μ1 = stationary_measure(F̂, s.g)
    # Step 6: the mass of entrants that clears the goods market
    M = mass_of_entrants(e.p, e.dec.y, μ1; ind.D̄)
    M > 0 || error("no entry at p = $(e.p): see slide 39")
    # store everything, with the measure scaled up: μ = M μ1
    return Equilibrium(ind, method, exit, e.p, M, M .* μ1, e.sol, χ, e.dec.n, e.dec.y, e.dec.π,
                       exit_threshold(e.sol, s.z).z̃, e.Ve, s, history)
end
# <<< solve_equilibrium
