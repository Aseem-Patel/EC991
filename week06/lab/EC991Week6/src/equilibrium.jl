# >>> output_per_firm
"""
    output_per_firm(sol, s, eco)

Net output of a firm in state (z_i, m = n_k), after it chooses n = n^d(z_i, n_k):
z_i n^η - c_f.
"""
output_per_firm(sol::FirmSolution, s, eco::Economy) = s.z .* s.n[sol.pol] .^ eco.η .- eco.c_f
# <<< output_per_firm

# >>> Equilibrium
"""
The stationary equilibrium: the economy `eco`, the price `p`, the mass of entrants
`M`, the measure `μ[i, k]` of firms that start the period with productivity z_i and
inherited employment m = n_k, the incumbent's solution `sol`, the objects from `setup` in
`s`, the value of entry `Ve` (zero up to the tolerance) and the (p, V^e) tried.
"""
struct Equilibrium
    eco::Economy
    p::Float64
    M::Float64
    μ::Matrix{Float64}
    sol::FirmSolution
    s::NamedTuple
    Ve::Float64
    history::Vector{Tuple{Float64, Float64}}
end
# <<< Equilibrium

# >>> solve_equilibrium
"""
    solve_equilibrium(eco = Economy(); kwargs...)

The stationary equilibrium, in the steps of tutorial slide 6:

  1. Solve the firm's problem at a trial price, and
  2. adjust the price by bisection until free entry holds (`solve_price`).
  3. Compute the measure for one unit of entrants, μ(p, 1)
     (`stationary_distribution`), then the mass of entrants from the goods
     market. Demand is C = D̄ / p and the measure scales with M, so

      M = (D̄ / p) / Σ_{i,k} μ(p, 1)_{ik} [z_i n^d(z_i, n_k)^η - c_f].

The labour market then clears by accounting: Π = p Y - N^d - T and p Y = D̄ give
N^d = D̄ - Π - T, the household's labour supply. `aggregates` checks it.
Keywords go to `solve_price`.
"""
function solve_equilibrium(eco::Economy = Economy(); kwargs...)
    # the grids z and n, the transition matrix F and the entrants' probabilities g
    s = setup(eco)
    # 1 and 2: the firm's problem at each trial price, and the price at which free entry holds
    e, history = solve_price(eco, s; kwargs...)
    # 3: the measure for one unit of entrants ...
    μ1 = stationary_distribution(e.sol, s)
    # ... then the mass of entrants that clears the goods market, C = D̄ / p
    Y1 = sum(output_per_firm(e.sol, s, eco) .* μ1)
    Y1 > 0 || error("output per unit of entrants is not positive at p = $(e.p)")
    M = (eco.D̄ / e.p) / Y1
    # store everything, with the measure scaled up: μ = M μ1
    return Equilibrium(eco, e.p, M, M .* μ1, e.sol, s, e.Ve, history)
end
# <<< solve_equilibrium
