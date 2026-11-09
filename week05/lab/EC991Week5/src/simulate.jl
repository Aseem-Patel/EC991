# >>> simulate_panel
"""
    simulate_panel(eq; T = 10, N = 1000, burn_in = 100, rng = Random.default_rng())

A panel of individual firms simulated from the equilibrium `eq`, for `T` years.
Each year has the timing of the law of motion μ' = F̂' μ + M g:

  1. entrants arrive, each drawing its productivity from g, and produce at age 0;
  2. every firm produces n(z), y(z) at the equilibrium price;
  3. each firm exits with probability χ(z) (0 or 1 on the grid, a share in the
     lottery cell);
  4. survivors draw next year's productivity from row z of F, and age by one year.

The number of entrants per year is N M / Σμ, so the industry settles at about
N firms. The simulation starts from an empty industry and runs `burn_in` years
before recording, so the firms' ages are known and the cross-section has reached
the stationary measure. Returns the panel as columns, one row per firm and year:
`firm` (an id), `year` (1 to T), `iz` (grid index), `z`, `n`, `y`, `age` and
`exit` (true if the firm leaves at the end of that year).
"""
function simulate_panel(eq::Equilibrium; T = 10, N = 1000, burn_in = 100, rng = Random.default_rng())
    (; F, g, z) = eq.s
    # entrants per year, so that the industry holds about N firms
    n_entrants = round(Int, N * eq.M / sum(eq.μ))
    # cumulative probabilities: a uniform draw u picks the first index where the sum exceeds u
    G = cumsum(g)
    Fcum = cumsum(F; dims = 2)
    draw(cdf) = min(searchsortedfirst(cdf, rand(rng)), length(cdf))
    # the firms alive at the start of the year: id, grid index, age
    ids, iz, age = Int[], Int[], Int[]
    next_id = 1
    # the panel, one entry per firm and recorded year
    panel = (; firm = Int[], year = Int[], iz = Int[], z = Float64[], n = Float64[],
             y = Float64[], age = Int[], exit = Bool[])
    for t in 1:(burn_in + T)
        # 1. entrants arrive, draw z from g, and start at age 0
        for _ in 1:n_entrants
            push!(ids, next_id); push!(iz, draw(G)); push!(age, 0)
            next_id += 1
        end
        # 3. each firm draws its exit decision: exit with probability χ at its grid point
        leaves = [rand(rng) < eq.χ[i] for i in iz]
        # 2. after the burn-in, record what every firm produces this year
        if t > burn_in
            append!(panel.firm, ids); append!(panel.year, fill(t - burn_in, length(ids)))
            append!(panel.iz, iz); append!(panel.z, z[iz])
            append!(panel.n, eq.n[iz]); append!(panel.y, eq.y[iz])
            append!(panel.age, age); append!(panel.exit, leaves)
        end
        # 4. survivors move to next year's productivity, drawn from their row of F, and age
        stay = .!leaves
        ids, age = ids[stay], age[stay] .+ 1
        iz = [draw(view(Fcum, i, :)) for i in iz[stay]]
    end
    return panel
end
# <<< simulate_panel
