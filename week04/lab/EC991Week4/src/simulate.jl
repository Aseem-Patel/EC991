# >>> simulate_panel
"""
    simulate_panel(g, a, y, P; n_households = 50_000, T = 1000, seed = 1)

Method 1 of slide 20, Monte Carlo: follow `n_households` households for `T` periods
under the saving policy `g` on the grid `a`, with a' read off by linear
interpolation and income drawn from P, and keep the final cross-section. Everyone
starts at the borrowing limit with income drawn from the stationary distribution,
so `T` must be long enough to forget that start. Returns each household's assets
and the index of its income state.
"""
function simulate_panel(g, a, y, P; n_households = 50_000, T = 1000, seed = 1)
    rng = Random.Xoshiro(seed)
    C = cumsum(P; dims = 2)
    ĝ = [LinearInterp(a, g[:, j]) for j in eachindex(y)]
    Cπ = cumsum(stationary(P))
    assets, income = fill(a[1], n_households), [min(searchsortedfirst(Cπ, rand(rng)), length(y)) for _ in 1:n_households]
    for _ in 1:T, h in 1:n_households
        j = income[h]
        assets[h] = clamp(ĝ[j](assets[h]), a[1], a[end])
        income[h] = min(searchsortedfirst(view(C, j, :), rand(rng)), length(y))
    end
    return (a = assets, j = income)
end
# <<< simulate_panel
