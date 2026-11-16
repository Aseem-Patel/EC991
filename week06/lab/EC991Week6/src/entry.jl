# >>> entry_value
"""
    entry_value(p, eco, s; V_start = nothing)

The first two steps of the algorithm at one price (tutorial slides 4 and 6): solve
the incumbent's problem at p, then the value of entering. An entrant pays c_e,
draws z with probabilities g_i = G(z_i) and starts next period with no inherited
workers, m = 0:

    V^e(p) = -c_e + β Σ_i g_i V(z_i, 0; p).

Free entry with a positive mass of entrants needs V^e(p) = 0.
"""
function entry_value(p, eco::Economy, s; V_start = nothing)
    # Step 1: the incumbent's problem at p
    sol = solve_firm(p, eco, s; V_start)
    # the value of entry: entrants start with no workers, the first column of V
    Ve = -eco.c_e + eco.β * sum(s.g .* sol.V[:, 1])
    return (; Ve, sol, p)
end
# <<< entry_value

# >>> solve_price
"""
    solve_price(eco, s; bracket = (0.5, 4.0), tol_p = 1e-8, callback = (p, Ve) -> nothing)

The price at which free entry holds, V^e(p) = 0, by bisection, as in week 5: a
higher p raises profit at every state, so V^e rises with p. If V^e(p) > 0 entry
pays and p must fall; if V^e(p) < 0 it must rise. Each value function starts from
the previous price's. Returns the result of `entry_value` at the price found and
the (p, V^e) tried, in order.
"""
function solve_price(eco::Economy, s; bracket = (0.5, 4.0), tol_p = 1e-8, callback = (p, Ve) -> nothing)
    history = Tuple{Float64, Float64}[]
    last = Ref{Any}(nothing)                      # the most recent result of entry_value
    function evaluate(p)
        # start the value function from the last price's V
        V_start = last[] === nothing ? nothing : last[].sol.V
        e = entry_value(p, eco, s; V_start)
        last[] = e
        # record the price tried and V^e there
        push!(history, (p, e.Ve))
        callback(p, e.Ve)
        return e
    end
    lo, hi = bracket
    # widen the bracket until entry is unprofitable at lo and profitable at hi
    while evaluate(lo).Ve > 0; lo /= 2; end
    while evaluate(hi).Ve < 0; hi *= 2; end
    e = last[]
    # bisect until the bracket is narrower than tol_p, relative to the price
    while hi - lo > tol_p * hi
        e = evaluate((lo + hi) / 2)
        # entry pays: the price is too high; entry loses: it is too low
        e.Ve > 0 ? (hi = e.p) : (lo = e.p)
    end
    return e, history
end
# <<< solve_price
