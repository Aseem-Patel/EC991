# >>> entry_value
"""
    entry_value(p, ind, s; method = :vfi, V_start = nothing)

Steps 1 to 3 at one price (lecture slides 34 to 36): the static decisions at p,
the incumbent's value function, then the value of entering,

    V^e(p) = -c_e + β Σ_i g_i V(z_i; p).

Free entry with a positive mass of entrants needs V^e(p) = 0. Returns V^e, the
firm's solution and the static decisions.
"""
function entry_value(p, ind::Industry, s; method = :vfi, V_start = nothing)
    (; η, β, c_e, c_f) = ind
    # Step 1: labour, output and profit at every grid point, at this price
    dec = static_decisions(p, s.z; η, c_f)
    # Step 2: the incumbent's value function and exit rule, given those profits
    sol = solve_firm(dec.π, s.F, β; method, V_start)
    # Step 3: pay c_e, draw z from g, and receive the discounted value of that z
    Ve = -c_e + β * sum(s.g .* sol.V)
    return (; Ve, sol, dec, p)
end
# <<< entry_value

# >>> solve_price
"""
    solve_price(ind, s; method = :vfi, bracket = (0.1, 10.0), tol_p = 1e-8, warm_start = true,
                callback = (p, Ve) -> nothing)

Step 4 (slide 36): the price at which free entry holds, V^e(p) = 0, by bisection.
V is increasing in p, so V^e is too: if V^e(p) > 0 entry is too attractive and p
falls; if V^e(p) < 0 it rises. The bracket is widened until V^e changes sign
across it. Stops when the bracket is narrower than `tol_p` times p. `warm_start`
starts each value function from the previous price's. Returns the result of
`entry_value` at the price found, and the (p, V^e) tried, in order.
"""
function solve_price(ind::Industry, s; method = :vfi, bracket = (0.1, 10.0), tol_p = 1e-8,
                     warm_start = true, callback = (p, Ve) -> nothing)
    history = Tuple{Float64, Float64}[]
    last = Ref{Any}(nothing)
    # one evaluation of V^e(p), warm-started from the last value function, and logged
    function evaluate(p)
        V_start = warm_start && last[] !== nothing ? last[].sol.V : nothing
        e = entry_value(p, ind, s; method, V_start)
        last[] = e
        push!(history, (p, e.Ve))
        callback(p, e.Ve)
        return e
    end
    lo, hi = bracket
    # widen the bracket until entry is unprofitable at lo and profitable at hi
    while evaluate(lo).Ve > 0; lo /= 2; end
    while evaluate(hi).Ve < 0; hi *= 2; end
    # halve the bracket until it is narrow enough
    e = last[]
    while hi - lo > tol_p * hi
        e = evaluate((lo + hi) / 2)
        # V^e > 0: entering pays, so the price must fall. Otherwise it must rise
        e.Ve > 0 ? (hi = e.p) : (lo = e.p)
    end
    return e, history
end
# <<< solve_price
