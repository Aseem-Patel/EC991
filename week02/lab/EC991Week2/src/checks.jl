# >>> euler_residuals
"""
    euler_residuals(cpath, hh)

The Euler inequality along a path (slide 7):  u'(c_t) - β R u'(c_{t+1}),
for t = 1, ..., T-1. Theory predicts it is ≥ 0, with equality while the
borrowing limit is slack. On a grid the equality holds only approximately.
"""
function euler_residuals(cpath, hh::Household)
    T = length(cpath)
    return [marginal_utility(cpath[t], hh) - hh.β * hh.R * marginal_utility(cpath[t + 1], hh)
            for t in 1:T - 1]
end
# <<< euler_residuals

# >>> cake_eating
"""
    cake_eating_consumption(a, t, T, β)

Closed form for the cake-eating problem (y_t = 0, R = 1, u = log),
slide 15:  c_t = (1 - β) / (1 - β^(T - t + 1)) · a_t.
"""
cake_eating_consumption(a, t, T, β) = (1 - β) / (1 - β^(T - t + 1)) * a

"""
    cake_eating_path(a0, T, β)

The exact asset and consumption paths of the cake-eating problem from a_1 = a0.
"""
function cake_eating_path(a0, T, β)
    apath = zeros(T + 1)
    cpath = zeros(T)
    apath[1] = a0
    for t in 1:T
        cpath[t] = cake_eating_consumption(apath[t], t, T, β)
        apath[t + 1] = apath[t] - cpath[t]
    end
    return apath, cpath
end
# <<< cake_eating

# >>> life_cycle_income
"""
    life_cycle_income(T; peak = 2 / 3, height = 1.5, retire = 0.4)

A hump-shaped income profile for the finite-horizon exercise: income rises
to `height` times its starting level at fraction `peak` of working life,
then drops to a pension of `retire` for the final quarter of life.
"""
function life_cycle_income(T; peak = 2 / 3, height = 1.5, retire = 0.4)
    Tw = round(Int, 0.75 * T)
    y = zeros(T)
    for t in 1:T
        if t <= Tw
            x = (t - 1) / (Tw - 1)
            y[t] = 1 + (height - 1) * (1 - ((x - peak) / peak)^2)
        else
            y[t] = retire
        end
    end
    return y
end
# <<< life_cycle_income
