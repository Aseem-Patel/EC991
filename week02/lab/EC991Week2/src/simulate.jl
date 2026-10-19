# >>> simulate
nearest_index(a, a0) = argmin(abs.(a .- a0))

"""
    simulate(sol::Solution, a0, T)

Follow the converged policy forward for T periods from the grid point
nearest to `a0`. Returns the asset path (length T+1) and the consumption
path (length T). Because a' is on the grid, the path never leaves it.
"""
function simulate(sol::Solution, a0, T)
    i = nearest_index(sol.a, a0)
    apath = zeros(eltype(sol.a), T + 1)
    cpath = zeros(eltype(sol.a), T)
    apath[1] = sol.a[i]
    for t in 1:T
        cpath[t] = sol.cpol[i]
        i = sol.pol[i]
        apath[t + 1] = sol.a[i]
    end
    return apath, cpath
end

"""
    simulate_finite(fsol::FiniteSolution, a0)

Step 3 of backward induction: follow the date-t policies from the grid
point nearest to `a0` at t = 1 through the terminal date.
"""
function simulate_finite(fsol::FiniteSolution, a0)
    T = size(fsol.pol, 2)
    i = nearest_index(fsol.a, a0)
    apath = zeros(eltype(fsol.a), T + 1)
    cpath = zeros(eltype(fsol.a), T)
    apath[1] = fsol.a[i]
    for t in 1:T
        cpath[t] = fsol.cpol[i, t]
        i = fsol.pol[i, t]
        apath[t + 1] = fsol.a[i]
    end
    return apath, cpath
end
# <<< simulate
