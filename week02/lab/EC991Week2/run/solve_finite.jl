# Finite horizon: backward induction, the cake-eating check, and a life cycle.
# Run from the package folder:  julia --project=. run/solve_finite.jl
using EC991Week2

# Cake eating: y_t = 0, R = 1, u = log. The closed form is exact; the grid is not.
T, a0 = 20, 1.0
cake = Household(β = 0.95, R = 1.0, y = 0.0, ϕ = 0.0, γ = 1.0)
println("Cake eating, T = $T, a_1 = $a0: max |c_numerical - c_closed form| by grid size")
for na in (100, 400, 1600)
    a = collect(range(0.0, a0; length = na))
    fsol = solve_backward(cake, a, zeros(T))
    apath, cpath = simulate_finite(fsol, a0)
    exact = [cake_eating_consumption(apath[t], t, T, cake.β) for t in 1:T]
    println("  na = $(lpad(na, 4)):  $(round(maximum(abs.(cpath .- exact)); sigdigits = 3))")
end

# Life cycle: hump-shaped income, borrowing limit, no bequest.
hh = Household()
T = 60
y = life_cycle_income(T)
a = asset_grid(hh; amax = 15.0, na = 600)
fsol = solve_backward(hh, a, y)
apath, cpath = simulate_finite(fsol, 0.0)
println("\nLife cycle, T = $T: peak assets $(round(maximum(apath); digits = 2)) at t = $(argmax(apath)); ",
        "terminal assets $(apath[end]); consumption range $(round.(extrema(cpath); digits = 3))")
