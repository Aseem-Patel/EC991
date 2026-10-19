# Figures for the lab deck (slide 12) and the notebook, written to Week-2/Figures/.
# Run from the package folder:  julia --project=. run/make_figures.jl
using EC991Week2, Plots

figdir = normpath(joinpath(@__DIR__, "..", "..", "..", "Figures"))
mkpath(figdir)
default(fontfamily = "Computer Modern", guidefontsize = 11, tickfontsize = 9,
        legendfontsize = 9, titlefontsize = 12, linewidth = 2, framestyle = :box,
        grid = false, legend = :topleft)

hh = Household()
a = asset_grid(hh; amax = 3.0, na = 300)
sol = solve_vfi(hh, a)
apath, cpath = simulate(sol, 2.5, 120)

p1 = plot(a, sol.V; label = "", xlabel = "a", title = "Value function V(a)")
p2 = plot(a, sol.apol; label = "a'(a)", xlabel = "a", title = "Savings policy")
plot!(p2, a, a; label = "45-degree line", linestyle = :dash, color = :gray)
p3 = plot(a, sol.cpol; label = "", xlabel = "a", title = "Consumption c(a)")
p4 = plot(0:120, apath; label = "", xlabel = "t", title = "Asset path from a0 = 2.5")
hline!(p4, [-hh.ϕ]; label = "", linestyle = :dash, color = :gray)
fig = plot(p1, p2, p3, p4; layout = (1, 4), size = (1700, 440), margin = 6Plots.mm, bottom_margin = 9Plots.mm)
savefig(fig, joinpath(figdir, "week02_vfi_solution.pdf"))

# Cake eating: numerical against the closed form, and the error by grid size.
T, a0 = 20, 1.0
cake = Household(β = 0.95, R = 1.0, y = 0.0, ϕ = 0.0, γ = 1.0)
nas = (50, 100, 200, 400, 800, 1600)
errs = Float64[]
for na in nas
    ag = collect(range(0.0, a0; length = na))
    fs = solve_backward(cake, ag, zeros(T))
    ap, cp = simulate_finite(fs, a0)
    push!(errs, maximum(abs.(cp .- [cake_eating_consumption(ap[t], t, T, cake.β) for t in 1:T])))
end
ag = collect(range(0.0, a0; length = 200))
fs = solve_backward(cake, ag, zeros(T))
ap, cp = simulate_finite(fs, a0)
exact = [cake_eating_consumption(ap[t], t, T, cake.β) for t in 1:T]
q1 = plot(1:T, exact; label = "closed form", xlabel = "t", title = "Cake eating, na = 200")
scatter!(q1, 1:T, cp; label = "backward induction", markersize = 4)
q2 = plot(collect(nas), errs; xscale = :log10, yscale = :log10, marker = :circle, label = "",
          xlabel = "grid points na", title = "Max error against the closed form")
savefig(plot(q1, q2; layout = (1, 2), size = (1000, 400), margin = 6Plots.mm),
        joinpath(figdir, "week02_cake_eating.pdf"))

# Life cycle.
y = life_cycle_income(60)
ag = asset_grid(hh; amax = 15.0, na = 600)
fs = solve_backward(hh, ag, y)
ap, cp = simulate_finite(fs, 0.0)
r1 = plot(1:60, y; label = "income", xlabel = "t", title = "Income and consumption")
plot!(r1, 1:60, cp; label = "consumption")
r2 = plot(1:61, ap; label = "", xlabel = "t", title = "Assets")
savefig(plot(r1, r2; layout = (1, 2), size = (1000, 400), margin = 6Plots.mm),
        joinpath(figdir, "week02_life_cycle.pdf"))
println("figures written to $figdir")
