# EC991 Week 5: a 10-year panel of firms simulated from the equilibrium.
# Prints the panel's averages next to the model's, and writes the panel to
# panel.csv (one row per firm and year). Run from the package folder:
#   julia --project=. run/simulate_panel.jl
using EC991Week5, Printf, Random

eq = solve_equilibrium(Industry())
st = industry_stats(eq)

# 10 years, about 1000 firms a year, a fixed seed so the numbers repeat
panel = simulate_panel(eq; T = 10, N = 1000, rng = MersenneTwister(1234))

# averages over all firm-years, against the model's stationary values
println("Simulated panel: $(length(unique(panel.firm))) firms, $(length(panel.firm)) firm-years")
println("                      panel     model")
@printf("  exit rate          %7.4f   %7.4f\n", sum(panel.exit) / length(panel.exit), st.exit_rate)
@printf("  entry rate         %7.4f   %7.4f\n", sum(panel.age .== 0) / length(panel.age), st.entry_rate)
@printf("  average size       %7.2f   %7.2f\n", sum(panel.n) / length(panel.n), st.avg_size)
@printf("  mean age           %7.2f   %7.2f\n", sum(panel.age) / length(panel.age), st.mean_age)

# write the panel as a CSV file, one row per firm and year
open("panel.csv", "w") do io
    println(io, "firm,year,z,n,y,age,exit")
    for r in eachindex(panel.firm)
        @printf(io, "%d,%d,%.6f,%.4f,%.4f,%d,%d\n", panel.firm[r], panel.year[r], panel.z[r],
                panel.n[r], panel.y[r], panel.age[r], panel.exit[r])
    end
end
println("Wrote panel.csv")
