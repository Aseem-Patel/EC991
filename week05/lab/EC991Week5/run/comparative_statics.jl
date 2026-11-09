# EC991 Week 5: comparative statics (lecture slides 42 and 43):
# the baseline, a higher entry cost (c_e = 60) and a higher fixed cost (c_f = 30).
# Then the exit rate as c_e rises, with the exit rule on the grid and with the lottery.
# Run from the package folder:  julia --project=. run/comparative_statics.jl
using EC991Week5, Printf

cases = [("baseline", Industry()), ("c_e = 60", Industry(c_e = 60.0)), ("c_f = 30", Industry(c_f = 30.0))]
stats = [industry_stats(solve_equilibrium(ind)) for (_, ind) in cases]
println("                       ", join((@sprintf("%12s", n) for (n, _) in cases)))
for (name, key, fmt) in [("price p", :p, "%12.4f"), ("exit threshold z̃", :z̃, "%12.4f"),
                         ("entry = exit rate", :entry_rate, "%12.4f"), ("mean age", :mean_age, "%12.2f"),
                         ("number of firms", :firms, "%12.4f"), ("average size", :avg_size, "%12.2f"),
                         ("output Y", :Y, "%12.3f"), ("aggregate profits", :profits, "%12.3f")]
    println(rpad("  " * name, 23), join(Printf.format(Printf.Format(fmt), getfield(st, key)) for st in stats))
end

println("\nExit rate as c_e rises: the exit rule on the grid jumps; the lottery does not")
println("   c_e    grid, n_z = 101   lottery, n_z = 101   grid, n_z = 1001")
for c_e in 36.0:2.0:48.0
    r(ind; kw...) = industry_stats(solve_equilibrium(ind; kw...)).entry_rate
    @printf("  %4.0f   %15.4f   %18.4f   %16.4f\n", c_e, r(Industry(; c_e)), r(Industry(; c_e); exit = :lottery),
            r(Industry(; c_e, n_z = 1001)))
end
