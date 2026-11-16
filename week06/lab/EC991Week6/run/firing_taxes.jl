# EC991 Week 6: what firing taxes do, and how much misallocation they cause.
#   1. The equilibrium at τ = 0, 0.1 and 1.0, side by side.
#   2. τ from 0 to 1: misallocation, output, the price and job reallocation (tutorial
#      slide 17); the graph goes to firing_taxes.png in this folder.
# Run from the package folder (14 equilibria):
#   julia --project=. run/firing_taxes.jl
using EC991Week6, Printf, LinearAlgebra
BLAS.set_num_threads(1)     # small matrices: threads only add overhead
include("plotting.jl")

# ---------------------------------------------------------------- 1. three equilibria
rows = [(τ, aggregates(eq), misallocation(eq)) for τ in (0.0, 0.1, 1.0) for eq in (solve_equilibrium(Economy(τ = τ)),)]
println("                                τ = 0     τ = 0.1     τ = 1")
# one row of the table: a label, then the statistic f at each τ in the format fmt
line(name, f, fmt) = println(rpad("  " * name, 30), join([Printf.format(Printf.Format(fmt), f(r)) for r in rows], ""))
line("price p", r -> r[2].p, "%10.4f")
line("output Y", r -> r[2].Y, "%10.3f")
line("labour supply N", r -> r[2].Ns, "%10.3f")
line("labour productivity Y / N", r -> r[2].Y / r[2].Ns, "%10.4f")
line("profits Π", r -> r[2].Π, "%10.3f")
line("firing tax revenue T", r -> r[2].T, "%10.3f")
line("mass of firms B", r -> r[2].B, "%10.4f")
line("mass of entrants M", r -> r[2].M, "%10.4f")
line("exit rate", r -> r[2].exit_rate, "%10.4f")
line("average size", r -> r[2].avg_size, "%10.2f")
line("job reallocation", r -> r[2].job_creation, "%10.4f")
line("misallocation, MPL gap (%)", r -> r[3].mpl_gap, "%10.3f")
line("output lost, 1 - Ω (%)", r -> r[3].output_lost, "%10.3f")
line("share of firms inactive", r -> r[3].share_inactive, "%10.3f")

# ---------------------------------------------------------------- 2. τ from 0 to 1
τs = collect(0.0:0.1:1.0)
sweep = map(τs) do τ
    eq = solve_equilibrium(Economy(τ = τ))
    (aggregates(eq), misallocation(eq))
end
Y0 = sweep[1][1].Y                                     # output at τ = 0, the reference
println("\n     τ    MPL gap %   output lost %    Y / Y(τ=0)   job reallocation")
for (τ, (a, m)) in zip(τs, sweep)
    @printf("  %4.1f   %9.3f   %13.3f   %11.4f   %16.4f\n", τ, m.mpl_gap, m.output_lost, a.Y / Y0, a.job_creation)
end
p1 = base(; xlabel = "firing tax, τ", ylabel = "%", legend = :topleft, title = "misallocation")
plot!(p1, τs, [m.mpl_gap for (a, m) in sweep]; color = c_low, label = "average MPL gap")
plot!(p1, τs, [m.output_lost for (a, m) in sweep]; color = c_high, label = "output lost, 1 - Ω")
p2 = base(; xlabel = "firing tax, τ", ylabel = "% change from τ = 0", legend = :bottomleft, title = "aggregates")
plot!(p2, τs, [100 * (a.Y / Y0 - 1) for (a, m) in sweep]; color = c_high, label = "output Y")
plot!(p2, τs, [100 * (a.Y / a.Ns / (Y0 / sweep[1][1].Ns) - 1) for (a, m) in sweep]; color = c_low, label = "output per worker Y / N")
plot!(p2, τs, [100 * (a.job_creation / sweep[1][1].job_creation - 1) for (a, m) in sweep]; color = c_dark,
      linestyle = :dash, label = "job reallocation")
savefig(plot(p1, p2; layout = (1, 2), size = (1250, 500)), "firing_taxes.png")
println("\nwrote firing_taxes.png")
