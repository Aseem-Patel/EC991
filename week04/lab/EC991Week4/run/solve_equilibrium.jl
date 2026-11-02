# The equilibrium by one method, printing Φ(r) at every step of the bisection, then the model
# statistics of lecture slide 32.
# Run from the package folder:  julia --project=. run/solve_equilibrium.jl [method] [dense]
#   method: egm (default), vfi_grid, vfi_fast, vfi_interp, vfi_interp_howard
#   dense:  add the word to use the grid behind the lecture's figures
# In the REPL, skip the script and call the function:
#   using EC991Week4;  eq = solve_equilibrium(Economy(); method = :vfi_fast);  model_stats(eq)
using EC991Week4, Printf

method = Symbol(get(ARGS, 1, "egm"))
econ = "dense" in ARGS ? dense_economy() : Economy()

show_step(r, Φ) = @printf "  r = %9.6f   Φ(r) = %12.5f\n" r Φ
println("method = $method, n_a = $(econ.n_a), n_dist = $(econ.n_dist), n_y = $(econ.n_y)")
t = @elapsed (eq = solve_equilibrium(econ; method, callback = show_step))
@printf "\n%d steps in %.1f s (includes compilation on the first run)\n\n" length(eq.history) t

m = model_stats(eq)
@printf "Equilibrium wage and interest rate  %8.4f %8.4f\n" m.w m.r
@printf "Capital demand and asset supply     %8.4f %8.4f\n" m.K m.Ea
@printf "Labour supply                       %8.3f\n" m.N
@printf "K/N                                 %8.3f\n" m.K_N
@printf "Output                              %8.3f\n" m.Y
@printf "Assets to output                    %8.3f\n" m.A_Y
@printf "Mass at the limit, constrained      %8.3f %8.3f\n" m.at_limit m.constrained
@printf "Mean and std of assets              %8.3f %8.3f\n" m.mean_a m.std_a
@printf "p25, p50, p95, p99 of assets        %8.3f %8.3f %8.3f %8.3f\n" m.p25 m.p50 m.p95 m.p99
@printf "Gini, top 10%% and top 1%% shares     %8.3f %8.3f %8.3f\n" m.gini m.top10 m.top1
