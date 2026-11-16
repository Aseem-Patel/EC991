# EC991Week6

Julia code for the week 6 lab: Hopenhayn and Rogerson (1993), the week 5 industry with firing costs. Firing a worker costs τ, so the employment a firm inherits, m, is a state next to its productivity z, and firms keep their workers inside an inaction region even when productivity changes. The household supplies labour with linear disutility, which gives week 5's demand C = D̄ / p. The package also measures the misallocation that firing costs cause. Slide numbers in the source are the pages of the week 6 tutorial, *Firing Costs: Hopenhayn and Rogerson*; `src/EC991Week6.jl` lists the files.

Layout: `src/` holds the model and solvers and never plots; `run/` holds scripts that call them, print results and draw figures; `test/` checks the code against known answers; `../week06_code_guide.html` reads each file against the slides.

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"   # once
julia --project=. run/solve_equilibrium.jl            # the equilibrium at τ = 0.1, step by step
julia --project=. run/solve_equilibrium.jl 0          # the same at any other τ
julia --project=. run/inaction_region.jl              # slides 13 and 14: inaction_region.png, inaction_region_high_tax.png
julia --project=. run/inaction_region.jl 0.5          # the second graph at another τ
julia --project=. run/firing_taxes.jl                 # τ = 0, 0.1, 1 side by side; τ from 0 to 1, firing_taxes.png (14 equilibria)
julia --project=. run/make_figures.jl                 # the slides' figures into ../../Figures/
julia --project=. -e "using Pkg; Pkg.test()"          # the checks
```

Or from the REPL, started with `julia --project=.` in this folder:

```julia
using EC991Week6
eq = solve_equilibrium(Economy())
aggregates(eq)
misallocation(eq)
inaction_region(eq.sol, eq.s)
```

Defaults: η = 2/3, β = 0.8, c_e = 20 (workers), c_f = 20 (output), τ = 0.1, D̄ = 1/A = 100; log z an AR(1) with mean 1, ρ = 0.9, σ = 0.2, by Tauchen with 101 points over ±4 standard deviations; entrants draw from the invariant distribution of the chain; employment on n = 0 and 499 points evenly spaced in log n from 0.1 to 8,000. Change any of them by name, as in `Economy(τ = 1.0)`. The firm's problem is solved by value function iteration with an exhaustive search over the employment grid: every candidate n is tried at every state (z, m), and choices stay on the grid.
