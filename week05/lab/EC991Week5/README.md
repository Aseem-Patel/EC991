# EC991Week5

Julia code for the week 5 lab: the Hopenhayn (1992) model of industry dynamics. Firms differ in productivity, which follows a Markov chain; they pay a fixed cost every period they produce and an entry cost to come in. The equilibrium has two blocks: the price comes from free entry alone, then the mass of entrants from the goods market. The package also covers the lecture's static economy: aggregation, misallocation from wedges and the planner's choice of entry. Slide numbers in the source are the pages of the week 5 lecture, *Firm Dynamics: the Hopenhayn Model*; `src/EC991Week5.jl` lists the files.

Layout: `src/` holds the model and solvers and never plots; `run/` holds scripts that call them, print results and draw figures; `test/` checks the code against known answers; `../week05_code_guide.html` reads each file against the slides.

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"   # once
julia --project=. run/static_examples.jl              # the static economy: two firms, wedges, entry
julia --project=. run/solve_equilibrium.jl            # the equilibrium by VFI, step by step
julia --project=. run/solve_equilibrium.jl howard     # the same with Howard (or policy_iteration)
julia --project=. run/compare_methods.jl              # VFI, Howard and policy iteration: iterations and time
julia --project=. run/comparative_statics.jl          # higher c_e, higher c_f; exit on the grid vs the lottery
julia --project=. run/simulate_panel.jl               # a 10-year panel of firms, written to panel.csv
julia --project=. run/make_figures.jl                 # the slides' figures into ../../Figures/
julia --project=. -e "using Pkg; Pkg.test()"          # the checks
```

Or from the REPL, started with `julia --project=.` in this folder:

```julia
using EC991Week5
eq = solve_equilibrium(Industry())
industry_stats(eq)
```

Defaults: η = 2/3, β = 0.8, c_e = 40, c_f = 20, D̄ = 100; log z an AR(1) with mean 1, ρ = 0.9, σ = 0.2, by Tauchen with 101 points over ±4 standard deviations; entrants draw from the invariant distribution of the chain. Change any of them by name, as in `Industry(c_e = 60)`. `solve_equilibrium(ind; exit = :lottery)` splits the grid cell that contains the exit threshold between exit and stay. Setting `EC991_FULL=1` makes `compare_methods.jl` write its table to `../../Figures/` for the lecture.
