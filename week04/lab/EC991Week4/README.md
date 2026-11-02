# EC991Week4

Julia code for the week 4 lab: the equilibrium of the Aiyagari model. The equilibrium loop is written once, and only the household step changes: value function iteration without interpolation (grid search), with interpolation, each with its speed-ups, and the endogenous grid method. From the policy the code computes the joint distribution of wealth and income, by histogram and by Monte Carlo, then the interest rate that clears the market and the model's statistics. Slide numbers in the source refer to the week 4 lecture, *Computing the Aiyagari Equilibrium*, unless they say week 3; `src/EC991Week4.jl` lists the files.

Layout: `src/` holds the model and solvers and never plots; `run/` holds scripts that call them, print results and draw figures; `test/` checks the code against known answers; the Pluto notebook `../week04_lab.jl` walks through the same code with the maths alongside; `../week04_code_guide.html` reads each file against the slides.

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"   # once
julia --project=. run/compare_household.jl            # five methods at one r: time and accuracy
julia --project=. run/solve_distribution.jl           # the joint distribution: histogram vs Monte Carlo
julia --project=. run/solve_equilibrium.jl            # the equilibrium by EGM, then the statistics
julia --project=. run/solve_equilibrium.jl vfi_fast   # the same by another method
julia --project=. run/compare_equilibrium.jl          # the race: every method, same bracket
julia --project=. run/make_figures.jl                 # the slides' figures into ../../Figures/
julia --project=. -e "using Pkg; Pkg.test()"          # the checks
```

Or from the REPL, started with `julia --project=.` in this folder:

```julia
using EC991Week4
eq = solve_equilibrium(Economy(); method = :egm)
model_stats(eq)
```

Lab defaults: β = 0.96, γ = 2, ϕ = 0, α = 0.33, δ = 0.05; 7-state Rouwenhorst with ρ = 0.9, σ = 0.1; 300 asset points on [0, 250] with growth rate ν = 0.025. `dense_economy()` is the grid behind the lecture's figures: 1000 points with ν = 0.008, and the distribution on 5000 points with ν = 0.0016. Setting `EC991_FULL=1` adds the dense grid to the two compare scripts, which then take some minutes, and writes their tables to `../../Figures/`, where `make_figures.jl` reads them for the speed figure.
