# EC991Week3

Julia code for the week 3 lab: the household problem with income risk, solved by value function iteration. The income process is discretised by Tauchen or Rouwenhorst; the maximisation is done by grid search or by interpolating (linear or cubic spline) and optimising (Brent or golden section). The source follows the tutorial slides, *Solving Aiyagari, Part 1*; `src/EC991Week3.jl` lists the files in slide order.

Layout: `src/` holds the model and solvers and never plots; `run/` holds scripts that call them, print results and draw figures (into `output/`); `test/` checks the code against known answers; the Pluto notebook `../week03_lab.jl` walks through the same code with the maths alongside; `../week03_code_guide.html` reads each file against the slides.

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"   # once
julia --project=. run/discretise_ar1.jl               # Tauchen vs Rouwenhorst
julia --project=. run/solve_household.jl              # the 3 x 3 example, then VFI by grid search
julia --project=. run/compare_grids.jl                # equal vs growth-rate spacing
julia --project=. run/compare_maximisation.jl         # grid search vs interpolate-then-optimise
julia --project=. run/make_figures.jl                 # the deck's figures into ../../Figures/
julia --project=. -e "using Pkg; Pkg.test()"          # the checks
```

Lab defaults: β = 0.96, γ = 2, ϕ = 0, r = 0.03 with w from the firm's first-order condition (α = 0.33, δ = 0.05); 3-state Rouwenhorst with ρ = 0.9, σ = 0.1; 300 asset points on [0, 50] with growth rate ν = 0.02.
