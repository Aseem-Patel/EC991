# EC991Week2

Julia code for the week 2 lab: the deterministic saving problem by value function iteration, and the finite-horizon problem by backward induction. The source follows the lab slides; `src/EC991Week2.jl` lists the files in slide order.

Layout: `src/` holds the model and solvers and never plots; `run/` holds scripts that call them and make figures; `test/` checks the solvers against theory; the Pluto notebook `../week02_lab.jl` walks through the same code with the maths alongside.

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"   # once
julia --project=. run/solve_infinite.jl               # VFI and the two predictions
julia --project=. run/solve_finite.jl                 # cake eating and a life cycle
julia --project=. run/make_figures.jl                 # figures into ../../Figures/
julia --project=. -e "using Pkg; Pkg.test()"          # the checks
```
