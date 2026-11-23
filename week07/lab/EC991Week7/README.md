# EC991Week7

Julia code for the week 7 lab: aggregate shocks and transition dynamics, solved two ways. Part A is the state-space method of Krusell and Smith (1998). Households forecast aggregate capital with a log-linear rule; the code simulates the economy, re-estimates the rule, and repeats until the rule reproduces itself. Part B is the sequence-space method. The week 4 Aiyagari economy is hit by an unexpected productivity shock (an MIT shock) or a labour tax reform, and the shooting algorithm finds the transition. Adding up impulse responses then simulates the economy with many shocks (Boppart, Krusell and Mitman, 2018). Slide numbers in the source are the pages of the week 7 lecture, *Aggregate Shocks and Transition Dynamics*; `src/EC991Week7.jl` lists the files.

Layout: `src/` holds the model and solvers and never plots; `run/` holds scripts that call them, print results and draw figures; `test/` checks the code against known answers; `../week07_code_guide.html` reads each file against the slides.

From this folder:

```
julia --project=. -e "using Pkg; Pkg.instantiate()"   # once
julia --project=. run/solve_krusell_smith.jl          # Part A: the rule, slides 23, 29, 30; graphs for slides 24 and 25
julia --project=. run/mit_shock.jl                    # Part B: the steady state, then an MIT shock (slides 35, 36)
julia --project=. run/mit_shock.jl 0.5                # the same with dampening ω = 0.5: it does not converge
julia --project=. run/labour_tax.jl                   # the transition after a labour tax of 0.2 (slide 38)
julia --project=. run/bkm_simulation.jl               # linearity, then 200 years by adding up responses (slide 44)
julia --project=. run/make_figures.jl                 # the code guide's figures into ../
julia --project=. -e "using Pkg; Pkg.test()"          # the checks
```

Or from the REPL, started with `julia --project=.` in this folder:

```julia
using EC991Week7
eq = solve_krusell_smith(KSEconomy())      # Part A
eq.θ, eq.R2
forecast_errors(eq)
ss = solve_steady_state(Economy())         # Part B
tr = mit_shock(ss; ε = 0.01, ρ_Z = 0.9)
tr.K
```

Part A defaults (lecture slides 18 to 22, quarterly): β = 0.99, log utility, α = 0.36, δ = 0.025; Z ∈ {0.99, 1.01} with P(Z' = Z | Z) = 0.875; y ∈ {0.25, 1}; unemployment 10% in recessions and 4% in booms. Assets on 100 points from 0 to 500, aggregate capital on 7 points within ±15% of the representative-agent steady state; the distribution on 1,000 points; 11,000 quarters simulated, the first 1,000 discarded. Change any of them by name, as in `KSEconomy(T_sim = 4_000)`.

Part B defaults: week 4's economy (β = 0.96, γ = 2, ϕ = 0, α = 0.33, δ = 0.05; 7-state Rouwenhorst with ρ = 0.9, σ = 0.1; 300 asset points on [0, 250]), annual, with productivity Z = 1 and no tax in the steady state. The MIT shock is log Z_1 = 0.01, dying out at ρ_Z = 0.9; transitions last T = 300 years; dampening ω = 0.2.
