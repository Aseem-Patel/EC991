"""
EC991 week 2 lab: the deterministic saving problem by value function
iteration, and its finite-horizon version by backward induction.

The files are in the order of the lab slides:

  household.jl   parameters (β, R, y, ϕ, γ) and the CRRA utility function
  grid.jl        the asset grid, a_1 = -ϕ
  bellman.jl     one application of the Bellman operator T: the double loop
  vfi.jl         iterate T to its fixed point (infinite horizon)
  backward.jl    walk T back from the terminal date (finite horizon)
  simulate.jl    follow the policy forward from a starting asset level
  checks.jl      predictions from theory and the cake-eating closed form

Nothing here plots. Plotting lives in the run scripts and the notebook.
"""
module EC991Week2

include("household.jl")
include("grid.jl")
include("bellman.jl")
include("vfi.jl")
include("backward.jl")
include("simulate.jl")
include("checks.jl")

export Household, utility, marginal_utility
export asset_grid
export bellman_step!
export Solution, initial_guess, solve_vfi
export FiniteSolution, solve_backward
export simulate, simulate_finite
export euler_residuals, cake_eating_consumption, cake_eating_path, life_cycle_income

end
