"""
EC991 week 4 lab: the Aiyagari equilibrium. The loop is written once; only the
household step changes, between value function iteration with and without
interpolation and the endogenous grid method.

From week 3, with the slide numbers still those of the week 3 tutorial:

  household.jl   parameters (β, γ, ϕ, r, w), R = 1 + r, CRRA utility, w_from_r
  grids.jl       the asset grid, equal or growth-rate spacing, a_1 = -ϕ
  markov.jl      Tauchen and Rouwenhorst, the stationary distribution, chain
                 moments, and a chain simulated from given shocks
  bellman.jl     one Bellman step by brute-force grid search: V P' and the loop
  interpolation.jl  linear interpolation and the natural cubic spline
  optimise.jl    golden-section search and Brent's method (maximisation)
  bellman_interp.jl one Bellman step by interpolate-then-optimise
  checks.jl      chain against AR(1), monotone policy, gap between two policies

New this week, in the order of the lecture slides:

  vfi_speedups.jl  monotonicity and concavity in the grid search; Howard's updates
  vfi.jl         iterate any Bellman step to its fixed point (from week 3, now
                 with Howard's updates and a warm start)
  egm.jl         the endogenous grid method
  euler_errors.jl  how accurate is a solution?
  distribution.jl  the histogram: Young's lottery, Q, the stationary distribution
  simulate.jl    the same distribution by Monte Carlo
  equilibrium.jl the economy, the methods behind one name, the loop on r
  statistics.jl  the table of model statistics, MPCs, densities, the joint table

Nothing here plots or prints. Plotting lives in the run scripts and the notebook.
"""
module EC991Week4

using LinearAlgebra
using Random

include("household.jl")
include("grids.jl")
include("markov.jl")
include("bellman.jl")
include("interpolation.jl")
include("optimise.jl")
include("bellman_interp.jl")
include("vfi_speedups.jl")
include("vfi.jl")
include("egm.jl")
include("euler_errors.jl")
include("distribution.jl")
include("simulate.jl")
include("equilibrium.jl")
include("statistics.jl")
include("checks.jl")

export Household, utility, w_from_r
export asset_grid
export normal_cdf, tauchen, rouwenhorst, stationary, chain_moments, simulate_chain
export bellman_step!
export LinearInterp, CubicSpline
export golden_section, brent
export bellman_step_interp!
export bellman_step_fast!, howard_update!
export Solution, initial_guess, solve_vfi
export marginal_utility, inverse_marginal_utility, endogenous_assets, egm_step!, solve_egm
export euler_error, euler_errors
export young_weights, policy_on_grid, build_Q, stationary_distribution, absorbing_states
export simulate_panel
export Economy, dense_economy, setup, capital_demand, solve_household, excess_demand, Equilibrium, solve_equilibrium
export percentile, wealth_inequality, model_stats, mpc, conditional_density, joint_table, panel_table
export chain_vs_ar1, policy_monotone, compare_policies

end
