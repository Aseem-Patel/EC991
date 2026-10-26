"""
EC991 week 3 lab: the household problem with income risk, solved by value
function iteration on a grid of assets and a Markov chain for income.

The files are in the order of the tutorial slides:

  household.jl   parameters (β, γ, ϕ, r, w), R = 1 + r, CRRA utility, w_from_r
  grids.jl       the asset grid, equal or growth-rate spacing, a_1 = -ϕ
  markov.jl      Tauchen and Rouwenhorst, the stationary distribution, chain
                 moments, and a chain simulated from given shocks
  bellman.jl     one Bellman step by brute-force grid search: V P' and the loop
  interpolation.jl  linear interpolation and the natural cubic spline
  optimise.jl    golden-section search and Brent's method (maximisation)
  bellman_interp.jl one Bellman step by interpolate-then-optimise
  vfi.jl         iterate either step to its fixed point
  checks.jl      chain against AR(1), monotone policy, gap between two policies

Nothing here plots or prints. Plotting lives in the run scripts and the notebook.
"""
module EC991Week3

using LinearAlgebra

include("household.jl")
include("grids.jl")
include("markov.jl")
include("bellman.jl")
include("interpolation.jl")
include("optimise.jl")
include("bellman_interp.jl")
include("vfi.jl")
include("checks.jl")

export Household, utility, w_from_r
export asset_grid
export normal_cdf, tauchen, rouwenhorst, stationary, chain_moments, simulate_chain
export bellman_step!
export LinearInterp, CubicSpline
export golden_section, brent
export bellman_step_interp!
export Solution, initial_guess, solve_vfi
export chain_vs_ar1, policy_monotone, compare_policies

end
