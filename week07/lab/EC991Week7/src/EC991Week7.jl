"""
EC991 week 7 lab: aggregate shocks and transition dynamics. Two ways to put
aggregate productivity Z_t into a heterogeneous-agent economy. Slide numbers in
the source are pages of the week 7 lecture, *Aggregate Shocks and Transition
Dynamics*.

Shared tools, from weeks 3 and 4:

  markov.jl          Rouwenhorst, the stationary distribution, a simulated chain
  tools.jl           the asset grid, linear interpolation, CRRA marginal utility,
                     Young's lottery, the Gini coefficient and top shares

Part A, the state-space method of Krusell and Smith (slides 3 to 30):

  ks_economy.jl      the parameters of slides 18 to 22, the 4 × 4 transition
                     matrix and aggregate consistency, factor prices
  ks_household.jl    the perceived law of motion and the household's problem by
                     EGM over (a, y, Z, K)
  ks_simulate.jl     the shocks, and K_t simulated with a histogram
  ks_equilibrium.jl  the regression, the update of the rule, forecast accuracy
  ks_statistics.jl   aggregate time series, the business-cycle and wealth tables

Part B, the sequence-space method (slides 31 to 45):

  aiyagari.jl        week 4's economy with Z and a labour tax; one EGM step
                     backwards in time; the steady state
  transition.jl      the shooting algorithm: households backwards, the
                     distribution forwards; the MIT shock and the tax reform
  bkm.jl             impulse responses per unit of shock and the simulation of
                     Boppart, Krusell and Mitman

Nothing here plots or prints. Plotting lives in the run scripts.
"""
module EC991Week7

using LinearAlgebra
using Random

include("markov.jl")
include("tools.jl")
include("ks_economy.jl")
include("ks_household.jl")
include("ks_simulate.jl")
include("ks_equilibrium.jl")
include("ks_statistics.jl")
include("aiyagari.jl")
include("transition.jl")
include("bkm.jl")

export rouwenhorst, stationary, simulate_chain
export asset_grid, LinearInterp, bracket, marginal_utility, inverse_marginal_utility,
       young_weights, move_assets!, gini, top_share
export KSEconomy, employment_transitions, joint_transition, ks_prices, ks_setup
export forecast_K, KSSolution, ks_egm_step!, solve_ks_household, individual_rule
export simulate_Z, policy_on_dist, initial_distribution, simulate_capital
export regress_law_of_motion, KSEquilibrium, solve_krusell_smith, forecast_errors
export aggregate_series, business_cycle_stats, wealth_stats
export Economy, setup, prices, income, egm_backward!, stationary_distribution, SteadyState, solve_steady_state
export backward_pass, forward_pass, Transition, solve_transition, transition_paths, mit_shock, tax_reform
export impulse_response, bkm_simulate

end
