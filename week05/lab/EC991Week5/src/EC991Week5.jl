"""
EC991 week 5 lab: the Hopenhayn (1992) model of industry dynamics. Firms differ
in productivity, which follows a Markov chain; they pay a fixed cost to stay
and an entry cost to come in. The equilibrium has two blocks: the price comes
from free entry alone, then the mass of entrants from the goods market.

From week 3, as they were:

  markov.jl       the normal CDF, Tauchen (now around a mean), the stationary
                  distribution, and a discretised normal for the entrants

New this week, in the order of the lecture slides:

  static.jl       the static economy: the efficient allocation, aggregate TFP,
                  misallocation from wedges, the planner's choice of entry
  industry.jl     the parameters, the grid and chain, the static profit problem
  bellman.jl      the incumbent's Bellman equation: VFI, Howard, policy iteration,
                  and the exit threshold between grid points
  entry.jl        the value of entry and the price that makes it zero
  distribution.jl the measure of firms: exits, entrants, cohorts
  equilibrium.jl  the mass of entrants and the equilibrium
  statistics.jl   the table, the size distribution, the age profile
  simulate.jl     a panel of individual firms simulated from the equilibrium

Nothing here plots or prints. Plotting lives in the run scripts.
"""
module EC991Week5

using LinearAlgebra
using Random

include("markov.jl")
include("static.jl")
include("industry.jl")
include("bellman.jl")
include("entry.jl")
include("distribution.jl")
include("equilibrium.jl")
include("statistics.jl")
include("simulate.jl")

export normal_cdf, tauchen, stationary, discretise_normal
export efficient_allocation, aggregate_tfp, allocation_with_wedges, misallocation, planner_entry
export Industry, setup, static_decisions
export bellman_step!, FirmSolution, evaluate_policy, solve_firm, solve_policy_iteration, exit_threshold, exit_lottery
export entry_value, solve_price
export exit_transition, stationary_measure, cohorts
export mass_of_entrants, Equilibrium, solve_equilibrium
export industry_stats, size_distribution, age_profile
export simulate_panel

end
