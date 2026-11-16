"""
EC991 week 6 lab: Hopenhayn and Rogerson (1993). The week 5 industry, with three
changes: firing a worker costs τ, so inherited employment is a state; the household
supplies labour; and the fixed cost is paid in output. Firing costs open an
inaction region, in which firms keep their workers although their productivity
has changed, so marginal products differ across firms: misallocation.

From weeks 3 and 5, as they were:

  markov.jl        the normal CDF, Tauchen around a mean, the stationary distribution

New this week, in the order of the tutorial slides:

  economy.jl       the parameters, the grids for z and n, the firing cost
  firm.jl          the incumbent's Bellman equation over (z, m), with m inherited
                   employment: the choice of n by exhaustive search over the
                   grid, VFI, the inaction region;
                   a faster search with the same answer, for comparison
  entry.jl         the value of entry and the price that makes it zero
  distribution.jl  the law of motion over (z, m) and its stationary measure
  equilibrium.jl   the mass of entrants and the equilibrium
  aggregates.jl    output, labour, taxes, profits, job flows; misallocation

Nothing here plots or prints. Plotting lives in the run scripts.
"""
module EC991Week6

using LinearAlgebra

include("markov.jl")
include("economy.jl")
include("firm.jl")
include("entry.jl")
include("distribution.jl")
include("equilibrium.jl")
include("aggregates.jl")

export normal_cdf, tauchen, stationary
export Economy, setup, firing_cost
export choice_value!, best_employment!, best_employment_fast!, bellman_step!, FirmSolution, solve_firm, inaction_region
export entry_value, solve_price
export law_of_motion!, stationary_distribution
export output_per_firm, Equilibrium, solve_equilibrium
export aggregates, misallocation, size_distribution

end
