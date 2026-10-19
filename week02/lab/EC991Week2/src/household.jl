# >>> Household
"""
    Household(; β = 1/1.05, R = 1.03, y = 1.0, ϕ = 0.02, γ = 2.0)

Parameters of the deterministic saving problem

    V(a) = max_{a' ≥ -ϕ} u(y + R a - a') + β V(a')

with CRRA utility of curvature γ. Defaults follow Moll's vfi_deterministic.m
(ρ = 0.05 so β = 1/1.05, r = 0.03), which gives βR < 1: the household
decumulates towards the borrowing limit.
"""
Base.@kwdef struct Household{T<:Real}
    β::T = 1 / 1.05   # discount factor
    R::T = 1.03       # gross interest rate, 1 + r
    y::T = 1.0        # constant income (wage normalised to one)
    ϕ::T = 0.02       # borrowing limit: a' ≥ -ϕ
    γ::T = 2.0        # CRRA curvature; γ = 1 is log
end
# <<< Household

# >>> utility
"""
    utility(c, γ)

CRRA felicity u(c) = (c^(1-γ) - 1) / (1 - γ), with u(c) = log(c) when γ = 1.
Only ever called with c > 0: the Bellman step skips infeasible choices.
"""
utility(c, γ) = γ == 1 ? log(c) : (c^(1 - γ) - 1) / (1 - γ)
utility(c, hh::Household) = utility(c, hh.γ)

"""
    marginal_utility(c, γ)

u'(c) = c^(-γ). Used only to test the Euler equation along a simulated path.
"""
marginal_utility(c, γ) = c^(-γ)
marginal_utility(c, hh::Household) = marginal_utility(c, hh.γ)
# <<< utility
