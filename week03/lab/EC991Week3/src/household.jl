# >>> w_from_r
"""
    w_from_r(r; α = 0.33, δ = 0.05)

The wage that goes with the interest rate r under Cobb-Douglas production
Y = K^α L^(1-α). The firm's first-order condition for capital,
r = α (K/L)^(α-1) - δ, gives K/L = (α / (r + δ))^(1/(1-α)); the one for
labour gives w = (1-α) (K/L)^α. Week 5 reuses this function when r is solved for.
"""
w_from_r(r; α = 0.33, δ = 0.05) = (1 - α) * (α / (r + δ))^(α / (1 - α))
# <<< w_from_r

# >>> Household
"""
    Household(; β = 0.96, γ = 2.0, ϕ = 0.0, r = 0.03, w = w_from_r(r))

Parameters of the household problem with income risk,

    V(a, y) = max_{a' ≥ -ϕ} u(R a + w y - a') + β E[V(a', y') | y],

with CRRA utility of curvature γ and R = 1 + r. The defaults are β = 0.96,
γ = 2, ϕ = 0 and r = 0.03; w follows from r. (At r = 0.04,
βR = 0.9984 and with the lab's income process the high-income household keeps
saving up to any upper bound of the asset grid; at r = 0.03 it stops near 25,
inside the lab grid [0, 50].) R is filled in
from r: pass r, not R. ϕ is the borrowing limit of the slides, a_1 = -ϕ.
"""
Base.@kwdef struct Household{T<:Real}
    β::T = 0.96            # discount factor
    γ::T = 2.0             # CRRA curvature; γ = 1 is log
    ϕ::T = 0.0             # borrowing limit: a' ≥ -ϕ
    r::T = 0.03            # net interest rate
    w::T = w_from_r(r)     # wage, from the firm's first-order condition
    R::T = 1 + r           # gross interest rate
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
# <<< utility
