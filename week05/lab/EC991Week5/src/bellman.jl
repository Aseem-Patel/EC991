# >>> bellman_step!
"""
    bellman_step!(V1, V0, π, F, β)

One application of the incumbent's Bellman operator (lecture slide 35), at
every grid point at once:

    V1(z_i) = π(z_i) + β max{ Σ_j F_ij V0(z_j), 0 }.

The sum is the continuation value: what staying is worth, before z' is known.
Exit is worth zero. Writes into V1.
"""
function bellman_step!(V1, V0, π, F, β)
    # the continuation value at every z_i: one matrix product, written into V1
    mul!(V1, F, V0)
    # stay if the continuation value is positive, exit (value 0) if not
    @. V1 = π + β * max(V1, 0.0)
    return V1
end
# <<< bellman_step!

# >>> FirmSolution
"""
The incumbent's solution at one price: the value function `V`, the continuation
value `EV` (EV_i = Σ_j F_ij V_j), the exit rule `χ` (true means exit, χ_i = EV_i < 0),
the number of iterations and the distance at every iteration.
"""
struct FirmSolution
    V::Vector{Float64}
    EV::Vector{Float64}
    χ::BitVector
    iterations::Int
    distances::Vector{Float64}
end
# <<< FirmSolution

# >>> evaluate_policy
"""
    evaluate_policy(χ, π, F, β)

The value of following a fixed exit rule χ forever: V = π + β (1 - χ) .* (F V),
a linear system, solved exactly:

    V = (I - β diag(1 - χ) F)⁻¹ π.

This is Howard's evaluation step (week 4) carried to the limit n_H → ∞. It is
cheap here because there is no continuous choice to hold fixed: the only
decision is a 0 or 1 at each z.
"""
function evaluate_policy(χ, π, F, β)
    # I - β diag(1 - χ) F: row i of F is scaled by 1 - χ_i, so exiters lose their future
    A = I - β .* (1 .- χ) .* F
    # solve the linear system A V = π in one step
    return A \ π
end
# <<< evaluate_policy

# >>> solve_firm
"""
    solve_firm(π, F, β; method = :vfi, n_H = 20, V_start = nothing, tol = 1e-9, maxiter = 5000)

The incumbent's value function and exit rule at one price (lecture slides 34 and 35; Howard and policy iteration, tutorial slides 5 and 6).

  :vfi               iterate the Bellman operator until max |V^{n+1} - V^n| < tol max |V^{n+1}|
  :howard            the same, with n_H evaluations of the current exit rule between steps
  :policy_iteration  improve the exit rule, evaluate it exactly, repeat until it stops changing

The tolerance is relative, because V is in units of
output and its size moves with the price. `V_start` warm-starts VFI and Howard.
"""
function solve_firm(π, F, β; method = :vfi, n_H = 20, V_start = nothing, tol = 1e-9, maxiter = 5000)
    method === :policy_iteration && return solve_policy_iteration(π, F, β; maxiter)
    (method === :vfi || method === :howard) || throw(ArgumentError("unknown method $method"))
    V0 = V_start === nothing ? copy(π) : copy(V_start)     # initial guess: today's profit
    V1 = similar(V0)
    EV = similar(V0)
    distances = Float64[]
    for n in 1:maxiter
        # one Bellman step: V1 = T V0
        bellman_step!(V1, V0, π, F, β)
        # how far the value moved, at the grid point where it moved most
        d = maximum(abs.(V1 .- V0))
        push!(distances, d)
        # converged: record the continuation values and the exit rule, and stop
        if d < tol * maximum(abs.(V1))
            mul!(EV, F, V1)
            return FirmSolution(V1, copy(EV), EV .< 0, n, distances)
        end
        if method === :howard
            # read the exit rule off the current value: stay where EV ≥ 0
            mul!(EV, F, V1)
            stay = EV .>= 0
            # hold that rule fixed and update the value n_H times, with no max
            for _ in 1:n_H
                mul!(EV, F, V1)
                # stayers: profit + discounted continuation value; exiters: profit only
                @. V1 = π + β * stay * EV
            end
        end
        V0, V1 = V1, V0                                    # the update becomes the guess
    end
    error("value function iteration did not converge in $maxiter iterations")
end
# <<< solve_firm

# >>> solve_policy_iteration
"""
    solve_policy_iteration(π, F, β; maxiter = 100)

Policy iteration on the exit rule. Start from "never exit", evaluate it exactly
with `evaluate_policy`, then improve it: exit wherever the continuation value is
negative. Stop when the rule no longer changes; the value is then the fixed point
of the Bellman operator, to machine precision. `distances` records how many grid
points changed their decision at each round.
"""
function solve_policy_iteration(π, F, β; maxiter = 100)
    χ = falses(length(π))                                  # start: no one exits
    distances = Float64[]
    for n in 1:maxiter
        # evaluate: the exact value of following χ forever
        V = evaluate_policy(χ, π, F, β)
        # the continuation value at every grid point under that value
        EV = F * V
        # improve: exit wherever the continuation value is negative
        χ_new = EV .< 0
        push!(distances, count(χ_new .!= χ))
        # stop when the improved rule is the rule just evaluated
        χ_new == χ && return FirmSolution(V, EV, χ, n, distances)
        χ = χ_new
    end
    error("policy iteration did not converge in $maxiter rounds")
end
# <<< solve_policy_iteration

# >>> exit_threshold
"""
    exit_threshold(sol, z)

The productivity z̃ at which the continuation value is zero (slide 28). On the
grid the exit rule switches between two neighbouring points; between them we
interpolate EV linearly in log z, the same idea as reading a policy between grid
points in week 4. Returns z̃, and the index of the first grid point where firms
stay. NaN if nobody exits.
"""
function exit_threshold(sol::FirmSolution, z)
    # the first grid point that stays; the one before it is the last that exits
    k = findfirst(.!sol.χ)
    (k === nothing || k == 1) && return (z̃ = NaN, k = k)
    # log z and the continuation value at those two neighbours
    lo, hi = log(z[k-1]), log(z[k])
    EVlo, EVhi = sol.EV[k-1], sol.EV[k]
    # the straight line between them crosses zero at log z̃
    return (z̃ = exp(lo + (hi - lo) * (0 - EVlo) / (EVhi - EVlo)), k = k)
end
# <<< exit_threshold

# >>> exit_lottery
"""
    exit_lottery(sol, z)

The exit rule as a share of each grid cell that exits, rather than a 0 or 1.
Grid point i stands for the cell of log z within half a step of log z_i (the
Tauchen bins). Every cell wholly below the threshold exits, every cell wholly
above stays, and the one cell that straddles log z̃ exits in proportion to the
part of it below log z̃. This is week 4's lottery between grid points, applied
to a discrete choice: statistics now move smoothly when z̃ moves inside a cell,
instead of jumping when z̃ crosses a grid point (tutorial slides 13 to 15).
"""
function exit_lottery(sol::FirmSolution, z)
    # start from the grid rule: 1 = exit, 0 = stay
    χ = Float64.(sol.χ)
    # the interpolated threshold z̃, and k, the first grid point that stays
    (; z̃, k) = exit_threshold(sol, z)
    isnan(z̃) && return χ
    # the grid step d in log z, and the threshold t in log z
    logz = log.(z)
    d = logz[2] - logz[1]
    t = log(z̃)
    # the cell that contains t: point k-1's cell if t is below their midpoint, else point k's
    i = t < logz[k-1] + d / 2 ? k - 1 : k
    # that cell exits in proportion to the part of it below t
    χ[i] = clamp((t - (logz[i] - d / 2)) / d, 0.0, 1.0)
    return χ
end
# <<< exit_lottery
