# >>> choice_value!
"""
    choice_value!(h, i, EV, p, s, eco)

The value of choosing employment n, for a firm with productivity z_i, before any
firing tax (tutorial slide 7):

    h(z_i, n) = p z_i n^η - n - p c_f + β max{ -τ n,  Σ_j F_ij V(z_j, n) }.

The max is the exit decision, taken after producing: exit fires all n workers
and costs τ n next period; staying is worth the continuation value. h does not
depend on inherited employment m, so it is computed once for each (z_i, n).
Writes h over the employment grid.
"""
function choice_value!(h, i, EV, p, s, eco::Economy)
    (; η, β, c_f, τ) = eco
    # this firm's productivity, and the employment grid
    z, n = s.z[i], s.n
    for ell in eachindex(n)
        # profit with n = n[ell] workers, plus the better of exit (fire them all) and stay
        h[ell] = p * z * n[ell]^η - n[ell] - p * c_f + β * max(-τ * n[ell], EV[i, ell])
    end
    return h
end
# <<< choice_value!

# >>> best_employment!
"""
    best_employment!(Vrow, polrow, h, n, τ)

The best employment n for every inherited employment m on the grid, at one z
(tutorial slide 7):

    V(z, m) = max_n { h(z, n) - τ max{0, m - n} }.

Exhaustive search: for each inherited m = n[k], try every candidate n = n[ell] on
the same grid and keep the best. That is n_n² evaluations at one z, so n_z n_n²
for one Bellman update. Choices stay on the grid: nothing is interpolated.
Writes the value into Vrow and the index of the chosen n into polrow.
"""
function best_employment!(Vrow, polrow, h, n, τ)
    # every inherited employment m = n[k] on the grid
    for k in eachindex(n)
        best_value = -Inf
        best_index = 0

        # try every candidate n = n[ell]: its value, net of the tax on the workers let go
        for ell in eachindex(n)
            value = h[ell] - τ * max(0.0, n[k] - n[ell])

            # on an exact tie, prefer the choice that fires nobody
            prefer_no_firing =
                value == best_value &&
                ell >= k &&
                best_index < k

            if value > best_value || prefer_no_firing
                best_value = value
                best_index = ell
            end
        end

        # the value at (z, n[k]) and the index of the chosen n
        Vrow[k] = best_value
        polrow[k] = best_index
    end

    return Vrow
end
# <<< best_employment!

# >>> best_employment_fast!
"""
    best_employment_fast!(Vrow, polrow, h, n, τ)

The same answer as `best_employment!`, in one pass over the grid instead of
n_n passes. It is here for comparison only: the solver uses `best_employment!`.

The firing tax depends on inherited employment m = n_k only through whether the
chosen n is below it. So split the candidates in two:

    fire nobody, n_ell ≥ n_k:   max_{ell ≥ k} h[ell]
    fire some, n_ell < n_k:     max_{ell < k} (h[ell] + τ n_ell) - τ n_k

Neither inner max needs a fresh search at each k. The first is a running maximum
from the top of the grid: the best over ell ≥ k is the better of h[k] and the
best over ell ≥ k + 1. The second is a running maximum from the bottom. All n_n
states at one z then cost about 2 n_n comparisons, not n_n². The split is exact
and uses no property of h, such as concavity.
"""
function best_employment_fast!(Vrow, polrow, h, n, τ)
    N = length(n)
    up, arg_up = similar(h), zeros(Int, N)
    # from the top: the best candidate at or above each k, and where it is
    up[N], arg_up[N] = h[N], N
    for k in N-1:-1:1
        if h[k] >= up[k+1]
            up[k], arg_up[k] = h[k], k
        else
            up[k], arg_up[k] = up[k+1], arg_up[k+1]
        end
    end
    # from the bottom: the best h[ell] + τ n[ell] strictly below each k, and where it is
    down, arg_down = -Inf, 0
    for k in 1:N
        fire_some = down - τ * n[k]
        # on a tie, prefer the choice that fires nobody
        if up[k] >= fire_some
            Vrow[k], polrow[k] = up[k], arg_up[k]
        else
            Vrow[k], polrow[k] = fire_some, arg_down
        end
        # now let n[k] join the candidates for every state above k
        if h[k] + τ * n[k] > down
            down, arg_down = h[k] + τ * n[k], k
        end
    end
    return Vrow
end
# <<< best_employment_fast!

# >>> bellman_step!
"""
    bellman_step!(V1, pol, EV, V0, p, s, eco, work)

One application of the incumbent's Bellman operator (tutorial slides 4 and 7) at
every (z_i, m = n_k) on the grid: the continuation values EV = F V0 (one matrix
product), then, at each z_i, the value h of every candidate n and the best n for
every inherited m.
"""
function bellman_step!(V1, pol, EV, V0, p, s, eco::Economy, work)
    # the continuation value of every (z_i, n), before next period's z is known
    mul!(EV, s.F, V0)
    for i in eachindex(s.z)
        # h at every candidate n, then the best n for every inherited m, at z_i
        choice_value!(work.h, i, EV, p, s, eco)
        best_employment!(view(V1, i, :), view(pol, i, :), work.h, s.n, eco.τ)
    end
    return V1
end
# <<< bellman_step!

# >>> FirmSolution
"""
The incumbent's solution at one price: the value `V[i, k]` = V(z_i, m = n_k), the
continuation value `EV` = F V, the employment policy `pol[i, k]` (the index of the
chosen n on the grid), the exit rule `χ[i, k]` (true: produce with n, then exit),
the number of iterations and the distance at each.
"""
struct FirmSolution
    V::Matrix{Float64}
    EV::Matrix{Float64}
    pol::Matrix{Int}
    χ::BitMatrix
    iterations::Int
    distances::Vector{Float64}
end
# <<< FirmSolution

# >>> solve_firm
"""
    solve_firm(p, eco, s; V_start = nothing, tol = 1e-9, maxiter = 5000)

The incumbent's value function, employment policy and exit rule at price p, by
value function iteration until max |V^(r+1) - V^(r)| < tol max |V^(r+1)|, with r
the iteration number. `V_start` warm-starts the iteration (the bisection on p
passes the last V).
"""
function solve_firm(p, eco::Economy, s; V_start = nothing, tol = 1e-9, maxiter = 5000)
    n_z, n_n = length(s.z), length(s.n)
    # initial guess: zero, or the value function passed in
    V0 = V_start === nothing ? zeros(n_z, n_n) : copy(V_start)
    V1, EV = similar(V0), similar(V0)
    pol = zeros(Int, n_z, n_n)
    work = (; h = zeros(n_n))                         # h over the n grid, reused at every z_i
    distances = Float64[]
    for it in 1:maxiter
        # one Bellman step: V1 = T V0, and the chosen n at every state
        bellman_step!(V1, pol, EV, V0, p, s, eco, work)
        # how far the value moved, at the state where it moved most
        d = maximum(abs.(V1 .- V0))
        push!(distances, d)
        if d < tol * maximum(abs.(V1))
            # converged: the continuation values at V, and the exit rule at the chosen n
            mul!(EV, s.F, V1)
            χ = falses(n_z, n_n)
            for k in 1:n_n, i in 1:n_z
                χ[i, k] = EV[i, pol[i, k]] < -eco.τ * s.n[pol[i, k]]
            end
            return FirmSolution(V1, copy(EV), pol, χ, it, distances)
        end
        V0, V1 = V1, V0                               # the update becomes the guess
    end
    error("value function iteration did not converge in $maxiter iterations")
end
# <<< solve_firm

# >>> inaction_region
"""
    inaction_region(sol, s)

The bounds of the inaction region at every z_i (tutorial slides 8, 13 and 14),
read off the employment policy n^d(z, m):

    n_L(z) = n^d(z, 0)       the hiring target: a firm that inherits fewer workers hires up to it
    n_H(z) = n^d(z, n_max)   the firing target: a firm that inherits more workers fires down to it

A firm that inherits n_L(z) ≤ m ≤ n_H(z) keeps its m workers. With τ = 0 the two
bounds coincide: chosen employment no longer depends on inherited employment.
"""
function inaction_region(sol::FirmSolution, s)
    nL = s.n[sol.pol[:, 1]]          # n^d(z, 0): a firm that inherits no workers hires up to n_L(z)
    nH = s.n[sol.pol[:, end]]        # n^d(z, n_max): a firm that inherits n_max fires down to n_H(z)
    return (; z = s.z, nL, nH)
end
# <<< inaction_region
