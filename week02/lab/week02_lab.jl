### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ e0ca747e-fd22-4d8a-9a73-c75a5512184e
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "EC991Week2"))
    Pkg.instantiate()
    using EC991Week2, Plots
    using Markdown: Markdown, @md_str
end

# ╔═╡ 68f52308-09d3-4bba-8da3-372611a6465f
md"""
# EC991 Week 2 lab: dynamic programming and value function iteration

**Aseem Patel, University of Essex, Autumn 2026**

This notebook follows the lab slides and runs the code that solves the problem on them. The code lives in the package `EC991Week2`, in the folder next to this notebook. Every section puts the maths first and the code that implements it directly underneath, so you can read them line against line.

**Open this file in Pluto**, not as a script. The first cell activates the package environment; Pluto then runs everything else. Three Pluto rules from week 1 still apply: cells are reactive, each name is defined in exactly one cell, and several lines in one cell go inside `begin ... end` or `let ... end`.

How the code is organised:

| Folder | What is in it | Rule |
|---|---|---|
| `EC991Week2/src/` | the model, the Bellman step, the two solvers, the checks | never plots, never prints |
| `EC991Week2/run/` | scripts that call the solvers and make the figures for the slides | no model logic |
| `EC991Week2/test/` | the predictions from theory, as automatic tests | run with `Pkg.test()` |
| this notebook | the same code, with the maths alongside | for learning |

Keeping the solver separate from the scripts that use it is the habit to take from today. A solver you can test is a solver you can trust.
"""

# ╔═╡ fab412c1-5f99-4594-9ddc-a7911fb08406
md"""
The helper below pulls a named block out of the package source so it can be shown next to the maths. It is only for reading; nothing in the notebook depends on it.
"""

# ╔═╡ 288a6d68-6df8-4cee-86a5-fd5d0895ad2b
function source(name)
    dir = dirname(pathof(EC991Week2))
    for f in readdir(dir; join = true)
        text = read(f, String)
        m = match(Regex("# >>> \\Q$name\\E\\n(.*?)# <<< \\Q$name\\E", "s"), text)
        m === nothing || return Markdown.parse("```julia\n" * m.captures[1] * "```")
    end
    error("no source block called $name")
end

# ╔═╡ 06ba8b5b-be2e-41be-9ec0-02b97b2f66d9
md"""
## 1. The problem and its state (slides 2 to 4)

The household chooses savings to maximise lifetime utility. Income is constant at ``y`` (the wage is normalised to one), the gross interest rate is ``R = 1 + r``, and borrowing is limited to ``\phi``:

```math
\max_{\{a_{t+1}\}_{t=0}^{\infty}} \ \sum_{t=0}^{\infty} \beta^t u(c_t)
\quad \text{s.t.} \quad
c_t + a_{t+1} \leq y + R a_t, \qquad a_{t+1} \geq -\phi .
```

Five numbers describe the household: ``\beta, R, y, \phi, \gamma``. In code they are the fields of one struct, so that every function receives them together instead of as five separate arguments. The defaults are Moll's: ``\beta = 1/1.05`` and ``R = 1.03``, so ``\beta R < 1`` and the household is impatient relative to the interest rate.
"""

# ╔═╡ 633b8bee-7173-4aa7-902e-0b87b97bd7f7
source("Household")

# ╔═╡ ae44c22b-8718-445e-8986-b73faa8413fb
hh = Household()

# ╔═╡ 06987487-d5f3-4ce3-8c8a-d679eaf0b583
md"""
Utility is CRRA, with the log as the ``\gamma = 1`` case:

```math
u(c) = \frac{c^{1-\gamma} - 1}{1 - \gamma}, \qquad u'(c) = c^{-\gamma}.
```

The code is a one-line function with a second method that reads ``\gamma`` out of the household. ``u'`` is only needed later, to test the Euler equation.
"""

# ╔═╡ 1ef60333-6ae4-42c1-9c1f-25179fb8282a
source("utility")

# ╔═╡ 96901ef4-ee32-498a-a9ed-0f56bf17b401
let c = range(0.2, 3; length = 200)
    plot(c, utility.(c, 1.0); label = "γ = 1 (log)", xlabel = "c", ylabel = "u(c)", lw = 2)
    plot!(c, utility.(c, 2.0); label = "γ = 2", lw = 2)
    plot!(c, utility.(c, 5.0); label = "γ = 5", lw = 2, legend = :bottomright)
end

# ╔═╡ a48684d8-cef3-4800-9117-2bf5d1e001f9
md"""
**The state.** The state is the minimal information needed today to evaluate current payoffs and to forecast the future. With constant income that is ``a`` alone: two households with the same assets face the same problem from here on, whatever their histories. So the value function has one argument, ``V(a)``, and so does every array in the code: one entry per asset level.
"""

# ╔═╡ 42f6a448-0688-47a8-aabe-ec62316abfc2
md"""
## 2. The Bellman equation on a grid (slides 3, 5 and 10)

Split the problem into today and everything after. If ``V(a)`` is the best lifetime utility attainable with assets ``a``, then

```math
V(a) \;=\; \max_{a' \,\geq\, -\phi} \ u\big(y + R a - a'\big) + \beta V(a') .
```

A computer cannot hold a function, so **Step 1** of the algorithm replaces the continuum of asset levels by a grid ``\mathcal{A} = \{a_1, \dots, a_{n_a}\}`` with ``a_1 = -\phi``. Both the state ``a`` and the choice ``a'`` are restricted to the grid. The unknown function ``V`` becomes a vector of ``n_a`` numbers, ``V_i = V(a_i)``.
"""

# ╔═╡ 485c65c1-24e9-48ba-891f-82c9022e6a9e
source("asset_grid")

# ╔═╡ 3b6ac019-67fe-4875-a0ab-c664793f63aa
a = asset_grid(hh; amax = 3.0, na = 300)

# ╔═╡ 11b8737b-3864-42d7-b5ae-04e1c6e65090
(first = a[1], last = a[end], step = a[2] - a[1], points = length(a))

# ╔═╡ fe8810c3-1dcf-44fc-a99d-7b8b804bc5d8
md"""
## 3. One application of the Bellman operator (slide 11)

Write the right-hand side of the Bellman equation as an operator ``T`` that takes a candidate ``V`` and returns a new function ``TV``. On the grid, with ``a'`` restricted to grid points,

```math
(TV)(a_i) \;=\; \max_{k = 1, \dots, n_a} \ u\big(y + R a_i - a_k\big) + \beta V(a_k), \qquad u(c) = -\infty \text{ when } c \leq 0 .
```

For each state ``a_i`` this is a search over ``n_a`` candidates: a double loop. The function below is the whole of Step 3 of the algorithm. Read it against the equation:

| Code | Maths |
|---|---|
| `for i in 1:na` | for each state ``a_i`` |
| `for k in 1:na` | for each candidate ``a'_k`` |
| `c = R * a[i] + y - a[k]` | ``c = y + R a_i - a'_k`` from the budget constraint |
| `c <= 0 && continue` | ``u(c) = -\infty`` for infeasible ``c``, so skip it |
| `v = utility(c, hh) + β * V0[k]` | the Bellman right-hand side ``u(c) + \beta V(a'_k)`` |
| `if v > best` | the ``\max`` |
| `V1[i], pol[i] = best, kbest` | ``(TV)(a_i)`` and the argmax, stored as an index |

`V0` is the function you put in, `V1` the function you get out; `pol[i]` is the **index** ``k`` of the chosen ``a'``, so the policy function is `a[pol]`. Income is a keyword argument so that the finite-horizon solver, where income varies by date, can reuse this function unchanged.
"""

# ╔═╡ 3f60c958-6fc8-4b84-b490-46c2a8fbcf8e
source("bellman_step!")

# ╔═╡ 99cffe12-c919-4daf-8488-0b9c92b02823
md"""
**Step 2, the initial guess.** Any guess works, but a good one starts close. The value of consuming interest plus income forever, ``c = r a + y`` at every date, is

```math
V_0(a) \;=\; \sum_{t=0}^{\infty} \beta^t u(ra + y) \;=\; \frac{u(ra + y)}{1 - \beta} .
```
"""

# ╔═╡ e39f3a18-b4a1-4fdd-b62a-2df0f437b4c6
source("initial_guess")

# ╔═╡ 03ae1633-5181-4fba-b86f-1393b5566dd3
md"""
Apply ``T`` once to the guess. The output lies above the input everywhere: ``V_0`` is the value of one feasible plan (never change assets), and ``T V_0`` picks the best plan given that continuation.
"""

# ╔═╡ f848bcb5-fc45-446d-9b7a-807bdabfe773
begin
    V0 = initial_guess(hh, a)
    V1 = similar(V0)
    pol1 = zeros(Int, length(a))
    bellman_step!(V1, pol1, V0, hh, a)
    plot(a, V0; label = "V₀, the guess", xlabel = "a", ylabel = "value", lw = 2)
    plot!(a, V1; label = "T V₀, one Bellman step", lw = 2, legend = :bottomright)
end

# ╔═╡ c52c657e-b353-4273-b1db-f2ebb24bea9c
(largest_increase = maximum(V1 .- V0), smallest_increase = minimum(V1 .- V0))

# ╔═╡ 1a01fa34-0054-42f0-be93-75a20fb15867
md"""
## 4. Iterate to the fixed point (slides 8 to 10)

A solution of the Bellman equation is a fixed point, ``V = TV``. Under our assumptions ``T`` is a contraction with modulus ``\beta``: for any two candidates,

```math
\| TV - TW \| \;\leq\; \beta\, \| V - W \| ,
\qquad \| f \| = \max_i |f(a_i)| .
```

Two things follow. The fixed point exists and is unique. And the sequence ``V_\ell = T V_{\ell-1}`` converges to it from any ``V_0``, with the distance between successive iterates shrinking by at least ``\beta`` each time:

```math
d_\ell = \| V_\ell - V_{\ell-1} \| \;\leq\; \beta\, d_{\ell - 1} .
```

That inequality is the algorithm. **Step 3** applies ``T``; **Step 4** stops when ``d_\ell`` is below a tolerance. The solver below is those two steps in a loop around `bellman_step!`. The line `V0, V1 = V1, V0` is the update: the function we just computed becomes the guess for the next round (the two arrays swap roles, so nothing is copied).
"""

# ╔═╡ 3cfc6963-6732-4e93-98a4-45725855b2b1
source("solve_vfi")

# ╔═╡ 271622c6-aec3-4e49-8156-b57cb90d1fe0
sol = solve_vfi(hh, a; tol = 1e-6)

# ╔═╡ 983f5146-3951-4037-bb26-29ad9a4acd65
(iterations = sol.iterations, last_change = sol.distances[end])

# ╔═╡ 00123ece-54b5-475d-ae37-d6985b805311
md"""
Only a few dozen iterations, which looks too good for a contraction of modulus ``0.95``. The plot explains it. Every ratio ``d_\ell / d_{\ell-1}`` is at most ``\beta``, as the theorem says, but the distances then collapse to zero: the guess ``V_0`` is **exact** at the borrowing limit (there the household really does consume ``ra + y`` forever), and from every grid point the optimal path reaches the limit in a finite number of periods. Once the iteration count passes that number, nothing is left to correct.

From a worse guess, ``V_0 = 0``, the same solver converges geometrically at rate ``\beta`` and needs a few hundred iterations. Both end at the same ``V``: the fixed point does not depend on where you start, only the number of steps does.
"""

# ╔═╡ 9d77f087-aa0d-4681-b2bb-ba1b7f6d9d71
begin
    V0_zero = zeros(length(a))
    V1_zero = similar(V0_zero)
    pol_zero = zeros(Int, length(a))
    d_zero = Float64[]
    for ℓ in 1:400
        bellman_step!(V1_zero, pol_zero, V0_zero, hh, a)
        push!(d_zero, maximum(abs.(V1_zero .- V0_zero)))
        V0_zero .= V1_zero                 # copy in place: the update becomes the guess
    end
    (converged_to_same_V = maximum(abs.(V0_zero .- sol.V)) < 1e-3,
     ratio_of_last_two_distances = d_zero[end] / d_zero[end - 1],
     β = hh.β)
end

# ╔═╡ e2ce2acb-d755-4d56-a16a-b151e08c65fb
let
    plot(1:length(sol.distances), sol.distances; yscale = :log10, lw = 2, marker = :circle, ms = 3,
         label = "from the good guess V₀ = u(ra + y)/(1 - β)", xlabel = "iteration ℓ", ylabel = "dₗ = max |Vₗ - Vₗ₋₁|")
    plot!(1:400, d_zero; lw = 2, label = "from V₀ = 0")
    plot!(1:400, d_zero[1] .* hh.β .^ (0:399); lw = 1, ls = :dash, color = :gray, label = "slope β", ylims = (1e-8, 1e3))
end

# ╔═╡ 1937b938-6d95-4bda-9f19-5fe9e9d48dc9
md"""
## 5. The solution (slide 12)

The converged ``V`` and the policies ``a'(a)`` and ``c(a) = y + Ra - a'(a)``. The savings policy lies below the 45-degree line everywhere: with ``\beta R < 1`` the household always runs assets down. The consumption function is a staircase, because ``a'`` can only jump between grid points. That is the inaccuracy week 3 removes.
"""

# ╔═╡ 91f7ba03-5a2d-4936-8168-119920cd92bc
let
    p1 = plot(a, sol.V; label = "", xlabel = "a", title = "V(a)")
    p2 = plot(a, sol.apol; label = "a'(a)", xlabel = "a", title = "savings policy")
    plot!(p2, a, a; label = "45-degree line", ls = :dash, color = :gray, legend = :topleft)
    p3 = plot(a, sol.cpol; label = "", xlabel = "a", title = "c(a)")
    plot(p1, p2, p3; layout = (1, 3), size = (900, 300), lw = 2, margin = 5Plots.mm)
end

# ╔═╡ e5061e0c-8a60-4858-bc2d-9776ef31ea0a
md"""
## 6. Test the output against theory (slides 7 and 12)

Never trust numerical output you have not tested against a prediction. The slides give two.

**Prediction 1.** With ``\beta R < 1``, assets drift down to the borrowing limit and stay there. To see it, follow the policy forward from a high starting level. On a grid this is index chasing: the next state is `a[pol[i]]`, so the path never leaves the grid.
"""

# ╔═╡ 0cc83c86-edb6-417a-88be-4132c94ff891
source("simulate")

# ╔═╡ e404ca9b-6fc2-4ce7-bac5-03a874e4179b
apath, cpath = simulate(sol, 2.5, 120)

# ╔═╡ 57ed20f0-98c3-45da-9e58-43047dc9ad7a
md"""
**Prediction 2.** The Euler equation derived from the Bellman equation (slide 7) is

```math
u'(c_t) \;=\; \beta R\, u'(c_{t+1}) + \mu_t \;\geq\; \beta R\, u'(c_{t+1}),
```

with equality while the borrowing limit is slack and a strict inequality once it binds (then ``\mu_t > 0``: the household would like to borrow and cannot). So the residual ``u'(c_t) - \beta R\, u'(c_{t+1})`` should be about zero at first and jump up when the path hits the limit. On a grid "about zero" means within the size of one consumption step, not exactly zero.
"""

# ╔═╡ 73375107-eeeb-4cb2-8aba-241f11636479
source("euler_residuals")

# ╔═╡ 998341f0-0612-4627-889f-86667d230d14
let res = euler_residuals(cpath, hh)
    p1 = plot(0:120, apath; lw = 2, label = "", xlabel = "t", title = "assets from a₀ = 2.5")
    hline!(p1, [-hh.ϕ]; ls = :dash, color = :gray, label = "borrowing limit")
    p2 = plot(1:119, res; lw = 2, label = "", xlabel = "t", title = "u'(cₜ) - βR u'(cₜ₊₁)")
    hline!(p2, [0]; ls = :dash, color = :gray, label = "")
    plot(p1, p2; layout = (1, 2), size = (900, 300), margin = 5Plots.mm)
end

# ╔═╡ 42d92cc3-715d-4989-85ec-73f2da6a5bff
(hits_limit_at = findfirst(==(a[1]), apath) - 1,
 min_residual_before = minimum(euler_residuals(cpath, hh)[1:20]),
 max_residual_after = maximum(euler_residuals(cpath, hh)[25:end]))

# ╔═╡ 3feb576f-c756-4ea2-9a30-61fcb9718a05
md"""
## 7. Finite horizon: backward induction (slides 13 and 14)

Now the household lives ``T`` periods, income ``y_t`` may vary by date, and the horizon shortens as time passes. The value function carries a time subscript (the state ``a`` still does not):

```math
V_t(a) \;=\; \max_{a' \,\geq\, -\phi} \ u\big(y_t + R a - a'\big) + \beta V_{t+1}(a'),
\qquad V_{T+1}(a) \equiv 0 .
```

Compare with the infinite-horizon equation: the right-hand side is the same operator, applied to ``V_{t+1}`` instead of to ``V`` itself. At ``t = T`` the continuation is known (zero), so ``V_T`` is a plain maximisation with no fixed point to find. Then ``V_T`` is known, so ``V_{T-1}`` follows, and so on back to ``t = 1``.

In code that is the same `bellman_step!`, called ``T`` times with the columns of a matrix instead of in a convergence loop. Read the solver against the infinite-horizon one in Section 4:

| Infinite horizon (`solve_vfi`) | Finite horizon (`solve_backward`) |
|---|---|
| `V0`, `V1`: two vectors that swap roles | `V`: an ``n_a \times (T+1)`` matrix, one column per date |
| start from a guess | start from the terminal condition, column ``T+1`` |
| loop `for ℓ in 1:maxiter` until `d < tol` | loop `for t in T:-1:1`, no stopping rule |
| `bellman_step!(V1, pol, V0, hh, a)` | `bellman_step!(V[:, t], pol[:, t], V[:, t+1], hh, a; y = y[t])` |
"""

# ╔═╡ ddb1ed04-cf15-470c-b3db-55d13e27504f
source("solve_backward")

# ╔═╡ 2177bc11-3d0d-4924-adec-c53544a4bb83
md"""
**Backward induction is value function iteration from the zero guess.** With constant income, ``V_t`` depends only on the number of periods left, ``T - t + 1``, and the recursion ``V_t = T V_{t+1}`` from ``V_{T+1} = 0`` is exactly the iteration ``V_\ell = T V_{\ell-1}`` from ``V_0 = 0`` in Section 4. So a long-lived household's early-life value functions converge to the infinite-horizon ``V``. The plot shows ``V_t`` for a few dates against the solution from Section 4.
"""

# ╔═╡ 13c36227-de8b-4c7d-ad71-d9d3427ee891
fsol_const = solve_backward(hh, a, fill(hh.y, 150))

# ╔═╡ 459b09fe-f810-4ede-bb05-64696dcaf429
let
    plot(a, sol.V; lw = 3, color = :black, label = "infinite horizon V", xlabel = "a", ylabel = "value", legend = :bottomright)
    for t in (150, 140, 120, 100, 1)
        plot!(a, fsol_const.V[:, t]; lw = 2, label = "V_t with $(150 - t + 1) periods left")
    end
    current()
end

# ╔═╡ 8940a1bc-1125-4eac-bf2f-69a39088c410
md"""
## 8. Validate against a known answer: cake eating (slide 15)

Set ``y_t = 0``, ``R = 1`` and ``u = \log``. The household eats a cake of size ``a_1`` over ``T`` periods. Two lines of algebra give the exact answer:

- Euler with equality: ``u'(c_t) = \beta u'(c_{t+1})``, so ``1/c_t = \beta / c_{t+1}`` and ``c_{t+1} = \beta c_t``.
- Remaining budget: ``\sum_{s=t}^{T} c_s = a_t``, so ``c_t (1 + \beta + \dots + \beta^{T-t}) = a_t``.

```math
c_t \;=\; \frac{1-\beta}{1-\beta^{\,T-t+1}}\; a_t .
```

The formula gives consumption as a share of **whatever assets the household has** at ``t``, so it can be checked at the numerical ``a_t`` date by date. The grid point ``a = 0`` has no feasible choice (nothing to eat, no income), which is why `bellman_step!` assigns value ``-\infty`` to a state with no feasible candidate: no earlier state will ever choose it.
"""

# ╔═╡ ba24a4a2-0ec3-41b1-9c62-604b9bf89505
source("cake_eating")

# ╔═╡ 74b97b22-8514-4545-86b8-4a78ec7661e8
begin
    T_cake, a0_cake = 20, 1.0
    cake = Household(β = 0.95, R = 1.0, y = 0.0, ϕ = 0.0, γ = 1.0)
    a_cake = collect(range(0.0, a0_cake; length = 200))
    fsol_cake = solve_backward(cake, a_cake, zeros(T_cake))
    apath_cake, cpath_cake = simulate_finite(fsol_cake, a0_cake)
    exact_cake = [cake_eating_consumption(apath_cake[t], t, T_cake, cake.β) for t in 1:T_cake]
    (max_abs_error = maximum(abs.(cpath_cake .- exact_cake)), cake_fully_eaten = apath_cake[end])
end

# ╔═╡ b3b95023-20e7-4d83-b50f-3bb66e879a2c
md"""
Close, not exact. The gap is grid coarseness: ``a'`` must land on a grid point, so ``c_t`` is off by up to one grid step. If that explanation is right the error must fall as the grid is refined; if it does not, the bug is in the code, not in the grid. The second panel is that check.
"""

# ╔═╡ 633e90fe-6960-48e4-9329-321b4b0f071f
let
    nas = (50, 100, 200, 400, 800, 1600)
    errs = map(nas) do na
        ag = collect(range(0.0, a0_cake; length = na))
        fs = solve_backward(cake, ag, zeros(T_cake))
        ap, cp = simulate_finite(fs, a0_cake)
        maximum(abs.(cp .- [cake_eating_consumption(ap[t], t, T_cake, cake.β) for t in 1:T_cake]))
    end
    p1 = plot(1:T_cake, exact_cake; lw = 2, label = "closed form", xlabel = "t", title = "consumption, na = 200")
    scatter!(p1, 1:T_cake, cpath_cake; label = "backward induction", ms = 4)
    p2 = plot(collect(nas), collect(errs); xscale = :log10, yscale = :log10, marker = :circle, lw = 2,
              label = "", xlabel = "grid points", title = "max error against the closed form")
    plot(p1, p2; layout = (1, 2), size = (900, 300), margin = 5Plots.mm)
end

# ╔═╡ 5e56e02b-7981-4bec-9401-077cc2a6e026
md"""
## 9. A life cycle

The finite-horizon solver takes any income sequence. With a hump-shaped profile and a pension, the household borrows against rising income when young (up to the limit), saves through the peak earning years, and runs the savings down in retirement. Consumption is far smoother than income: the two motives from the lecture, intertemporal substitution and smoothing, both at work without any risk.
"""

# ╔═╡ 98c4ec02-d5d7-4cf4-bfac-2ac150d79e60
source("life_cycle_income")

# ╔═╡ 57c8ff2e-3c48-435f-b34a-d21de227a195
let
    T = 60
    y_lc = life_cycle_income(T)
    a_lc = asset_grid(hh; amax = 15.0, na = 600)
    fs = solve_backward(hh, a_lc, y_lc)
    ap, cp = simulate_finite(fs, 0.0)
    p1 = plot(1:T, y_lc; lw = 2, label = "income", xlabel = "t", title = "income and consumption")
    plot!(p1, 1:T, cp; lw = 2, label = "consumption")
    p2 = plot(0:T, ap; lw = 2, label = "", xlabel = "t", title = "assets")
    plot(p1, p2; layout = (1, 2), size = (900, 300), margin = 5Plots.mm)
end

# ╔═╡ b0244f1b-67b8-4360-a4c7-b58c0b4da383
md"""
## 10. Exercises

Work in new cells below. Each one reuses the functions above; none needs new solver code.

1. **Patience.** Set ``\beta R = 1`` (for example `Household(β = 1/1.03)`), solve, and simulate from ``a_0 = 2.5``. What happens to the drift in assets, and why? Check the savings policy against the 45-degree line.
2. **Grid coarseness.** Solve with `na = 50` and with `na = 1000`. Compare the consumption functions and the size of the Euler residuals before the limit binds. Which direction does the error move, and does it move at the rate the cake-eating plot suggests?
3. **A bequest motive.** In the life-cycle problem, replace the terminal condition by ``V_{T+1}(a) = \phi_b\, u(a + \phi)`` through the `terminal` keyword of `solve_backward` (the shift by ``\phi`` keeps the argument positive at the borrowing limit). How does terminal wealth change with ``\phi_b``?
4. **Where does the time go?** Use `@time solve_vfi(hh, a)` for `na` of 100, 300 and 1000. The double loop makes each Bellman step cost ``n_a^2`` operations. Does the timing scale that way? This is the cost week 3 attacks.
"""

# ╔═╡ Cell order:
# ╟─68f52308-09d3-4bba-8da3-372611a6465f
# ╠═e0ca747e-fd22-4d8a-9a73-c75a5512184e
# ╟─fab412c1-5f99-4594-9ddc-a7911fb08406
# ╠═288a6d68-6df8-4cee-86a5-fd5d0895ad2b
# ╟─06ba8b5b-be2e-41be-9ec0-02b97b2f66d9
# ╠═633b8bee-7173-4aa7-902e-0b87b97bd7f7
# ╠═ae44c22b-8718-445e-8986-b73faa8413fb
# ╟─06987487-d5f3-4ce3-8c8a-d679eaf0b583
# ╠═1ef60333-6ae4-42c1-9c1f-25179fb8282a
# ╠═96901ef4-ee32-498a-a9ed-0f56bf17b401
# ╟─a48684d8-cef3-4800-9117-2bf5d1e001f9
# ╟─42f6a448-0688-47a8-aabe-ec62316abfc2
# ╠═485c65c1-24e9-48ba-891f-82c9022e6a9e
# ╠═3b6ac019-67fe-4875-a0ab-c664793f63aa
# ╠═11b8737b-3864-42d7-b5ae-04e1c6e65090
# ╟─fe8810c3-1dcf-44fc-a99d-7b8b804bc5d8
# ╠═3f60c958-6fc8-4b84-b490-46c2a8fbcf8e
# ╟─99cffe12-c919-4daf-8488-0b9c92b02823
# ╠═e39f3a18-b4a1-4fdd-b62a-2df0f437b4c6
# ╟─03ae1633-5181-4fba-b86f-1393b5566dd3
# ╠═f848bcb5-fc45-446d-9b7a-807bdabfe773
# ╠═c52c657e-b353-4273-b1db-f2ebb24bea9c
# ╟─1a01fa34-0054-42f0-be93-75a20fb15867
# ╠═3cfc6963-6732-4e93-98a4-45725855b2b1
# ╠═271622c6-aec3-4e49-8156-b57cb90d1fe0
# ╠═983f5146-3951-4037-bb26-29ad9a4acd65
# ╟─00123ece-54b5-475d-ae37-d6985b805311
# ╠═9d77f087-aa0d-4681-b2bb-ba1b7f6d9d71
# ╠═e2ce2acb-d755-4d56-a16a-b151e08c65fb
# ╟─1937b938-6d95-4bda-9f19-5fe9e9d48dc9
# ╠═91f7ba03-5a2d-4936-8168-119920cd92bc
# ╟─e5061e0c-8a60-4858-bc2d-9776ef31ea0a
# ╠═0cc83c86-edb6-417a-88be-4132c94ff891
# ╠═e404ca9b-6fc2-4ce7-bac5-03a874e4179b
# ╟─57ed20f0-98c3-45da-9e58-43047dc9ad7a
# ╠═73375107-eeeb-4cb2-8aba-241f11636479
# ╠═998341f0-0612-4627-889f-86667d230d14
# ╠═42d92cc3-715d-4989-85ec-73f2da6a5bff
# ╟─3feb576f-c756-4ea2-9a30-61fcb9718a05
# ╠═ddb1ed04-cf15-470c-b3db-55d13e27504f
# ╟─2177bc11-3d0d-4924-adec-c53544a4bb83
# ╠═13c36227-de8b-4c7d-ad71-d9d3427ee891
# ╠═459b09fe-f810-4ede-bb05-64696dcaf429
# ╟─8940a1bc-1125-4eac-bf2f-69a39088c410
# ╠═ba24a4a2-0ec3-41b1-9c62-604b9bf89505
# ╠═74b97b22-8514-4545-86b8-4a78ec7661e8
# ╟─b3b95023-20e7-4d83-b50f-3bb66e879a2c
# ╠═633e90fe-6960-48e4-9329-321b4b0f071f
# ╟─5e56e02b-7981-4bec-9401-077cc2a6e026
# ╠═98c4ec02-d5d7-4cf4-bfac-2ac150d79e60
# ╠═57c8ff2e-3c48-435f-b34a-d21de227a195
# ╟─b0244f1b-67b8-4360-a4c7-b58c0b4da383
