### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 118bd5d9-1be5-4c79-91c1-f0603e7dcc4b
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "EC991Week3"))
    Pkg.instantiate()
    using EC991Week3, Plots, Printf, Random
    using Markdown: Markdown, @md_str
end

# ╔═╡ fc6b42bb-5e30-405c-ad23-54dd8b4ba681
md"""
# EC991 Week 3 lab: solving the household problem with income risk

**Aseem Patel, University of Essex, Autumn 2026**

This notebook follows the week 3 tutorial slides, *Solving Aiyagari, Part 1*, and runs the code that does what they describe: discretise the asset space and the income process, then solve the household problem by value function iteration, first by grid search and then by interpolating and optimising. The code lives in the package `EC991Week3`, in the folder next to this notebook. Every section puts the maths first and the code underneath.

**Open this file in Pluto**, not as a script. The first cell activates the package environment. The week 1 rules still apply: cells are reactive, each name is defined in exactly one cell, and several lines in one cell go inside `begin ... end` or `let ... end`.

| Folder | What is in it | Rule |
|---|---|---|
| `EC991Week3/src/` | grids, Markov chains, the two Bellman steps, interpolation, optimisers, VFI, checks | never plots, never prints |
| `EC991Week3/run/` | scripts that call the solvers, print results and draw figures | no model logic |
| `EC991Week3/test/` | checks against known answers | run with `Pkg.test()` |
| this notebook | the same code, with the maths alongside | for learning |

The notation is the slides': endowment ``y``, its transition matrix ``P``, ``R = 1 + r``, borrowing limit ``\phi`` (written `ϕ` in the code), grid sizes ``n_a`` and ``n_y``.
"""

# ╔═╡ 7b1bc735-1a6e-4709-9211-308bfd93a213
md"""
The helper below pulls a named block out of the package source so it can be shown next to the maths. It is only for reading.
"""

# ╔═╡ 861f0a9c-a9e1-4484-bdd3-e74c12d69473
function source(name)
    dir = dirname(pathof(EC991Week3))
    for f in readdir(dir; join = true)
        text = read(f, String)
        m = match(Regex("# >>> \\Q$name\\E\\n(.*?)# <<< \\Q$name\\E", "s"), text)
        m === nothing || return Markdown.parse("```julia\n" * m.captures[1] * "```")
    end
    error("no source block called $name")
end

# ╔═╡ abf56767-1ad0-402a-a1ab-f95bddf4d938
md"""
## 1. From week 2 to week 3 (slides 2, 20 to 23)

Week 2's household had a constant income. Now the endowment ``y`` follows a Markov chain, and the Bellman equation gains one state and an expectation:

```math
V(a, y) \;=\; \max_{a' \,\geq\, -\phi} \ u\big(R a + w y - a'\big) + \beta\, \mathbb{E}\big[V(a', y') \mid y\big].
```

The household takes ``r`` and ``w`` as given. In week 5 they come out of equilibrium; this week ``r`` is fixed and ``w`` follows from the firm's first-order conditions with ``\alpha = 0.33`` and ``\delta = 0.05``:
"""

# ╔═╡ 12d69934-a81e-49d6-a40a-08f4199890c1
source("w_from_r")

# ╔═╡ 6c7ca34d-33e6-4756-b7a9-d73c71d4a602
md"""
The household's parameters sit in one struct, as in week 2, now with ``r`` and ``w`` in place of a fixed income. The lab default is ``r = 0.03``.
"""

# ╔═╡ b145ecee-e291-4d90-b0f9-81db0b848124
source("Household")

# ╔═╡ a4676424-d5ef-4349-8b8b-4bc2ee0e38b3
hh = Household()

# ╔═╡ 094accf0-189e-4e3d-b4fc-6f2f97398780
md"""
**What changes in the code.** On the grids ``a_1, \dots, a_{n_a}`` and ``y_1, \dots, y_{n_y}``, ``V`` is an ``n_a \times n_y`` table, ``V_{ij} = V(a_i, y_j)``. The expectation for a choice ``a_k`` and income today ``y_j`` is a weighted sum over income tomorrow, and all of them at once are one matrix product (slide 23):

```math
\mathbb{E}\big[V(a_k, y') \mid y_j\big] \;=\; \sum_{m} V_{km} P_{jm} \;=\; \big(V P^{\top}\big)_{kj}.
```

| Code | Maths |
|---|---|
| `EV = V0 * P'` | ``\mathbb{E}[V(a_k, y') \mid y_j] = (V P^{\top})_{kj}`` for every ``k`` and ``j`` |
| `for j in 1:n_y, i in 1:n_a` | for each state ``(a_i, y_j)`` |
| `for k in 1:n_a` | for each candidate ``a'_k`` |
| `c = R * a[i] + w * y[j] - a[k]` | the budget constraint |
| `c <= 0 && continue` | ``u(c) = -\infty`` when ``c \leq 0`` |
| `utility(c, hh) + β * EV[k, j]` | the right-hand side of the Bellman equation |

Apart from the matrix product and the extra loop over ``j``, it is week 2's Bellman step.
"""

# ╔═╡ 0eb2ce11-c63c-4819-8312-4d72b5e008ad
source("bellman_step!")

# ╔═╡ 68c2f283-0245-4ec8-b07e-0b7fc14ce53f
md"""
## 2. The asset grid (slides 6 to 9)

The grid runs from the borrowing limit ``a_1 = -\phi`` to an upper bound ``a_{n_a}`` that must never bind. Equal spacing wastes points where the policy is nearly straight; growth-rate spacing makes each gap ``1 + \nu`` times the one before, so points bunch near the constraint, where the policy bends:

```math
a_i \;=\; a_1 + \big(a_{n_a} - a_1\big)\, \frac{(1+\nu)^{i-1} - 1}{(1+\nu)^{n_a - 1} - 1}, \qquad i = 1, \dots, n_a .
```
"""

# ╔═╡ 1792e005-d843-4cbb-a767-42245cd8df63
source("asset_grid")

# ╔═╡ ab281dd5-fcdd-483a-960d-d9eacfc6afd6
md"""
The lab default: ``n_a = 300`` points on ``[0, 50]`` with ``\nu = 0.02``.
"""

# ╔═╡ df349408-2853-44df-ba0c-961cb1cfe78d
a = asset_grid(300, 50.0)

# ╔═╡ 3239b518-9c24-4d41-848b-d2e332202f69
let a_eq = asset_grid(300, 50.0; spacing = :equal)
    p = plot(; xlabel = "a", yticks = ([1, 2], ["growth", "equal"]), ylims = (0.5, 2.5), legend = false, size = (800, 220))
    scatter!(p, a_eq, fill(2, 300); ms = 2, msw = 0, color = :gray40)
    scatter!(p, a, fill(1, 300); ms = 2, msw = 0, color = :steelblue)
end

# ╔═╡ a6243bc5-650c-404f-bd02-f768e2eb9b81
(first_gap = a[2] - a[1], last_gap = a[end] - a[end-1], points_below_5 = count(a .< 5))

# ╔═╡ 5b74ea44-b005-4e4c-8e54-bc67efcd8b69
md"""
## 3. The income process (slides 10 to 18)

Log income follows an AR(1), ``\log y' = \rho \log y + \sigma \varepsilon``, ``\varepsilon \sim N(0, 1)``. To use it in the Bellman equation we replace it with a Markov chain: a grid ``\log y_1, \dots, \log y_{n_y}`` and a transition matrix ``P``.

**Tauchen (slides 11 to 13).** Put ``n_y`` equally spaced points on ``\pm m \sigma/\sqrt{1-\rho^2}``, at distance ``d``. Given ``\log y_j``, tomorrow's log income is normal with mean ``\rho \log y_j`` and standard deviation ``\sigma``; ``P_{jk}`` is the area under that density in the bin around ``\log y_k``:

```math
P_{jk} = \Phi\!\left(\frac{\log y_k - \rho \log y_j + d/2}{\sigma}\right) - \Phi\!\left(\frac{\log y_k - \rho \log y_j - d/2}{\sigma}\right),
```

with the two outer bins running to ``\pm\infty``.
"""

# ╔═╡ d60ad043-8688-4ca9-b974-b77c8fea08bf
source("tauchen")

# ╔═╡ a2f56154-a5bc-4927-afb3-f432a2ff19d6
let (logy, P) = tauchen(7, 0.9, 0.1), j = 5
    d = logy[2] - logy[1]
    μ = 0.9 * logy[j]
    x = range(logy[1] - d, logy[end] + d; length = 600)
    p = plot(x, exp.(-((x .- μ) ./ 0.1) .^ 2 ./ 2); color = :black, label = "density of log y' given log y_5",
             xlabel = "log y'", yticks = false, legend = :topleft, size = (800, 300))
    vline!(p, (logy[1:end-1] .+ logy[2:end]) ./ 2; color = :gray, linestyle = :dash, label = "bin edges")
    scatter!(p, logy, zeros(7); color = :black, label = "grid points")
    annotate!(p, [(logy[k], 1.08, text(@sprintf("%.2f", P[j, k]), 9)) for k in 1:7])
end

# ╔═╡ 617980f8-5244-4fec-bbc2-63bca7c41aef
md"""
**Rouwenhorst (slides 14 to 17, backup slide 34).** Put ``n_y`` equally spaced points on ``\pm\sqrt{n_y - 1}\,\sigma/\sqrt{1-\rho^2}``. Start from the ``2 \times 2`` matrix with ``p = (1+\rho)/2`` on the diagonal; to go from ``n - 1`` to ``n`` states, place ``p P``, ``(1-p) P``, ``(1-p) P`` and ``p P`` in the four corners of an ``n \times n`` matrix, add them, and halve the interior rows.
"""

# ╔═╡ f830df32-3557-4739-bd20-d6e5d211cc0f
source("rouwenhorst")

# ╔═╡ c1bbec2a-7b36-4d02-bb43-1557756e23b7
rouwenhorst(2, 0.9, 0.1)

# ╔═╡ 79ab9f42-5759-4bad-a790-d33ed5145c62
md"""
The three-state chain with ``\rho = 0.9`` and ``\sigma = 0.1`` is the one on slides 16 and 17: the matrix below, and ``y \approx \{0.723, 1, 1.383\}``.
"""

# ╔═╡ 2ec569bf-dce2-44b4-932a-d820b202c105
chain = rouwenhorst(3, 0.9, 0.1)

# ╔═╡ fab64c68-ff3b-46e7-a28c-830dd929af14
round.(exp.(chain[1]); digits = 3)

# ╔═╡ a9dc5b8a-51ad-4371-8235-7b55907ed684
md"""
**Which is better?** Both chains should reproduce the standard deviation ``\sigma/\sqrt{1-\rho^2}`` and the autocorrelation ``\rho`` of the AR(1). Each chain's moments come from its stationary distribution:
"""

# ╔═╡ f29f2a67-eeca-4a28-ace6-a51e9512490c
source("chain_moments")

# ╔═╡ 4e31b5ba-8d04-470a-be89-8fd48b4e54b2
let rows = ["| ``\\rho`` | AR(1) sd | Tauchen sd | Tauchen ac | Tauchen ``P_{33}`` | Rouwenhorst sd | Rouwenhorst ac |", "|---|---|---|---|---|---|---|"]
    for ρ in (0.5, 0.9, 0.99)
        lT, PT = tauchen(5, ρ, 0.1); lR, PR = rouwenhorst(5, ρ, 0.1)
        mT, mR = chain_moments(lT, PT), chain_moments(lR, PR)
        push!(rows, @sprintf("| %.2f | %.4f | %.4f | %.4f | %.3f | %.4f | %.4f |", ρ, 0.1 / sqrt(1 - ρ^2), mT.sd, mT.autocorr, PT[3, 3], mR.sd, mR.autocorr))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 2534480c-89e3-4125-9921-bf319d11c04b
md"""
Rouwenhorst matches both moments exactly, at every ``\rho``. Tauchen is close at ``\rho = 0.5``; at ``\rho = 0.99`` its grid is so wide that the middle state is absorbing, ``P_{33} = 1.000`` (slide 15): the chain never moves. Driving the AR(1) and both chains with the same shocks shows it (slide 18):
"""

# ╔═╡ 5fd4f5b6-14e1-4ea0-8b62-818e39a28944
source("simulate_chain")

# ╔═╡ cfe47d6f-a720-4ebf-a9c4-34ec75131319
let T = 200, ε = randn(Random.MersenneTwister(991), T), ρ = 0.9
    lT, PT = tauchen(5, ρ, 0.1); lR, PR = rouwenhorst(5, ρ, 0.1)
    s = zeros(T)
    for t in 2:T; s[t] = ρ * s[t-1] + 0.1 * ε[t]; end
    p = plot(s; color = :gray40, label = "AR(1)", xlabel = "t", ylabel = "log y", size = (800, 300))
    plot!(p, simulate_chain(lT, PT, normal_cdf.(ε))[1]; seriestype = :steppost, label = "Tauchen")
    plot!(p, simulate_chain(lR, PR, normal_cdf.(ε))[1]; seriestype = :steppost, label = "Rouwenhorst")
end

# ╔═╡ 0a0ec04d-2c81-4ece-aac8-9acf7a16e406
md"""
## 4. VFI with grid search (slides 21 to 25)

The lab default income process is the three-state Rouwenhorst chain above.
"""

# ╔═╡ cee0bc7b-c1ec-4ed4-b447-645b8b30630d
P = chain[2]

# ╔═╡ 4f011ad6-e40c-4828-a701-1bb93b2c1231
y = exp.(chain[1])

# ╔═╡ aec4c763-fc25-4b25-a037-ca14cdd1ac23
md"""
**The ``3 \times 3`` example by hand (slides 22 and 23).** Three asset points ``\{0, 1, 2\}`` and the three income levels. Starting from the guess ``V^0(a, y) = u(r a + w y)/(1-\beta)``, the update at a state tries each of the three choices and keeps the best; a choice with ``c \leq 0`` gets ``-\infty``. At ``(a_1, y_1)`` only ``a' = 0`` is affordable; at ``(a_3, y_3)`` all three are:
"""

# ╔═╡ b79ddbe4-9dd2-4468-bb27-f8136fe11c7b
let a3 = [0.0, 1.0, 2.0]
    EV = initial_guess(hh, a3, y) * P'
    rows = ["| state | choice ``a'`` | ``c`` | ``u(c) + \\beta \\sum_m P_{jm} V_{km}`` |", "|---|---|---|---|"]
    for (i, j) in ((1, 1), (3, 3)), k in 1:3
        c = hh.R * a3[i] + hh.w * y[j] - a3[k]
        v = c > 0 ? @sprintf("%.4f", utility(c, hh) + hh.β * EV[k, j]) : "``-\\infty``"
        push!(rows, @sprintf("| (a_%d, y_%d) | %.0f | %.3f | %s |", i, j, a3[k], c, v))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 706e57cd-2d29-4d41-9c64-953d6ca75532
md"""
**The iteration (slide 24).** Guess ``V^0``, apply the Bellman step, stop when ``\max_{i,j} |V^{n+1}_{ij} - V^n_{ij}|`` is below the tolerance. The operator is a contraction with modulus ``\beta``, so the distance falls at least geometrically. The same `solve_vfi` takes either Bellman step; grid search is the default.
"""

# ╔═╡ e62546fd-82e0-415d-b6d5-2fd1e8508d4d
source("solve_vfi")

# ╔═╡ 7580fb8d-e73c-47c0-92d9-a0020c4de1b9
sol = solve_vfi(hh, a, y, P)

# ╔═╡ 7645286d-4509-4e11-80cc-dd2926377636
let c_y = [:steelblue, :gray40, :darkorange]
    p1 = plot(; xlabel = "a", title = "net saving a'(a, y) - a", legend = :topright)
    p2 = plot(; xlabel = "a", title = "consumption c(a, y)", legend = :bottomright)
    for j in 1:3
        plot!(p1, a, sol.apol[:, j] .- a; color = c_y[j], label = "y_$j")
        plot!(p2, a, sol.cpol[:, j]; color = c_y[j], label = "y_$j")
    end
    hline!(p1, [0]; color = :black, linestyle = :dash, label = "")
    p3 = plot(sol.distances; yscale = :log10, xlabel = "iteration n", title = "max |V^{n+1} - V^n|", label = "")
    plot(p1, p2, p3; layout = (1, 3), size = (1000, 300), margin = 4Plots.mm)
end

# ╔═╡ 8eb6ccab-08d7-4890-bca2-e7362dedd842
(iterations = sol.iterations, monotone = policy_monotone(sol), top_of_grid = sol.apol[end, :])

# ╔═╡ 888ee8f2-3ce8-465e-94e2-8e72c9e045f9
md"""
The policy rises in assets and in income, the constraint binds only at low assets with low income, and the upper bound does not bind. The saw-tooth in net saving is the issue on slide 25: the policy can only take grid values, so where the grid is coarse the choice jumps from one point to the next.
"""

# ╔═╡ 2e7f3267-9e42-4769-85bc-ab777278f1d4
md"""
## 5. Interpolate, then optimise (slides 26 to 33)

To choose ``a'`` between grid points we need ``V^n`` between grid points. For each income level tomorrow, interpolate the column ``V^n(\cdot, y_m)`` over ``a'``; today's ``y_j`` enters only through the weights ``P_{jm}``.

**Linear interpolation (slide 28).** For ``a_k \leq a' \leq a_{k+1}``,

```math
\hat V^n(a', y_m) = V^n_{km} + \frac{a' - a_k}{a_{k+1} - a_k}\big(V^n_{k+1,m} - V^n_{km}\big).
```
"""

# ╔═╡ c9712fdc-3283-4ad7-bc5d-3cedad623c92
source("LinearInterp")

# ╔═╡ 54ffe7d4-d54a-40d6-82b2-164b1ecb2ba1
md"""
The worked number from slide 28: stored values ``10`` at ``a = 2`` and ``14`` at ``a = 4``.
"""

# ╔═╡ ce1014b0-99c3-4631-98fe-8a6f73d1df29
LinearInterp([2.0, 4.0], [10.0, 14.0])(2.5)

# ╔═╡ b2183834-2454-4d89-b667-ab0750bf1e45
md"""
**Cubic spline (slide 29, backup slides 35 and 36).** Each gap gets its own cubic; the pieces pass through the dots and share slope and curvature where they meet; zero curvature at both ends closes the system. The coefficients come from one tridiagonal linear system, solved once when the spline is built:
"""

# ╔═╡ 4032f226-5852-4fea-829d-dbb3ef52e24a
source("CubicSpline")

# ╔═╡ be4953a6-72de-4f3d-b9b3-209281a894a4
md"""
One column of the converged ``V``, seven stored points on ``[0, 6]``, both interpolants drawn through them. The 300-point solution stands in for the truth between the dots.
"""

# ╔═╡ ae5af0e5-89ca-4a92-9b4a-01a93adc9de4
let truth = CubicSpline(a, sol.V[:, 1]), xs = collect(range(0, 6; length = 7))
    vs = truth.(xs)
    lin, spl = LinearInterp(xs, vs), CubicSpline(xs, vs)
    xx = range(0, 6; length = 601)
    p1 = plot(xx, truth.(xx); color = :black, lw = 1, label = "300 points", xlabel = "a'", title = "V(a', y_1)", legend = :bottomright)
    plot!(p1, xx, lin.(xx); label = "linear, 7 nodes"); plot!(p1, xx, spl.(xx); label = "spline, 7 nodes")
    scatter!(p1, xs, vs; color = :black, label = "")
    p2 = plot(xx, lin.(xx) .- truth.(xx); label = "linear", xlabel = "a'", title = "error between the dots")
    plot!(p2, xx, spl.(xx) .- truth.(xx); label = "spline")
    plot(p1, p2; layout = (1, 2), size = (900, 300), margin = 4Plots.mm)
end

# ╔═╡ 1427cc97-3839-45d9-8bf4-c5876ccb1b98
md"""
**The objective and the optimiser (slides 26 and 32).** At a state ``(a_i, y_j)`` the objective is now a continuous function of ``a'``,

```math
\varphi(a') = u\big(R a_i + w y_j - a'\big) + \beta \sum_m P_{jm}\, \hat V^n(a', y_m),
```

and a one-dimensional routine finds its peak. Brent's method fits parabolas through its best points and falls back on golden-section steps when a parabola cannot be trusted:
"""

# ╔═╡ 7e4d3d01-bf50-43f4-afc6-d7b3004bbb1a
source("brent")

# ╔═╡ 83936439-8f24-4a40-bab1-956ebc012c28
let i = 150, j = 2
    cash = hh.R * a[i] + hh.w * y[j]
    V̂ = [LinearInterp(a, sol.V[:, m]) for m in 1:3]
    trials = Float64[]
    φ(ap) = (push!(trials, ap); utility(cash - ap, hh) + hh.β * sum(P[j, m] * V̂[m](ap) for m in 1:3))
    xb, fb, nb = brent(φ, a[1], min(a[end], cash - 1e-10))
    k = argmax([a[kk] < cash ? utility(cash - a[kk], hh) + hh.β * sum(P[j, m] * sol.V[kk, m] for m in 1:3) : -Inf for kk in eachindex(a)])
    win = range(xb - 0.3, xb + 0.3; length = 301)
    p = plot(win, [utility(cash - x, hh) + hh.β * sum(P[j, m] * V̂[m](x) for m in 1:3) for x in win]; color = :black, label = "φ(a')",
             xlabel = "a'", title = @sprintf("a = %.2f, y_2: Brent's %d trial points", a[i], nb), size = (800, 320))
    tr = filter(x -> first(win) <= x <= last(win), trials)
    scatter!(p, tr, [utility(cash - x, hh) + hh.β * sum(P[j, m] * V̂[m](x) for m in 1:3) for x in tr]; ms = 3, label = "Brent's trials in this window")
    vline!(p, [a[k]]; color = :steelblue, label = @sprintf("grid search: %.4f", a[k]))
    vline!(p, [xb]; color = :darkorange, linestyle = :dash, label = @sprintf("Brent: %.4f", xb))
end

# ╔═╡ 7738cb4a-284d-4d03-b532-ca3bf10b61d0
md"""
Construct once, evaluate many times: the Bellman step below builds one interpolant per income level at the start of each iteration and then calls them inside the optimiser.
"""

# ╔═╡ 65e5310d-08f5-41b9-b215-2e59fe8cf6aa
source("bellman_step_interp!")

# ╔═╡ ec3e9e51-85a9-4d27-a79d-a2cbf4241c56
sol_lin = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :linear)

# ╔═╡ 9c3f6e94-ab00-4a00-9132-e6369751ace3
sol_cub = solve_vfi(hh, a, y, P; step = bellman_step_interp!, interp = :cubic)

# ╔═╡ be806d6c-d49c-4d00-b1fd-f921118af8b4
let
    p1 = plot(a, sol.apol[:, 2] .- a; seriestype = :steppost, color = :gray40, label = "grid search", xlabel = "a", title = "net saving at y_2")
    plot!(p1, a, sol_lin.apol[:, 2] .- a; label = "linear + Brent")
    plot!(p1, a, sol_cub.apol[:, 2] .- a; linestyle = :dash, label = "cubic + Brent")
    p2 = plot(a, sol_lin.apol[:, 2] .- sol.apol[:, 2]; label = "linear", xlabel = "a", title = "interpolated minus grid search")
    plot!(p2, a, sol_cub.apol[:, 2] .- sol.apol[:, 2]; label = "cubic")
    plot!(p2, a[2:end], diff(a) ./ 2; color = :gray, linestyle = :dash, label = "half the grid gap")
    plot(p1, p2; layout = (1, 2), size = (900, 320), margin = 4Plots.mm)
end

# ╔═╡ c4c96413-ce1d-4eb9-8ff7-81f3de75cd3a
(linear_vs_grid = compare_policies(sol_lin, sol).gap, cubic_vs_grid = compare_policies(sol_cub, sol).gap, linear_vs_cubic = compare_policies(sol_lin, sol_cub).gap)

# ╔═╡ 66979a6a-b3e4-4c37-a717-476746df3ebc
md"""
The two interpolated policies agree with each other to about 0.01, and each differs from grid search by at most about one grid gap, mostly by less than half of one: grid search can only land on a grid point, the interpolated policies can land anywhere. That is slide 33's "more accurate at the same ``n_a``".

**The libraries.** `Interpolations.jl` and `Optim.jl` do the same jobs. They are not in this lab's environment; in an environment of your own you could check them against ours:

```julia
using Interpolations, Optim
itp = linear_interpolation(a, sol.V[:, 1])          # same numbers as LinearInterp(a, sol.V[:, 1])
res = optimize(x -> -itp(x), a[1], a[end], Brent())   # Brent's method, minimising -f
```
"""

# ╔═╡ 74aa6ec8-098f-49c7-b9d9-20e0427c18ba
md"""
## 6. Exercises

Work in new cells below; each reuses the functions above.

1. **Tauchen freezes.** Set ``\rho = 0.99`` with ``n_y = 5`` and simulate both chains on common shocks. How often does Tauchen's chain change state in 200 periods? Rouwenhorst's?
2. **Chebyshev.** Using the formulas on backup slide 37, write a function that interpolates one column of ``V`` on ``[0, a_{n_a}]`` through ``n`` Chebyshev nodes. Compare it with the 300-point solution for ``n = 5, 9, 15``. Where does it do worst, and why there?
3. **Golden section against Brent.** Solve with `routine = :golden` instead of `:brent`. Do the policies change? Count the evaluations each routine needs at one state (both return the count).
4. **The practical check (slide 33).** Double ``n_a`` to 600 and solve again with grid search and with linear + Brent. How much does each saving policy move? Which method's answer is more stable?
"""

# ╔═╡ Cell order:
# ╠═118bd5d9-1be5-4c79-91c1-f0603e7dcc4b
# ╟─fc6b42bb-5e30-405c-ad23-54dd8b4ba681
# ╟─7b1bc735-1a6e-4709-9211-308bfd93a213
# ╠═861f0a9c-a9e1-4484-bdd3-e74c12d69473
# ╟─abf56767-1ad0-402a-a1ab-f95bddf4d938
# ╠═12d69934-a81e-49d6-a40a-08f4199890c1
# ╟─6c7ca34d-33e6-4756-b7a9-d73c71d4a602
# ╠═b145ecee-e291-4d90-b0f9-81db0b848124
# ╠═a4676424-d5ef-4349-8b8b-4bc2ee0e38b3
# ╟─094accf0-189e-4e3d-b4fc-6f2f97398780
# ╠═0eb2ce11-c63c-4819-8312-4d72b5e008ad
# ╟─68c2f283-0245-4ec8-b07e-0b7fc14ce53f
# ╠═1792e005-d843-4cbb-a767-42245cd8df63
# ╟─ab281dd5-fcdd-483a-960d-d9eacfc6afd6
# ╠═df349408-2853-44df-ba0c-961cb1cfe78d
# ╠═3239b518-9c24-4d41-848b-d2e332202f69
# ╠═a6243bc5-650c-404f-bd02-f768e2eb9b81
# ╟─5b74ea44-b005-4e4c-8e54-bc67efcd8b69
# ╠═d60ad043-8688-4ca9-b974-b77c8fea08bf
# ╠═a2f56154-a5bc-4927-afb3-f432a2ff19d6
# ╟─617980f8-5244-4fec-bbc2-63bca7c41aef
# ╠═f830df32-3557-4739-bd20-d6e5d211cc0f
# ╠═c1bbec2a-7b36-4d02-bb43-1557756e23b7
# ╟─79ab9f42-5759-4bad-a790-d33ed5145c62
# ╠═2ec569bf-dce2-44b4-932a-d820b202c105
# ╠═fab64c68-ff3b-46e7-a28c-830dd929af14
# ╟─a9dc5b8a-51ad-4371-8235-7b55907ed684
# ╠═f29f2a67-eeca-4a28-ace6-a51e9512490c
# ╠═4e31b5ba-8d04-470a-be89-8fd48b4e54b2
# ╟─2534480c-89e3-4125-9921-bf319d11c04b
# ╠═5fd4f5b6-14e1-4ea0-8b62-818e39a28944
# ╠═cfe47d6f-a720-4ebf-a9c4-34ec75131319
# ╟─0a0ec04d-2c81-4ece-aac8-9acf7a16e406
# ╠═cee0bc7b-c1ec-4ed4-b447-645b8b30630d
# ╠═4f011ad6-e40c-4828-a701-1bb93b2c1231
# ╟─aec4c763-fc25-4b25-a037-ca14cdd1ac23
# ╠═b79ddbe4-9dd2-4468-bb27-f8136fe11c7b
# ╟─706e57cd-2d29-4d41-9c64-953d6ca75532
# ╠═e62546fd-82e0-415d-b6d5-2fd1e8508d4d
# ╠═7580fb8d-e73c-47c0-92d9-a0020c4de1b9
# ╠═7645286d-4509-4e11-80cc-dd2926377636
# ╠═8eb6ccab-08d7-4890-bca2-e7362dedd842
# ╟─888ee8f2-3ce8-465e-94e2-8e72c9e045f9
# ╟─2e7f3267-9e42-4769-85bc-ab777278f1d4
# ╠═c9712fdc-3283-4ad7-bc5d-3cedad623c92
# ╟─54ffe7d4-d54a-40d6-82b2-164b1ecb2ba1
# ╠═ce1014b0-99c3-4631-98fe-8a6f73d1df29
# ╟─b2183834-2454-4d89-b667-ab0750bf1e45
# ╠═4032f226-5852-4fea-829d-dbb3ef52e24a
# ╟─be4953a6-72de-4f3d-b9b3-209281a894a4
# ╠═ae5af0e5-89ca-4a92-9b4a-01a93adc9de4
# ╟─1427cc97-3839-45d9-8bf4-c5876ccb1b98
# ╠═7e4d3d01-bf50-43f4-afc6-d7b3004bbb1a
# ╠═83936439-8f24-4a40-bab1-956ebc012c28
# ╟─7738cb4a-284d-4d03-b532-ca3bf10b61d0
# ╠═65e5310d-08f5-41b9-b215-2e59fe8cf6aa
# ╠═ec3e9e51-85a9-4d27-a79d-a2cbf4241c56
# ╠═9c3f6e94-ab00-4a00-9132-e6369751ace3
# ╠═be806d6c-d49c-4d00-b1fd-f921118af8b4
# ╠═c4c96413-ce1d-4eb9-8ff7-81f3de75cd3a
# ╟─66979a6a-b3e4-4c37-a717-476746df3ebc
# ╟─74aa6ec8-098f-49c7-b9d9-20e0427c18ba
