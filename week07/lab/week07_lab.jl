### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 1e3eff16-3eef-5694-aeca-0782f8608b3a
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "EC991Week7"))
    Pkg.instantiate()
    using EC991Week7, Plots, Printf, LinearAlgebra, Random
    using Markdown: Markdown, @md_str
    BLAS.set_num_threads(1)
end

# ╔═╡ f810e26f-da5c-573d-b751-d2d1a57660ab
md"""
# EC991 Week 7 lab: aggregate shocks and transitions

**Aseem Patel, University of Essex, Autumn 2026**

This notebook follows the week 7 lecture, *Aggregate Shocks and Transition Dynamics*, and runs the code that does what it describes. The code lives in the package `EC991Week7`, in the folder next to this notebook. Every section puts the maths first and the code underneath. Slide numbers are the lecture's.

**The one idea.** Productivity ``Z_t`` now moves, for everyone at once, so prices move too, and households must forecast them. There are two ways to solve this:

| | Part A: state space (Krusell–Smith) | Part B: sequence space |
|---|---|---|
| households know | ``Z``, ``K`` and a rule ``\log K' = \theta_{0Z} + \theta_{1Z} \log K`` | the whole path of prices |
| the unknown | four numbers, ``\theta`` | a path, ``\{K_t\}_{t=2}^{T-1}`` |
| the loop | solve, simulate, regress, update ``\theta`` | households backwards, distribution forwards, update ``K_t`` |
| functions | `solve_krusell_smith` | `solve_transition`, `mit_shock`, `tax_reform` |

**Open this file in Pluto**, not as a script. Cells are reactive, each name is defined in exactly one cell, and several lines in one cell go inside `begin ... end` or `let ... end`. The Krusell–Smith economy takes about a minute to solve and each transition several seconds, so the notebook takes a while when it opens.
"""

# ╔═╡ 95e91251-13a6-593f-bb6f-d078aa34a9b3
md"""
The helper below pulls a named block out of the package source so it can be shown next to the maths. It is only for reading.
"""

# ╔═╡ 22859aeb-93c3-564d-87ff-54f13b5ed778
function source(name)
    dir = dirname(pathof(EC991Week7))
    for f in readdir(dir; join = true)
        text = read(f, String)
        m = match(Regex("# >>> \\Q$name\\E\\n(.*?)# <<< \\Q$name\\E", "s"), text)
        m === nothing || return Markdown.parse("```julia\n" * m.captures[1] * "```")
    end
    error("no source block called $name")
end

# ╔═╡ 5cbaf980-f3e0-551c-921d-ce3e14b509af
md"""
# Part A: Krusell and Smith

## 1. The economy (slides 18 to 22)

One period is a quarter. ``\beta = 0.99``, log utility, ``\alpha = 0.36``, ``\delta = 0.025``; ``Z \in \{0.99, 1.01\}`` with ``P(Z' = Z \mid Z) = 0.875``; the unemployed have ``y_l = 0.25``, the employed ``y_h = 1``. Index 1 is always the low state: a recession, or unemployment.

Slides 20 and 21 give the probability that the unemployed stay unemployed. The probability that the employed lose their job follows from **consistency** (slide 22): with ``u(Z)`` the unemployment rate (10% in a recession, 4% in a boom),

```math
u(Z') = P(y_l \mid y_l, Z, Z')\, u(Z) + P(y_l \mid y_h, Z, Z')\, \big(1 - u(Z)\big),
```

one equation in one unknown:
"""

# ╔═╡ 313bd69b-0000-551c-abe2-ed5682787cce
source("employment_transitions")

# ╔═╡ cea56214-83b7-50af-a366-b0fe880f28ab
eco = KSEconomy()

# ╔═╡ 15931561-ec01-5ab3-b382-e2c8d60268c1
md"""
The ``4 \times 4`` matrix ``P(y', Z' \mid y, Z)`` of slide 6, with the states in the order ``(y_l, Z_l), (y_h, Z_l), (y_l, Z_h), (y_h, Z_h)``:
"""

# ╔═╡ aa80f9b4-3073-5d6b-bb17-4b6e2158dddb
Π = joint_transition(eco)

# ╔═╡ 79e0b47d-b2ac-5852-b25b-7372d94200b7
md"""
**Check it.** Every row sums to one, and summing a row over ``y'`` gives ``P(Z' \mid Z)``, whatever ``y`` is: 0.875 for the same state, 0.125 for the other.
"""

# ╔═╡ e232144e-9028-5235-80ff-3a02872c8537
let PZ = [Π[s, 1] + Π[s, 2] for s in 1:4]
    (; row_sums = vec(sum(Π; dims = 2)), prob_recession_next = PZ)
end

# ╔═╡ 72e3d690-f0d4-5142-9fe9-3d4bdca30a26
md"""
## 2. The household (slides 10, 13 and 14)

The distribution is replaced by aggregate capital ``K``, and households forecast tomorrow's with the rule

```math
\log K' = \theta_{0Z} + \theta_{1Z} \log K .
```

The household is solved by week 4's endogenous grid method, with the Euler equation of slide 10,

```math
u'(c) = \beta\, \mathbb{E}\Big[ u'(c')\, \big(1 + r(Z', K')\big) \,\Big|\, y, Z \Big],
```

on a grid over ``(a, y, Z, K)``. Tomorrow's ``K'`` comes from the rule and lies between two points of the ``K`` grid, so tomorrow's consumption is interpolated there:
"""

# ╔═╡ 5134b8eb-a916-5f4a-9bc6-250b5b3d7bc0
source("ks_egm_step!")

# ╔═╡ 3a463b10-44a4-5ddb-9aac-24f1d08cc251
md"""
## 3. Simulate, regress, repeat (slides 15 to 17)

Slide 15 simulates a panel of households. The code moves a histogram instead, week 4's distribution on a fine asset grid: a panel with infinitely many households, and so no sampling noise.
"""

# ╔═╡ a937050f-ea51-52d1-be16-65c205f51be8
source("simulate_capital")

# ╔═╡ a5d1bc15-962e-5219-bdea-0d8dd1fcc11f
md"""
Then, for each ``Z``, regress ``\log K_{t+1}`` on ``\log K_t`` over the dates with ``Z_t = Z``:
"""

# ╔═╡ 3207a333-b542-5562-aff4-98c63f6cff62
source("regress_law_of_motion")

# ╔═╡ aa5bca9c-6153-555a-a313-691c0ee8d364
md"""
And the loop, starting from slide 17's guess ``\theta_{0Z} = \log K_{ss}``, ``\theta_{1Z} = 0``, with dampening ``\theta \leftarrow \omega \theta^{\text{new}} + (1 - \omega) \theta``:
"""

# ╔═╡ 1ac519ad-e27a-5070-aec2-8f69e691aac2
source("solve_krusell_smith")

# ╔═╡ 8cabdb2a-7197-53a0-8e89-cee827117387
md"""
## 4. The approximate equilibrium (slide 23)

Edit `ω` and the equilibrium is solved again. With ``\omega = 0.3`` it takes 30 iterations; try 0.5 and 0.7.
"""

# ╔═╡ f35de239-cb3b-5652-94dc-9503fce0c853
ω = 0.3

# ╔═╡ ac9f4a4e-0286-554b-8174-4273d6f591b7
eq = solve_krusell_smith(eco; ω)

# ╔═╡ ebfe5314-4216-595f-bb56-a153383ca529
let rows = join([@sprintf("| %s | %.4f | %.4f | %.7f | %.4f%% |", name, eq.θ[1, z], eq.θ[2, z], eq.R2[z], 100 * eq.σ[z])
                 for (z, name) in ((2, "boom"), (1, "recession"))], "\n")
    Markdown.parse("| | ``\\theta_{0Z}`` | ``\\theta_{1Z}`` | ``R^2`` | s.d. of residuals |\n|---|---|---|---|---|\n" * rows *
                   "\n\n$(length(eq.history)) iterations.")
end

# ╔═╡ 7d5e987b-2a37-5fcd-a09e-0fde5ecd3c11
md"""
Krusell and Smith find slopes of 0.962 in a boom and 0.965 in a recession, with ``R^2 = 0.999998``. The intercepts differ from theirs because they measure labour in hours, so their ``K`` is about a third of ours; the slopes do not depend on units.

**How far ahead does the rule see?** `forecast_errors` starts from the actual ``K`` and iterates the rule forward, looking only at the shocks. Errors are in %:
"""

# ╔═╡ 6af1d4bc-4a5d-58eb-b974-debc72d5d4bf
forecast_errors(eq)[(:dynamic_mean, :dynamic_max, :horizon, :horizon_mean, :horizon_max)]

# ╔═╡ 062b097d-40a2-5118-8a24-e3151f19ff0f
md"""
## 5. Your turn: the graphs on slides 24 and 25

**Slide 24.** Tomorrow's against today's aggregate capital: the rule for each ``Z``, the simulated pairs, and the 45° line.
"""

# ╔═╡ 3723b561-47a3-5068-9e2d-c4ba7f543c73
let T0 = eq.eco.T_burn + 1, K = eq.K[T0:end], z = eq.zpath[T0:end]
    lo, hi = extrema(K)
    grid = range(lo, hi; length = 100)
    p = plot(xlabel = "today's aggregate capital, K", ylabel = "tomorrow's aggregate capital, K'",
             legend = :topleft, xlims = (lo, hi), ylims = (lo, hi))
    plot!(p, [lo, hi], [lo, hi]; color = :gray, linestyle = :dash, label = "45° line")
    for (zz, name) in ((2, "boom"), (1, "recession"))
        ts = findall(==(zz), z[1:end-1])
        scatter!(p, K[ts], K[ts .+ 1]; markersize = 2, markerstrokewidth = 0, alpha = 0.4, label = "simulated, $name")
        plot!(p, grid, forecast_K.(grid, zz, Ref(eq.θ)); color = :black, linewidth = 1, label = "")
    end
    p
end

# ╔═╡ 67e4a842-8885-592f-89f3-fe94e313983e
md"""
**Slide 25.** Tomorrow's against today's individual assets, for the employed and the unemployed, in a boom (solid) and a recession (dashed), at aggregate capital `K_at`. Lower `a_top` to zoom in on the borrowing limit, where the rules bend.
"""

# ╔═╡ 0bc1232d-a6ac-59c2-a50e-a20d56a002c8
K_at = sum(eq.K[eq.eco.T_burn+1:end]) / (length(eq.K) - eq.eco.T_burn)

# ╔═╡ b9ccc290-2163-56c5-b24a-b1cf8fb2a9dd
a_top = 80.0

# ╔═╡ a97252f7-3011-554b-8bb5-f511b0db0635
let a = eq.s.a, keep = eq.s.a .<= a_top
    p = plot(xlabel = "today's assets, a", ylabel = "tomorrow's assets, a'", legend = :topleft,
             title = @sprintf("K = %.2f", K_at))
    plot!(p, [0, a_top], [0, a_top]; color = :gray, linestyle = :dash, label = "45° line")
    for (j, who) in ((2, "employed"), (1, "unemployed"))
        plot!(p, a[keep], individual_rule(eq.sol, eq.s, j, 2, K_at)[keep]; linewidth = 2.5, label = "$who, boom")
        plot!(p, a[keep], individual_rule(eq.sol, eq.s, j, 1, K_at)[keep]; linewidth = 2.5, linestyle = :dash,
              label = "$who, recession")
    end
    p
end

# ╔═╡ 1bf53099-c12e-54b3-a2ac-1bdfe87e376f
md"""
Away from the borrowing limit the rules are straight and parallel: every household saves the same share of an extra unit of wealth. So aggregate saving depends on the distribution only through its mean, and ``K`` alone forecasts ``K'`` (slide 26).

## 6. Wealth and the business cycle (slides 29 and 30)
"""

# ╔═╡ 7de128cc-afa7-5741-b7d5-7ffd7686a0bd
wealth_stats(eq)

# ╔═╡ 7c5eb1e5-c3ac-5df9-bb87-1b0f9df75133
business_cycle_stats(eq)

# ╔═╡ 22721e59-9a4e-5c56-961c-42c838e5bfac
md"""
Krusell and Smith's benchmark: top 1% hold 3%, top 10% hold 19%, Gini 0.25, against 30%, 64% and 0.79 in the data. With one discount factor, income risk alone does not make anyone very rich (slide 28). Their incomplete-markets economy has corr``(C_t, Y_t) = 0.701`` and corr``(Y_t, Y_{t-4}) = 0.481``.
"""

# ╔═╡ cb93de4d-c517-5396-aece-82095aa046e1
md"""
# Part B: the sequence space

## 7. The steady state (slides 33 and 34)

Part B uses the week 4 economy (``\beta = 0.96``, ``\gamma = 2``, ``\alpha = 0.33``, ``\delta = 0.05``, seven income states, one period a year), with productivity ``Z`` in front of the production function and a labour tax ``\tau`` paid back as a lump sum. With ``Z = 1`` and ``\tau = 0`` it is exactly week 4's economy, and the steady state should give week 4's ``r = 0.0377``:
"""

# ╔═╡ b9b31e1b-c945-5ccf-a6ab-c2ed5d110e24
ss = solve_steady_state(Economy())

# ╔═╡ 3cf4b126-1dd4-5888-b8d0-5ac6be42e04b
(; r = ss.r, w = ss.w, K = ss.K, Ea = ss.Ea)

# ╔═╡ 4f55a956-7de1-58d3-92a8-43674aa82698
md"""
## 8. The shooting algorithm (slides 39 to 41)

Along a transition, today's return ``R_t`` and tomorrow's ``R_{t+1}`` differ. That is the one change to week 4's EGM step:
"""

# ╔═╡ c1cfddcc-0f6f-59e1-9622-e74c20c47e49
source("egm_backward!")

# ╔═╡ d6d51344-a561-5e71-a23d-cab5bd8da8a9
md"""
**(ii) and (iii).** Guess the capital path, compute prices, and solve the households backwards from the final steady state's policy at ``T``:
"""

# ╔═╡ 6a2a671b-88d7-58ad-963f-381c8e163c80
source("backward_pass")

# ╔═╡ 2e9ceb9d-75e0-5d7b-a754-2d2ebfa480d5
md"""
**(iv) and (v).** Move the distribution forwards from the initial steady state, ``\lambda_{t+1} = Q_t' \lambda_t``, and add up assets: ``K^s_t = \sum_j a_j \lambda_{t,j}``.
"""

# ╔═╡ 67815a6a-4e67-5958-bb03-4417112186ff
source("forward_pass")

# ╔═╡ 06ef1f51-89e4-5f43-abb2-d1be2e255c81
md"""
**(vi).** Compare implied with guessed capital, and update with dampening, ``K^g \leftarrow \omega K^s + (1 - \omega) K^g``:
"""

# ╔═╡ 0ca03b64-cf15-5585-a33c-bd9829992d4a
source("solve_transition")

# ╔═╡ 63f45b8d-c7bd-5b5a-adb5-639933766617
md"""
## 9. Your turn: an MIT shock (slides 33, 35 and 36)

At ``t = 1``, ``\log Z_1 = \varepsilon``, and then ``\log Z_t = \rho_Z \log Z_{t-1}``. Nobody expected it, and nobody expects another. Edit the shock and its persistence:
"""

# ╔═╡ d7249dfd-b2e0-5c63-ba9b-cbfd1a2f5196
ε_shock = 0.01

# ╔═╡ 6bb0f466-0282-5390-b6e9-8f71c2439d4f
ρ_Z = 0.9

# ╔═╡ 35170363-1775-53c8-aa7e-6d7f09592f36
tr = mit_shock(ss; ε = ε_shock, ρ_Z)

# ╔═╡ 51ed62bf-af76-5ad8-b4c7-827213bbe719
let t = 0:150, Ȳ = ss.K^ss.eco.α * ss.s.N^(1 - ss.eco.α), C̄ = sum(ss.c .* ss.λ)
    path(x, x̄) = 100 .* ([x̄; x[1:150]] ./ x̄ .- 1)
    plot(plot(t, path(tr.Z, 1.0); title = "productivity Z (%)"),
         plot(t, path(tr.K, ss.K); title = "capital K (%)"),
         plot(t, 100 .* ([ss.r; tr.r[1:150]] .- ss.r); title = "interest rate r (p.p.)"),
         plot(t, path(tr.w, ss.w); title = "wage w (%)"),
         plot(t, path(tr.Y, Ȳ); title = "output Y (%)"),
         plot(t, path(tr.C, C̄); title = "consumption C (%)");
         layout = (2, 3), legend = false, linewidth = 2.5, xlabel = "t", size = (900, 500))
end

# ╔═╡ a7dddb3d-b7a0-5677-b6df-142df5d9ac6d
md"""
Capital cannot move at ``t = 1``: it was chosen before the shock. Households save part of the extra income, so capital builds up, then pushes the interest rate below its steady state once productivity has faded.

**Slide 41: check the last date.** If ``T`` is long enough, the capital households supply at ``T`` is the final steady state's. `end_gap` is ``K^s_T - K_T``; `iterations` is how many passes the algorithm took.
"""

# ╔═╡ bef71f02-517b-5316-9020-3dd4bf26b6de
(; iterations = tr.iterations, end_gap = tr.end_gap, peak_K_percent = 100 * (maximum(tr.K) / ss.K - 1), at_t = argmax(tr.K))

# ╔═╡ 2e69f4c3-9d91-587b-aae6-a2fa55fe57ce
md"""
Slide 36: inequality along the transition, as changes from the steady state.
"""

# ╔═╡ 45ba22be-0bcb-535f-a11f-0c6fe0777f6e
let t = 1:150, d(x) = x[t] .- x[end]
    plot(plot(t, d(tr.gini); title = "Gini of wealth, all"),
         plot(t, d(tr.gini_low); title = "Gini, lowest income"),
         plot(t, d(tr.gini_high); title = "Gini, highest income"),
         plot(t, 100 .* d(tr.constrained); title = "at the limit (p.p.)");
         layout = (2, 2), legend = false, linewidth = 2.5, xlabel = "t", size = (800, 500))
end

# ╔═╡ ac440984-04c4-552a-8ccd-b612a0359d35
md"""
## 10. Your turn: a labour tax reform (slides 37 and 38)

The economy is in its steady state with ``\tau = 0``; at ``t = 1`` the tax rises for ever to `τ_new`, and the revenue goes back to everyone as the same lump sum, ``T = \tau w N``. Two steady states now, and the transition between them:
"""

# ╔═╡ 6a148a98-cd6f-57bd-9168-69671b59f573
τ_new = 0.2

# ╔═╡ f1768133-5495-5fe2-baf7-90b1087836ac
ss_tax = solve_steady_state(Economy(τ = τ_new))

# ╔═╡ c2d16d6f-8be9-5632-a614-340cca651831
tr_tax = tax_reform(ss, ss_tax)

# ╔═╡ c09e312d-57b6-5499-ae0d-52ba20d99f50
let t = 0:150, pct(x, x0) = 100 .* (x ./ x0 .- 1)
    p1 = plot(t, [0.0; tr_tax.τ[1:150]]; title = "labour tax τ", linetype = :steppost)
    p2 = plot(t, pct([ss.K; tr_tax.K[1:150]], ss.K); title = "capital K (% from initial)")
    hline!(p2, [pct(ss_tax.K, ss.K)]; color = :gray, linestyle = :dash)
    p3 = plot(t, 100 .* ([ss.r; tr_tax.r[1:150]] .- ss.r); title = "interest rate r (p.p.)")
    hline!(p3, [100 * (ss_tax.r - ss.r)]; color = :gray, linestyle = :dash)
    p4 = plot(t, pct([ss.w; tr_tax.w[1:150]], ss.w); title = "wage w (% from initial)")
    hline!(p4, [pct(ss_tax.w, ss.w)]; color = :gray, linestyle = :dash)
    plot(p1, p2, p3, p4; layout = (2, 2), legend = false, linewidth = 2.5, xlabel = "t", size = (800, 500))
end

# ╔═╡ ee844c1e-9e51-54fd-b7aa-8c7e7cc75a34
md"""
The tax with a lump-sum transfer is insurance: it takes from high incomes and gives to low ones. Households face less risk, save less for precautionary reasons, and capital falls.

## 11. From one shock to many (slides 42 to 45)

Record capital's response at each horizon per unit of shock, ``h_s = (K_{s+1} - \bar K) / \varepsilon``, then add up the effects of a sequence of shocks:
"""

# ╔═╡ dfa985e9-5062-590c-bf7f-6084ffff46e1
source("bkm_simulate")

# ╔═╡ 8030ff33-217a-56f9-8bc3-e294638442e1
md"""
This works only if the response is close to linear. Compare the response per unit of shock for ``+\varepsilon`` and ``-\varepsilon``:
"""

# ╔═╡ 4f762a68-22c9-5da6-9cb0-029cd099eccf
tr_neg = mit_shock(ss; ε = -ε_shock, ρ_Z)

# ╔═╡ f94bfea7-206f-5db5-ba76-ef85426f5c64
let h = impulse_response(tr.K, ss.K, ε_shock), hn = impulse_response(tr_neg.K, ss.K, -ε_shock)
    p = plot(0:149, 100 .* [h[1:150] hn[1:150]] ./ ss.K; label = ["ε > 0" "ε < 0"], linewidth = 2.5,
             xlabel = "horizon s", ylabel = "% of K per unit of ε", title = "capital's response per unit of shock")
    (; plot = p, largest_gap_share_of_peak = maximum(abs.(h .- hn)) / maximum(abs.(h)))
end

# ╔═╡ 6355a7f7-1748-5b3c-9322-5e227e593f06
md"""
Now draw `years` innovations with standard deviation `σ_ε` and simulate capital and output by adding up their effects (slide 44). Change `seed` for another history.
"""

# ╔═╡ 1689fea7-c9f1-5b3e-aedc-40262e6c436b
σ_ε = 0.01

# ╔═╡ 0f5d93dc-a83b-51c5-a2e0-b398f9a734f4
years = 200

# ╔═╡ fb781c60-4fef-5002-aa33-bacee91de8ef
seed = 7

# ╔═╡ 6b3d4c28-06df-55c8-be53-ea34148597e4
let ε = σ_ε .* randn(Random.Xoshiro(seed), years), Ȳ = ss.K^ss.eco.α * ss.s.N^(1 - ss.eco.α)
    K = 100 .* bkm_simulate(impulse_response(tr.K, ss.K, ε_shock), ε) ./ ss.K
    Y = 100 .* bkm_simulate(impulse_response(tr.Y, Ȳ, ε_shock), ε) ./ Ȳ
    p = plot(1:years, [Y K]; label = ["output Y" "capital K"], linewidth = [1.5 2.5],
             xlabel = "year", ylabel = "% deviation from steady state", title = "Boppart, Krusell and Mitman")
    hline!(p, [0]; color = :gray, label = "")
end

# ╔═╡ e52b8db8-851b-51cf-85d5-0ea6db1d04fb
md"""
## 12. Try

1. In Part A, set `ω = 0.7`. What happens to the distance at each iteration, and why does dampening help?
2. Solve `KSEconomy(γ = 2.0)` (more risk aversion). Does mean capital rise, as on slide 30? Does the ``R^2`` of the rule stay as high?
3. Set `ρ_Z = 0.5` and then `ρ_Z = 0.99`. How does the timing of capital's peak change? At `ρ_Z = 0.99` look at `end_gap`: is ``T = 300`` still long enough?
4. Call `mit_shock(ss; ω = 0.5)`. Then try `ω = 0.1`: slower, but does it converge?
5. Set `ε_shock = 0.05`. How large is the gap between the responses to ``+\varepsilon`` and ``-\varepsilon`` now? What does that say about the BKM simulation for large shocks?
6. Raise `τ_new` to 0.4. Is the fall in capital twice as large? How long does the economy take to cover half of the way?
"""

# ╔═╡ Cell order:
# ╠═1e3eff16-3eef-5694-aeca-0782f8608b3a
# ╟─f810e26f-da5c-573d-b751-d2d1a57660ab
# ╟─95e91251-13a6-593f-bb6f-d078aa34a9b3
# ╠═22859aeb-93c3-564d-87ff-54f13b5ed778
# ╟─5cbaf980-f3e0-551c-921d-ce3e14b509af
# ╠═313bd69b-0000-551c-abe2-ed5682787cce
# ╠═cea56214-83b7-50af-a366-b0fe880f28ab
# ╟─15931561-ec01-5ab3-b382-e2c8d60268c1
# ╠═aa80f9b4-3073-5d6b-bb17-4b6e2158dddb
# ╟─79e0b47d-b2ac-5852-b25b-7372d94200b7
# ╠═e232144e-9028-5235-80ff-3a02872c8537
# ╟─72e3d690-f0d4-5142-9fe9-3d4bdca30a26
# ╠═5134b8eb-a916-5f4a-9bc6-250b5b3d7bc0
# ╟─3a463b10-44a4-5ddb-9aac-24f1d08cc251
# ╠═a937050f-ea51-52d1-be16-65c205f51be8
# ╟─a5d1bc15-962e-5219-bdea-0d8dd1fcc11f
# ╠═3207a333-b542-5562-aff4-98c63f6cff62
# ╟─aa5bca9c-6153-555a-a313-691c0ee8d364
# ╠═1ac519ad-e27a-5070-aec2-8f69e691aac2
# ╟─8cabdb2a-7197-53a0-8e89-cee827117387
# ╠═f35de239-cb3b-5652-94dc-9503fce0c853
# ╠═ac9f4a4e-0286-554b-8174-4273d6f591b7
# ╠═ebfe5314-4216-595f-bb56-a153383ca529
# ╟─7d5e987b-2a37-5fcd-a09e-0fde5ecd3c11
# ╠═6af1d4bc-4a5d-58eb-b974-debc72d5d4bf
# ╟─062b097d-40a2-5118-8a24-e3151f19ff0f
# ╠═3723b561-47a3-5068-9e2d-c4ba7f543c73
# ╟─67e4a842-8885-592f-89f3-fe94e313983e
# ╠═0bc1232d-a6ac-59c2-a50e-a20d56a002c8
# ╠═b9ccc290-2163-56c5-b24a-b1cf8fb2a9dd
# ╠═a97252f7-3011-554b-8bb5-f511b0db0635
# ╟─1bf53099-c12e-54b3-a2ac-1bdfe87e376f
# ╠═7de128cc-afa7-5741-b7d5-7ffd7686a0bd
# ╠═7c5eb1e5-c3ac-5df9-bb87-1b0f9df75133
# ╟─22721e59-9a4e-5c56-961c-42c838e5bfac
# ╟─cb93de4d-c517-5396-aece-82095aa046e1
# ╠═b9b31e1b-c945-5ccf-a6ab-c2ed5d110e24
# ╠═3cf4b126-1dd4-5888-b8d0-5ac6be42e04b
# ╟─4f55a956-7de1-58d3-92a8-43674aa82698
# ╠═c1cfddcc-0f6f-59e1-9622-e74c20c47e49
# ╟─d6d51344-a561-5e71-a23d-cab5bd8da8a9
# ╠═6a2a671b-88d7-58ad-963f-381c8e163c80
# ╟─2e9ceb9d-75e0-5d7b-a754-2d2ebfa480d5
# ╠═67815a6a-4e67-5958-bb03-4417112186ff
# ╟─06ef1f51-89e4-5f43-abb2-d1be2e255c81
# ╠═0ca03b64-cf15-5585-a33c-bd9829992d4a
# ╟─63f45b8d-c7bd-5b5a-adb5-639933766617
# ╠═d7249dfd-b2e0-5c63-ba9b-cbfd1a2f5196
# ╠═6bb0f466-0282-5390-b6e9-8f71c2439d4f
# ╠═35170363-1775-53c8-aa7e-6d7f09592f36
# ╠═51ed62bf-af76-5ad8-b4c7-827213bbe719
# ╟─a7dddb3d-b7a0-5677-b6df-142df5d9ac6d
# ╠═bef71f02-517b-5316-9020-3dd4bf26b6de
# ╟─2e69f4c3-9d91-587b-aae6-a2fa55fe57ce
# ╠═45ba22be-0bcb-535f-a11f-0c6fe0777f6e
# ╟─ac440984-04c4-552a-8ccd-b612a0359d35
# ╠═6a148a98-cd6f-57bd-9168-69671b59f573
# ╠═f1768133-5495-5fe2-baf7-90b1087836ac
# ╠═c2d16d6f-8be9-5632-a614-340cca651831
# ╠═c09e312d-57b6-5499-ae0d-52ba20d99f50
# ╟─ee844c1e-9e51-54fd-b7aa-8c7e7cc75a34
# ╠═dfa985e9-5062-590c-bf7f-6084ffff46e1
# ╟─8030ff33-217a-56f9-8bc3-e294638442e1
# ╠═4f762a68-22c9-5da6-9cb0-029cd099eccf
# ╠═f94bfea7-206f-5db5-ba76-ef85426f5c64
# ╟─6355a7f7-1748-5b3c-9322-5e227e593f06
# ╠═1689fea7-c9f1-5b3e-aedc-40262e6c436b
# ╠═0f5d93dc-a83b-51c5-a2e0-b398f9a734f4
# ╠═fb781c60-4fef-5002-aa33-bacee91de8ef
# ╠═6b3d4c28-06df-55c8-be53-ea34148597e4
# ╟─e52b8db8-851b-51cf-85d5-0ea6db1d04fb
