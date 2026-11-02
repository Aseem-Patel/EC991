### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 5ec47728-55ce-551a-85bc-10d0787ad4b1
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "EC991Week4"))
    Pkg.instantiate()
    using EC991Week4, Plots, Printf
    using Markdown: Markdown, @md_str
end

# ╔═╡ 6da29021-3033-5200-bb66-9552b655a71e
md"""
# EC991 Week 4 lab: the Aiyagari equilibrium

**Aseem Patel, University of Essex, Autumn 2026**

This notebook follows the week 4 lecture, *Computing the Aiyagari Equilibrium*, and runs the code that does what it describes. The code lives in the package `EC991Week4`, in the folder next to this notebook. Every section puts the maths first and the code underneath. Slide numbers are the week 4 lecture's.

**The one idea.** The equilibrium loop is written once. Only Step 3, the household problem, changes, and every method hands the same object to Step 4: the saving policy. So we climb a ladder of household methods, each answering a problem the one before exposes.

| Method | `method =` | Where interpolation enters |
|---|---|---|
| VFI by grid search | `:vfi_grid` | nowhere: ``a'`` stays on the grid |
| the same with speed-ups | `:vfi_fast` | nowhere |
| VFI, interpolate and optimise | `:vfi_interp` | ``V`` is interpolated, so the search can pick ``a'`` off the grid |
| the same with Howard's updates | `:vfi_interp_howard` | as above |
| endogenous grid method | `:egm` | no search at all; the saving policy is interpolated back onto the grid |

**Open this file in Pluto**, not as a script. Cells are reactive, each name is defined in exactly one cell, and several lines in one cell go inside `begin ... end` or `let ... end`. This notebook prints no timings, because Pluto's compile times would mislead: run `run/compare_household.jl` and `run/compare_equilibrium.jl` for those.
"""

# ╔═╡ 6a3c5257-3c71-5c46-9d87-5170c687d3ce
md"""
The helper below pulls a named block out of the package source so it can be shown next to the maths. It is only for reading.
"""

# ╔═╡ b1130a05-969e-547b-b5cd-0a434b3e7985
function source(name)
    dir = dirname(pathof(EC991Week4))
    for f in readdir(dir; join = true)
        text = read(f, String)
        m = match(Regex("# >>> \\Q$name\\E\\n(.*?)# <<< \\Q$name\\E", "s"), text)
        m === nothing || return Markdown.parse("```julia\n" * m.captures[1] * "```")
    end
    error("no source block called $name")
end

# ╔═╡ 9c34ddbb-2c7f-5c9f-9791-702db2329575
md"""
## 1. Set up once (slide 28)

Before the interest rate loop we fix the income process, labour supply and the asset grids. `Economy` holds the choices and `setup` builds the objects. The defaults are the lab grid: 300 points on ``[0, 250]`` with ``\nu = 0.025``, and 7 income states.
"""

# ╔═╡ 6b92f067-a549-5910-9021-ce187b01469a
source("Economy")

# ╔═╡ b76d192b-02dc-57d4-a183-22a134cbce07
source("setup")

# ╔═╡ f1794833-eef0-5a7e-b3ce-2508d990af97
econ = Economy()

# ╔═╡ 3b9540b2-1481-5c59-84af-57f4123e1e8b
s = setup(econ)

# ╔═╡ 71c5d900-ee62-5a84-acee-e0c7aea36d39
hh = Household(r = 0.0377)

# ╔═╡ 2897fad5-a07f-5cd7-b0a4-c2686f780d80
md"""
We solve the household at ``r = 0.0377``, close to the equilibrium, until section 7. The wage comes from the firm through `w_from_r`, as in week 3.

## 2. Without interpolation: grid search and its speed-ups (slides 5 and 6)

Week 3's grid search tries every ``a'_k`` at every state. Two shortcuts make each sweep cheaper. If the policy is monotone, start at the previous state's best choice. If the objective is concave, stop at the first fall.
"""

# ╔═╡ 10847a73-92b8-58be-8b58-2ec667d53cb9
source("bellman_step_fast!")

# ╔═╡ f215b648-d879-5073-ad6c-635bcae9bdbc
md"""
Howard's improvement makes sweeps rarer: hold the policy fixed and update the value ``n_H`` times with no maximisation,

```math
W^{k+1}_{ij} \;=\; u\big(R a_i + w y_j - g_{ij}\big) + \beta \sum_m P_{jm}\, W^k\big(g_{ij}, y_m\big).
```
"""

# ╔═╡ f6b8f5b1-d55f-58fd-907f-5c70d231bb81
source("howard_update!")

# ╔═╡ bddb6210-bb5c-58c7-847d-e8d2db813e41
sol_grid = solve_household(hh, s; method = :vfi_grid)

# ╔═╡ dc06d3c1-10ac-55f6-b545-905f9fa5d059
sol_fast = solve_household(hh, s; method = :vfi_fast)

# ╔═╡ 4b6d3f88-874c-53ab-8ce6-5a024b0b3f31
md"""
The speed-ups do not change the answer here. The next cell gives the sweeps each method needed and the largest gap between the two policies, against the largest grid gap on the asset range people actually occupy.
"""

# ╔═╡ 6db86c1c-20af-5ef4-87cc-dbf265f9de7a
(sweeps_plain = sol_grid.iterations, sweeps_fast = sol_fast.iterations,
 gap = compare_policies(sol_fast, sol_grid; a_hi = 50.0).gap)

# ╔═╡ 8e788b11-f14b-5439-87bc-e5ce475793db
md"""
## 3. With interpolation (slides 26 and 32 of week 3)

Interpolating ``V`` lets the search choose ``a'`` between grid points. This is week 3's step. Howard's updates work here too, with one caution: the interpolated objective has kinks and Brent's method assumes one peak, so far from the solution the policy can jump between sweeps, and evaluating a poor policy many times can stop the iteration from settling. So `:vfi_interp_howard` waits until the distance has fallen below 0.01.
"""

# ╔═╡ 06bad61b-8a7a-5477-a796-b195b598a5a7
source("solve_vfi")

# ╔═╡ cd0164d0-590f-5278-91eb-8771811b2aed
sol_interp = solve_household(hh, s; method = :vfi_interp_howard)

# ╔═╡ 2ccf620e-2a02-5976-9857-931cf8107208
md"""
## 4. The endogenous grid method (slides 8 to 14)

**EGM does not search, and it is not a grid search. It does interpolate.** Read each grid point as tomorrow's assets ``a'_i``. Then, for every pair ``(a'_i, y_j)``:

```math
\text{RHS}(a'_i, y_j) = \beta R \sum_m P_{jm}\, u'\big(g^n_c(a'_i, y_m)\big), \qquad
\tilde c = u'^{-1}\big(\text{RHS}\big), \qquad
\hat a_i = \frac{\tilde c + a'_i - w y_j}{R}.
```

The pairs ``(\hat a_i, a'_i)`` describe the policy at the generated assets. Interpolation brings it back to the grid, the one place EGM needs it. No maximisation and no root finding.

| Code | Maths |
|---|---|
| `EMU = marginal_utility.(c0, γ) * P'` | ``\sum_m P_{jm}\, u'(g^n_c(a'_i, y_m))``, every ``i`` and ``j`` at once |
| `inverse_marginal_utility(β * R * EMU[i, j], γ)` | ``\tilde c = u'^{-1}(\text{RHS})`` |
| `endogenous_assets(c̃, a[i], w * y[j], R)` | ``\hat a_i = (\tilde c + a'_i - w y_j) / R`` |
| `ĝ = LinearInterp(â, a)` | the policy ``a'(\hat a)``, read at the grid points |
| `max(ĝ(a[i]), a[1])` | the borrowing constraint (slide 14) |
"""

# ╔═╡ a4e5f69c-ce29-5572-90cb-cf1e4d160dda
source("egm_step!")

# ╔═╡ 3f264f3f-92a0-5531-83ff-12fd2cbcc235
md"""
**The worked number on slide 10.** With ``\tilde c = 1.5``, ``a' = 2``, ``w y_3 = 1`` and ``R = 1.02``:
"""

# ╔═╡ 5182999a-4d17-53d6-8eef-9eab12822a1f
endogenous_assets(1.5, 2.0, 1.0, 1.02)

# ╔═╡ 114d3339-8766-5e71-bae8-e2db4749eeaa
sol_egm = solve_household(hh, s; method = :egm)

# ╔═╡ 3ca555e0-51a8-573c-9349-441a8623782f
md"""
Grid search, interpolation and EGM at the lowest income, near the limit. Grid search moves in steps; the other two lie on top of each other.
"""

# ╔═╡ a3d6a6d6-0d61-588a-ad8c-0bcff0c2bf1f
let keep = s.a .<= 2
    p = plot(xlabel = "assets, a", ylabel = "a'(a, y_1)", legend = :topleft)
    plot!(p, s.a[keep], sol_grid.apol[keep, 1]; seriestype = :steppost, color = :gray40, label = "grid search")
    plot!(p, s.a[keep], sol_interp.apol[keep, 1]; color = :steelblue, label = "interpolation")
    plot!(p, s.a[keep], sol_egm.apol[keep, 1]; color = :darkorange, linestyle = :dash, label = "EGM")
end

# ╔═╡ 148bf57f-f005-5a4a-84e7-039fa2fd47de
md"""
## 5. How accurate? Euler equation errors (slides 18 and 19)

```math
\varepsilon \;=\; 1 - \frac{u'^{-1}\big(\beta R\, \mathbb{E}[u'(c')]\big)}{c}.
```

Check it against slide 19's example: consumption 2 today, five equally likely values tomorrow, log utility, ``\beta = 0.95``, ``R = 1``.
"""

# ╔═╡ 163f7ddb-a9f1-594f-acf6-b1cec4d689bc
euler_error(2.0, [1.5, 1.8, 2.0, 2.2, 2.5], fill(0.2, 5), 0.95, 1.0, 1.0)

# ╔═╡ f00b7e6d-4c6d-51d0-aad7-97428b9c9a2d
md"""
Now each method, on a finer grid than it was solved on and only where the constraint is slack. The values are ``\log_{10} |\varepsilon|``, so ``-3`` is an error of one part in a thousand of consumption.
"""

# ╔═╡ 6da4e03d-3a90-5f66-b7a4-dbb70ed9e713
Markdown.parse(join(vcat(["| method | mean ``\\log_{10}|\\varepsilon|`` | worst |", "|---|---|---|"],
    [let e = euler_errors(so, hh, s.P)
        @sprintf("| %s | %.2f | %.2f |", name, e.mean_log10, e.max_log10)
    end for (name, so) in (("grid search", sol_grid), ("interpolation", sol_interp), ("EGM", sol_egm))]), "\n"))

# ╔═╡ 800d0d7a-8793-51c3-a9b6-dc264f5cfe17
md"""
## 6. The joint distribution of wealth and income (slides 20 to 26)

**The lottery (slide 23).** A choice ``a'`` between two grid points splits the mass so that the mean is preserved. Slide 25's example: ``a' = 1.3`` between nodes 1 and 2 sends 0.7 down and 0.3 up.
"""

# ╔═╡ 200af993-5b91-56a5-b2d3-ba75e911faf6
source("young_weights")

# ╔═╡ daaceee0-cdba-59fe-9819-1d5287d03c96
young_weights(1.3, [0.0, 1.0, 2.0, 3.0, 4.0])

# ╔═╡ 0d363a1c-37ff-565f-9fb2-6b5610e3e045
md"""
Multiply by ``P`` and we have ``Q``, the transition function. `build_Q` writes it out for small grids, so you can see it is sparse. `stationary_distribution` does the same multiplication in two parts without storing it, and iterates ``\lambda_{t+1} = Q' \lambda_t``.
"""

# ╔═╡ b50c084e-eaf6-5993-8ad0-bf17b186c991
source("stationary_distribution")

# ╔═╡ 81fce46f-6d07-5653-8cf4-e408368e39b4
dist = stationary_distribution(policy_on_grid(sol_egm, s.a_dist), s.a_dist, s.P)

# ╔═╡ f5c83082-bf94-5627-aedc-233c97ff820e
md"""
`dist.λ` is the joint distribution: entry ``[i, j]`` is the share of households with assets ``a_i`` and income ``y_j``. Grouped into wealth bins:
"""

# ╔═╡ 90a4e10e-949a-534c-b3d9-69ec49fec02b
edges = [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, Inf]

# ╔═╡ eec59662-b72f-5c60-a6fb-88072638406a
let T = joint_table(dist.λ, s.a_dist, edges)
    head = "| assets | " * join((@sprintf("%.2f", v) for v in s.y), " | ") * " |"
    rows = [@sprintf("| %g to %g | ", edges[b], edges[b+1]) * join((@sprintf("%.2f", 100v) for v in T[b, :]), " | ") * " |" for b in axes(T, 1)]
    Markdown.parse(join(vcat(head, "|" * "---|"^(length(s.y) + 1), rows), "\n"))
end

# ╔═╡ 28cccb72-a7f7-55cc-8df5-74d46e5264f7
md"""
**The same thing by Monte Carlo.** Method 1 on slide 20 follows households through time under the same policy. The final cross-section should agree with the histogram, up to sampling noise. It takes far longer and is noisier, which is why we use the histogram. This cell uses a small panel so the notebook stays quick.
"""

# ╔═╡ 2e2d20a3-5e1a-5380-be7c-10c794e5e490
source("simulate_panel")

# ╔═╡ dd97c28c-22a1-59df-8bb4-67066ab65720
panel = simulate_panel(policy_on_grid(sol_egm, s.a_dist), s.a_dist, s.y, s.P; n_households = 10_000, T = 500)

# ╔═╡ 95c4406d-2a4c-5a43-82de-33892dc8a3c9
(mean_histogram = sum(s.a_dist .* sum(dist.λ; dims = 2)), mean_monte_carlo = sum(panel.a) / length(panel.a),
 largest_gap_in_table = maximum(abs.(joint_table(dist.λ, s.a_dist, edges) .- panel_table(panel, edges, length(s.y)))))

# ╔═╡ bc805e9a-0347-56a2-9837-230999c6725c
md"""
**A problem only grid search has.** When ``a'`` must lie on the grid, a wide gap makes every income state round to the same point. At those assets ``a' = a`` for everyone: a household that arrives never leaves. These *absorbing states* mean the long-run distribution depends on where it started, so we start everyone at the borrowing limit. EGM and interpolation have none.
"""

# ╔═╡ fb585e89-c6a1-5e3e-b379-efb647392281
source("absorbing_states")

# ╔═╡ ca5eace0-e3fb-5d46-9c51-124156318c76
(grid_search = length(absorbing_states(sol_grid)), first_at_a = sol_grid.a[first(absorbing_states(sol_grid))],
 egm = length(absorbing_states(sol_egm)))

# ╔═╡ 5b35101d-35cb-5434-af78-e81076f33db6
md"""
## 7. The equilibrium (slides 27 to 30)

Steps 2 to 5 give excess demand at one interest rate, ``\Phi(r) = \mathbb{E}a(r) - K(r)``. `solve_household` is the only line that knows which method you chose.
"""

# ╔═╡ 37a532fd-1074-519f-827d-fc60d2389b71
source("solve_household")

# ╔═╡ fac5bffd-66cb-513c-8443-1b2a9dd24557
source("excess_demand")

# ╔═╡ 86dd4011-34b2-5e12-b456-b527b9009a50
md"""
Step 6 updates ``r`` by bisection: if ``\Phi(r) > 0`` households supply more assets than firms demand, so ``r`` falls.
"""

# ╔═╡ 1da4936b-a239-5ea4-b9c7-9fdc59d01099
source("solve_equilibrium")

# ╔═╡ 6e0c81af-2754-59de-b3cb-c3b821809966
eq = solve_equilibrium(econ; method = :egm)

# ╔═╡ 93bec9b4-3312-531f-88d5-e1ee685b2294
md"""
The bracket shrinks on the crossing of asset supply and capital demand:
"""

# ╔═╡ 1e6d6beb-0468-5857-ab33-332c59d27884
let
    rs, Φs = first.(eq.history), last.(eq.history)
    plot(1:length(rs), rs; marker = :circle, xlabel = "step", ylabel = "r tried", label = "", title = "bisection on r")
end

# ╔═╡ 1e3fe368-cc1b-5126-a4b1-4830fe869f21
md"""
## 8. What the model delivers (slides 15, 16, 26, 32 and 33)

These are the lecture's results on the lab grid. The slides use the dense grid, `dense_economy()`, with the same code.
"""

# ╔═╡ 3a04f6f6-75d5-5456-a158-9c12d1896c91
let
    p1 = plot(xlabel = "a", ylabel = "c", title = "consumption", legend = :bottomright)
    p2 = plot(xlabel = "a", ylabel = "a'", title = "saving", legend = false)
    keep = eq.s.a .<= 20
    for j in eachindex(eq.s.y)
        plot!(p1, eq.s.a[keep], eq.sol.cpol[keep, j]; label = @sprintf("y = %.2f", eq.s.y[j]))
        plot!(p2, eq.s.a[keep], eq.sol.apol[keep, j])
    end
    plot(p1, p2; layout = (1, 2), size = (850, 380))
end

# ╔═╡ ec64f950-e7e0-56d4-9861-4f528ea35d1e
stats = model_stats(eq)

# ╔═╡ bfd7523a-cd83-5eed-92ba-775910818c96
md"""
**The MPC (slide 33).** Extra assets of ``\Delta a`` are ``R \Delta a`` of extra resources, so the MPC is ``\Delta c / (R \Delta a)``. It is one where the household spends any windfall.
"""

# ╔═╡ 1e3e934c-63fe-5219-a482-e3f7e00c017e
source("mpc")

# ╔═╡ aabeb516-1dbf-55d6-a74e-b8ca0b9d8908
let M = mpc(eq.sol, eq.hh), keep = eq.s.a[1:end-1] .<= 50
    p = plot(xlabel = "assets, a", ylabel = "MPC", legend = :topright)
    plot!(p, eq.s.a[1:end-1][keep], M[keep, 1]; label = "lowest income")
    plot!(p, eq.s.a[1:end-1][keep], M[keep, end]; label = "highest income")
end

# ╔═╡ b56a3a50-628b-57b4-bfa8-0db31cd77f68
md"""
## 9. Exercises

1. **Howard's trade-off.** Solve `solve_vfi(hh, s.a, s.y, s.P; step = bellman_step_fast!, n_H = n_H)` for `n_H` in 0, 5, 50 and 500, and time each in the REPL with `@time`. Where does the gain stop?
2. **Refine the grid.** Set `n_a = 600` and `ν = 0.0125` in `Economy`. How far do ``r^*`` and the Euler errors move under EGM? Under `:vfi_fast`? Which method's answer can you trust at 300 points?
3. **Absorbing states.** Count them under grid search as you refine the grid. Where does the first one sit, and why does it move?
4. **Loosen the borrowing limit.** Solve the equilibrium with `ϕ = 1.0`. Does ``r^*`` rise, as the lecture predicts? Keep ``a_1 = -\phi`` in mind when you read the statistics.
5. **More risk.** Set `σ = 0.2`. What happens to ``r^*``, the Gini and the top shares? Check that the upper bound of the grid still does not bind.
6. **Race your own variant.** Add `:vfi_interp` with cubic splines to `solve_household`, then run `run/compare_equilibrium.jl`.
"""

# ╔═╡ Cell order:
# ╠═5ec47728-55ce-551a-85bc-10d0787ad4b1
# ╟─6da29021-3033-5200-bb66-9552b655a71e
# ╟─6a3c5257-3c71-5c46-9d87-5170c687d3ce
# ╠═b1130a05-969e-547b-b5cd-0a434b3e7985
# ╟─9c34ddbb-2c7f-5c9f-9791-702db2329575
# ╠═6b92f067-a549-5910-9021-ce187b01469a
# ╠═b76d192b-02dc-57d4-a183-22a134cbce07
# ╠═f1794833-eef0-5a7e-b3ce-2508d990af97
# ╠═3b9540b2-1481-5c59-84af-57f4123e1e8b
# ╠═71c5d900-ee62-5a84-acee-e0c7aea36d39
# ╟─2897fad5-a07f-5cd7-b0a4-c2686f780d80
# ╠═10847a73-92b8-58be-8b58-2ec667d53cb9
# ╟─f215b648-d879-5073-ad6c-635bcae9bdbc
# ╠═f6b8f5b1-d55f-58fd-907f-5c70d231bb81
# ╠═bddb6210-bb5c-58c7-847d-e8d2db813e41
# ╠═dc06d3c1-10ac-55f6-b545-905f9fa5d059
# ╟─4b6d3f88-874c-53ab-8ce6-5a024b0b3f31
# ╠═6db86c1c-20af-5ef4-87cc-dbf265f9de7a
# ╟─8e788b11-f14b-5439-87bc-e5ce475793db
# ╠═06bad61b-8a7a-5477-a796-b195b598a5a7
# ╠═cd0164d0-590f-5278-91eb-8771811b2aed
# ╟─2ccf620e-2a02-5976-9857-931cf8107208
# ╠═a4e5f69c-ce29-5572-90cb-cf1e4d160dda
# ╟─3f264f3f-92a0-5531-83ff-12fd2cbcc235
# ╠═5182999a-4d17-53d6-8eef-9eab12822a1f
# ╠═114d3339-8766-5e71-bae8-e2db4749eeaa
# ╟─3ca555e0-51a8-573c-9349-441a8623782f
# ╠═a3d6a6d6-0d61-588a-ad8c-0bcff0c2bf1f
# ╟─148bf57f-f005-5a4a-84e7-039fa2fd47de
# ╠═163f7ddb-a9f1-594f-acf6-b1cec4d689bc
# ╟─f00b7e6d-4c6d-51d0-aad7-97428b9c9a2d
# ╠═6da4e03d-3a90-5f66-b7a4-dbb70ed9e713
# ╟─800d0d7a-8793-51c3-a9b6-dc264f5cfe17
# ╠═200af993-5b91-56a5-b2d3-ba75e911faf6
# ╠═daaceee0-cdba-59fe-9819-1d5287d03c96
# ╟─0d363a1c-37ff-565f-9fb2-6b5610e3e045
# ╠═b50c084e-eaf6-5993-8ad0-bf17b186c991
# ╠═81fce46f-6d07-5653-8cf4-e408368e39b4
# ╟─f5c83082-bf94-5627-aedc-233c97ff820e
# ╠═90a4e10e-949a-534c-b3d9-69ec49fec02b
# ╠═eec59662-b72f-5c60-a6fb-88072638406a
# ╟─28cccb72-a7f7-55cc-8df5-74d46e5264f7
# ╠═2e2d20a3-5e1a-5380-be7c-10c794e5e490
# ╠═dd97c28c-22a1-59df-8bb4-67066ab65720
# ╠═95c4406d-2a4c-5a43-82de-33892dc8a3c9
# ╟─bc805e9a-0347-56a2-9837-230999c6725c
# ╠═fb585e89-c6a1-5e3e-b379-efb647392281
# ╠═ca5eace0-e3fb-5d46-9c51-124156318c76
# ╟─5b35101d-35cb-5434-af78-e81076f33db6
# ╠═37a532fd-1074-519f-827d-fc60d2389b71
# ╠═fac5bffd-66cb-513c-8443-1b2a9dd24557
# ╟─86dd4011-34b2-5e12-b456-b527b9009a50
# ╠═1da4936b-a239-5ea4-b9c7-9fdc59d01099
# ╠═6e0c81af-2754-59de-b3cb-c3b821809966
# ╟─93bec9b4-3312-531f-88d5-e1ee685b2294
# ╠═1e6d6beb-0468-5857-ab33-332c59d27884
# ╟─1e3fe368-cc1b-5126-a4b1-4830fe869f21
# ╠═3a04f6f6-75d5-5456-a158-9c12d1896c91
# ╠═ec64f950-e7e0-56d4-9861-4f528ea35d1e
# ╟─bfd7523a-cd83-5eed-92ba-775910818c96
# ╠═1e3e934c-63fe-5219-a482-e3f7e00c017e
# ╠═aabeb516-1dbf-55d6-a74e-b8ca0b9d8908
# ╟─b56a3a50-628b-57b4-bfa8-0db31cd77f68
