### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ e799a78b-531e-5286-b9ea-7055eb31d4a8
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "EC991Week6"))
    Pkg.instantiate()
    using EC991Week6, Plots, Printf, LinearAlgebra
    using Markdown: Markdown, @md_str
    BLAS.set_num_threads(1)
end

# ╔═╡ 0b74341c-d495-536c-bee3-c0b715661c69
md"""
# EC991 Week 6 lab: firing costs

**Aseem Patel, University of Essex, Autumn 2026**

This notebook follows the week 6 tutorial slides, *Firing Costs: Hopenhayn and Rogerson*, and runs the code that does what they describe. The code lives in the package `EC991Week6`, in the folder next to this notebook. Every section puts the maths first and the code underneath. Slide numbers are the tutorial's.

**The one idea.** Firing a worker costs ``\tau``, so the employment a firm inherits, ``m``, is a state next to its productivity ``z``. Firms then keep their workers inside an **inaction region**, marginal products differ across firms, and labour is misallocated.

| Step (slide 6) | What | Function |
|---|---|---|
| set-up | grids for ``z`` and employment, transition matrix ``F``, entrants' probabilities ``g_i = G(z_i)`` | `Economy`, `setup` |
| 1 | the firm's problem over ``(z, m)`` at a trial price | `solve_firm` |
| 2 | the price, by bisection until the value of entry is zero | `entry_value`, `solve_price` |
| 3 | the distribution for one unit of entrants, then the mass of entrants | `stationary_distribution`, `solve_equilibrium` |

**Open this file in Pluto**, not as a script. Cells are reactive, each name is defined in exactly one cell, and several lines in one cell go inside `begin ... end` or `let ... end`. Every equilibrium is slow to solve, so the notebook takes a while when it opens.
"""

# ╔═╡ 832d1582-373d-5783-aa40-15335b88ea87
md"""
The helper below pulls a named block out of the package source so it can be shown next to the maths. It is only for reading.
"""

# ╔═╡ 6efc738b-8df8-5b9b-bd3d-b606c8d82e5c
function source(name)
    dir = dirname(pathof(EC991Week6))
    for f in readdir(dir; join = true)
        text = read(f, String)
        m = match(Regex("# >>> \\Q$name\\E\\n(.*?)# <<< \\Q$name\\E", "s"), text)
        m === nothing || return Markdown.parse("```julia\n" * m.captures[1] * "```")
    end
    error("no source block called $name")
end

# ╔═╡ c152910f-03e7-580c-9508-66f79f8e94ff
md"""
## 1. The firm's problem (slides 4 and 7)

A firm with productivity ``z`` that inherits ``m`` workers chooses employment ``n``, pays ``\tau`` for every worker it lets go, produces, and then decides whether to exit, which means firing all ``n`` workers:

```math
V(z, m) = \max_{n} \Big\{ \underbrace{p z n^{\eta} - n - p c_f + \beta \max\Big\{ -\tau n,\ \mathbb{E}\big[V(z_{t+1}, n) \mid z_t = z\big] \Big\}}_{h(z,\, n)} \;-\; \tau \max\{0, m - n\} \Big\}.
```

On the grid, inherited employment is ``m = n_k`` and a candidate choice is ``n = n_\ell``. First, ``h`` at every candidate. It does not depend on inherited employment, so it is computed once for each ``(z_i, n_\ell)``:
"""

# ╔═╡ f9177506-9ea8-574c-a140-ae475b0147e1
source("choice_value!")

# ╔═╡ 23754fa5-57cf-58c3-bf03-4ab88cd10fb1
md"""
Then the search. For every inherited ``n_k``, try **every** candidate ``n_\ell`` and keep the best: ``V[i, k] = \max_{\ell} \{ h[i, \ell] - \tau \max(0, n_k - n_\ell) \}``. Choices are restricted to the grid, and nothing is interpolated:
"""

# ╔═╡ 5d423646-82d3-5938-ba6e-9b5fa8051940
source("best_employment!")

# ╔═╡ a57d1bcc-7be7-5754-829e-019e20498784
md"""
**Check it by hand.** Four grid points, ``n = [0, 1, 2, 3]``, with ``h = [1, 6, 5, 2]`` and ``\tau = 2``. A firm that inherits 3 workers can keep them (worth 2), fire one (``5 - 2 = 3``), fire two (``6 - 4 = 2``) or fire all (``1 - 6 = -5``): it fires one. The values should be ``[6, 6, 5, 3]`` and the choices ``[1, 1, 2, 2]``.
"""

# ╔═╡ d047d4a2-79f4-5366-b81c-1ea39c369a66
let n = [0.0, 1.0, 2.0, 3.0], h = [1.0, 6.0, 5.0, 2.0], τ = 2.0
    V, pol = zeros(length(n)), zeros(Int, length(n))
    best_employment!(V, pol, h, n, τ)
    (; values = V, chosen = n[pol])
end

# ╔═╡ 21aa4366-7255-55ee-94b4-00cbb347abc4
md"""
## 2. The equilibrium (slides 6 and 12)

The default `Economy()` has ``\tau = 0.1``. Change any parameter by name, as in `Economy(τ = 1.0)`.
"""

# ╔═╡ 1ee81c03-546a-5337-b6a0-8739cfddb6b4
eq = solve_equilibrium(Economy())

# ╔═╡ a17ddfc9-6deb-58d5-8da3-12a04705d124
aggregates(eq)

# ╔═╡ 2463e3c2-fd89-5636-bfad-9157c7d4a512
md"""
Check the output:

- `Y` equals `C`: the goods market clears, by construction of ``M``.
- `Ns` equals `Nd`: the labour market clears too, by accounting (``\Pi = pY - N^d - T`` and ``pY = \bar D``).
- `entry_rate` equals `exit_rate`, and `job_creation` equals `job_destruction`: the industry is stationary.
"""

# ╔═╡ 735f9a89-6778-5cac-99df-7e101fe42365
md"""
## 3. The inaction region (slides 8 and 9)

The hiring target ``n_L(z)`` is what a firm that inherits no workers chooses; the firing target ``n_H(z)`` is what a firm that inherits the largest workforce on the grid chooses. In between, firms keep their workers, ``n = m``:
"""

# ╔═╡ 4a5e6d94-44dc-5be4-a9ac-9d4d7f8b5b0b
source("inaction_region")

# ╔═╡ cbda37d2-a4d9-5dc7-bde7-c36fa8f14640
md"""
**The policy at one productivity.** Choose the grid point nearest ``z = 6``, and plot chosen employment ``n`` against inherited employment ``m``: flat, the line ``n = m``, flat.
"""

# ╔═╡ 0264ae68-461b-584e-93bc-ff60a6c4ed21
let i = argmin(abs.(eq.s.z .- 6.0)), keep = 50 .<= eq.s.n .<= 600
    p = plot(xlabel = "inherited employment, m", ylabel = "chosen employment, n", legend = :topleft,
             title = @sprintf("z = %.2f, τ = %.1f", eq.s.z[i], eq.eco.τ))
    plot!(p, [50, 600], [50, 600]; color = :gray, label = "n = m")
    plot!(p, eq.s.n[keep], eq.s.n[eq.sol.pol[i, keep]]; linewidth = 2.5, label = "n = nᵈ(z, m)")
end

# ╔═╡ 57f76af2-7f96-5dbf-89d7-27cc025adf30
md"""
## 4. Your turn: the graphs on slides 13 and 14

`plot_inaction` draws the two targets against ``z``. The first graph uses `eq`, at ``\tau = 0.1``. The second uses `τ_high`: edit it, and the graph redraws.
"""

# ╔═╡ ba695637-ecb6-5b89-a80f-4342137d45a9
function plot_inaction(eq; zmax = 12, nmax = 1000)
    r = inaction_region(eq.sol, eq.s)
    keep = r.z .<= zmax
    p = plot(xlabel = "productivity, z", ylabel = "employment", legend = :topleft,
             xlims = (0, zmax), ylims = (0, nmax), title = "Inaction region, τ = $(eq.eco.τ)")
    plot!(p, r.z[keep], r.nL[keep]; fillrange = r.nH[keep], fillalpha = 0.15, linealpha = 0, label = "inaction: keep m")
    plot!(p, r.z[keep], r.nH[keep]; linewidth = 2.5, linestyle = :dash, label = "n_H(z): fire down to it")
    plot!(p, r.z[keep], r.nL[keep]; linewidth = 2.5, label = "n_L(z): hire up to it")
    return p
end

# ╔═╡ c399203a-8149-5893-a67c-785c0cd0b7ae
plot_inaction(eq)

# ╔═╡ 805f978f-9123-5f9f-9248-889e075cea2c
τ_high = 1.0

# ╔═╡ 8852d5ee-0db2-50bb-8dd6-d9845381c4a6
eq_high = solve_equilibrium(Economy(τ = τ_high))

# ╔═╡ c37390fe-3907-5810-a0fa-947b57a0b116
plot_inaction(eq_high)

# ╔═╡ 1f3b83d2-1161-59d2-bc27-e209bed35d2e
md"""
With ``\tau = 1`` both curves jump near ``z = 3``. That is where firms start to exit: a firm that will exit pays ``\tau`` for every worker it still has, so it hires less now and puts off firing until it exits. For a range of ``z``, firms that inherit few workers exit but large ones stay: the exit rule depends on ``m`` as well as ``z``.
"""

# ╔═╡ a5f55edc-aca8-594b-8002-6e9373bfd1b5
md"""
## 5. Misallocation (slides 16 and 17)

Two measures. The average **MPL gap**, ``|p \eta z n^{\eta-1} - 1| \times 100`` with ``n = n^d(z, m)``, is how far each firm's marginal product is from the real wage ``1/p``. **Output lost**, ``1 - \Omega``, compares output with what the same firms and workers would produce if the workers were allocated efficiently.
"""

# ╔═╡ f4ffeabf-0cc8-56e1-a412-5ad91c3bf46e
source("misallocation")

# ╔═╡ f1b842f7-55be-56f5-a313-399c438cdaec
let rows = [(τ, misallocation(e), aggregates(e)) for (τ, e) in ((0.1, eq), (τ_high, eq_high))]
    md_rows = join([@sprintf("| %.2f | %.4f | %.2f | %.3f | %.2f |", τ, a.p, m.mpl_gap, m.output_lost, 100m.share_inactive)
                    for (τ, m, a) in rows], "\n")
    Markdown.parse("| ``\\tau`` | price | MPL gap (%) | output lost (%) | firms with ``n = m`` (%) |\n|---|---|---|---|---|\n" * md_rows)
end

# ╔═╡ 00832fb2-2bfa-53b0-9b0f-72d196756580
md"""
**Grid error.** With ``\tau = 0`` the MPL gap should be zero, but firms must choose ``n`` on the grid. Double the grid and the gap halves; the gap with ``\tau > 0`` does not move.
"""

# ╔═╡ 9c05a806-1cac-5146-b485-3f6757b1be18
[(n_n, round(misallocation(solve_equilibrium(Economy(τ = 0.0, n_n = n_n))).mpl_gap; digits = 3)) for n_n in (250, 500)]

# ╔═╡ 1c6868b5-a0c2-5c54-a878-5320ae024524
md"""
## 6. Try

1. Lower the persistence of productivity: `Economy(ρ = 0.6)`. Does the inaction region widen or narrow? Why might it?
2. Make the tax a cost of **hiring** instead of firing. Which lines of `best_employment!` change?
3. Set `τ_high = 0.5`. Is output lost at ``\tau = 0.5`` closer to its value at ``\tau = 0.1`` or at ``\tau = 1``? What does that say about losses from wedges?
4. Run `run/firing_taxes.jl` from the package folder. Output per worker falls by more than ``1 - \Omega``: where does the rest come from?
"""

# ╔═╡ Cell order:
# ╠═e799a78b-531e-5286-b9ea-7055eb31d4a8
# ╟─0b74341c-d495-536c-bee3-c0b715661c69
# ╟─832d1582-373d-5783-aa40-15335b88ea87
# ╠═6efc738b-8df8-5b9b-bd3d-b606c8d82e5c
# ╟─c152910f-03e7-580c-9508-66f79f8e94ff
# ╠═f9177506-9ea8-574c-a140-ae475b0147e1
# ╟─23754fa5-57cf-58c3-bf03-4ab88cd10fb1
# ╠═5d423646-82d3-5938-ba6e-9b5fa8051940
# ╟─a57d1bcc-7be7-5754-829e-019e20498784
# ╠═d047d4a2-79f4-5366-b81c-1ea39c369a66
# ╟─21aa4366-7255-55ee-94b4-00cbb347abc4
# ╠═1ee81c03-546a-5337-b6a0-8739cfddb6b4
# ╠═a17ddfc9-6deb-58d5-8da3-12a04705d124
# ╟─2463e3c2-fd89-5636-bfad-9157c7d4a512
# ╟─735f9a89-6778-5cac-99df-7e101fe42365
# ╠═4a5e6d94-44dc-5be4-a9ac-9d4d7f8b5b0b
# ╟─cbda37d2-a4d9-5dc7-bde7-c36fa8f14640
# ╠═0264ae68-461b-584e-93bc-ff60a6c4ed21
# ╟─57f76af2-7f96-5dbf-89d7-27cc025adf30
# ╠═ba695637-ecb6-5b89-a80f-4342137d45a9
# ╠═c399203a-8149-5893-a67c-785c0cd0b7ae
# ╠═805f978f-9123-5f9f-9248-889e075cea2c
# ╠═8852d5ee-0db2-50bb-8dd6-d9845381c4a6
# ╠═c37390fe-3907-5810-a0fa-947b57a0b116
# ╟─1f3b83d2-1161-59d2-bc27-e209bed35d2e
# ╟─a5f55edc-aca8-594b-8002-6e9373bfd1b5
# ╠═f4ffeabf-0cc8-56e1-a412-5ad91c3bf46e
# ╠═f1b842f7-55be-56f5-a313-399c438cdaec
# ╟─00832fb2-2bfa-53b0-9b0f-72d196756580
# ╠═9c05a806-1cac-5146-b485-3f6757b1be18
# ╟─1c6868b5-a0c2-5c54-a878-5320ae024524
