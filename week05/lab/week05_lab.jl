### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 07e4bed0-3fcc-5daf-bd7b-9fe9208e080a
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "EC991Week5"))
    Pkg.instantiate()
    using EC991Week5, Plots, Printf, Random
    using Markdown: Markdown, @md_str
end

# ╔═╡ 1eb8f064-25b1-5264-a992-8d64ba848555
md"""
# EC991 Week 5 lab: the Hopenhayn model

**Aseem Patel, University of Essex, Autumn 2026**

This notebook follows the week 5 lecture, *Firm Dynamics: the Hopenhayn Model*, and the tutorial slides, *Solving Hopenhayn*, and runs the code that does what they describe. The code lives in the package `EC991Week5`, in the folder next to this notebook. Every section puts the maths first and the code underneath. Slide numbers are the lecture's unless marked as tutorial slides.

**The one idea.** The equilibrium comes in two blocks. Free entry alone pins down the price ``p``, without knowing how many firms there are. Then the goods market gives the mass of entrants ``M``, in one line, because the measure of firms is proportional to ``M``.

| Block | Step | Function |
|---|---|---|
| set-up | grid ``z``, transition matrix ``F``, entrants' probabilities ``g`` | `Industry`, `setup` |
| 1. the price | Step 1: profits at ``p`` | `static_decisions` |
| | Step 2: the incumbent's Bellman equation | `solve_firm` |
| | Step 3: the value of entry ``V^e(p)`` | `entry_value` |
| | Step 4: update ``p`` until ``V^e(p) = 0`` | `solve_price` |
| 2. the firms | Step 5: the measure for one unit of entrants, ``\bar\mu`` | `stationary_measure` |
| | Step 6: the mass of entrants ``M`` | `mass_of_entrants` |

**Open this file in Pluto**, not as a script. Cells are reactive, each name is defined in exactly one cell, and several lines in one cell go inside `begin ... end` or `let ... end`. This notebook prints no timings, because Pluto's compile times would mislead: run `run/compare_methods.jl` for those.

The notation is the slides': productivity ``z``, curvature ``\eta``, transition matrix ``F`` with ``F_{ij} = \Pr(z' = z_j \mid z = z_i)``, entrants' probabilities ``g``, exit rule ``\chi`` (1 means exit), measure of firms ``\mu``, total mass of firms ``B``, and the wage as numeraire, ``w = 1``.
"""

# ╔═╡ dced05db-9918-5092-ab5b-706a487a0aa2
md"""
The helper below pulls a named block out of the package source so it can be shown next to the maths. It is only for reading.
"""

# ╔═╡ 37b009c3-0957-5b44-acc1-18f0ae3810d8
function source(name)
    dir = dirname(pathof(EC991Week5))
    for f in readdir(dir; join = true)
        text = read(f, String)
        m = match(Regex("# >>> \\Q$name\\E\\n(.*?)# <<< \\Q$name\\E", "s"), text)
        m === nothing || return Markdown.parse("```julia\n" * m.captures[1] * "```")
    end
    error("no source block called $name")
end

# ╔═╡ 63c1a185-0814-50c0-961a-45da097768ca
md"""
## 1. The static economy (slides 9 to 22)

A fixed set of firms with ``y_i = z_i n_i^{\eta}``. Output is highest when every firm has the same marginal product of labour, so more productive firms hire more:

```math
n_i \;\propto\; z_i^{\frac{1}{1-\eta}} .
```
"""

# ╔═╡ 672b23c4-94d5-57a1-a02f-8d7116e88af4
source("efficient_allocation")

# ╔═╡ 496f92ab-86ea-53b4-8fd4-b20f63a9233d
md"""
**The two-firm example on slide 12.** Firms with ``z = (1, 2)``, ``\eta = 2/3`` and ``N = 9`` workers. Firm 2 should hire ``2^3 = 8`` times as many workers as firm 1, so ``n = (1, 8)`` and ``Y = 9``.
"""

# ╔═╡ 9a11e5e2-cd02-5902-99f1-1b4f161e62e1
two_firms = efficient_allocation([1.0, 2.0], 9.0; η = 2 / 3)

# ╔═╡ 603162bd-4d6b-5df4-9d86-b57f1af02e84
md"""
Aggregate output has the same form as each firm's technology, ``Y = Z N^{\eta}``. Aggregate TFP ``Z`` is a power mean of firm productivity, so productive firms count for more than their number (slide 11):

```math
Z \;=\; \Big( \sum_i z_i^{\frac{1}{1-\eta}} \Big)^{1-\eta}.
```

Here ``Z = 9^{1/3}`` and ``Z N^{\eta} = 9``, the same output as above.
"""

# ╔═╡ 374fda3d-54e2-507b-935f-689cbaa0766c
let Z = aggregate_tfp([1.0, 2.0]; η = 2 / 3)
    (Z = Z, Z_times_N_to_the_η = Z * 9.0^(2 / 3))
end

# ╔═╡ 9829bf0e-4d25-5c0a-ba6e-1afc08a60000
md"""
**Wedges (slides 14 to 16).** A firm with wedge ``\tau_i`` hires as if its productivity were ``(1 + \tau_i) z_i``, but produces with ``z_i``. Unequal wedges move workers to the wrong firms, and output falls to ``Y = Z \Omega N^{\eta}`` with ``\Omega \le 1``. With ``x_i = z_i^{1/(1-\eta)}``:

```math
\Omega \;=\; \frac{\sum_i x_i (1+\tau_i)^{\frac{\eta}{1-\eta}}}{\big(\sum_i x_i (1+\tau_i)^{\frac{1}{1-\eta}}\big)^{\eta}\big(\sum_i x_i\big)^{1-\eta}} .
```
"""

# ╔═╡ fe7c75fb-d2e0-5815-9fd5-4bc548842df1
source("misallocation")

# ╔═╡ 31f98b40-0788-57c5-9777-0c21dcf51853
md"""
Give firm 2 a tax that exactly offsets its higher productivity, ``\tau_2 = -1/2``. Both firms then hire as if ``z = 1``, the workers split equally, and output is 9.1% lower: ``\Omega \approx 0.909``.
"""

# ╔═╡ 1cfec12b-925f-5516-acd9-ec7ff42ddb5e
let z = [1.0, 2.0], τ = [0.0, -0.5]
    w = allocation_with_wedges(z, τ, 9.0; η = 2 / 3)
    (n = w.n, Y = w.Y, Ω = misallocation(z, τ; η = 2 / 3), Y_over_efficient = w.Y / two_firms.Y)
end

# ╔═╡ 9ae96fc2-c15a-5932-8bab-7b399e0ad58b
md"""
**Entry (slides 20 to 22).** Each new firm costs ``c_e`` workers. The planner puts the share ``1 - \eta`` of workers into creating firms and the share ``\eta`` into production:

```math
N_p = \eta N, \qquad M = \frac{(1-\eta) N}{c_e}, \qquad Y = Z\, \eta^{\eta} (1-\eta)^{1-\eta} c_e^{-(1-\eta)} N .
```

Doubling ``c_e`` halves the number of firms and lowers output per worker by about 21%, with firms' technologies unchanged.
"""

# ╔═╡ 518a19b4-815a-5d4d-b84d-3dcab0c5bb42
let rows = [planner_entry(100.0, c_e; η = 2 / 3) for c_e in (1.0, 2.0)]
    (firms = [r.M for r in rows], output_per_worker = [r.Y / 100 for r in rows],
     fall = 1 - rows[2].Y / rows[1].Y)
end

# ╔═╡ 692911a3-c4ad-557c-a782-e3621971c184
md"""
## 2. The dynamic model: set up once (slides 34 and 41)

From here on productivity changes over time. Log productivity is an AR(1) around a mean of 1,

```math
\log z' - 1 \;=\; \rho\, (\log z - 1) + \sigma \varepsilon', \qquad \varepsilon' \sim N(0, 1),
```

discretised by Tauchen, as in week 3. `Industry` holds the parameters and `setup` builds the grid ``z``, the transition matrix ``F`` and the entrants' probabilities ``g``. The defaults are ``\eta = 2/3``, ``\beta = 0.8``, ``c_e = 40``, ``c_f = 20``, demand ``D(p) = \bar D / p`` with ``\bar D = 100``, ``\rho = 0.9`` and ``\sigma = 0.2``, with 101 grid points over ``\pm 4`` standard deviations. Entrants draw from the stationary distribution of the incumbents' chain. Change any of them by name, as in `Industry(c_e = 60)`.
"""

# ╔═╡ 1d4d988e-a002-5547-9631-8775cfeacfb1
source("Industry")

# ╔═╡ d3b65022-17f0-50f2-8ba1-dd3451b7ae07
ind = Industry()

# ╔═╡ 8c3a25c2-de3e-58bf-9be4-a613f87090bc
s = setup(ind)

# ╔═╡ da801948-aa4d-5d14-99ba-288e39f004e7
md"""
## 3. Step 1: profits at one price (slide 26)

Labour is chosen period by period, so the static problem has a closed form. With ``w = 1``, the first-order condition ``\eta p z n^{\eta - 1} = 1`` gives

```math
n(z; p) = (\eta p z)^{\frac{1}{1-\eta}}, \qquad y(z; p) = z\, n^{\eta}, \qquad \pi(z; p) = p\, y - n - c_f .
```
"""

# ╔═╡ 91e1bb8a-256d-5e0d-9c4d-14031aaec73a
source("static_decisions")

# ╔═╡ 6f5f8bf9-6fbe-5900-b5cd-271cd02d5254
md"""
We start from a guess of the price, ``p^0 = 1.5``.
"""

# ╔═╡ 1d12c31c-b66c-555f-9e71-e0d7a78c5cf4
p0 = 1.5

# ╔═╡ 37603b90-b3ac-5fd7-8cba-116e63e2c2b2
dec = static_decisions(p0, s.z; η = ind.η, c_f = ind.c_f)

# ╔═╡ eafbf8ad-a0a6-56af-af69-109c6c7cd4b3
md"""
## 4. Step 2: stay or exit (slides 27, 28 and 35)

The only dynamic choice is whether to stay. Exit is worth zero, so the incumbent stays when the continuation value is positive:

```math
V(z_i) \;=\; \pi(z_i; p) + \beta \max\Big\{ \sum_j F_{ij} V(z_j),\; 0 \Big\},
\qquad
\chi(z_i) \;=\; \mathbf{1}\Big\{ \sum_j F_{ij} V(z_j) < 0 \Big\}.
```

| Code | Maths |
|---|---|
| `mul!(V1, F, V0)` | ``\sum_j F_{ij} V^n(z_j)`` for every ``i``: the continuation value |
| `max(V1, 0.0)` | stay or exit; exit is worth zero |
| `π + β * …` | profit today, plus the discounted best option |
"""

# ╔═╡ cc5084b2-6d11-5815-9eab-a3eb4ae0608a
source("bellman_step!")

# ╔═╡ a1d8d63c-c15d-5412-9b35-afcdf42564be
sol_vfi = solve_firm(dec.π, s.F, ind.β; method = :vfi)

# ╔═╡ beccf1d0-a402-5a8d-98db-60263acb1c26
md"""
**Howard and policy iteration (tutorial slides 5 and 6).** Hold the exit rule ``\chi`` fixed and the Bellman equation has no maximum left. It is linear in ``V``, so the value of following one rule forever solves a linear system exactly:

```math
\big[ I - \beta\, \mathrm{diag}(1 - \chi)\, F \big]\, V \;=\; \pi .
```

Howard applies the fixed-rule update ``n_H`` times between Bellman steps. Policy iteration solves the system, improves the rule, and repeats until the rule stops changing.
"""

# ╔═╡ 10b067c2-e6ba-51a2-b6ed-dbf4bc82d8ea
source("evaluate_policy")

# ╔═╡ 3c61c819-a91a-5e6b-b4ff-19d7cf90f92f
sol_howard = solve_firm(dec.π, s.F, ind.β; method = :howard, n_H = 20)

# ╔═╡ 02496a8a-f741-5b30-b1fc-b5a46ebaac3c
sol_pi = solve_firm(dec.π, s.F, ind.β; method = :policy_iteration)

# ╔═╡ bd0e1cf4-6094-52f5-9b99-a681416e5326
md"""
The three methods agree. Policy iteration needs the fewest rounds and VFI the most, but fewer rounds is not less time here: the maximum is only ``\max\{\cdot, 0\}``, so a fixed-rule update costs as much as a full Bellman step, and policy iteration pays for a linear solve every round.
"""

# ╔═╡ ce7ff881-db7a-5eeb-9352-8de61c2fe42e
(iterations = (vfi = sol_vfi.iterations, howard = sol_howard.iterations, policy_iteration = sol_pi.iterations),
 largest_gap_in_V = max(maximum(abs.(sol_howard.V .- sol_pi.V)), maximum(abs.(sol_vfi.V .- sol_pi.V))) / maximum(abs.(sol_pi.V)),
 same_exit_rule = sol_vfi.χ == sol_howard.χ == sol_pi.χ)

# ╔═╡ 1b4f3080-5fb4-5e5c-b0cf-153c59dc6387
md"""
**The exit threshold (slide 28).** Firms exit below ``\tilde z``. On the grid the exit rule switches between two neighbouring points; between them we interpolate the continuation value linearly in ``\log z`` and find where it is zero.
"""

# ╔═╡ 25b47c25-9781-5e2c-a73f-f5e02772014b
source("exit_threshold")

# ╔═╡ 25b7f4a3-ad3d-5d70-b4da-c33ddd0b4ee1
threshold = exit_threshold(sol_vfi, s.z)

# ╔═╡ b63605a8-a98e-5465-83ef-3abccf510cbd
md"""
Below ``\tilde z`` the firm produces this period and leaves, so ``V = \pi``. Just above ``\tilde z`` profit is still negative, but the firm stays: ``z`` is persistent, and staying keeps the option of better draws (tutorial slide 9).
"""

# ╔═╡ d9a4a0b8-b03f-54c8-83cc-49ca8a684a2e
let keep = s.z .<= 6
    p = plot(xlabel = "productivity, z", ylabel = "value", legend = :topleft)
    plot!(p, s.z[keep], sol_vfi.V[keep]; label = "V(z)", color = :steelblue)
    plot!(p, s.z[keep], dec.π[keep]; label = "π(z; p⁰)", color = :darkorange, linestyle = :dash)
    hline!(p, [0.0]; color = :gray50, label = "")
    vline!(p, [threshold.z̃]; color = :firebrick, linestyle = :dot, label = "exit threshold z̃")
end

# ╔═╡ 34a774cf-e26e-5929-92ff-24bc2e2d289d
md"""
## 5. Steps 3 and 4: the free entry price (slides 29 and 36)

An entrant pays ``c_e``, draws ``z`` from ``g`` and starts producing next period. Free entry with a positive mass of entrants needs the value of entry to be zero:

```math
V^e(p) \;=\; -c_e + \beta \sum_i g_i\, V(z_i; p) \;=\; 0 .
```
"""

# ╔═╡ 639ac76c-422f-578e-a554-9000aa005274
source("entry_value")

# ╔═╡ 2502abd5-e69d-5ffc-aa6f-f75a718eb504
md"""
At our guess ``p^0 = 1.5`` entry pays, so the price must fall:
"""

# ╔═╡ 6e7ba788-67f7-53d5-b09b-1731f8104ab6
entry_value(p0, ind, s).Ve

# ╔═╡ 3c95844d-6aa3-5882-93ab-b5a34119b337
md"""
A higher price raises profit at every ``z``, so ``V^e(p)`` is increasing and has one root. `solve_price` finds it by bisection, as week 4 found ``r``: if ``V^e(p) > 0`` entry is too attractive and ``p`` falls, otherwise it rises. Demand ``\bar D`` never enters.
"""

# ╔═╡ dc983ed4-9dff-5d2e-a19a-b1abdf4297a2
source("solve_price")

# ╔═╡ c1e7b414-2dfc-545f-a269-7be84cda9225
free_entry, price_history = solve_price(ind, s)

# ╔═╡ bb4dc8ba-65ab-5875-a54b-410984e09ad9
md"""
The curve ``V^e(p)`` and the prices bisection tried, which close in on the root (tutorial slide 10):
"""

# ╔═╡ 7c6901ea-7da9-5de0-8e7d-ef797fe9e2a3
let ps = range(1.3, 1.7; length = 41)
    p = plot(xlabel = "price, p", ylabel = "Vᵉ(p)", legend = :topleft)
    plot!(p, ps, [entry_value(q, ind, s).Ve for q in ps]; color = :steelblue, label = "Vᵉ(p)")
    tried = filter(h -> 1.3 <= h[1] <= 1.7, price_history)
    scatter!(p, first.(tried), last.(tried); color = :darkorange, label = "prices tried")
    hline!(p, [0.0]; color = :gray50, label = "")
end

# ╔═╡ cd7799fe-9396-53bf-8bf5-4f8202622ffa
md"""
## 6. Steps 5 and 6: how many firms (slides 30, 37 and 38)

Survivors move with ``F``, exiters drop out, and entrants arrive with ``g``. ``\hat F`` is ``F`` with the rows of exiting firms set to zero, ``\hat F_{ij} = (1 - \chi(z_i)) F_{ij}``, and the stationary measure solves a linear system:

```math
\mu \;=\; \hat F' \mu + M g \qquad \Longrightarrow \qquad \mu \;=\; M (I - \hat F')^{-1} g .
```

Two differences from week 4's ``\lambda``. Mass moves with the transpose ``\hat F'``, because ``F_{ij}`` is the move from ``z_i`` to ``z_j``. And ``\mu`` is a measure, not a probability: it does not sum to one, and doubling ``M`` doubles it.
"""

# ╔═╡ 063ede0e-a3d7-5a78-81ed-e8ba6b1de34f
source("stationary_measure")

# ╔═╡ b9b742db-daaa-5e00-b564-ee754512b20d
F̂ = exit_transition(s.F, Float64.(free_entry.sol.χ))

# ╔═╡ d2410b43-74f7-5099-969e-846adbc0137e
md"""
Step 5 computes the measure for one unit of entrants, ``\bar\mu = (I - \hat F')^{-1} g``. Iterating the law of motion from ``\mu = 0``, as in week 4, gives the same answer:
"""

# ╔═╡ b603ec3e-b46d-5ca7-82f9-7208a806c6ea
μ̄ = stationary_measure(F̂, s.g)

# ╔═╡ cb56aae9-5f53-576a-a7c4-91d38c6c22b7
maximum(abs.(stationary_measure(F̂, s.g; method = :iterate) .- μ̄))

# ╔═╡ a38348b1-2bc7-5756-94d0-928b15fe4a80
md"""
Step 6 clears the goods market, ``D(p) = M \sum_i y(z_i; p)\, \bar\mu_i``, which gives ``M`` directly:

```math
M \;=\; \frac{\bar D / p}{\sum_i y(z_i; p)\, \bar\mu_i}, \qquad \mu = M \bar\mu, \qquad B = M \sum_i \bar\mu_i .
```
"""

# ╔═╡ e0695e4e-6882-5bf4-8c69-97546492bb72
M = mass_of_entrants(free_entry.p, free_entry.dec.y, μ̄; D̄ = ind.D̄)

# ╔═╡ e58069f4-805b-59e6-be8c-e6fdab3b4d2f
md"""
`solve_equilibrium` does all six steps in one call. The price block never needs ``M`` or ``\mu``.
"""

# ╔═╡ 12554610-ce34-51c8-a402-9fca0d60e7b2
source("solve_equilibrium")

# ╔═╡ 2c88b047-b128-5041-8efd-569c560b8458
eq = solve_equilibrium(ind)

# ╔═╡ 39244395-1692-598a-a500-09580fee6187
(same_price = eq.p == free_entry.p, same_M = eq.M ≈ M, B = sum(eq.μ))

# ╔═╡ 03775b92-c81e-521e-9bc4-85130dbc810a
md"""
## 7. What the equilibrium delivers (tutorial slides 11 and 12, slide 45)

`industry_stats` gives the table on tutorial slide 12.
"""

# ╔═╡ b3580ff6-0b2a-55c5-a594-0bf56ead9efa
stats = industry_stats(eq)

# ╔═╡ f945909c-6d87-558c-b7df-17b4e2417141
md"""
Two identities to check. The entry rate ``M / B`` equals the exit rate, because entrants replace exiting firms one for one. And production labour is exactly ``\eta \bar D``, because firms pay the share ``\eta`` of revenue ``p Y = \bar D`` to production workers.
"""

# ╔═╡ e805747a-e115-519e-8d88-77e97d2a80ca
(entry_rate = stats.entry_rate, exit_rate = stats.exit_rate, N_prod = stats.N_prod, η_times_D̄ = ind.η * ind.D̄)

# ╔═╡ d119916f-e72e-5dc3-9254-c2422b31c8dc
md"""
**Firm size (tutorial slide 11).** Most firms are small, but most workers are in large firms. Size means production workers ``n(z)``.
"""

# ╔═╡ d886851f-287f-50bf-9179-5e2393345518
let edges = [0, 10, 20, 50, 100, 500, Inf], sd = size_distribution(eq, edges)
    rows = [@sprintf("| %g to %g | %.1f | %.1f |", edges[k], edges[k+1], 100sd.firms[k], 100sd.employment[k]) for k in eachindex(sd.firms)]
    Markdown.parse(join(vcat("| workers | % of firms | % of employment |", "|---|---|---|", rows), "\n"))
end

# ╔═╡ f78c6b06-4f40-573d-a00c-ecebfde7a2cb
md"""
**A cohort of entrants (slide 45).** Follow one cohort as it ages: ``(\hat F')^a g`` is the cohort at age ``a``. Survivors grow, but no firm grows on purpose: low draws leave, so the firms that remain are the productive ones.
"""

# ╔═╡ 91181f7b-a27d-59d6-8256-02b8ce67bb5c
ages = age_profile(eq, 20)

# ╔═╡ f365b683-3a59-5c28-a0a3-51acf529a3c0
let
    p1 = plot(ages.age, ages.survival; xlabel = "age", title = "share still producing", label = "", marker = :circle)
    p2 = plot(ages.age, ages.avg_size; xlabel = "age", title = "average workers", label = "", marker = :circle)
    p3 = plot(ages.age, ages.exit_rate; xlabel = "age", title = "exit rate", label = "", marker = :circle)
    plot(p1, p2, p3; layout = (1, 3), size = (900, 300))
end

# ╔═╡ b1674b24-2a75-520c-97d4-77f8d85a36a0
md"""
## 8. Exit on a grid: the lottery (tutorial slides 13 to 15)

On the grid the exit rule can switch only at a grid point. When ``c_e`` rises, ``\tilde z`` falls smoothly, but the computed exit rate stays put until ``\tilde z`` crosses a grid point, then jumps. The fix is to read each grid point as a cell of ``\log z``, within half a step of ``\log z_i``. Cells wholly below ``\tilde z`` exit, cells wholly above stay, and the cell that contains ``\tilde z`` exits in proportion to its part below ``\tilde z``:

```math
\text{exit share} \;=\; \frac{\log \tilde z - (\log z_i - d/2)}{d}, \qquad d = \log z_{i+1} - \log z_i .
```
"""

# ╔═╡ 7478ec18-bf03-5a1e-be0c-9b81ad1174f5
source("exit_lottery")

# ╔═╡ b6253997-d5ad-5f7f-b10b-001e7a83996b
let χ = exit_lottery(eq.sol, s.z), i = findfirst(x -> 0 < x < 1, χ)
    (cell = i, z_at_cell = s.z[i], z̃ = eq.z̃, exit_share = χ[i])
end

# ╔═╡ 6280b726-1866-5a6e-9424-3d702a53a917
md"""
The exit rate as ``c_e`` rises (tutorial slide 14): the binary rule on 101 points moves in steps, the lottery on the same grid is smooth and close to the binary rule on 1001 points. The price is the same with and without the lottery, because the lottery is used only for the measure.
"""

# ╔═╡ 3cfee0a4-8a02-5805-90cf-852c8683a41d
ce_values = 30.0:1.0:50.0

# ╔═╡ 2227db2f-f45e-51c2-9681-1a6332ef50bd
exit_rates = let rate(ind; kw...) = industry_stats(solve_equilibrium(ind; kw...)).exit_rate
    (grid = [rate(Industry(; c_e)) for c_e in ce_values],
     lottery = [rate(Industry(; c_e); exit = :lottery) for c_e in ce_values],
     fine = [rate(Industry(; c_e, n_z = 1001)) for c_e in ce_values])
end

# ╔═╡ 97432e7a-1463-5baa-9195-b1aca1b972a4
let p = plot(xlabel = "entry cost, cₑ", ylabel = "exit rate", legend = :topright)
    plot!(p, ce_values, exit_rates.grid; seriestype = :steppost, color = :gray40, label = "grid, n_z = 101")
    plot!(p, ce_values, exit_rates.lottery; color = :steelblue, label = "lottery, n_z = 101")
    plot!(p, ce_values, exit_rates.fine; color = :darkorange, linestyle = :dash, label = "grid, n_z = 1001")
end

# ╔═╡ a0c462a6-f1c5-5840-8f4d-c2e24f886aeb
md"""
## 9. Comparative statics (slides 42 and 43)

A higher entry cost raises the price, so incumbents survive at lower ``z``: the threshold falls, turnover falls and firms get older. A higher fixed cost also raises the price, but makes staying more costly, so the threshold rises and turnover rises. The price alone cannot tell the two apart; turnover can.
"""

# ╔═╡ 38838567-584f-523a-a3c9-07cc24bd34eb
let cases = [("baseline", Industry()), ("c_e = 60", Industry(c_e = 60.0)), ("c_f = 30", Industry(c_f = 30.0))]
    st = [industry_stats(solve_equilibrium(c)) for (_, c) in cases]
    rows = [("price p", :p, "%.4f"), ("exit threshold z̃", :z̃, "%.3f"), ("entry rate = exit rate", :entry_rate, "%.3f"),
            ("mean age", :mean_age, "%.1f"), ("mass of firms B", :firms, "%.3f"), ("average workers", :avg_size, "%.1f"),
            ("output Y", :Y, "%.2f")]
    lines = [string("| ", name, " | ", join((Printf.format(Printf.Format(fmt), getfield(x, key)) for x in st), " | "), " |")
             for (name, key, fmt) in rows]
    Markdown.parse(join(vcat("| | " * join(first.(cases), " | ") * " |", "|---|---|---|---|", lines), "\n"))
end

# ╔═╡ 408fb976-2bee-5150-82f0-1eac4247a827
md"""
## 10. A simulated panel of firms (tutorial slide 16)

The equilibrium gives masses, not firms. To get data shaped like a firm panel, follow individual firms through the same law of motion. Each year entrants arrive and draw ``z`` from ``g``, every firm produces, each exits with probability ``\chi(z)``, and survivors draw ``z'`` from their row of ``F``. With ``N M / B`` entrants a year the industry settles at about ``N`` firms.
"""

# ╔═╡ 2b6919c7-a480-5546-ad38-ba41611508e2
source("simulate_panel")

# ╔═╡ 1e5322c2-7b22-57f3-b4f6-cef3aed53e6c
panel = simulate_panel(eq; T = 10, N = 1000, rng = MersenneTwister(1234))

# ╔═╡ 6b37768f-852b-51bf-b42f-48854c35dd97
md"""
The panel's averages against the model's. The gaps are sampling noise and shrink as ``N`` grows.
"""

# ╔═╡ 1a0591ea-6545-513d-821a-1394420ad765
let k = length(panel.firm)
    (exit_rate = (panel = sum(panel.exit) / k, model = stats.exit_rate),
     entry_rate = (panel = sum(panel.age .== 0) / k, model = stats.entry_rate),
     average_workers = (panel = sum(panel.n) / k, model = stats.avg_size),
     mean_age = (panel = sum(panel.age) / k, model = stats.mean_age))
end

# ╔═╡ 4ccc4e20-cb2f-5771-a14a-56ee92b3cf86
md"""
## 11. Exercises

1. **Howard's trade-off.** Solve `solve_firm(dec.π, s.F, ind.β; method = :howard, n_H = n_H)` for `n_H` in 0, 5, 20 and 200, and time each in the REPL with `@time`. The iterations fall; does the time? In week 4 Howard paid off. Why not here?
2. **Demand only scales the industry.** Solve with `Industry(D̄ = 200)`. Which statistics change and which do not? Explain with the two blocks.
3. **Refine the grid.** Compare the exit rate on the grid and with `exit = :lottery` for `n_z` in 51, 101, 201 and 1001. Which settles first? Check against the backup slide of the tutorial.
4. **Entrants who start lower.** Use `Industry(entrants = :lognormal, μ_g = 0.8, σ_g = 0.2)`, so entrants draw ``\log z`` below the incumbents' mean. Redraw the age profile. Do entrants start smaller, and how do the exit rate and mean age change?
5. **More persistent productivity.** Set `ρ = 0.95`. What happens to the price, the exit rate and the share of employment in firms with 100 or more workers?
6. **Exit by age in the panel.** From `panel`, compute the exit rate at each age from 0 to 10 and plot it against `ages.exit_rate`. Why are the oldest ages the noisiest?
"""

# ╔═╡ Cell order:
# ╠═07e4bed0-3fcc-5daf-bd7b-9fe9208e080a
# ╟─1eb8f064-25b1-5264-a992-8d64ba848555
# ╟─dced05db-9918-5092-ab5b-706a487a0aa2
# ╠═37b009c3-0957-5b44-acc1-18f0ae3810d8
# ╟─63c1a185-0814-50c0-961a-45da097768ca
# ╠═672b23c4-94d5-57a1-a02f-8d7116e88af4
# ╟─496f92ab-86ea-53b4-8fd4-b20f63a9233d
# ╠═9a11e5e2-cd02-5902-99f1-1b4f161e62e1
# ╟─603162bd-4d6b-5df4-9d86-b57f1af02e84
# ╠═374fda3d-54e2-507b-935f-689cbaa0766c
# ╟─9829bf0e-4d25-5c0a-ba6e-1afc08a60000
# ╠═fe7c75fb-d2e0-5815-9fd5-4bc548842df1
# ╟─31f98b40-0788-57c5-9777-0c21dcf51853
# ╠═1cfec12b-925f-5516-acd9-ec7ff42ddb5e
# ╟─9ae96fc2-c15a-5932-8bab-7b399e0ad58b
# ╠═518a19b4-815a-5d4d-b84d-3dcab0c5bb42
# ╟─692911a3-c4ad-557c-a782-e3621971c184
# ╠═1d4d988e-a002-5547-9631-8775cfeacfb1
# ╠═d3b65022-17f0-50f2-8ba1-dd3451b7ae07
# ╠═8c3a25c2-de3e-58bf-9be4-a613f87090bc
# ╟─da801948-aa4d-5d14-99ba-288e39f004e7
# ╠═91e1bb8a-256d-5e0d-9c4d-14031aaec73a
# ╟─6f5f8bf9-6fbe-5900-b5cd-271cd02d5254
# ╠═1d12c31c-b66c-555f-9e71-e0d7a78c5cf4
# ╠═37603b90-b3ac-5fd7-8cba-116e63e2c2b2
# ╟─eafbf8ad-a0a6-56af-af69-109c6c7cd4b3
# ╠═cc5084b2-6d11-5815-9eab-a3eb4ae0608a
# ╠═a1d8d63c-c15d-5412-9b35-afcdf42564be
# ╟─beccf1d0-a402-5a8d-98db-60263acb1c26
# ╠═10b067c2-e6ba-51a2-b6ed-dbf4bc82d8ea
# ╠═3c61c819-a91a-5e6b-b4ff-19d7cf90f92f
# ╠═02496a8a-f741-5b30-b1fc-b5a46ebaac3c
# ╟─bd0e1cf4-6094-52f5-9b99-a681416e5326
# ╠═ce7ff881-db7a-5eeb-9352-8de61c2fe42e
# ╟─1b4f3080-5fb4-5e5c-b0cf-153c59dc6387
# ╠═25b47c25-9781-5e2c-a73f-f5e02772014b
# ╠═25b7f4a3-ad3d-5d70-b4da-c33ddd0b4ee1
# ╟─b63605a8-a98e-5465-83ef-3abccf510cbd
# ╠═d9a4a0b8-b03f-54c8-83cc-49ca8a684a2e
# ╟─34a774cf-e26e-5929-92ff-24bc2e2d289d
# ╠═639ac76c-422f-578e-a554-9000aa005274
# ╟─2502abd5-e69d-5ffc-aa6f-f75a718eb504
# ╠═6e7ba788-67f7-53d5-b09b-1731f8104ab6
# ╟─3c95844d-6aa3-5882-93ab-b5a34119b337
# ╠═dc983ed4-9dff-5d2e-a19a-b1abdf4297a2
# ╠═c1e7b414-2dfc-545f-a269-7be84cda9225
# ╟─bb4dc8ba-65ab-5875-a54b-410984e09ad9
# ╠═7c6901ea-7da9-5de0-8e7d-ef797fe9e2a3
# ╟─cd7799fe-9396-53bf-8bf5-4f8202622ffa
# ╠═063ede0e-a3d7-5a78-81ed-e8ba6b1de34f
# ╠═b9b742db-daaa-5e00-b564-ee754512b20d
# ╟─d2410b43-74f7-5099-969e-846adbc0137e
# ╠═b603ec3e-b46d-5ca7-82f9-7208a806c6ea
# ╠═cb56aae9-5f53-576a-a7c4-91d38c6c22b7
# ╟─a38348b1-2bc7-5756-94d0-928b15fe4a80
# ╠═e0695e4e-6882-5bf4-8c69-97546492bb72
# ╟─e58069f4-805b-59e6-be8c-e6fdab3b4d2f
# ╠═12554610-ce34-51c8-a402-9fca0d60e7b2
# ╠═2c88b047-b128-5041-8efd-569c560b8458
# ╠═39244395-1692-598a-a500-09580fee6187
# ╟─03775b92-c81e-521e-9bc4-85130dbc810a
# ╠═b3580ff6-0b2a-55c5-a594-0bf56ead9efa
# ╟─f945909c-6d87-558c-b7df-17b4e2417141
# ╠═e805747a-e115-519e-8d88-77e97d2a80ca
# ╟─d119916f-e72e-5dc3-9254-c2422b31c8dc
# ╠═d886851f-287f-50bf-9179-5e2393345518
# ╟─f78c6b06-4f40-573d-a00c-ecebfde7a2cb
# ╠═91181f7b-a27d-59d6-8256-02b8ce67bb5c
# ╠═f365b683-3a59-5c28-a0a3-51acf529a3c0
# ╟─b1674b24-2a75-520c-97d4-77f8d85a36a0
# ╠═7478ec18-bf03-5a1e-be0c-9b81ad1174f5
# ╠═b6253997-d5ad-5f7f-b10b-001e7a83996b
# ╟─6280b726-1866-5a6e-9424-3d702a53a917
# ╠═3cfee0a4-8a02-5805-90cf-852c8683a41d
# ╠═2227db2f-f45e-51c2-9681-1a6332ef50bd
# ╠═97432e7a-1463-5baa-9195-b1aca1b972a4
# ╟─a0c462a6-f1c5-5840-8f4d-c2e24f886aeb
# ╠═38838567-584f-523a-a3c9-07cc24bd34eb
# ╟─408fb976-2bee-5150-82f0-1eac4247a827
# ╠═2b6919c7-a480-5546-ad38-ba41611508e2
# ╠═1e5322c2-7b22-57f3-b4f6-cef3aed53e6c
# ╟─6b37768f-852b-51bf-b42f-48854c35dd97
# ╠═1a0591ea-6545-513d-821a-1394420ad765
# ╟─4ccc4e20-cb2f-5771-a14a-56ee92b3cf86
