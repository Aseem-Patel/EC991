# EC991 Week 1 smoke test.
# Run from a terminal:   julia week01_check.jl
# Or in VS Code: open this file and press Ctrl+F5 (Cmd+F5 on a Mac).
# It uses only Julia's standard library, so it runs in any environment.
# Every line should end in PASS. If one says FAIL, read the hint next to it.

using LinearAlgebra, Statistics, Printf, Pkg

results = Tuple{String,Bool,String}[]
check(name, ok, hint = "") = push!(results, (name, ok, hint))

# 1. Julia version
check("Julia 1.10 or newer", VERSION >= v"1.10", "Run `juliaup update`, then `juliaup default release`.")

# 2. Pluto is installed in the default environment
has_pluto = haskey(Pkg.project().dependencies, "Pluto") ||
            any(d -> d.name == "Pluto", values(Pkg.dependencies()))
check("Pluto installed", has_pluto, "In the REPL press ] and type: add Pluto")

# 3. Arrays, broadcasting and comprehensions behave as expected
a = range(0.0, 10.0, length = 5)           # a grid, like the asset grid in week 2
u(c; γ = 2.0) = c^(1 - γ) / (1 - γ)        # CRRA utility
vals = u.(a .+ 1.0)                        # broadcast over the grid
check("Broadcasting works", length(vals) == 5 && isapprox(vals[1], -1.0))

# 4. A tiny maximisation over a grid, the shape of what the labs do
β, R, y = 0.96, 1.02, 1.0
V0 = u.(y .+ (R - 1) .* a) ./ (1 - β)     # a value for each grid point
best = similar(a)
for (i, ai) in enumerate(a)
    # for each a' on the grid: consumption c = y + R a - a'; infeasible choices get -Inf
    candidates = [(c = y + R * ai - ap; c > 0 ? u(c) + β * V0[j] : -Inf) for (j, ap) in enumerate(a)]
    best[i] = maximum(candidates)
end
check("Maximisation over a grid works", all(isfinite, best) && issorted(best))

# 5. Backward loop syntax used in the finite-horizon code
T = 5; s = 0
for t in T:-1:1
    global s += t
end
check("for t in T:-1:1 loop works", s == 15)

# 6. Closed form check (cake eating): c_t = (1-β)/(1-β^(T-t+1)) a_t
c1 = (1 - β) / (1 - β^T) * 10.0
check("Closed-form arithmetic", isapprox(c1, 10.0 * (1 - β) / (1 - β^T)))

# 7. Threads (informational, never fails)
check("Threads available: $(Threads.nthreads())", true)

println("\nEC991 week 1 check, Julia $(VERSION) on $(Sys.MACHINE)\n")
allok = true
for (name, ok, hint) in results
    global allok &= ok
    @printf("  %-36s %s\n", name, ok ? "PASS" : "FAIL   <- " * hint)
end
println()
println(allok ? "All checks passed. You are ready for the week 2 lab." :
                "Some checks failed. Fix the hints above, or ask in the tutorial.")
