# Infinite horizon: solve by VFI, then test the output against theory.
# Run from the package folder:  julia --project=. run/solve_infinite.jl
using EC991Week2

hh = Household()                          # β = 1/1.05, R = 1.03: βR < 1
a = asset_grid(hh; amax = 3.0, na = 300)
sol = solve_vfi(hh, a; tol = 1e-6)

println("Converged in $(sol.iterations) iterations; last change $(sol.distances[end])")
d = sol.distances
println("Contraction check: every ratio d_ℓ / d_{ℓ-1} is at most β = $(round(hh.β; digits = 4)); ",
        "largest observed $(round(maximum(d[2:end] ./ d[1:end-1]); digits = 4))")
println("  (few iterations because the guess is exact at the borrowing limit and paths reach it in finite time)")

apath, cpath = simulate(sol, 2.5, 200)
println("\nPrediction 1: with βR = $(round(hh.β * hh.R; digits = 4)) < 1 assets drift to the limit -ϕ = $(-hh.ϕ)")
println("  a_0 = $(apath[1]), a_50 = $(round(apath[51]; digits = 3)), a_200 = $(apath[end])")

res = euler_residuals(cpath, hh)
println("\nPrediction 2: u'(c_t) - βR u'(c_{t+1}) ≥ 0 along the path")
println("  minimum residual $(round(minimum(res); digits = 5)); binding from t = ",
        findfirst(t -> apath[t + 1] == a[1], 1:200))
