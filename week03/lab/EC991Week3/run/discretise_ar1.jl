# Tauchen vs Rouwenhorst for log y' = ρ log y + σ ε at ρ = 0.5, 0.9, 0.99, with
# n_y = 5 and σ = 0.1 (slides 10 to 18). Prints the chain's standard deviation and
# autocorrelation against the AR(1)'s, and draws the AR(1) and both chains on
# common shocks, one panel per ρ, into output/discretise_ar1_paths.pdf.
# Run from the package folder:  julia --project=. run/discretise_ar1.jl
using EC991Week3, Plots, Printf, Random

n_y, σ, T = 5, 0.1, 200
ρs = (0.5, 0.9, 0.99)

println("n_y = $n_y, σ = $σ: standard deviation and autocorrelation of log y")
@printf "%6s  %8s %8s  %8s %8s %6s  %8s %8s\n" "ρ" "AR(1) sd" "ac" "Tauchen" "ac" "P_33" "Rouwen." "ac"
for ρ in ρs
    logyT, PT = tauchen(n_y, ρ, σ)
    logyR, PR = rouwenhorst(n_y, ρ, σ)
    mT, mR = chain_moments(logyT, PT), chain_moments(logyR, PR)
    @printf "%6.2f  %8.4f %8.4f  %8.4f %8.4f %6.3f  %8.4f %8.4f\n" ρ σ / sqrt(1 - ρ^2) ρ mT.sd mT.autocorr PT[3, 3] mR.sd mR.autocorr
end

# Common shocks: ε drives the AR(1); u = Φ(ε) picks the next state of each chain.
Random.seed!(991)
ε = randn(T)
u = normal_cdf.(ε)
panels = map(ρs) do ρ
    logyT, PT = tauchen(n_y, ρ, σ)
    logyR, PR = rouwenhorst(n_y, ρ, σ)
    logy_ar1 = zeros(T)
    for t in 2:T
        logy_ar1[t] = ρ * logy_ar1[t-1] + σ * ε[t]
    end
    p = plot(1:T, logy_ar1; color = :gray40, linewidth = 1.2, label = "AR(1)", title = "ρ = $ρ",
             xlabel = "period, t", ylabel = "log y", legend = :outertopright)
    plot!(p, 1:T, simulate_chain(logyT, PT, u)[1]; seriestype = :steppost, linewidth = 2, label = "Tauchen")
    plot!(p, 1:T, simulate_chain(logyR, PR, u)[1]; seriestype = :steppost, linewidth = 2, label = "Rouwenhorst")
    p
end
outdir = joinpath(@__DIR__, "..", "output"); mkpath(outdir)
savefig(plot(panels...; layout = (3, 1), size = (900, 900)), joinpath(outdir, "discretise_ar1_paths.pdf"))
println("\npaths on common shocks written to output/discretise_ar1_paths.pdf")
