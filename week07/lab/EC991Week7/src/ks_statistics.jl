# >>> aggregate_series
"""
    aggregate_series(eq)

The aggregate time series after the burn-in: productivity Z_t, capital K_t,
labour N_t = N(Z_t), output Y_t = Z_t K_t^α N_t^(1-α), investment
I_t = K_{t+1} - (1 - δ) K_t and consumption C_t = Y_t - I_t (the goods market).
The last date is dropped, because I_t needs K_{t+1}.
"""
function aggregate_series(eq::KSEquilibrium)
    (; eco, s, zpath, K) = eq
    ts = eco.T_burn+1:length(K)-1
    Z = eco.Z[zpath[ts]]
    N = s.N[zpath[ts]]
    Kt = K[ts]
    Y = Z .* Kt .^ eco.α .* N .^ (1 - eco.α)
    I = K[ts .+ 1] .- (1 - eco.δ) .* Kt
    return (; t = collect(ts), Z, N, K = Kt, Y, I, C = Y .- I)
end
# <<< aggregate_series

# >>> business_cycle_stats
"""
    business_cycle_stats(eq)

The columns of slide 30 (Krusell and Smith's Table 2): the mean of K,
corr(C_t, Y_t), the standard deviation of I_t and corr(Y_t, Y_{t-4}), from the
levels of the simulated series.
"""
function business_cycle_stats(eq::KSEquilibrium)
    x = aggregate_series(eq)
    mean(v) = sum(v) / length(v)
    sd(v) = sqrt(sum((v .- mean(v)) .^ 2) / (length(v) - 1))
    corr(u, v) = sum((u .- mean(u)) .* (v .- mean(v))) / ((length(u) - 1) * sd(u) * sd(v))
    return (; mean_K = mean(x.K), corr_CY = corr(x.C, x.Y), sd_I = sd(x.I),
            corr_YY4 = corr(x.Y[5:end], x.Y[1:end-4]), mean_Y = mean(x.Y), sd_logY = sd(log.(x.Y)))
end
# <<< business_cycle_stats

# >>> wealth_stats
"""
    wealth_stats(eq)

The wealth distribution at the end of the simulation (slide 29): the shares of
wealth held by the top 1, 5, 10, 20 and 30%, the share of households at the
borrowing limit, and the Gini coefficient. The distribution's mass over assets
is λ summed over endowments.
"""
function wealth_stats(eq::KSEquilibrium)
    a, m = eq.s.a_dist, vec(sum(eq.λ; dims = 2))
    top = [100 * top_share(a, m, x) for x in (0.01, 0.05, 0.10, 0.20, 0.30)]
    return (; top1 = top[1], top5 = top[2], top10 = top[3], top20 = top[4], top30 = top[5],
            at_limit = m[1] / sum(m), gini = gini(a, m))
end
# <<< wealth_stats
