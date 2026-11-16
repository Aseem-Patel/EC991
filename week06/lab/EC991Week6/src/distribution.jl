# >>> law_of_motion!
"""
    law_of_motion!(μ1, μ0, sol, s, M, tmp)

One period of the law of motion for the measure of firms over (z, m) (tutorial
slides 5 and 10), with m the employment a firm inherits:

    μ_{t+1}(z_j, m) = Σ_i Σ_k μ_t(z_i, n_k) (1 - χ(z_i, n_k)) F_ij 1[m = n^d(z_i, n_k)]  +  M g_j 1[m = 0].

Two moves. First, every surviving firm carries its mass to the column of the n
it chose. Employment choices lie on the grid, so firm mass is assigned directly
to the chosen employment point. Then productivity moves with the transpose of F,
as in week 5. Entrants arrive at m = 0.
"""
function law_of_motion!(μ1, μ0, sol::FirmSolution, s, M, tmp)
    fill!(tmp, 0.0)
    n_z, n_n = size(μ0)
    # 1. survivors keep their z_i for now and move to the column of the n they chose
    for k in 1:n_n, i in 1:n_z
        sol.χ[i, k] || (tmp[i, sol.pol[i, k]] += μ0[i, k])
    end
    # 2. productivity moves: mass at z_i spreads over z_j with F_ij
    mul!(μ1, s.F', tmp)
    # 3. entrants draw z from g and start with no workers
    μ1[:, 1] .+= M .* s.g
    return μ1
end
# <<< law_of_motion!

# >>> stationary_distribution
"""
    stationary_distribution(sol, s; M = 1.0, tol = 1e-12, maxiter = 100_000)

The stationary measure μ(z, m) for a mass M of entrants, by iterating the law of
motion from μ = 0 until it stops changing (non-stochastic simulation,
tutorial slide 10). With inherited employment a state, the matrix of the law of motion has
(n_z n_n)² entries, so we do not invert it as in week 5. μ is proportional to M.
"""
function stationary_distribution(sol::FirmSolution, s; M = 1.0, tol = 1e-12, maxiter = 100_000)
    # start from no firms at all
    μ0 = zeros(length(s.z), length(s.n))
    μ1, tmp = similar(μ0), similar(μ0)
    for it in 1:maxiter
        # one period: survivors move, productivity moves, entrants arrive
        law_of_motion!(μ1, μ0, sol, s, M, tmp)
        # stop when no state's mass changes, relative to the largest
        maximum(abs.(μ1 .- μ0)) < tol * maximum(μ1) && return μ1
        μ0, μ1 = μ1, μ0                               # the update becomes the guess
    end
    error("the distribution did not converge in $maxiter iterations")
end
# <<< stationary_distribution
