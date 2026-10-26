# Equal against growth-rate spacing of the asset grid (slides 6 to 9), both with 300 points on
# [0, 50], at the lab default. Draws the two grids and the two saving policies, and prints the
# largest gap between the policies and where it occurs. Plots go to output/compare_grids.pdf.
# Run from the package folder:  julia --project=. run/compare_grids.jl
using EC991Week3, Plots, Printf

hh = Household()
logy, P = rouwenhorst(3, 0.9, 0.1)
y = exp.(logy)
a_eq = asset_grid(300, 50.0; spacing = :equal)
a_gr = asset_grid(300, 50.0; spacing = :growth)              # ν = 0.02
@printf "equal spacing: every gap %.3f\n" a_eq[2] - a_eq[1]
@printf "growth-rate spacing: first gap %.4f, last gap %.3f; %d of the 300 points lie below a = 5\n" a_gr[2] - a_gr[1] a_gr[end] - a_gr[end-1] count(a_gr .< 5)

sol_eq = solve_vfi(hh, a_eq, y, P)
sol_gr = solve_vfi(hh, a_gr, y, P)
g = compare_policies(sol_gr, sol_eq)
@printf "largest gap between the two saving policies: %.4f, at a = %.3f with y_%d\n" g.gap g.a g.j
for (lo, hi) in ((0.0, 5.0), (5.0, 50.0))
    gs = maximum(abs(sol_gr.apol[i, j] - LinearInterp(a_eq, sol_eq.apol[:, j])(a_gr[i]))
                 for i in findall(lo .<= a_gr .<= hi), j in 1:3)
    @printf "  on [%2.0f, %2.0f]: %.4f\n" lo hi gs
end

p1 = plot(xlabel = "a", yticks = ([1, 2], ["growth", "equal"]), ylims = (0.5, 2.5), legend = false, title = "the two grids")
scatter!(p1, a_eq, fill(2, 300); markersize = 2, markerstrokewidth = 0, color = :gray40)
scatter!(p1, a_gr, fill(1, 300); markersize = 2, markerstrokewidth = 0, color = :steelblue)
p2 = plot(xlabel = "a", ylabel = "a'(a, y_1) - a", title = "net saving at y_1, near the constraint", xlims = (0, 5))
plot!(p2, a_eq, sol_eq.apol[:, 1] .- a_eq; label = "equal", color = :gray40, marker = :circle, markersize = 2, markercolor = :gray40, markerstrokewidth = 0)
plot!(p2, a_gr, sol_gr.apol[:, 1] .- a_gr; label = "growth", color = :steelblue, marker = :circle, markersize = 2, markercolor = :steelblue, markerstrokewidth = 0)
outdir = joinpath(@__DIR__, "..", "output"); mkpath(outdir)
savefig(plot(p1, p2; layout = (2, 1), size = (900, 700)), joinpath(outdir, "compare_grids.pdf"))
println("plots written to output/compare_grids.pdf")
