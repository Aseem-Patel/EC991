# EC991 Week 6: plotting helpers shared by run/inaction_region.jl, run/firing_taxes.jl
# and run/make_figures.jl. No model logic here: everything plotted comes from the package.
ENV["GKSwstype"] = "100"          # draw without opening a window
using Plots

default(fontfamily = "Computer Modern", linewidth = 2.5, framestyle = :box, grid = false)
const c_low, c_high, c_grid, c_text = colorant"#2a78d6", colorant"#eb6834", "#d9d8d4", "#52514e"
const c_dark = colorant"#3d3c3a"

base(; kwargs...) = plot(; framestyle = :grid, grid = :y, gridlinewidth = 0.8, foreground_color_grid = c_grid,
                         gridalpha = 1.0, foreground_color_text = c_text, foreground_color_axis = c_grid,
                         foreground_color_border = c_grid, tick_direction = :none,
                         guidefontsize = 16, tickfontsize = 12, legendfontsize = 12, titlefontsize = 16,
                         margin = 3Plots.mm, left_margin = 6Plots.mm, bottom_margin = 7Plots.mm, kwargs...)

"""
    plot_inaction(eq; title, zmax = 12, nmax = 1000)

Tutorial slides 13 and 14: the hiring target n_L(z) and the firing target n_H(z),
with the inaction region between them shaded.
"""
function plot_inaction(eq; title = "", zmax = 12, nmax = 1000)
    r = inaction_region(eq.sol, eq.s)
    keep = r.z .<= zmax
    z, nL, nH = r.z[keep], r.nL[keep], r.nH[keep]
    p = base(; xlabel = "productivity, z", ylabel = "employment", legend = :topleft, title,
             xlims = (0, zmax), ylims = (0, nmax), size = (820, 520),
             legendfontsize = 15, tickfontsize = 14, guidefontsize = 18)
    plot!(p, z, nL; fillrange = nH, fillcolor = c_low, fillalpha = 0.12, linealpha = 0, label = "inaction: keep m")
    plot!(p, z, nH; color = c_dark, linestyle = :dash, label = "\$n_H(z)\$: fire down to it")
    plot!(p, z, nL; color = c_low, label = "\$n_L(z)\$: hire up to it")
    return p
end
