# >>> asset_grid
"""
    asset_grid(hh; amax = 3.0, na = 300)

Step 1 of the algorithm: a uniform grid of `na` asset levels from the
borrowing limit, a_1 = -ϕ, to `amax`. Both the state a and the choice a'
live on this grid.
"""
asset_grid(hh::Household; amax = 3.0, na = 300) = collect(range(-hh.ϕ, amax; length = na))
# <<< asset_grid
