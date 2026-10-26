# >>> golden_section
"""
    golden_section(f, lo, hi; tol = 1e-8)

Maximise f on [lo, hi] by golden-section search. Keep two interior points at
the golden ratios of the bracket, drop the end beyond the worse one, and
reuse the better one, so each step costs one evaluation and shrinks the
bracket by 0.618. Assumes f has one peak on [lo, hi]. Returns the maximiser,
the maximum and the number of evaluations.
"""
function golden_section(f, lo, hi; tol = 1e-8)
    g = (sqrt(5) - 1) / 2
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = f(x1), f(x2)
    evals = 2
    while hi - lo > tol
        if f1 >= f2                          # the peak is in [lo, x2]
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo); f1 = f(x1)
        else                                 # the peak is in [x1, hi]
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo); f2 = f(x2)
        end
        evals += 1
    end
    return f1 >= f2 ? (x1, f1, evals) : (x2, f2, evals)
end
# <<< golden_section

# >>> brent
"""
    brent(f, lo, hi; tol = 1e-8)

Maximise f on [lo, hi] by Brent's method: fit a parabola through the best
three points found so far and jump to its peak; fall back on a golden-section
step whenever the parabola is untrustworthy (its peak outside the bracket, or
the step not shrinking). Near a smooth peak the parabolic steps converge much
faster than golden section. Returns the maximiser, the maximum and the
number of evaluations. (Brent 1973; the layout follows Press et al.,
Numerical Recipes, section 10.3, written for maximisation.)
"""
function brent(f, lo, hi; tol = 1e-8, maxiter = 200)
    cg = (3 - sqrt(5)) / 2                   # golden-section fraction
    a, b = lo, hi
    x = w = v = a + cg * (b - a)             # x: best, w: second best, v: previous w
    fx = fw = fv = f(x)
    evals = 1
    d = e = 0.0
    for _ in 1:maxiter
        xm = (a + b) / 2
        tol1 = tol * abs(x) + 1e-12
        tol2 = 2tol1
        abs(x - xm) <= tol2 - (b - a) / 2 && break
        parabolic = false
        if abs(e) > tol1                     # try a parabola through x, w, v
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            q > 0 && (p = -p)
            q = abs(q)
            if abs(p) < abs(q * e / 2) && p > q * (a - x) && p < q * (b - x)
                e, d = d, p / q
                u = x + d
                (u - a < tol2 || b - u < tol2) && (d = xm >= x ? tol1 : -tol1)
                parabolic = true
            end
        end
        if !parabolic                        # golden-section step into the larger part
            e = x >= xm ? a - x : b - x
            d = cg * e
        end
        u = abs(d) >= tol1 ? x + d : x + (d > 0 ? tol1 : -tol1)
        fu = f(u); evals += 1
        if fu >= fx                          # u is the new best
            u >= x ? (a = x) : (b = x)
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else
            u < x ? (a = u) : (b = u)
            if fu >= fw || w == x
                v, fv, w, fw = w, fw, u, fu
            elseif fu >= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    return x, fx, evals
end
# <<< brent
