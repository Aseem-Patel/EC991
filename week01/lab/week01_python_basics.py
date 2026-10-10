"""EC991 Week 1: Julia basics for the week 2 lab, Python edition.

Mirrors week01_julia_basics.jl section by section so that you can read the two
side by side. Needs numpy; matplotlib is optional. Run with:  python3 week01_python_basics.py
Exercises are marked TODO; the check after each one prints PASS or NOT YET.
"""
import numpy as np

def check(name, ok):
    print(f"{'PASS' if ok else 'NOT YET'}  {name}")

# ---- 1. Numbers and variables -------------------------------------------
# Python has int and float too; 2 / 2 is a float (1.0) just as in Julia.
print(type(2), type(2.0), type(2 / 2))
beta, gamma, R = 0.96, 2.0, 1.02          # no Greek letters by convention; spell them out
print("beta * R < 1:", beta * R < 1)

# ---- 2. Functions ----------------------------------------------------------
def u(c):                                  # CRRA utility
    return c ** (1 - gamma) / (1 - gamma)

def u_general(c, g):
    return np.log(c) if g == 1 else c ** (1 - g) / (1 - g)

def pv(y, beta=0.96, T=10):                # keyword arguments with defaults
    return sum(beta ** t * y for t in range(T + 1))

print(u(1.0), u(2.0), u_general(2.0, 1.0))
print(pv(1.0), pv(1.0, T=1000), 1 / (1 - 0.96))

# Exercise 2: write u_log(c) returning log(c)
def u_log(c):
    return None                            # TODO
check("Exercise 2", u_log(3.0) is not None and np.isclose(u_log(3.0), u_general(3.0, 1.0)))

# ---- 3. Arrays, ranges and indexing ---------------------------------------
# Python indexes from 0 and a slice a[1:4] excludes position 4.
n_a = 7
a_grid = np.linspace(0.0, 3.0, n_a)
print(a_grid[0], a_grid[-1], a_grid[1:4], len(a_grid))
M = np.array([[1.0, 2.0], [3.0, 4.0]])
print(M[1, 0], M[:, 1], M.shape)
Vmat = np.zeros((n_a, 4))

# ---- 4. Broadcasting --------------------------------------------------------
# NumPy broadcasts automatically: no dot needed, but u must be written with
# array-friendly operations (** and / are fine; `if` on an array is not).
print(u(a_grid + 1.0))

# Exercise 4: utility of consuming all cash on hand y + R a at every grid point
y = 1.0
u_cash = None                              # TODO: one line
check("Exercise 4", u_cash is not None and np.allclose(u_cash, u(y + R * a_grid)))

# ---- 5. Loops, conditionals, comprehensions ---------------------------------
T = 5
out = []
for t in range(T, 0, -1):                  # Julia: for t in T:-1:1
    out.append(t)
print(out)

x, tol, it = 1.0, 1e-6, 0
while x > tol:
    x /= 2
    it += 1
print({"x": x, "iterations": it})

print([u(c) if c > 0 else -np.inf for c in np.arange(-1.0, 1.01, 0.5)])

v = np.array([0.3, -np.inf, 1.7, 0.9])
print(v.max(), v.argmax())

# Exercise 5: present value with a loop and a running total
def pv_loop(y, beta, T):
    total = 0.0
    # TODO: add beta**t * y to total for t = 0, 1, ..., T
    return None
r = pv_loop(1.0, 0.96, 10)
check("Exercise 5", r is not None and np.isclose(r, pv(1.0, beta=0.96, T=10)))

# ---- 6. Plotting (optional) ------------------------------------------------
try:
    import matplotlib.pyplot as plt
    c = np.linspace(0.2, 3.0, 100)
    plt.plot(c, u(c), label=f"u(c), gamma = {gamma}")
    plt.plot(c, np.log(c), "--", label="log(c)")
    plt.xlabel("c"); plt.ylabel("utility"); plt.legend(); plt.savefig("week01_utility.png", dpi=120)
    print("Saved week01_utility.png")
except ImportError:
    print("matplotlib not installed: skip the plot, or `pip install matplotlib`")

# ---- 8. Two habits -----------------------------------------------------------
import time
t0 = time.perf_counter(); s = np.sum(u(np.linspace(0.1, 10.0, 1_000_000))); dt = time.perf_counter() - t0
print(f"sum over a million points: {s:.4f} in {dt*1000:.1f} ms")
