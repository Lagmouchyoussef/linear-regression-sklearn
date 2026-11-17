"""Generate the extra figures and the terminal screenshot used in the README.

Run with:  python generate_assets.py
Outputs go to the ./assets folder.
"""
import html
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

X = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float).reshape(-1, 1)
y = np.array([22, 27, 34, 40, 45, 52, 57], dtype=float)

BLUE, RED, ORANGE, GREEN, GRAY = "#2563eb", "#ef4444", "#f59e0b", "#10b981", "#64748b"

model = LinearRegression().fit(X, y)
theta0, theta1 = model.intercept_, model.coef_[0]
y_pred = model.predict(X)


def style(ax):
    ax.grid(True, linestyle="--", alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# 1) Residuals ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
residuals = y - y_pred
ax.axhline(0, color=GRAY, linewidth=1.2)
ax.vlines(X.ravel(), 0, residuals, color=ORANGE, linewidth=2)
ax.scatter(X, residuals, color=RED, zorder=3, s=55)
ax.set_title("Residuals (actual - predicted)", fontweight="bold")
ax.set_xlabel("Experience (years)")
ax.set_ylabel("Residual (kDH)")
style(ax)
fig.tight_layout()
fig.savefig(ASSETS / "residuals_plot.png", dpi=200)
plt.close(fig)

# 2) Cost function surface (contour) -----------------------------------------
t0 = np.linspace(0, 32, 200)
t1 = np.linspace(2, 10, 200)
T0, T1 = np.meshgrid(t0, t1)
J = np.zeros_like(T0)
for xi, yi in zip(X.ravel(), y):
    J += (T0 + T1 * xi - yi) ** 2
J /= 2 * len(y)

fig, ax = plt.subplots(figsize=(8, 5))
cs = ax.contourf(T0, T1, np.log10(J + 1e-3), levels=25, cmap="viridis")
ax.contour(T0, T1, np.log10(J + 1e-3), levels=25, colors="white", linewidths=0.3, alpha=0.5)
ax.scatter([theta0], [theta1], color=RED, marker="*", s=260, edgecolor="white", zorder=5,
           label=f"Optimum ({theta0:.2f}, {theta1:.2f})")
fig.colorbar(cs, ax=ax, label="log10 J(theta0, theta1)")
ax.set_title("Cost function J(theta0, theta1)", fontweight="bold")
ax.set_xlabel("theta0 (intercept)")
ax.set_ylabel("theta1 (slope)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(ASSETS / "cost_function.png", dpi=200)
plt.close(fig)

# 3) Gradient descent: regression lines for each learning rate --------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color=BLUE, zorder=3, label="Actual data", s=55)
xs = np.linspace(1, 7, 100)
ax.plot(xs, theta0 + theta1 * xs, color=RED, linewidth=2.8, label="Analytical (LinearRegression)")
for eta0, color, ls in [(1e-4, GRAY, ":"), (1e-2, GREEN, "--"), (1e-1, ORANGE, "-.")]:
    sgd = SGDRegressor(loss="squared_error", penalty=None, learning_rate="constant",
                       eta0=eta0, max_iter=10000, tol=None, random_state=42).fit(X, y)
    ax.plot(xs, sgd.intercept_[0] + sgd.coef_[0] * xs, color=color, linestyle=ls, linewidth=2,
            label=f"SGD eta0={eta0:g}")
ax.set_title("Analytical solution vs. gradient descent", fontweight="bold")
ax.set_xlabel("Experience (years)")
ax.set_ylabel("Salary (kDH)")
ax.legend()
style(ax)
fig.tight_layout()
fig.savefig(ASSETS / "gradient_descent_comparison.png", dpi=200)
plt.close(fig)

# 4) Terminal screenshot (SVG) built from the real output of main.py ---------
proc = subprocess.run([sys.executable, str(Path(__file__).parent / "main.py")],
                      capture_output=True, text=True, encoding="utf-8",
                      env={**__import__("os").environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"})
lines = [l for l in proc.stdout.splitlines()]
# Wrap the two long explanatory sentences so the window stays readable
wrapped = []
for line in lines:
    while len(line) > 96:
        cut = line.rfind(" ", 0, 96)
        wrapped.append(line[:cut])
        line = line[cut + 1:]
    wrapped.append(line)
lines = wrapped

lh, pad_x, top = 20, 26, 64
height = top + lh * (len(lines) + 2) + 24
width = 860
rows = []
for i, line in enumerate(lines):
    y_pos = top + lh * (i + 2)
    color = "#e2e8f0"
    if line.rstrip().endswith(":") and not line.startswith(" "):
        color = "#38bdf8"
    elif "theta" in line or "R^2" in line or "MSE" in line or "Equation" in line:
        color = "#a7f3d0"
    elif "Prediction" in line:
        color = "#fde68a"
    rows.append(f'<text x="{pad_x}" y="{y_pos}" fill="{color}">{html.escape(line) or " "}</text>')

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="12" fill="#0f172a"/>
  <rect width="{width}" height="40" rx="12" fill="#1e293b"/>
  <rect y="28" width="{width}" height="12" fill="#1e293b"/>
  <circle cx="24" cy="20" r="7" fill="#ef4444"/><circle cx="46" cy="20" r="7" fill="#f59e0b"/><circle cx="68" cy="20" r="7" fill="#10b981"/>
  <text x="{width / 2}" y="25" text-anchor="middle" fill="#94a3b8" font-family="Consolas, Menlo, monospace" font-size="13">python main.py</text>
  <g xml:space="preserve" font-family="Consolas, Menlo, 'DejaVu Sans Mono', monospace" font-size="13">
    <text x="{pad_x}" y="{top}" fill="#10b981">$ python main.py</text>
    {chr(10).join(rows)}
  </g>
</svg>"""
(ASSETS / "terminal_output.svg").write_text(svg, encoding="utf-8")
print("Assets generated in", ASSETS)
