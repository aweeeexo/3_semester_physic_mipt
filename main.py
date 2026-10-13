import numpy as np
import matplotlib.pyplot as plt
import physics as ph

v0 = 300.0
alpha = 1.0
h0 = 0.0
m = 1.0
g = ph.G
k_lin = 1.1
k_quad = 0.1

if v0 < 0 or h0 < 0 or k_lin < 0 or k_quad < 0:
    raise ValueError("v0, h0, k_lin, k_quad не могут быть отрицательными")
if m <= 0 or g <= 0:
    raise ValueError("m и g должны быть больше нуля")
if not 0 <= alpha <= 90:
    raise ValueError("угол alpha должен быть от 0 до 90 градусов")
if h0 == 0 and (v0 == 0 or alpha == 0):
    print("h0 = 0 и бросок не вверх: камень сразу лежит на земле, полёт длится 0 с\n")

MODELS = ["none", "linear", "quadratic"]
COLORS = {"none": "gray", "linear": "tab:blue", "quadratic": "tab:red"}
K = {"none": 0.0, "linear": k_lin, "quadratic": k_quad}

res = {md: ph.fly(v0, alpha, h0, m, K[md], md, g) for md in MODELS}

print(f"{'модель':<20}{'L, м':>10}{'H, м':>10}{'t_top, с':>10}{'T, с':>10}{'v_пад, м/с':>12}")
for md in MODELS:
    r = res[md]
    print(f"{ph.MODEL_NAMES[md]:<20}{r['L']:>10.3f}{r['H']:>10.3f}{r['t_top']:>10.3f}"
          f"{r['T']:>10.3f}{r['v_end']:>12.3f}")

ex = ph.exact_no_drag_values(v0, alpha, h0, g)
print(f"\nбез сопротивления, теория: L = {ex['L']:.6f}, H = {ex['H']:.6f}, T = {ex['T']:.6f}")
if k_lin > 0:
    ex_lin = ph.exact_linear_values(v0, alpha, h0, m, k_lin, g)
    print(f"вязкое, теория: t_top = {ex_lin['t_top']:.6f}, H = {ex_lin['H']:.6f}, "
          f"x_inf = {ex_lin['x_inf']:.3f}, v_пред = {ex_lin['v_term']:.3f}")

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

def plot_trajectories(ax, models):
    for md in models:
        ax.plot(res[md]["x"], res[md]["y"], color=COLORS[md], label=ph.MODEL_NAMES[md])
    x_span = max(res[md]["x"].max() for md in models)
    y_span = max(res[md]["y"].max() for md in models)
    ax.set_xlim(-0.03 * x_span, x_span * 1.03 if x_span > 0 else 1)
    ax.set_ylim(0, y_span * 1.1 if y_span > 0 else 1)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")


big_x = res["none"]["x"].max()
big_y = res["none"]["y"].max()
small = [md for md in ["linear", "quadratic"]
         if res[md]["x"].max() < 0.3 * big_x or res[md]["y"].max() < 0.3 * big_y]

ax = axs[0, 0]
if small:
    cell = ax.get_subplotspec().subgridspec(1, 2, wspace=0.35)
    ax.remove()
    ax = axs[0, 0] = fig.add_subplot(cell[0])
    ax_zoom = fig.add_subplot(cell[1])
    plot_trajectories(ax_zoom, small)
    ax_zoom.set_title("Увеличено: с сопротивлением", fontsize=10)
    ax_zoom.grid(alpha=0.3)
plot_trajectories(ax, MODELS)
ax.set_title("Траектории")
ax.legend(fontsize=7)

ax = axs[0, 1]
for md in MODELS:
    r = res[md]
    ax.plot(r["t"], np.hypot(r["vx"], r["vy"]), color=COLORS[md], label=ph.MODEL_NAMES[md])
if k_lin > 0:
    ax.axhline(m * g / k_lin, color=COLORS["linear"], ls="--", label="mg/k")
if k_quad > 0:
    ax.axhline(np.sqrt(m * g / k_quad), color=COLORS["quadratic"], ls="--", label="√(mg/k)")
ax.set_ylim(bottom=0)
ax.set_title("Модуль скорости")
ax.set_xlabel("t, с")
ax.set_ylabel("v, м/с")
ax.legend(fontsize=8)

ax = axs[0, 2]
r = res["none"]
x_ex, y_ex = ph.exact_no_drag(r["t"], v0, alpha, h0, g)
ax.semilogy(r["t"][1:], np.hypot(r["x"] - x_ex, r["y"] - y_ex)[1:] + 1e-18, color=COLORS["none"],
            label="без сопротивления")
r = res["linear"]
x_ex, y_ex = ph.exact_linear(r["t"], v0, alpha, h0, m, k_lin, g)
ax.semilogy(r["t"][1:], np.hypot(r["x"] - x_ex, r["y"] - y_ex)[1:] + 1e-18, color=COLORS["linear"],
            label="вязкое")
ax.set_title("Ошибка: |r_числ − r_теор|")
ax.set_xlabel("t, с")
ax.set_ylabel("м")
ax.legend()

ax = axs[1, 0]
angles = np.linspace(0, 90, 91)
for md in MODELS:
    L = np.array([ph.fly(v0, a, h0, m, K[md], md, g, n_points=20)["L"] for a in angles])
    i = np.argmax(L)
    ax.plot(angles, L, color=COLORS[md], label=f"{ph.MODEL_NAMES[md]}, опт. {angles[i]:.0f}°")
    ax.plot(angles[i], L[i], "o", color=COLORS[md])
ax.set_title("Дальность от угла")
ax.set_xlabel("α, °")
ax.set_ylabel("L, м")
ax.legend(fontsize=8)

ax = axs[1, 1]
for a in [15, 30, 45, 60, 75, 90]:
    r = ph.fly(v0, a, h0, m, k_quad, "quadratic", g)
    ax.plot(r["x"], r["y"], label=f"α = {a}°")
ax.set_title("Лобовое: разные углы")
ax.set_xlabel("x, м")
ax.set_ylabel("y, м")
ax.legend(fontsize=8)

ax = axs[1, 2]
ks = np.logspace(-6, 0, 25)
for md in ["linear", "quadratic"]:
    L = [ph.fly(v0, alpha, h0, m, kk, md, g, n_points=20)["L"] for kk in ks]
    ax.semilogx(ks, L, "o-", ms=3, color=COLORS[md], label=ph.MODEL_NAMES[md])
ax.axhline(ex["L"], color="k", ls="--", label="теория без сопр.")
ax.set_title("Предельный переход k → 0")
ax.set_xlabel("k")
ax.set_ylabel("L, м")
ax.legend(fontsize=8)

for ax in axs.flat:
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
