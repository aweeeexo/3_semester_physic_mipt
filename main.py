import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import physics as ph

n_up = 4.0
n_down = 3.0
u = 4400.0
theta = 0.0
m_dry = 20000.0

if u <= 0 or m_dry <= 0:
    raise ValueError("u и m_dry должны быть больше нуля")
if not 0 <= theta <= 180:
    raise ValueError("угол theta должен быть от 0 до 180 градусов")
n_min_up = ph.surface_gravity(ph.GM_EARTH, ph.R_EARTH) / ph.G0
n_min_down = ph.surface_gravity(ph.GM_MARS, ph.R_MARS) / ph.G0
if n_up <= n_min_up:
    raise ValueError(f"n_up должно быть больше {n_min_up:.4f}: иначе тяга не превышает вес и корабль не взлетит")
if n_down <= n_min_down:
    raise ValueError(f"n_down должно быть больше {n_min_down:.4f}: иначе тяга не удержит корабль у Марса")
for stage, n in [("взлёте", n_up), ("посадке", n_down)]:
    if n > ph.N_HUMAN_MAX:
        print(f"ВНИМАНИЕ: перегрузка {n:g} g при {stage} больше {ph.N_HUMAN_MAX:g} g — опасно для экипажа\n")

AU, DAY = ph.AU, ph.DAY
C_BURN, C_COAST = "tab:orange", "tab:blue"
C_EARTH, C_MARS = "tab:blue", "tab:red"

hoh = ph.hohmann(ph.A_EARTH, ph.A_MARS)
v_inf0 = ph.required_v_inf(0.0)
v_inf = ph.required_v_inf(theta)
s1 = ph.takeoff(n_up, v_inf, u)
s2 = ph.transfer(v_inf, theta)
ref = s2 if theta == 0 else ph.transfer(v_inf0, 0.0)
s3 = ph.landing(s2["v_rel"], n_down, u)


def wrap(deg):
    return (deg + 180) % 360 - 180


def wrap_rad(a):
    return np.angle(np.exp(1j * a))


def km_unit(length):
    return (1e9, "млн км") if length >= 1e9 else (1e6, "тыс. км") if length >= 1e6 else (1e3, "км")


print("ЭТАП 1. Вертикальный взлёт с Земли")
print(f"  перегрузка {n_up:g} g, скорость истечения {u / 1e3:.2f} км/с, угол вылета θ = {theta:g}°")
print(f"  нужная скорость на границе сферы действия Земли ({ph.R_SOI_EARTH / 1e6:.0f} тыс. км): "
      f"{v_inf / 1e3:.4f} км/с")
print(f"  двигатель выключен через {s1['t_burn']:.1f} с на высоте {(s1['r_burn'] - ph.R_EARTH) / 1e3:.3f} км "
      f"(теория {(s1['r_burn_theory'] - ph.R_EARTH) / 1e3:.3f} км), v = {s1['v_burn'] / 1e3:.4f} км/с")
print(f"  затрачено Δv = {s1['dv_engine'] / 1e3:.3f} км/с (без гравитационных потерь было бы "
      f"{s1['dv_ideal'] / 1e3:.3f}), осталось {100 * s1['m_burn']:.3g}% массы")
print(f"  граница сферы действия достигнута через {s1['t_end'] / DAY:.2f} сут, "
      f"v = {s1['v_end'] / 1e3:.6f} км/с (цель {v_inf / 1e3:.6f})")

print("\nЭТАП 2. Перелёт в поле тяготения Солнца")
print(f"  стартовая скорость относительно Солнца {s2['v_start'] / 1e3:.4f} км/с")
print(f"  время перелёта {s2['T'] / DAY:.3f} сут (Гоман, теория: {hoh['T'] / DAY:.3f} сут), "
      f"угол перелёта {np.degrees(s2['phi_arr']):.3f}°")
print(f"  остановка: {s2['stopped_by']}, промах по радиусу орбиты Марса {s2['miss']:.1f} м")
print(f"  скорость относительно Марса {s2['v_rel'] / 1e3:.4f} км/с (Гоман, теория: {hoh['dv2'] / 1e3:.4f} км/с)")
print(f"  Марс при старте должен опережать Землю на {wrap(np.degrees(s2['phi_mars0'])):.2f}°")
if theta != 0:
    d = np.hypot(s2["x"][-1] - ref["x"][-1], s2["y"][-1] - ref["y"][-1])
    print(f"  по сравнению с θ = 0: Δv_inf = {v_inf - v_inf0:+.4g} м/с, ΔT = {(s2['T'] - ref['T']) / DAY:+.4g} сут, "
          f"точка прилёта сдвинута на {d / 1e6:.4g} тыс. км")

print("\nЭТАП 3. Мягкая посадка на Марс")
print(f"  перегрузка {n_down:g} g, свободное падение от границы сферы действия ({ph.R_SOI_MARS / 1e6:.0f} тыс. км) "
      f"{s3['t_ign'] / DAY:.2f} сут")
print(f"  двигатель включён на высоте {(s3['r_ign'] - ph.R_MARS) / 1e3:.3f} км при v = {abs(s3['v_ign']) / 1e3:.4f} км/с")
print(f"  торможение {s3['t_brake']:.1f} с, затрачено Δv = {s3['dv_engine'] / 1e3:.3f} км/с, "
      f"без торможения удар на {s3['v_crash'] / 1e3:.3f} км/с")
print(f"  остановка: {s3['stopped_by']}, высота {s3['h_end']:.2e} м, остаточная скорость {s3['v_touch']:.2e} м/с")

m_land = m_dry / s3["m_end"]
m_start = m_land / s1["m_burn"]
print("\nМАССА (одна ступень, без учёта баков)")
print(f"  на Марс садится {m_dry / 1e3:.4g} т, перед посадкой {m_land / 1e3:.4g} т, на старте {m_start / 1e3:.4g} т")

fig, axs = plt.subplots(3, 3, figsize=(19, 12), gridspec_kw=dict(width_ratios=[1, 1.25, 1]))

ax = axs[0, 0]
h = (s1["r"] - ph.R_EARTH) / 1e3
burn = s1["thrust"] > 0
ax.plot(h[burn], s1["v"][burn] / 1e3, color=C_BURN, lw=2, label="разгон, двигатель включён")
ax.plot(h[~burn], s1["v"][~burn] / 1e3, color=C_COAST, lw=2, label="полёт по инерции")
r_need = ph.R_EARTH + np.concatenate([[0], np.geomspace(1, h.max() * 1e3, 400)])
v_need = np.sqrt(v_inf**2 + 2 * ph.GM_EARTH * (1 / r_need - 1 / ph.R_SOI_EARTH))
ax.plot((r_need - ph.R_EARTH) / 1e3, v_need / 1e3, "k--", lw=1, label="скорость, нужная на этой высоте")
ax.axhline(v_inf / 1e3, color="gray", ls=":", label=f"v_∞ = {v_inf / 1e3:.3f} км/с")
hb = (s1["r_burn"] - ph.R_EARTH) / 1e3
ax.plot(hb, s1["v_burn"] / 1e3, "o", color="k")
ax.annotate(f"выключение двигателя\nh = {hb:.0f} км, t = {s1['t_burn']:.0f} с\nv = {s1['v_burn'] / 1e3:.2f} км/с",
            (hb, s1["v_burn"] / 1e3), xytext=(0.62, 0.55), textcoords="axes fraction", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xscale("symlog", linthresh=1, linscale=0.5)
ax.set_xlim(0, h.max() * 1.5)
ax.set_ylim(0, v_need[0] / 1e3 * 1.45)
ax.set_title("Этап 1: взлёт с Земли")
ax.set_xlabel("высота h, км (до 1 км — линейно, дальше — логарифм)")
ax.set_ylabel("v, км/с")
ax.legend(fontsize=7, loc="upper left")


def plot_overload(ax, t, res, gm, t_on, t_off, n_before, n_after, title):
    span = t_off - t_on
    mask = (t >= t_on - 0.2 * span) & (t <= t_off + 0.2 * span)
    tt = t[mask] - t_on
    felt = res["thrust"][mask] / ph.G0
    kin = (res["thrust"][mask] - gm / res["r"][mask]**2) / ph.G0
    ax.plot(np.r_[-0.2 * span, tt[0]], [n_before, n_before], color="k", lw=2.5)
    ax.plot(tt, felt, color="k", lw=2.5, label="ощущаемая перегрузка n")
    ax.plot(np.r_[tt[-1], 1.2 * span], [n_after, n_after], color="k", lw=2.5)
    ax.plot(tt, kin, color="tab:green", ls="--", lw=1.5, label="ускорение dv/dt (вверх +), в g")
    ax.axhline(ph.N_HUMAN_MAX, color="red", ls=":", label=f"предел для экипажа ≈ {ph.N_HUMAN_MAX:g} g")
    ax.axhline(1, color="gray", ls=":", lw=1, label="1 g (вес на Земле)")
    ax.axvspan(0, span, color=C_BURN, alpha=0.12, label=f"двигатель работает {span:.0f} с")
    ax.annotate(f"n = {felt.max():.3g} g", (0.5 * span, felt.max()), xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)
    for x, n, text in [(-0.1 * span, n_before, "до"), (1.1 * span, n_after, "после")]:
        if n > 0:
            ax.annotate(f"{text}: {n:.3g} g", (x, n), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-0.2 * span, 1.2 * span)
    top = max(ph.N_HUMAN_MAX, felt.max(), kin.max())
    ax.set_ylim(min(-1.0, kin.min()) - 0.1 * top, top * 1.2)
    ax.set_title(title)
    ax.set_xlabel("время от включения двигателя, с")
    ax.set_ylabel("n, g")
    ax.legend(fontsize=7, loc="upper right")


plot_overload(axs[1, 0], s1["t"], s1, ph.GM_EARTH, 0.0, s1["t_burn"], 1.0, 0.0, "Этап 1: перегрузка при разгоне")

ax = axs[2, 0]
ns = np.unique(np.r_[np.linspace(1.08 * n_min_up, max(12, 1.3 * n_up), 50), n_up])
dv = np.array([ph.takeoff(n, v_inf, u, coast=False, n_points=5)["dv_engine"] for n in ns])
ax.semilogy(ns, dv / 1e3, color="k", label="затраты Δv двигателя")
ax.axhline(s1["dv_ideal"] / 1e3, color="gray", ls="--", label=f"без гравитационных потерь: {s1['dv_ideal'] / 1e3:.2f}")
ax.axvspan(n_min_up * 0.9, n_min_up, color="gray", alpha=0.3, label="не взлетит")
ax.axvspan(ph.N_HUMAN_MAX, ns[-1] * 1.05, color="red", alpha=0.1, label="опасно для экипажа")
ax.plot(n_up, s1["dv_engine"] / 1e3, "o", color=C_BURN,
        label=f"выбрано: n = {n_up:g}, m0/m = {1 / s1['m_burn']:.3g}")
ax.set_xlim(n_min_up * 0.9, ns[-1] * 1.05)
ax.set_title("Этап 1: цена разгона от перегрузки")
ax.set_xlabel("перегрузка n, g")
ax.set_ylabel("Δv, км/с")
ax.legend(fontsize=7, loc="upper right")

ax = axs[0, 1]
circle = np.linspace(0, 2 * np.pi, 2001)
arr = np.array([s2["x"][-1], s2["y"][-1]]) / AU
arr_ref = np.array([ref["x"][-1], ref["y"][-1]]) / AU


def at(r, phi):
    return r * np.cos(phi) / AU, r * np.sin(phi) / AU


def draw_map(a, legend):
    a.plot(0, 0, "o", color="gold", ms=12, mec="darkorange", label="Солнце" if legend else None)
    a.plot(*at(ph.A_EARTH, circle), color=C_EARTH, ls="--", lw=0.8, label="орбита Земли" if legend else None)
    a.plot(*at(ph.A_MARS, circle), color=C_MARS, ls="--", lw=0.8, label="орбита Марса" if legend else None)
    if theta != 0:
        a.plot(ref["x"] / AU, ref["y"] / AU, color="gray", lw=1.5, label="θ = 0 (гомановский)" if legend else None)
        a.plot(*arr_ref, "s", color="gray", ms=5)
    a.plot(s2["x"] / AU, s2["y"] / AU, color="k", lw=1.5, label=f"корабль, θ = {theta:g}°" if legend else None)
    a.plot(*at(ph.A_EARTH, 0.0), "o", color=C_EARTH, label="Земля при старте" if legend else None)
    a.plot(*at(ph.A_EARTH, s2["phi_earth1"]), "o", color=C_EARTH, mfc="none",
           label="Земля при прилёте" if legend else None)
    a.plot(*at(ph.A_MARS, s2["phi_mars0"]), "o", color=C_MARS, mfc="none",
           label=f"Марс при старте (впереди на {wrap(np.degrees(s2['phi_mars0'])):.1f}°)" if legend else None)
    a.plot(*arr, "o", color=C_MARS, label="встреча с Марсом" if legend else None)


draw_map(ax, True)
ax.plot(*arr, "o", ms=16, mfc="none", mec="k", label="см. врезку")
lim = 1.15 * ph.A_MARS / AU
ax.set_xlim(-lim, lim * 2.75)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title(f"Этап 2: перелёт, T = {s2['T'] / DAY:.1f} сут")
ax.set_xlabel("x, а.е.")
ax.set_ylabel("y, а.е.")
ax.legend(fontsize=6, loc="upper right", labelspacing=0.3, borderaxespad=0.3)

ax_in = ax.inset_axes([0.7, 0.11, 0.29, 0.31])
dphi = wrap_rad(s2["phi_arr"] - ref["phi_arr"])
phi_c = ref["phi_arr"] + dphi / 2
ds = ph.A_MARS * abs(dphi)
ws = min(max(2.5 * ds, 5e7), 0.95 * np.pi * ph.A_MARS) if theta != 0 else 0.02 * ph.A_MARS

curves = []
for res, c, lw in ([(ref, "gray", 3)] if theta != 0 else []) + [(s2, "k", 1.5)]:
    s = ph.A_MARS * wrap_rad(np.arctan2(res["y"], res["x"]) - phi_c)
    dr = np.hypot(res["x"], res["y"]) - ph.A_MARS
    near = np.abs(s) <= ws
    curves.append((np.where(near, s, np.nan), np.where(near, dr, np.nan), s[-1], dr[-1], c, lw, res is s2))
dr_min = min(np.nanmin(cv[1]) for cv in curves)
sx, nx = km_unit(ws)
sy, ny = km_unit(-dr_min)
for s, dr, s_end, dr_end, c, lw, main_res in curves:
    ax_in.plot(s / sx, dr / sy, color=c, lw=lw)
    ax_in.plot(s_end / sx, dr_end / sy, "o" if main_res else "s", color=C_MARS if main_res else "gray")
ax_in.axhline(0, color=C_MARS, ls="--", lw=0.8)
ax_in.set_xlim(-ws / sx, ws / sx)
ax_in.set_ylim(1.1 * dr_min / sy, -0.15 * dr_min / sy)
ax_in.tick_params(labelsize=6)
ax_in.set_xlabel(f"вдоль орбиты Марса, {nx}", fontsize=6, labelpad=1)
ax_in.set_ylabel(f"r − a_Марса, {ny}", fontsize=6, labelpad=1)
su, nu = km_unit(ds)
ax_in.set_title(f"подлёт к орбите Марса: сдвиг {ds / su:.4g} {nu}" if theta != 0
                else "подлёт к орбите Марса: касание", fontsize=6.5, pad=2)
ax_in.grid(alpha=0.3)

ax = axs[1, 1]
thetas = np.unique(np.r_[0, 0.25, 0.5, 1, 2, 3, 5, 7, np.linspace(10, 180, 35), theta, theta / 2, min(2 * theta, 180)])
sweep = [(ph.required_v_inf(th), th) for th in thetas]
sweep = [(v, ph.transfer(v, th, n_points=5)) for v, th in sweep]
dv_inf = np.array([v - v_inf0 for v, _ in sweep])
dv_rel = np.array([r["v_rel"] - ref["v_rel"] for _, r in sweep])
T_days = np.array([r["T"] / DAY for _, r in sweep])
i = int(np.argmin(np.abs(thetas - theta)))
ax.plot(thetas, dv_inf, "o-", ms=3, color=C_EARTH, label="прирост v_∞ при вылете")
ax.plot(thetas, dv_rel, "s-", ms=3, color=C_MARS, label="прирост скорости у Марса")
ax.plot(theta, dv_inf[i], "o", ms=9, mfc="none", color="k")
ax.plot(theta, dv_rel[i], "s", ms=9, mfc="none", color="k",
        label=f"θ = {theta:g}°: {dv_inf[i]:+.3g} / {dv_rel[i]:+.3g} м/с")
lin_y = min([1.0] + [0.5 * abs(d) for d in (dv_inf[i], dv_rel[i]) if d != 0])
ax.set_xscale("symlog", linthresh=theta if 0 < theta < 1 else 1, linscale=0.6)
ax.set_yscale("symlog", linthresh=lin_y, linscale=0.6)
ax.set_xlim(0, 200)
ax.set_title("Этап 2: зависимость от угла вылета θ")
ax.set_xlabel(f"θ, ° (до {theta if 0 < theta < 1 else 1:g}° — линейно, дальше — логарифм)")
ax.set_ylabel("Δv относительно θ = 0, м/с")
ax_t = ax.twinx()
ax_t.plot(thetas, T_days, color="gray", ls="--", label="время перелёта")
ax_t.plot(theta, T_days[i], "D", color="gray", label=f"T = {T_days[i]:.2f} сут")
ax_t.set_ylabel("T, сут")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax_t.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")

ax = axs[2, 1]
err_E2, err_L2 = ph.transfer_invariant_errors(s2)
for x, err, c, ls, label in [
        (s1["t"] / s1["t_end"], ph.takeoff_energy_error(s1), C_EARTH, "-", "этап 1: энергия vs теория"),
        (s2["t"] / s2["T"], err_E2, "k", "-", "этап 2: энергия"),
        (s2["t"] / s2["T"], err_L2, "gray", "--", "этап 2: момент импульса"),
        (s3["t"] / s3["t_end"], ph.landing_energy_error(s3), C_MARS, "-", "этап 3: энергия vs теория")]:
    ax.semilogy(x[1:], err[1:] + 1e-18, color=c, ls=ls, lw=1, label=label)
ax.set_xscale("symlog", linthresh=1e-3, linscale=0.5)
ax.set_xlim(0, 1)
ax.set_title("Точность интегрирования")
ax.set_xlabel("доля времени этапа t / T (до 0.001 — линейно)")
ax.set_ylabel("относительная ошибка")
ax.legend(fontsize=7)

ax = axs[0, 2]
h = (s3["r"] - ph.R_MARS) / 1e3
burn = s3["thrust"] > 0
ax.plot(h[~burn], -s3["v"][~burn] / 1e3, color=C_COAST, lw=2, label="свободное падение")
ax.plot(h[burn], -s3["v"][burn] / 1e3, color=C_BURN, lw=2, label="торможение двигателем")
r_fall = ph.R_MARS + np.concatenate([[0], np.geomspace(1, s3["r_ign"] - ph.R_MARS, 300)])
ax.plot((r_fall - ph.R_MARS) / 1e3, np.sqrt(2 * (s3["e0"] + ph.GM_MARS / r_fall)) / 1e3, "k--", lw=1,
        label=f"без торможения: удар на {s3['v_crash'] / 1e3:.2f} км/с")
hi = (s3["r_ign"] - ph.R_MARS) / 1e3
ax.plot(hi, -s3["v_ign"] / 1e3, "o", color="k")
ax.annotate(f"включение двигателя\nh = {hi:.0f} км\nv = {-s3['v_ign'] / 1e3:.2f} км/с", (hi, -s3["v_ign"] / 1e3),
            xytext=(0.66, 0.55), textcoords="axes fraction", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.plot(max(s3["h_end"], 0) / 1e3, s3["v_touch"] / 1e3, "*", ms=14, color="tab:green",
        label=f"посадка: v = {s3['v_touch']:.1e} м/с")
ax.set_xscale("symlog", linthresh=1, linscale=0.5)
ax.set_xlim(h.max() * 1.5, -0.3)
ax.set_ylim(0, s3["v_crash"] / 1e3 * 1.45)
ax.set_title("Этап 3: посадка на Марс")
ax.set_xlabel("высота h, км (до 1 км — линейно, дальше — логарифм)")
ax.set_ylabel("|v|, км/с")
ax.legend(fontsize=7, loc="upper left")

plot_overload(axs[1, 2], s3["t"], s3, ph.GM_MARS, s3["t_ign"], s3["t_end"], 0.0, n_min_down,
              "Этап 3: перегрузка при торможении")

ax = axs[2, 2]
ns = np.unique(np.r_[np.linspace(1.08 * n_min_down, max(12, 1.3 * n_down), 50), n_down])
dv = np.array([ph.landing(s2["v_rel"], n, u, n_points=5)["dv_engine"] for n in ns])
ax.semilogy(ns, dv / 1e3, color="k", label="затраты Δv двигателя")
ax.axhline(s3["v_crash"] / 1e3, color="gray", ls="--", label=f"без гравитационных потерь: {s3['v_crash'] / 1e3:.2f}")
ax.axvspan(0, n_min_down, color="gray", alpha=0.3, label="не сядет")
ax.axvspan(ph.N_HUMAN_MAX, ns[-1] * 1.05, color="red", alpha=0.1, label="опасно для экипажа")
ax.plot(n_down, s3["dv_engine"] / 1e3, "o", color=C_BURN,
        label=f"выбрано: n = {n_down:g}, m0/m = {1 / s3['m_end']:.3g}")
ax.set_xlim(0, ns[-1] * 1.05)
ax.set_title("Этап 3: цена торможения от перегрузки")
ax.set_xlabel("перегрузка n, g")
ax.set_ylabel("Δv, км/с")
ax.legend(fontsize=7, loc="upper right")

for ax in [axs[2, 0], axs[2, 2]]:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax.tick_params(axis="y", which="minor", labelsize=7)
for ax in axs.flat:
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
