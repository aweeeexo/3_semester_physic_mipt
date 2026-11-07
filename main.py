import numpy as np
import matplotlib.pyplot as plt
import physics as ph

N = 100
k = 100.0
m = 1.0
a = 0.1
j0 = 25
ends = "free"
A = 0.00175
tau = 40 / ph.omega0(k, m)
gamma = 0.05
F0 = 0.002
modes = [1, 2, 3, 5, 25, 50, 75]
show_modes = [1, 2, 3]

if N < 100:
    raise ValueError("по условию в цепочке должно быть не меньше 100 грузов")
if k <= 0 or m <= 0 or a <= 0:
    raise ValueError("k, m и a должны быть больше нуля")
if A <= 0 or F0 <= 0:
    raise ValueError("A и F0 должны быть больше нуля, иначе волны нет")
if gamma <= 0:
    raise ValueError("gamma должна быть больше нуля, иначе в резонансе амплитуда бесконечна")
if not 1 <= j0 <= N:
    raise ValueError("j0 должен быть от 1 до N")
if ends not in ["free", "fixed"]:
    raise ValueError("ends может быть только free или fixed")
if not all(1 <= n <= N for n in modes + show_modes):
    raise ValueError("номера мод должны быть от 1 до N")
if tau * ph.omega0(k, m) < 1:
    raise ValueError("импульс слишком короткий: нужно tau >= 1/omega0")

w0 = ph.omega0(k, m)
c_th = ph.wave_speed(k, m, a)
w1 = ph.mode_omega(1, N, k, m)

if tau * w0 < 10:
    print("импульс короткий: заметна дисперсия, скорость будет меньше a·ω0\n")
if gamma > w1:
    print("трение большое: соседние резонансы сливаются, пиков может не быть\n")
if gamma < w1 / 100:
    print("трение очень маленькое: установление идёт долго, счёт будет медленным\n")


def pulse_speed(N, k, m, a, j0, tau):
    r = ph.run_pulse(N, k, m, j0, A, tau, ends)
    dist, times = ph.arrival_times(r, k, m, j0, A, tau)
    c, b = ph.measure_speed(dist, times, a)
    return r, dist, times, c, b


res, dist, times, c_num, c_shift = pulse_speed(N, k, m, a, j0, tau)
if np.isnan(c_num):
    print("импульс слишком длинный для такой цепочки: скорость не измерена")
if ph.masses_collide(res["u"], a):
    print("смещения больше расстояния между грузами: грузы столкнулись бы")

print("Часть 1. Скорость волны")
print(f"{'N':>5}{'k':>8}{'m':>8}{'a':>8}{'τ·ω0':>8}{'c_числ':>10}{'c_теор':>10}{'ошибка, %':>11}")
speed_cases = [(N, k, m, a, j0, tau * w0),
               (N, 4 * k, m, a, j0, 40),
               (N, k, 4 * m, a, j0, 40),
               (N, k, m, 2 * a, j0, 40),
               (2 * N, k, m, a, 2 * j0, 40),
               (N, k, m, a, j0, 5)]
for Nc, kc, mc, ac, jc, tw in speed_cases:
    tc = tw / ph.omega0(kc, mc)
    c = pulse_speed(Nc, kc, mc, ac, jc, tc)[3]
    th = ph.wave_speed(kc, mc, ac)
    print(f"{Nc:>5}{kc:>8.1f}{mc:>8.1f}{ac:>8.2f}{tw:>8.0f}{c:>10.4f}{th:>10.4f}"
          f"{100 * (c - th) / th:>11.2f}")

print("\nЧасть 2. Резонансы (идёт счёт, это занимает около минуты)")
scans = {}
for n in modes:
    if ph.is_node(n, N, j0):
        print(f"мода {n} не возбуждается: груз j0 = {j0} стоит в её узле")
    else:
        scans[n] = ph.scan_mode(n, N, k, m, gamma, j0, F0)

print(f"{'n':>4}{'ω_n':>10}{'Ω_числ':>10}{'ошибка, %':>11}{'A_числ':>12}{'A_теор':>12}")
for n, s in scans.items():
    w = ph.mode_omega(n, N, k, m)
    if np.isnan(s["W"]):
        print(f"{n:>4}{w:>10.4f}   пика нет: груз близко к узлу моды или трение велико")
        continue
    i = np.argmax(s["amps"])
    a_th = ph.exact_amplitude(N, k, m, gamma, j0, F0, s["Omegas"][i]).max()
    print(f"{n:>4}{w:>10.4f}{s['W']:>10.4f}{100 * (s['W'] - w) / w:>11.2f}"
          f"{s['amps'][i]:>12.3e}{a_th:>12.3e}")

print("\nПервый резонанс при других параметрах")
print(f"{'N':>5}{'k':>8}{'m':>8}{'γ':>8}{'ω_1':>10}{'Ω_числ':>10}{'ошибка, %':>11}")
res_cases = [(N, 4 * k, m, gamma), (N, k, 4 * m, gamma / 2), (N + 50, k, m, gamma / 2)]
for Nc, kc, mc, gc in res_cases:
    W = ph.scan_mode(1, Nc, kc, mc, gc, j0, F0)["W"]
    th = ph.mode_omega(1, Nc, kc, mc)
    print(f"{Nc:>5}{kc:>8.1f}{mc:>8.1f}{gc:>8.3f}{th:>10.4f}{W:>10.4f}"
          f"{100 * (W - th) / th:>11.2f}")

stand = {n: ph.run_forced(N, k, m, gamma, j0, F0, ph.mode_omega(n, N, k, m)) for n in show_modes}
if any(ph.masses_collide(s["u"], a) for s in stand.values()):
    print("амплитуда стоячей волны слишком большая: грузы столкнулись бы")

side, room = ph.long_side(N, j0)
t_reach = max(room / w0, tau)
t_snaps = [tau, (tau + t_reach) / 2, t_reach]
pulse_len = w0 * tau

fig1, axs = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("Часть 1: бегущая волна")

ax = axs[0, 0]
for ts, color in zip(t_snaps, ["tab:blue", "tab:orange", "tab:green"]):
    i = np.argmin(np.abs(res["t"] - ts))
    ax.plot(res["j"], res["u"][:, i], color=color, label=f"t = {ts:.2f} с")
ax.set_title("Снимки всей цепочки")
ax.set_xlabel("номер груза j")
ax.set_ylabel("смещение u, м")
ax.legend(fontsize=8)

ax = axs[0, 1]
ts = t_snaps[1]
i = np.argmin(np.abs(res["t"] - ts))
center = j0 + side * (w0 * ts - pulse_len / 2)
left, right = center - 0.6 * pulse_len, center + 0.6 * pulse_len
inside = (res["j"] >= left) & (res["j"] <= right)
u_snap = res["u"][:, i]
ax.plot(res["j"][inside], u_snap[inside], "o", ms=4, color="tab:orange", label="расчёт")
ax.plot(res["j"][inside], ph.exact_pulse(ts, N, k, m, j0, A, tau)[inside], color="k", ls="--",
        label="теория (без дисперсии)")
ax.set_title(f"Кроп: импульс при t = {ts:.2f} с")
ax.set_xlabel("номер груза j")
ax.set_ylabel("смещение u, м")
ax.legend(fontsize=8)

ax = axs[1, 0]
u_max = np.abs(res["u"]).max()
img = ax.imshow(res["u"], aspect="auto", origin="lower", cmap="RdBu_r", vmin=-u_max, vmax=u_max,
                extent=[0, res["t"][-1], 1, N])
t_line = np.array([tau / 2, tau / 2 + room / w0])
ax.plot(t_line, j0 + side * w0 * (t_line - tau / 2), color="k", ls="--", label="c = a√(k/m)")
ax.set_title(f"Диаграмма j–t, концы: {ends}")
ax.set_xlabel("t, с")
ax.set_ylabel("номер груза j")
ax.legend(fontsize=8)
fig1.colorbar(img, ax=ax, label="u, м")

ax = axs[1, 1]
ax.plot(times, dist * a, "o", ms=3, color="tab:blue", label="момент прихода фронта")
ax.plot(times, c_num * times + c_shift, color="tab:red", label="прямая по точкам")
ax.set_title(f"c_числ = {c_num:.4f} м/с, c_теор = {c_th:.4f} м/с")
ax.set_xlabel("t, с")
ax.set_ylabel("расстояние от груза j0, м")
ax.legend(fontsize=8)

fig2, axs = plt.subplots(2, 3, figsize=(16, 9))
fig2.suptitle("Часть 2: резонанс и стоячие волны")

ax = axs[0, 0]
Om_all = np.linspace(0.3 * w1, 2.3 * w0, 1500)
ax.semilogy(Om_all, [ph.exact_amplitude(N, k, m, gamma, j0, F0, W).max() for W in Om_all],
            color="tab:blue", label="теория")
for s in scans.values():
    ax.semilogy(s["Omegas"], s["amps"], "o", ms=3, color="tab:red")
ax.plot([], [], "o", ms=3, color="tab:red", label="расчёт")
ax.axvline(2 * w0, color="k", ls="--", label="2ω0")
ax.set_title("Резонансная кривая, вся полоса")
ax.set_xlabel("Ω, рад/с")
ax.set_ylabel("наибольшая амплитуда, м")
ax.legend(fontsize=8)

ax = axs[0, 1]
first = sorted(modes)[:4]
Om_lo = 0.5 * ph.mode_omega(first[0], N, k, m)
Om_hi = ph.mode_omega(first[-1], N, k, m) + 0.5 * w1
Om_zoom = np.linspace(Om_lo, Om_hi, 800)
ax.plot(Om_zoom, [ph.exact_amplitude(N, k, m, gamma, j0, F0, W).max() for W in Om_zoom],
        color="tab:blue", label="теория")
for n in first:
    ax.axvline(ph.mode_omega(n, N, k, m), color="gray", ls=":")
    if n in scans:
        ax.plot(scans[n]["Omegas"], scans[n]["amps"], "o", ms=3, color="tab:red")
ax.plot([], [], "o", ms=3, color="tab:red", label="расчёт")
ax.plot([], [], color="gray", ls=":", label="ω_n")
ax.set_xlim(Om_lo, Om_hi)
ax.set_ylim(bottom=0)
ax.set_title("Кроп: первые резонансы")
ax.set_xlabel("Ω, рад/с")
ax.set_ylabel("наибольшая амплитуда, м")
ax.legend(fontsize=8)

ax = axs[0, 2]
n_all = np.arange(1, N + 1)
ax.plot(n_all, ph.mode_omega(n_all, N, k, m), color="tab:blue", label="ω_n = 2ω0·sin(q_n/2)")
ax.plot(n_all, w0 * ph.mode_q(n_all, N), color="gray", ls="--", label="длинные волны ω = ω0·q")
ax.plot(list(scans), [s["W"] for s in scans.values()], "o", ms=4, color="tab:red",
        label="пики из расчёта")
ax.set_ylim(0, 2.2 * w0)
ax.set_title("Закон дисперсии")
ax.set_xlabel("номер моды n")
ax.set_ylabel("ω, рад/с")
ax.legend(fontsize=8)

for ax, n in zip(axs[1], show_modes):
    s = stand[n]
    amp_th = ph.exact_amplitude(N, k, m, gamma, j0, F0, ph.mode_omega(n, N, k, m))
    for i in range(0, len(s["t"]), 25):
        ax.plot(s["j"], s["u"][:, i], color="tab:blue", lw=0.7, alpha=0.6)
    ax.plot(s["j"], amp_th, color="k", ls="--", label="теория |U_j|")
    ax.plot(s["j"], -amp_th, color="k", ls="--")
    ax.plot(s["j"], s["amp"], "o", ms=2, color="tab:red", label="расчёт")
    ax.set_title(f"Стоячая волна, мода n = {n}")
    ax.set_xlabel("номер груза j")
    ax.set_ylabel("смещение u, м")
    ax.legend(fontsize=8)

for ax in list(fig1.axes) + list(fig2.axes):
    if ax.get_label() != "<colorbar>":
        ax.grid(alpha=0.3)
fig1.tight_layout()
fig2.tight_layout()
plt.show()
