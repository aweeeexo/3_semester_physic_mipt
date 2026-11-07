import numpy as np
from scipy.integrate import solve_ivp


def omega0(k, m):
    return np.sqrt(k / m)


def wave_speed(k, m, a):
    return a * omega0(k, m)


def mode_q(n, N):
    return np.pi * n / (N + 1)


def mode_omega(n, N, k, m):
    return 2 * omega0(k, m) * np.sin(mode_q(n, N) / 2)


def is_node(n, N, j0):
    return abs(np.sin(mode_q(n, N) * j0)) < 1e-9


def long_side(N, j0):
    if N - j0 >= j0 - 1:
        return 1, N - j0
    return -1, j0 - 1


def spring_acc(u, k, m, ends):
    if ends == "fixed":
        left, right = 0.0, 0.0
    else:
        left, right = u[0], u[-1]
    full = np.concatenate(([left], u, [right]))
    return k / m * (full[2:] - 2 * u + full[:-2])


def pulse_law(t, A, tau):
    w = 2 * np.pi / tau
    pos = A / 2 * (1 - np.cos(w * t))
    vel = A * w / 2 * np.sin(w * t)
    acc = A * w**2 / 2 * np.cos(w * t)
    return pos, vel, acc


def pulse_derivatives(t, state, k, m, ends, j0, A, tau):
    N = len(state) // 2
    vel = state[N:].copy()
    acc = spring_acc(state[:N], k, m, ends)
    if t < tau:
        _, vel[j0 - 1], acc[j0 - 1] = pulse_law(t, A, tau)
    return np.concatenate((vel, acc))


def forced_derivatives(t, state, k, m, gamma, j0, F0, Omega):
    N = len(state) // 2
    vel = state[N:]
    acc = spring_acc(state[:N], k, m, "fixed") - gamma * vel
    acc[j0 - 1] += F0 / m * np.cos(Omega * t)
    return np.concatenate((vel, acc))


def run_pulse(N, k, m, j0, A, tau, ends, n_frames=3000, rtol=1e-8):
    t_end = tau + 2 * N / omega0(k, m)
    t = np.linspace(0, t_end, n_frames)
    sol = solve_ivp(pulse_derivatives, (0, t_end), np.zeros(2 * N),
                    args=(k, m, ends, j0, A, tau), method="DOP853",
                    t_eval=t, rtol=rtol, atol=1e-9 * A)
    return dict(t=t, j=np.arange(1, N + 1), u=sol.y[:N])


def exact_pulse(t, N, k, m, j0, A, tau):
    j = np.arange(1, N + 1)
    s = t - np.abs(j - j0) / omega0(k, m)
    return np.where((s > 0) & (s < tau), pulse_law(s, A, tau)[0], 0.0)


def arrival_times(res, k, m, j0, A, tau):
    side, room = long_side(len(res["j"]), j0)
    last = int(room - omega0(k, m) * tau / 4)
    dist, times = [], []
    for d in range(3, last):
        u = np.abs(res["u"][j0 - 1 + side * d])
        if u.max() >= A / 2:
            dist.append(d)
            times.append(res["t"][np.argmax(u >= A / 2)])
    return np.array(dist), np.array(times)


def measure_speed(dist, times, a):
    if len(dist) < 5:
        return np.nan, np.nan
    c, b = np.polyfit(times, dist * a, 1)
    return c, b


def run_forced(N, k, m, gamma, j0, F0, Omega, n_frames=200, rtol=1e-6):
    t_settle = 16 / gamma
    t_end = t_settle + 2 * np.pi / Omega
    t = np.linspace(t_settle, t_end, n_frames)
    sol = solve_ivp(forced_derivatives, (0, t_end), np.zeros(2 * N),
                    args=(k, m, gamma, j0, F0, Omega), method="DOP853",
                    t_eval=t, rtol=rtol, atol=1e-9 * F0 / k)
    u = sol.y[:N]
    return dict(t=t, j=np.arange(1, N + 1), u=u, amp=np.abs(u).max(axis=1))


def exact_amplitude(N, k, m, gamma, j0, F0, Omega):
    n = np.arange(1, N + 1)
    q = mode_q(n, N)
    w = mode_omega(n, N, k, m)
    b = 2 / (N + 1) * F0 / m * np.sin(q * j0) / (w**2 - Omega**2 + 1j * gamma * Omega)
    j = n.reshape(-1, 1)
    return np.abs(np.sin(q * j) @ b)


def find_peak(Omegas, amps):
    i = np.argmax(amps)
    if i == 0 or i == len(amps) - 1:
        return np.nan
    p = np.polyfit(Omegas[i - 1:i + 2], amps[i - 1:i + 2], 2)
    return -p[1] / (2 * p[0])


def scan_mode(n, N, k, m, gamma, j0, F0, n_points=7):
    w = mode_omega(n, N, k, m)
    gap = w - mode_omega(n - 1, N, k, m)
    Omegas = np.linspace(w - gap / 3, w + gap / 3, n_points)
    amps = np.array([run_forced(N, k, m, gamma, j0, F0, W)["amp"].max() for W in Omegas])
    return dict(Omegas=Omegas, amps=amps, W=find_peak(Omegas, amps))


def masses_collide(u, a):
    return np.diff(u, axis=0).min() <= -a
