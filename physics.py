import numpy as np
from scipy.integrate import solve_ivp

G = 9.81

MODEL_NAMES = {
    "none": "Без сопротивления",
    "linear": "Вязкое  F = k·v",
    "quadratic": "Лобовое  F = k·v²",
}


def derivatives(t, state, m, g, k, model):
    x, y, vx, vy = state
    if model == "linear":
        ax = -k / m * vx
        ay = -g - k / m * vy
    elif model == "quadratic":
        v = np.hypot(vx, vy)
        ax = -k / m * v * vx
        ay = -g - k / m * v * vy
    else:
        ax = 0.0
        ay = -g
    return [vx, vy, ax, ay]


def flight_time_no_drag(v0, alpha, h0, g=G):
    vy0 = v0 * np.sin(np.radians(alpha))
    return (vy0 + np.sqrt(vy0**2 + 2 * g * h0)) / g


def make_result(t, Y, T, t_top, H):
    x, y, vx, vy = Y
    return dict(t=t, x=x, y=y, vx=vx, vy=vy, T=T, L=x[-1], H=H, t_top=t_top,
                v_end=float(np.hypot(vx[-1], vy[-1])))


def hit_ground(t, s, *args):
    return s[1]


hit_ground.terminal = True
hit_ground.direction = -1


def top_point(t, s, *args):
    return s[3]


top_point.terminal = False
top_point.direction = -1


def fly(v0, alpha, h0, m, k, model, g=G, n_points=600, rtol=1e-9, atol=1e-9):
    if k == 0:
        model = "none"

    a = np.radians(alpha)
    vx0 = v0 * np.cos(a)
    vy0 = v0 * np.sin(a)
    if abs(vx0) < 1e-12 * max(v0, 1.0):
        vx0 = 0.0
    if abs(vy0) < 1e-12 * max(v0, 1.0):
        vy0 = 0.0
    state = [0.0, h0, vx0, vy0]

    if h0 <= 0 and vy0 <= 0:
        return make_result(np.array([0.0]), np.array([state]).T, 0.0, 0.0, max(h0, 0.0))

    length = max(h0, v0**2 / g, 1e-12)
    speed = max(v0, np.sqrt(2 * g * h0), 1e-12)
    atol = [atol * length, atol * length, atol * speed, atol * speed]

    t_start = 0.0
    t_end = 2 * flight_time_no_drag(v0, alpha, h0, g) + 1
    t_top, H = 0.0, h0
    found_top = False
    ts, ys = [], []

    for _ in range(60):
        sol = solve_ivp(derivatives, (t_start, t_end), state, args=(m, g, k, model),
                        method="RK23", events=(hit_ground, top_point),
                        dense_output=True, rtol=rtol, atol=atol)
        t_grid = np.linspace(sol.t[0], sol.t[-1], n_points)
        ts.append(t_grid)
        ys.append(sol.sol(t_grid))

        if not found_top and len(sol.t_events[1]) > 0:
            t_top = sol.t_events[1][0]
            H = sol.y_events[1][0][1]
            found_top = True

        if len(sol.t_events[0]) > 0:
            break
        t_start, state = sol.t[-1], sol.y[:, -1]
        t_end = t_start + 2 * t_end

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    Y[1, -1] = 0.0
    return make_result(t, Y, t[-1], t_top, H)


def exact_no_drag(t, v0, alpha, h0, g=G):
    a = np.radians(alpha)
    x = v0 * np.cos(a) * t
    y = h0 + v0 * np.sin(a) * t - g * t**2 / 2
    return x, y


def exact_no_drag_values(v0, alpha, h0, g=G):
    a = np.radians(alpha)
    T = flight_time_no_drag(v0, alpha, h0, g)
    return dict(L=v0 * np.cos(a) * T, H=h0 + (v0 * np.sin(a))**2 / (2 * g), T=T)


def phi(u):
    u = np.asarray(u, dtype=float)
    series = u**2 / 2 - u**3 / 6 + u**4 / 24 - u**5 / 120
    return np.where(u < 1e-3, series, u + np.expm1(-u))


def exact_linear(t, v0, alpha, h0, m, k, g=G):
    if k == 0:
        return exact_no_drag(t, v0, alpha, h0, g)
    a = np.radians(alpha)
    tau = m / k
    u = np.asarray(t, dtype=float) / tau
    e = -np.expm1(-u)
    x = v0 * np.cos(a) * tau * e
    y = h0 + v0 * np.sin(a) * tau * e - g * tau**2 * phi(u)
    return x, y


def exact_linear_values(v0, alpha, h0, m, k, g=G):
    a = np.radians(alpha)
    vx0, vy0 = v0 * np.cos(a), v0 * np.sin(a)
    tau = m / k
    t_top, H = 0.0, h0
    if vy0 > 0:
        t_top = tau * np.log1p(vy0 / (g * tau))
        H = exact_linear(t_top, v0, alpha, h0, m, k, g)[1]
    return dict(t_top=float(t_top), H=float(H), x_inf=vx0 * tau, v_term=g * tau)
