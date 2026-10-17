import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G0 = 9.80665
AU = 1.495978707e11
DAY = 86400.0

GM_SUN = 1.32712440018e20
GM_EARTH = 3.986004418e14
GM_MARS = 4.282837e13
R_EARTH = 6.371e6
R_MARS = 3.3895e6
A_EARTH = 1.0 * AU
A_MARS = 1.523679 * AU

N_HUMAN_MAX = 6.0
RTOL = 1e-10


def sphere_of_influence(a, gm):
    return a * (gm / GM_SUN) ** 0.4


R_SOI_EARTH = sphere_of_influence(A_EARTH, GM_EARTH)
R_SOI_MARS = sphere_of_influence(A_MARS, GM_MARS)


def surface_gravity(gm, R):
    return gm / R**2


def event(fun, direction, terminal=True):
    fun.direction = direction
    fun.terminal = terminal
    return fun


def time_grid(t0, t1, n):
    dt = t1 - t0
    if dt <= 0:
        return np.array([t0])
    g = np.geomspace(1e-9, 1, n) * dt
    return np.unique(np.concatenate([np.linspace(t0, t1, n), t0 + g, t1 - g]))


def integrate(rhs, t0, state, args, events, t_guess, scale, n_points=400):
    atol = RTOL * np.asarray(scale, dtype=float)
    t_start, span = t0, t_guess
    ts, ys = [], []
    for _ in range(60):
        sol = solve_ivp(rhs, (t_start, t_start + span), state, args=args, method="DOP853",
                        events=events, dense_output=True, rtol=RTOL, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        grid = time_grid(sol.t[0], sol.t[-1], n_points)
        ts.append(grid)
        ys.append(sol.sol(grid))
        if sol.status == 1:
            hit = next(i for i, te in enumerate(sol.t_events) if len(te) > 0)
            Y = np.concatenate(ys, axis=1)
            Y[:, -1] = sol.y[:, -1]
            return np.concatenate(ts), Y, hit
        t_start, state = sol.t[-1], sol.y[:, -1]
        span *= 2
    raise RuntimeError("событие так и не наступило")


def radial_rhs(t, s, gm, a_thrust, u):
    r, v, m = s
    return [v, -gm / r**2 + a_thrust, -m * a_thrust / u]


def burnout_radius_theory(v_inf, a_t, gm=GM_EARTH, R=R_EARTH, r_soi=R_SOI_EARTH):
    return R + (v_inf**2 / 2 - gm / r_soi + gm / R) / a_t


def ignition_radius_theory(v_inf, a_t, gm=GM_MARS, R=R_MARS, r_soi=R_SOI_MARS):
    return R + (v_inf**2 / 2 - gm / r_soi + gm / R) / a_t


def takeoff(n_up, v_inf, u, coast=True, n_points=400):
    gm, R, r_soi = GM_EARTH, R_EARTH, R_SOI_EARTH
    a_t = n_up * G0
    g_surf = surface_gravity(gm, R)
    if a_t <= g_surf:
        raise ValueError("тяга не превышает вес: корабль не оторвётся от Земли")

    e_target = v_inf**2 / 2 - gm / r_soi
    v_ideal = np.sqrt(2 * (e_target + gm / R))
    r_b = burnout_radius_theory(v_inf, a_t)
    if r_b >= r_soi:
        raise ValueError("разгон не успевает закончиться внутри сферы действия Земли")
    scale = [R, v_ideal, 1.0]

    burnout = event(lambda t, s, *a: s[1]**2 / 2 - gm / s[0] - e_target, +1)
    t1, y1, _ = integrate(radial_rhs, 0.0, [R, 0.0, 1.0], (gm, a_t, u), [burnout],
                          2 * v_ideal / (a_t - g_surf), scale, n_points)
    res = dict(t=t1, r=y1[0], v=y1[1], m=y1[2], thrust=np.full(t1.size, a_t), a_t=a_t,
               t_burn=t1[-1], r_burn=y1[0, -1], v_burn=y1[1, -1], m_burn=y1[2, -1],
               r_burn_theory=r_b, dv_engine=a_t * t1[-1], dv_ideal=v_ideal,
               e_target=e_target, v_target=v_inf)
    if not coast:
        return res

    leave = event(lambda t, s, *a: s[0] - r_soi, +1)
    t2, y2, _ = integrate(radial_rhs, t1[-1], y1[:, -1], (gm, 0.0, u), [leave],
                          r_soi / max(v_inf, 1.0), scale, n_points)
    res.update(t=np.concatenate([t1, t2]), r=np.concatenate([y1[0], y2[0]]),
               v=np.concatenate([y1[1], y2[1]]), m=np.concatenate([y1[2], y2[2]]),
               thrust=np.concatenate([res["thrust"], np.zeros(t2.size)]),
               t_end=t2[-1], v_end=y2[1, -1])
    return res


def landing(v_inf, n_down, u, n_points=400):
    gm, R, r_soi = GM_MARS, R_MARS, R_SOI_MARS
    a_t = n_down * G0
    if a_t <= surface_gravity(gm, R):
        raise ValueError("тяга не превышает вес на Марсе: мягкая посадка невозможна")

    e0 = v_inf**2 / 2 - gm / r_soi
    r_ign = ignition_radius_theory(v_inf, a_t)
    if r_ign >= r_soi:
        raise ValueError("тормозить нужно начинать раньше входа в сферу действия Марса")
    v_crash = np.sqrt(2 * (e0 + gm / R))
    scale = [R, v_crash, 1.0]

    ignite = event(lambda t, s, *a: s[0] - r_ign, -1)
    t1, y1, _ = integrate(radial_rhs, 0.0, [r_soi, -v_inf, 1.0], (gm, 0.0, u), [ignite],
                          r_soi / max(v_inf, 1.0), scale, n_points)

    stop = event(lambda t, s, *a: s[1], +1)
    touch = event(lambda t, s, *a: s[0] - R, -1)
    t2, y2, hit = integrate(radial_rhs, t1[-1], y1[:, -1], (gm, a_t, u), [stop, touch],
                            2 * v_crash / (a_t - surface_gravity(gm, R)), scale, n_points)

    return dict(t=np.concatenate([t1, t2]), r=np.concatenate([y1[0], y2[0]]),
                v=np.concatenate([y1[1], y2[1]]), m=np.concatenate([y1[2], y2[2]]),
                thrust=np.concatenate([np.zeros(t1.size), np.full(t2.size, a_t)]), a_t=a_t,
                e0=e0, r_ign=r_ign, t_ign=t1[-1], v_ign=y1[1, -1], t_end=t2[-1],
                t_brake=t2[-1] - t1[-1], h_end=y2[0, -1] - R, v_touch=abs(y2[1, -1]),
                m_end=y2[2, -1], dv_engine=a_t * (t2[-1] - t1[-1]), v_crash=v_crash,
                stopped_by=["скорость обнулилась", "касание поверхности"][hit])


def hohmann(r1, r2, mu=GM_SUN):
    a = (r1 + r2) / 2
    v_p = np.sqrt(mu * (2 / r1 - 1 / a))
    v_a = np.sqrt(mu * (2 / r2 - 1 / a))
    return dict(a=a, v_p=v_p, v_a=v_a, T=np.pi * np.sqrt(a**3 / mu),
                dv1=v_p - np.sqrt(mu / r1), dv2=np.sqrt(mu / r2) - v_a)


def departure_velocity(v_inf, theta, r1=A_EARTH, mu=GM_SUN):
    th = np.radians(theta)
    return v_inf * np.sin(th), np.sqrt(mu / r1) + v_inf * np.cos(th)


def apoapsis(r, vr, vt, mu=GM_SUN):
    energy = (vr**2 + vt**2) / 2 - mu / r
    if energy >= 0:
        return np.inf
    a = -mu / (2 * energy)
    e = np.sqrt(max(0.0, 1 + 2 * energy * (r * vt)**2 / mu**2))
    return a * (1 + e)


def required_v_inf(theta, r1=A_EARTH, r2=A_MARS, mu=GM_SUN):
    def miss(v):
        return min(apoapsis(r1, *departure_velocity(v, theta, r1, mu), mu), 10 * r2) - r2

    grid = np.linspace(0, 3 * np.sqrt(mu / r1), 1501)
    vals = np.array([miss(v) for v in grid])
    i = int(np.argmax(vals >= 0))
    if vals[i] < 0:
        raise ValueError("не удалось подобрать скорость для полёта к Марсу")
    if i == 0:
        return 0.0
    return brentq(miss, grid[i - 1], grid[i], xtol=1e-10, rtol=1e-15)


def sun_rhs(t, s, mu):
    x, y, vx, vy = s
    r3 = np.hypot(x, y)**3
    return [vx, vy, -mu * x / r3, -mu * y / r3]


def transfer(v_inf, theta, n_points=400):
    mu, r1, r2 = GM_SUN, A_EARTH, A_MARS
    vr, vt = departure_velocity(v_inf, theta)
    v_scale = np.sqrt(mu / r1)

    reach = event(lambda t, s, *a: np.hypot(s[0], s[1]) - r2, +1)
    apo = event(lambda t, s, *a: s[0] * s[2] + s[1] * s[3], -1)
    t, Y, hit = integrate(sun_rhs, 0.0, [r1, 0.0, vr, vt], (mu,), [reach, apo],
                          2 * hohmann(r1, r2, mu)["T"], [r1, r1, v_scale, v_scale], n_points)
    x, y, vx, vy = Y

    phi = np.unwrap(np.arctan2(y, x))
    T = t[-1]
    v_mars = np.sqrt(mu / r2) * np.array([-np.sin(phi[-1]), np.cos(phi[-1])])
    v_rel = np.array([vx[-1], vy[-1]]) - v_mars
    return dict(t=t, x=x, y=y, vx=vx, vy=vy, T=T, phi_arr=phi[-1],
                miss=r2 - np.hypot(x[-1], y[-1]), v_rel=float(np.hypot(*v_rel)),
                phi_mars0=phi[-1] - T * np.sqrt(mu / r2**3), phi_earth1=T * np.sqrt(mu / r1**3),
                v_start=np.hypot(vr, vt), v_arr=float(np.hypot(vx[-1], vy[-1])),
                stopped_by=["пересечение орбиты Марса", "афелий на орбите Марса"][hit])


def takeoff_energy_error(res, gm=GM_EARTH, R=R_EARTH):
    E = res["v"]**2 / 2 - gm / res["r"]
    E_th = -gm / R + res["a_t"] * (np.minimum(res["r"], res["r_burn"]) - R)
    return np.abs(E - E_th) / (gm / R)


def transfer_invariant_errors(res, mu=GM_SUN):
    E = (res["vx"]**2 + res["vy"]**2) / 2 - mu / np.hypot(res["x"], res["y"])
    L = res["x"] * res["vy"] - res["y"] * res["vx"]
    return np.abs(E - E[0]) / abs(E[0]), np.abs(L - L[0]) / abs(L[0])


def landing_energy_error(res, gm=GM_MARS, R=R_MARS):
    E = res["v"]**2 / 2 - gm / res["r"]
    E_th = res["e0"] - res["a_t"] * np.maximum(res["r_ign"] - res["r"], 0.0)
    return np.abs(E - E_th) / (gm / R)
