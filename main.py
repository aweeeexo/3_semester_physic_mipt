"""Задача M11: охлаждение радиатора. Локальный сайт + консольная проверка метода.

    python main.py            - запустить сайт на http://127.0.0.1:8011 (откроется сам)
    python main.py --check    - проверить метод на геометриях с точным ответом (в консоли)
    python main.py --port 9000
"""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import physics as ph

WEB = Path(__file__).with_name("web")

ROD_LEN = 161           # длина стержня (клеток) в проверке 1
ROD_N0 = 200_000        # сколько порций кладём в центр стержня
PLATE = (40, 3)         # размер пластины (клеток) в проверке 2
PLATE_DT0 = 50.0        # начальный перегрев пластины, К

DEFAULTS = dict(
    mode="radiator", material="al", h_mm=2.0, depth_mm=80.0, q=10,
    n_fins=8, fin_h_mm=40.0, fin_t_mm=2.0, gap_mm=8.0, base_mm=6.0, cpu_mm=30.0,
)
DEFAULT_ENV = dict(power=15.0, alpha=7.0, t_air=25.0, cpu_on=True)


class World:
    """Одна симуляция на весь сервер + то, что нужно для ответа сайту."""

    def __init__(self):
        self.lock = threading.Lock()
        self.env = dict(DEFAULT_ENV)
        self.reset(DEFAULTS)

    def reset(self, params):
        p = {**DEFAULTS, **(params or {})}
        h = float(p["h_mm"]) / 1000
        cells_of = lambda mm: max(1, int(round(float(mm) / p["h_mm"])))
        self.mode = p["mode"]
        if self.mode == "rod":
            cells, cpu = ph.build_rod(ROD_LEN)
        elif self.mode == "plate":
            cells, cpu = ph.build_plate(*PLATE)
        else:
            self.mode = "radiator"
            cells, cpu = ph.build_radiator(int(p["n_fins"]), cells_of(p["fin_h_mm"]) if float(p["fin_h_mm"]) > 0 else 0,
                                           cells_of(p["fin_t_mm"]), cells_of(p["gap_mm"]),
                                           cells_of(p["base_mm"]), cells_of(p["cpu_mm"]))
        self.sim = ph.Simulation(cells, cpu, p["material"], h, float(p["depth_mm"]) / 1000, int(p["q"]))
        self.params = p
        self.apply_env(self.env)
        if self.mode == "rod":
            self.sim.n[len(self.sim.n) // 2] = ROD_N0
            self.sim.q_in = ROD_N0
        elif self.mode == "plate":
            self.sim.heat_all(PLATE_DT0)
        self.view = self.sim.n.astype(float)
        self.p_in = self.p_out = 0.0

    def apply_env(self, env):
        self.env.update(env or {})
        self.sim.alpha = float(self.env["alpha"])
        self.sim.power = float(self.env["power"]) if self.env["cpu_on"] else 0.0

    def step(self, k):
        s = self.sim
        q_in, q_out, t0 = s.q_in, s.q_out, s.t
        self.view = s.step(k)
        if s.t > t0:
            self.p_in = (s.q_in - q_in) * s.eps / (s.t - t0)
            self.p_out = (s.q_out - q_out) * s.eps / (s.t - t0)

    def state(self):
        s, t_air = self.sim, float(self.env["t_air"])
        est = s.estimates()
        if self.mode == "rod":
            values, unit = s.grid(self.view), "порций"
        else:
            values, unit = t_air + s.grid(self.view) / s.q, "°C"
        excess = s.n / s.q   # числа - по текущему моменту; картинка (values) - среднее за кадр, чтобы меньше мерцала
        cpu_T = t_air + float(excess[s.cpu_k].mean()) if len(s.cpu_k) else None
        out = dict(
            mode=self.mode, rows=s.cells.shape[0], cols=s.cells.shape[1],
            cells="".join(map(str, s.cells.ravel().tolist())),
            values=np.round(values, 2).ravel().tolist(), unit=unit,
            cpu=np.flatnonzero(s.cpu_mask.ravel()).tolist(),
            t=s.t, steps=s.steps, dt=s.dt, h=s.h, eps=s.eps, q=s.q, t_air=t_air,
            s_escape=ph.escape_probability(s.alpha, s.h, s.mat), a=ph.diffusivity(s.mat),
            material=s.mat["name"], n_cells=len(s.n),
            T_cpu=cpu_T,
            T_mean=t_air + float(excess.mean()) if len(s.n) else t_air,
            T_max=t_air + float(excess.max()) if len(s.n) else t_air,
            P_in=self.p_in, P_out=self.p_out,
            E_in=s.q_in * s.eps, E_out=s.q_out * s.eps, E_stored=int(s.n.sum()) * s.eps,
            est=dict(S=est["S"], C=est["C"], mass=est["mass"], tau=est["tau"],
                     T_simple=t_air + est["dT_simple"] if s.power > 0 else None),
            env=self.env,
        )
        if self.mode == "rod" and s.n.sum() > 0:
            j = np.arange(len(s.n)) - len(s.n) // 2
            x = j * s.h
            out["check"] = dict(
                j=j.tolist(), sim=s.n.tolist(), theory=np.round(ph.rod_theory(j, s.steps, ROD_N0), 2).tolist(),
                sigma2=float((s.n * x ** 2).sum() / s.n.sum()), sigma2_theory=ph.rod_sigma2(s.t, s.mat),
                done=bool(np.sqrt(s.steps / 4) > ROD_LEN / 8),
            )
        elif self.mode == "plate":
            out["check"] = dict(dT=float(excess.mean()) if len(s.n) else 0.0,
                                dT_theory=float(ph.plate_theory(s.t, PLATE_DT0, est["tau"])), tau=est["tau"])
        return out


WORLD = World()


def finite(obj):
    """Заменяет бесконечности и NaN на None: JSON (и браузер) их не понимает."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(finite(obj)).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with WORLD.lock:
                self._json(WORLD.state())
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
        w = WORLD
        with w.lock:
            if "env" in data:
                w.apply_env(data["env"])
            if self.path == "/api/reset":
                w.reset(data.get("params"))
            elif self.path == "/api/step":
                w.step(max(0, min(int(data.get("steps", 1)), 20000)))
            elif self.path == "/api/poke":
                w.sim.poke(int(data["row"]), int(data["col"]), data["tool"],
                           int(data.get("radius", 1)), float(data.get("amount", 15)))
                w.view = w.sim.n.astype(float)
            elif self.path == "/api/heat_all":
                w.sim.heat_all(float(data.get("temp", 80)) - float(w.env["t_air"]))
                w.view = w.sim.n.astype(float)
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._json(w.state())


# ---------------------------------------------------------------- проверка метода в консоли

def check():
    h = 0.002
    print("Проверка 1: тепловое пятно в теплоизолированном стержне (алюминий, h = 2 мм)")
    print(f"{'шагов':>7}{'t, с':>9}{'sigma^2 числ, мм^2':>21}{'2at, мм^2':>12}{'ошибка':>9}{'макс. откл. профиля':>22}")
    cells, cpu = ph.build_rod(ROD_LEN)
    sim = ph.Simulation(cells, cpu, "al", h, 0.08, 10, seed=1)
    sim.n[ROD_LEN // 2] = ROD_N0
    j = np.arange(ROD_LEN) - ROD_LEN // 2
    done = 0
    for k in (100, 200, 400, 800):
        sim.step(k - done)
        done = k
        s2 = (sim.n * (j * h) ** 2).sum() / sim.n.sum()
        s2t = ph.rod_sigma2(sim.t, sim.mat)
        th = ph.rod_theory(j, k, ROD_N0)
        dev = np.abs(sim.n - th).max() / th.max()
        print(f"{k:>7}{sim.t:>9.3f}{s2 * 1e6:>21.4f}{s2t * 1e6:>12.4f}{abs(s2 / s2t - 1):>9.2%}{dev:>22.2%}")

    print("\nПроверка 2: остывание тонкой пластины 80 x 6 мм в воздухе (alpha = 50 Вт/(м^2 К))")
    cells, cpu = ph.build_plate(*PLATE)
    sim = ph.Simulation(cells, cpu, "al", h, 0.08, 10, seed=2)
    sim.alpha = 50.0
    sim.heat_all(PLATE_DT0)
    tau = sim.estimates()["tau"]
    print(f"tau = C/(alpha S) = {tau:.2f} с")
    print(f"{'t/tau':>7}{'t, с':>9}{'dT числ, К':>13}{'dT теор, К':>13}{'ошибка':>9}")
    for m in (0.5, 1, 2, 3):
        sim.step(int(round(m * tau / sim.dt)) - sim.steps)
        dT = sim.n.mean() / sim.q
        dTt = ph.plate_theory(sim.t, PLATE_DT0, tau)
        print(f"{m:>7}{sim.t:>9.1f}{dT:>13.3f}{dTt:>13.3f}{abs(dT / dTt - 1):>9.2%}")

    print("\nПроверка 3: баланс энергии в радиаторе (пришло - ушло = запасено, точно)")
    cells, cpu = ph.build_radiator(8, 20, 1, 4, 3, 15)
    sim = ph.Simulation(cells, cpu, "al", h, 0.08, 10, seed=3)
    sim.power, sim.alpha = 15.0, 7.0
    sim.step(5000)
    print(f"пришло {sim.q_in} порций, ушло {sim.q_out}, лежит в металле {sim.n.sum()} "
          f"-> {'сходится' if sim.q_in - sim.q_out == sim.n.sum() else 'НЕ сходится'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="проверка метода в консоли")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Сайт: {url}   (остановить: Ctrl+C)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
