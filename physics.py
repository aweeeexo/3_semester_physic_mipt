"""Задача M11: охлаждение радиатора методом случайных блужданий порций энергии.

Идея модели:
  * радиатор (его поперечный разрез) разбит на квадратные клетки со стороной h;
  * тепло хранится в клетках в виде одинаковых порций энергии ("квантов") размером eps;
  * за один шаг времени dt каждая порция подбрасывает монетку: с вероятностью 1/2 сидит на месте,
    с вероятностью 1/2 прыгает к одному из 4 соседей (каждый с вероятностью 1/8);
  * если сосед - тоже металл, порция туда переходит;
    если сосед - воздух, порция с вероятностью s уходит в воздух, иначе отскакивает (остаётся);
    если сосед - плата (теплоизолятор), порция просто отскакивает;
  * процессор каждый шаг подкидывает в свои клетки P*dt/eps новых порций.

Температура клетки: T = T_воздуха + (число порций) / q, где q - "порций на 1 кельвин в клетке".
Связь с физикой (выводится в SOLUTION.md):
  dt = h^2 / (8a),  a = lam / (c*rho)      - чтобы случайное блуждание давало правильную теплопроводность
  s  = alpha * h / lam                     - чтобы через поверхность уходил правильный поток alpha*S*dT
  eps = c*rho*h^2*D / q                    - энергия одной порции (D - длина радиатора вдоль рёбер)
"""
import numpy as np

AIR, METAL, BOARD = 0, 1, 2              # типы клеток: воздух, металл, плата (теплоизолятор)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # вверх, вниз, влево, вправо (строка, столбец)
TO_AIR, TO_WALL = -1, -2                 # "номер соседа", если сосед не металл

MATERIALS = {
    "al": dict(name="Алюминий", lam=200.0, c=900.0, rho=2700.0),
    "cu": dict(name="Медь", lam=390.0, c=385.0, rho=8900.0),
    "steel": dict(name="Сталь", lam=50.0, c=500.0, rho=7800.0),
}


# ---------------------------------------------------------------- связь блужданий с физикой

def diffusivity(mat):
    """Температуропроводность a = lam / (c*rho), м^2/с."""
    return mat["lam"] / (mat["c"] * mat["rho"])


def time_step(h, mat):
    """Сколько реальных секунд длится один шаг блуждания: dt = h^2 / (8a)."""
    return h * h / (8 * diffusivity(mat))


def escape_probability(alpha, h, mat):
    """Вероятность, что порция, толкнувшаяся в воздух, уйдёт: s = alpha*h/lam (не больше 1)."""
    return min(1.0, alpha * h / mat["lam"])


def cell_capacity(h, depth, mat):
    """Теплоёмкость одной клетки C = c*rho*h*h*D, Дж/К."""
    return mat["c"] * mat["rho"] * h * h * depth


# ---------------------------------------------------------------- геометрии (всё в клетках)

def build_radiator(n_fins, fin_h, fin_t, gap, base_t, cpu_w, margin=4):
    """Радиатор "гребёнка": основание + n_fins вертикальных рёбер, снизу плата, на основании процессор.

    Возвращает (cells, cpu): cells - двумерный массив типов клеток, cpu - маска клеток процессора.
    """
    n_fins, fin_h, fin_t, gap, base_t = max(1, n_fins), max(0, fin_h), max(1, fin_t), max(1, gap), max(1, base_t)
    width = n_fins * fin_t + (n_fins - 1) * gap
    rows = margin + fin_h + base_t + 1
    cols = width + 2 * margin
    cells = np.full((rows, cols), AIR, dtype=np.int8)
    base_top = margin + fin_h
    cells[base_top:base_top + base_t, margin:margin + width] = METAL
    for k in range(n_fins):
        c0 = margin + k * (fin_t + gap)
        cells[margin:base_top, c0:c0 + fin_t] = METAL
    cells[-1, :] = BOARD
    cpu = np.zeros_like(cells, dtype=bool)
    w = max(1, min(cpu_w, width))
    c0 = margin + (width - w) // 2
    cpu[base_top + base_t - 1, c0:c0 + w] = True
    return cells, cpu


def build_rod(length):
    """Проверка 1: стержень в один ряд клеток, со всех сторон теплоизоляция (плата)."""
    cells = np.full((3, length + 2), BOARD, dtype=np.int8)
    cells[1, 1:-1] = METAL
    return cells, np.zeros_like(cells, dtype=bool)


def build_plate(length, thick, margin=4):
    """Проверка 2: тонкая пластина length x thick клеток, со всех сторон воздух."""
    cells = np.full((thick + 2 * margin, length + 2 * margin), AIR, dtype=np.int8)
    cells[margin:margin + thick, margin:margin + length] = METAL
    return cells, np.zeros_like(cells, dtype=bool)


# ---------------------------------------------------------------- моделирование

class Simulation:
    """Случайное блуждание порций энергии по клеткам металла."""

    def __init__(self, cells, cpu, material="al", h=0.002, depth=0.08, q=10, seed=None):
        self.cells = np.array(cells, dtype=np.int8)
        self.cpu_mask = np.array(cpu, dtype=bool)
        self.material = material
        self.mat = MATERIALS[material]
        self.h, self.depth, self.q = h, depth, q
        self.dt = time_step(h, self.mat)
        self.eps = cell_capacity(h, depth, self.mat) / q
        self.rng = np.random.default_rng(seed)
        self.alpha = 7.0      # коэффициент теплоотдачи в воздух, Вт/(м^2*К)
        self.power = 0.0      # мощность процессора, Вт
        self.steps = 0        # сколько шагов сделано
        self.q_in = 0         # сколько порций пришло (процессор, паяльник)
        self.q_out = 0        # сколько порций ушло (в воздух, лёд, ластик)
        self._index(np.zeros(self.cells.shape, dtype=np.int64))

    @property
    def t(self):
        return self.steps * self.dt

    def _index(self, grid_n):
        """Строит список клеток металла и для каждой - номера соседей по 4 направлениям."""
        H, W = self.cells.shape
        metal = self.cells == METAL
        self.ij = np.argwhere(metal)                 # координаты клеток металла (строка, столбец)
        M = len(self.ij)
        num = np.full((H, W), -1, dtype=np.int64)    # номер клетки в списке
        num[metal] = np.arange(M)
        self.nb = np.empty((4, M), dtype=np.int64)
        for d, (di, dj) in enumerate(DIRS):
            i, j = self.ij[:, 0] + di, self.ij[:, 1] + dj
            inside = (i >= 0) & (i < H) & (j >= 0) & (j < W)
            code = np.full(M, TO_AIR, dtype=np.int64)  # за краем картинки - воздух
            ii, jj = i[inside], j[inside]
            kind = self.cells[ii, jj]
            code[inside] = np.where(kind == METAL, num[ii, jj], np.where(kind == BOARD, TO_WALL, TO_AIR))
            self.nb[d] = code
        self.n = grid_n[metal].astype(np.int64)      # порции в каждой клетке металла
        self.cpu_k = num[self.cpu_mask & metal]      # номера клеток процессора
        self.air_faces = int((self.nb == TO_AIR).sum())

    def grid(self, values=None):
        """Раскладывает значения по клеткам металла обратно в двумерную картинку (вне металла 0)."""
        out = np.zeros(self.cells.shape, dtype=float if values is not None else np.int64)
        out[self.ij[:, 0], self.ij[:, 1]] = self.n if values is None else values
        return out

    def step(self, k=1):
        """Делает k шагов. Возвращает среднее за эти шаги число порций в каждой клетке."""
        M = len(self.n)
        if M == 0 or k <= 0:
            self.steps += max(k, 0)
            return self.n.astype(float)
        s = escape_probability(self.alpha, self.h, self.mat)
        per_cpu = self.power * self.dt / self.eps / len(self.cpu_k) if len(self.cpu_k) else 0.0
        rng = self.rng
        acc = np.zeros(M)
        for _ in range(k):
            n = self.n
            if per_cpu > 0:                                   # процессор греет
                add = rng.poisson(per_cpu, len(self.cpu_k))
                n[self.cpu_k] += add
                self.q_in += int(add.sum())
            jump = rng.binomial(n, 0.5)                       # монетка: кто прыгает
            new = n - jump
            rest = jump
            for d in range(4):                                # куда прыгает: каждый сосед 1/4
                go = rng.binomial(rest, 1.0 / (4 - d)) if d < 3 else rest
                rest = rest - go
                to = self.nb[d]
                ok = to >= 0
                new += np.bincount(to[ok], weights=go[ok], minlength=M).astype(np.int64)
                stuck = np.where(ok, 0, go)                   # упёрлись в воздух или плату
                if s > 0:
                    esc = rng.binomial(np.where(to == TO_AIR, stuck, 0), s)
                    stuck = stuck - esc
                    self.q_out += int(esc.sum())
                new += stuck                                  # отскочили - остались на месте
            self.n = new
            acc += new
            self.steps += 1
        return acc / k

    # ---- взаимодействие (мышка на сайте)

    def _change(self, ks, new_values):
        """Ставит в клетках ks новое число порций и честно учитывает, сколько добавили/убрали."""
        diff = int(np.sum(new_values) - np.sum(self.n[ks]))
        if diff > 0:
            self.q_in += diff
        else:
            self.q_out -= diff
        self.n[ks] = new_values

    def heat_all(self, dT):
        """Нагреть весь металл до перегрева dT над воздухом."""
        ks = np.arange(len(self.n))
        self._change(ks, np.full(len(ks), int(round(dT * self.q)), dtype=np.int64))

    def poke(self, row, col, tool, radius=1, amount=15.0):
        """Действие мышкой в клетке (row, col): heat / cool / metal / erase."""
        H, W = self.cells.shape
        ii, jj = np.mgrid[0:H, 0:W]
        brush = (ii - row) ** 2 + (jj - col) ** 2 <= radius ** 2
        if tool in ("heat", "cool"):
            ks = np.flatnonzero(brush[self.ij[:, 0], self.ij[:, 1]])
            if len(ks):
                new = self.n[ks] + int(round(amount * self.q)) if tool == "heat" else np.zeros(len(ks), np.int64)
                self._change(ks, new)
            return
        grid_n = self.grid()
        if tool == "metal":
            self.cells[brush & (self.cells == AIR)] = METAL
        elif tool == "erase":
            gone = brush & (self.cells == METAL)
            self.q_out += int(grid_n[gone].sum())
            grid_n[gone] = 0
            self.cells[gone] = AIR
        self._index(grid_n)

    # ---- то, что считается "на бумаге"

    def estimates(self):
        """Оценки из закона Ньютона-Рихмана для радиатора, у которого вся температура одинаковая."""
        M = len(self.n)
        S = self.air_faces * self.h * self.depth                  # площадь, отдающая тепло, м^2
        C = M * cell_capacity(self.h, self.depth, self.mat)       # теплоёмкость металла, Дж/К
        mass = M * self.h * self.h * self.depth * self.mat["rho"]
        aS = self.alpha * S
        return dict(S=S, C=C, mass=mass,
                    dT_simple=self.power / aS if aS > 0 else float("inf"),
                    tau=C / aS if aS > 0 else float("inf"))


# ---------------------------------------------------------------- точные ответы для проверки

def rod_theory(j, steps, n0):
    """Проверка 1. Сколько порций в клетке j (от центра) через steps шагов, если в начале n0 в центре.

    Это "колокол" (гауссиана) с шириной sigma^2 = 2at = steps*h^2/4.
    """
    j = np.asarray(j, dtype=float)
    if steps == 0:
        return np.where(j == 0, float(n0), 0.0)
    return n0 / np.sqrt(np.pi * steps / 2) * np.exp(-2 * j ** 2 / steps)


def rod_sigma2(t, mat):
    """Проверка 1. Квадрат ширины пятна sigma^2 = 2at, м^2."""
    return 2 * diffusivity(mat) * t


def plate_theory(t, dT0, tau):
    """Проверка 2. Остывание тонкой пластины: dT = dT0 * exp(-t/tau), tau = C/(alpha*S)."""
    return dT0 * np.exp(-np.asarray(t, dtype=float) / tau)
