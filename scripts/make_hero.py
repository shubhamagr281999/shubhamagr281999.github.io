#!/usr/bin/env python3
"""Generate content/hero-art.svg, the animated full-viewport illustration behind
the home hero. Standard library + numpy only. Run from the repo root:

    python3 scripts/make_hero.py            # writes content/hero-art.svg
    python3 scripts/make_hero.py out.svg    # or anywhere else

Like make_figures.py, nothing is traced or faked; every mark and every keyframe
is computed here:

* The floor plan is rasterised to a 2.5 cm occupancy grid and given an exact
  Euclidean distance transform (Felzenszwalb & Huttenlocher).
* The route is a backward Dijkstra from the goal over a clearance-weighted grid
  (the first search a D* Lite planner makes), relaxed as an elastic band against
  the distance field. The robot drives it out and back on a trapezoidal speed
  profile and turns in place at each end; its keyframes sample that profile.
* Riding on the robot's frame: the distance field as a heat patch, seen through a
  window that moves and turns with it, and a fan of constant-curvature rollouts
  (a dynamic-window sample set); the one that scores best on heading to the route,
  clearance and speed is drawn amber at every keyframe.
* In the circular detail view: a generic 6-DoF arm (textbook layout, round link
  lengths, not any product) in side elevation, doing a planar pick-and-place.
  Shoulder, elbow and wrist angles come from closed-form planar IK at via-poses
  and are interpolated in joint space with the same easing curve CSS uses, so the
  drawn end-effector path and the action ticks sit exactly under the gripper.
* A policy block emits action chunks: k future end-effector waypoints, consumed
  one per control step and re-planned every m steps (a receding horizon).

Everything is one inline SVG: classes, ids, keyframes and custom properties are
prefixed `ha-`, colours come only from the page's theme tokens, and
prefers-reduced-motion stops all motion on a complete, legible frame.
"""

from __future__ import annotations

import heapq
import math
import pathlib
import sys

import numpy as np

OUT = pathlib.Path("content/hero-art.svg")

VB_W, VB_H = 1600, 1000         # viewBox; xMaxYMin slice: right edge and top strip anchored, crop falls at the bottom
PX = 100.0                      # plan scale: svg units per metre

LAYOUT = dict(
    start=(0.6, 2.95), goal=(12.05, 7.2),   # route ends (metres)
    robot_static=0.6,                       # reduced-motion frame: fraction of the route
    loupe=(1364, 334, 220),                 # arm detail view: centre x, y, radius
    model=(1330, 640),                      # policy block: top-left of its frame
)


# =================================================================== world
RES = 0.025                                   # occupancy grid, m / cell
GX0, GY0, GX1, GY1 = -1.5, -1.5, 17.5, 11.5   # grid extent (m), larger than the view


def wall(x0, y0, x1, y1, t=0.12, gaps=()):
    """Axis-aligned wall as rectangles, with door gaps along its axis."""
    a, b = sorted((x0, x1)) if y0 == y1 else sorted((y0, y1))
    cuts = [a] + [v for g in sorted(gaps) for v in g] + [b]
    out = []
    for s, e in zip(cuts[0::2], cuts[1::2]):
        if e - s > 1e-6:
            out.append((s, y0 - t / 2, e, y0 + t / 2) if y0 == y1 else (x0 - t / 2, s, x0 + t / 2, e))
    return out


class Plan:
    def __init__(self, walls, furniture, doors):
        self.walls, self.furniture, self.doors = walls, furniture, doors
        self.nx = int(round((GX1 - GX0) / RES))
        self.ny = int(round((GY1 - GY0) / RES))
        xs = GX0 + (np.arange(self.nx) + 0.5) * RES
        ys = GY0 + (np.arange(self.ny) + 0.5) * RES
        X, Y = np.meshgrid(xs, ys)
        occ = np.zeros((self.ny, self.nx), bool)
        for (x0, y0, x1, y1) in walls:
            occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        for f in furniture:
            if f[0] == "rect":
                occ |= (X >= f[1]) & (X <= f[3]) & (Y >= f[2]) & (Y <= f[4])
            else:
                occ |= (X - f[1]) ** 2 + (Y - f[2]) ** 2 <= f[3] ** 2
        self.occ = occ
        self.dist = edt(occ) * RES
        self.gy, self.gx = np.gradient(self.dist, RES)

    def sample(self, A, X, Y):
        """Vectorised bilinear sample of a grid array at world points."""
        fx = (np.asarray(X) - GX0) / RES - 0.5
        fy = (np.asarray(Y) - GY0) / RES - 0.5
        i0 = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx, ty = np.clip(fx - i0, 0, 1), np.clip(fy - j0, 0, 1)
        return ((1 - tx) * (1 - ty) * A[j0, i0] + tx * (1 - ty) * A[j0, i0 + 1]
                + (1 - tx) * ty * A[j0 + 1, i0] + tx * ty * A[j0 + 1, i0 + 1])

    def d(self, x, y):
        return float(self.sample(self.dist, x, y))


def _edt1d(f):
    """1-D squared distance transform of a sampled function (F&H 2012)."""
    n = len(f)
    d, v, z = [0.0] * n, [0] * n, [0.0] * (n + 1)
    k, z[0], z[1] = 0, -1e30, 1e30
    for q in range(1, n):
        fq = f[q] + q * q
        while True:
            vk = v[k]
            s = (fq - (f[vk] + vk * vk)) / (2 * q - 2 * vk)
            if s <= z[k]:
                k -= 1
                continue
            break
        k += 1
        v[k], z[k], z[k + 1] = q, s, 1e30
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def edt(occ):
    """Exact Euclidean distance (cells) from every cell to the nearest occupied one."""
    f = np.where(occ, 0.0, 1e12)
    g = np.empty_like(f)
    for i in range(f.shape[1]):
        g[:, i] = _edt1d(f[:, i].tolist())
    for j in range(f.shape[0]):
        g[j, :] = _edt1d(g[j, :].tolist())
    return np.sqrt(g)


def plan_route(plan, start, goal, radius, step=0.05):
    """Backward Dijkstra from the goal (D* Lite's initial search) over an
    8-connected grid whose edge cost rises inside 0.9 m of obstacles, then
    steepest descent of the cost-to-go from the start."""
    sub = int(round(step / RES))
    Dg = plan.dist[sub // 2::sub, sub // 2::sub]
    ny, nx = Dg.shape
    free = Dg > radius + 0.12          # inflate: never squeeze through a gap
    pen = np.clip((0.9 - Dg) / 0.9, 0, None) ** 2 * 6.0

    def cell(p):
        return (int((p[1] - GY0) / step), int((p[0] - GX0) / step))

    s, t = cell(start), cell(goal)
    assert free[s] and free[t], "start/goal must be in free space"
    g = np.full((ny, nx), np.inf)
    g[t] = 0.0
    pq = [(0.0, t)]
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while pq:
        c, (j, i) = heapq.heappop(pq)
        if c > g[j, i]:
            continue
        if (j, i) == s:
            break
        for dj, di in nb:
            nj, ni = j + dj, i + di
            if 0 <= nj < ny and 0 <= ni < nx and free[nj, ni]:
                nc = c + math.hypot(dj, di) * (1.0 + 0.5 * (pen[j, i] + pen[nj, ni]))
                if nc < g[nj, ni]:
                    g[nj, ni] = nc
                    heapq.heappush(pq, (nc, (nj, ni)))
    cur, out = s, []
    while True:
        out.append((GX0 + (cur[1] + 0.5) * step, GY0 + (cur[0] + 0.5) * step))
        if cur == t:
            break
        j, i = cur
        cur = min(((j + dj, i + di) for dj, di in nb
                   if 0 <= j + dj < ny and 0 <= i + di < nx), key=lambda c: g[c])
    out[0], out[-1] = tuple(start), tuple(goal)
    return np.array(out)


def resample(P, ds):
    P = np.asarray(P, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    q = np.linspace(0, s[-1], max(2, int(s[-1] / ds) + 1))
    return np.stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])], 1)


def elastic_band(plan, P, keep=0.42, iters=6000, k_s=0.22, k_o=0.35, ds=0.05):
    """Tension pulls the band taut; the distance-field gradient holds it `keep`
    metres off obstacles. Endpoints fixed."""
    P = resample(P, ds)
    for it in range(iters):
        P[1:-1] += k_s * (P[:-2] - 2 * P[1:-1] + P[2:])
        X, Y = P[1:-1, 0], P[1:-1, 1]
        push = np.clip(keep - plan.sample(plan.dist, X, Y), 0, None)
        g = np.stack([plan.sample(plan.gx, X, Y), plan.sample(plan.gy, X, Y)], 1)
        P[1:-1] += k_o * push[:, None] * g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        if it % 250 == 249:
            P = resample(P, ds)
    return P



def build_plan():
    """A generic indoor floor: two rooms along the top, a hall down the right,
    a workshop bottom-right, and one large open room where the headline sits
    (no interior wall crosses it)."""
    walls = (wall(-1.5, 3.5, 9.6, 3.5, gaps=[(6.6, 7.5)])
             + wall(4.6, -1.5, 4.6, 3.5, gaps=[(2.0, 2.9)])
             + wall(9.6, -1.5, 9.6, 11.5, gaps=[(2.1, 3.0)])
             + wall(11.1, -1.5, 11.1, 5.8, gaps=[(3.7, 4.6)])
             + wall(9.6, 5.8, 17.5, 5.8, gaps=[(9.85, 10.85), (14.6, 15.5)]))
    furn = [("rect", .2, .9, 2.2, 1.45), ("circ", 2.0, 2.0, .3), ("rect", 3.6, .9, 4.3, 1.6),
            ("circ", 7.0, 2.65, .45), ("rect", 5.4, .9, 6.6, 1.4), ("rect", 8.7, .9, 9.4, 1.5),
            ("rect", 11.2, 5.86, 12.9, 6.5), ("rect", 15.8, 6.3, 16.3, 9.3),
            ("rect", 9.95, 8.75, 10.65, 9.45), ("rect", 12.25, 8.35, 12.85, 8.95), ("circ", 14.7, 8.9, .25),
            ("rect", 11.8, .9, 13.9, 1.6), ("rect", 14.6, 2.6, 16.6, 3.3)]
    doors = [((7.5, 3.5), (0, -1), (-1, 0)), ((4.6, 2.9), (1, 0), (0, -1)), ((9.6, 3.0), (-1, 0), (0, -1)),
             ((11.1, 4.6), (1, 0), (0, -1)), ((9.85, 5.8), (0, 1), (1, 0)), ((14.6, 5.8), (0, -1), (1, 0))]
    return Plan(walls, furn, doors)



# ==================================================================== draw
def f(v, nd=1):
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return "0" if s in ("-0", "", "-") else s


def nums(vals, nd=1):
    """Join numbers SVG-style, letting '-' and a second '.' act as separators."""
    out, last = "", ""
    for v in vals:
        s = v if isinstance(v, str) else f(v, nd)
        if out and not (s[0] == "-" or (s[0] == "." and "." in last)):
            out += " "
        out += s
        last = s
    return out


def pd(pts, closed=False, nd=1):
    """Polyline path data in relative coordinates."""
    q = [(round(float(x), nd), round(float(y), nd)) for x, y in pts]
    out = "M" + nums(q[0], nd)
    body, (px, py) = [], q[0]
    for x, y in q[1:]:
        dx, dy = round(x - px, nd), round(y - py, nd)
        if dx or dy:
            body += [dx, dy]
            px, py = x, y
    if body:
        out += "l" + nums(body, nd)
    return out + ("z" if closed else "")


def rdp(pts, eps):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = math.hypot(*ab)
    dd = (np.hypot(*(pts - a).T) if n < 1e-12 else
          np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n)
    k = int(np.argmax(dd))
    if dd[k] > eps:
        return np.vstack([rdp(pts[:k + 1], eps)[:-1], rdp(pts[k:], eps)])
    return np.vstack([a, b])


def smooth_d(pts, nd=1):
    """Catmull-Rom through points, emitted as relative cubic Beziers."""
    P = [np.asarray(p, float) for p in pts]
    out = "M" + nums([round(P[0][0], nd), round(P[0][1], nd)], nd) + "c"
    body = []
    for i in range(len(P) - 1):
        p0, p1, p2 = P[max(i - 1, 0)], P[i], P[i + 1]
        p3 = P[min(i + 2, len(P) - 1)]
        c1, c2 = p1 + (p2 - p0) / 6, p2 - (p3 - p1) / 6
        body += [*(c1 - p1), *(c2 - p1), *(p2 - p1)]
    return out + nums([round(v, nd) for v in body], nd)


def ellipse_d(cx, cy, rx, ry, rot):
    t = math.radians(rot)
    ux, uy = rx * math.cos(t), rx * math.sin(t)
    return (f"M{nums([cx + ux, cy + uy])}A{nums([rx, ry, f(rot, 0), '0', '1', cx - ux, cy - uy])}"
            f"A{nums([rx, ry, f(rot, 0), '0', '1', cx + ux, cy + uy])}z")



def _sweep(a, q):
    return "1" if a[0] * q[1] - a[1] * q[0] > 0 else "0"


def rect_d(a, b, c, d):
    return f"M{nums([a, b])}h{f(c - a)}v{f(d - b)}h{f(a - c)}z"


def circle_d(cx, cy, r):
    return (f"M{nums([cx + r, cy])}A{nums([r, r, '0', '0', '1', cx - r, cy])}"
            f"A{nums([r, r, '0', '0', '1', cx + r, cy])}z")


def corner_ticks(x0, y0, x1, y1, c=10):
    return (f"M{nums([x0, y0 + c])}V{f(y0)}H{f(x0 + c)}M{nums([x1 - c, y0])}H{f(x1)}V{f(y0 + c)}"
            f"M{nums([x1, y1 - c])}V{f(y1)}H{f(x1 - c)}M{nums([x0 + c, y1])}H{f(x0)}V{f(y1 - c)}")


def cubic(p0, p1, p2, p3):
    return f"M{nums(p0)}C{nums([*p1, *p2, *p3])}"


def on_circle(cx, cy, r, deg):
    a = math.radians(deg)
    return (cx + r * math.cos(a), cy + r * math.sin(a))


def pct(t, period):
    return f(100.0 * t / period, 2) + "%"


def ease(x1, y1, x2, y2):
    """CSS cubic-bezier(x1, y1, x2, y2) as a function of normalised time."""
    def bx(u):
        return 3 * (1 - u) ** 2 * u * x1 + 3 * (1 - u) * u * u * x2 + u ** 3

    def run(t):
        lo, hi = 0.0, 1.0
        for _ in range(48):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if bx(m) < t else (lo, m)
        u = (lo + hi) / 2
        return 3 * (1 - u) ** 2 * u * y1 + 3 * (1 - u) * u * u * y2 + u ** 3
    return run


EASE_CSS = "cubic-bezier(.45,0,.55,1)"
EASE = ease(.45, 0, .55, 1)


# ================================================================== robot
R_BOT = 0.2
V_MAX, A_MAX = 0.5, 0.4                          # m/s, m/s^2
KF_DS = 0.4                                      # keyframe spacing along the route (m)
T_DWELL, T_TURN = 1.4, 2.6                       # s
FAN_K = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)   # rollout curvatures, 1/m (w = v k)
FAN_LEN = 0.7                                    # rollout length: 1.4 s at 0.5 m/s
WIN = (-0.7, -0.85, 1.5, 0.85)                   # heat window, robot frame (m): x0 y0 x1 y1


def trapezoid_time(s, L, v=V_MAX, a=A_MAX):
    """Time to reach arc length s on a rest-to-rest trapezoidal speed profile."""
    da = v * v / (2 * a)
    ta = v / a
    if s <= da:
        return math.sqrt(2 * s / a)
    if s <= L - da:
        return ta + (s - da) / v
    return ta + (L - 2 * da) / v + ta - math.sqrt(max(0.0, 2 * (L - s) / a))


def fan_local():
    """Constant-curvature rollouts in the robot frame (x ahead, y to its right)."""
    u = np.linspace(0, FAN_LEN, 12)
    return [np.stack([u, 0 * u], 1) if k == 0 else
            np.stack([np.sin(k * u) / k, (1 - np.cos(k * u)) / k], 1) for k in FAN_K]


def choose_arc(plan, x, y, th, target, fan):
    """Dynamic-window choice among the rollouts: reject any that comes within the
    robot radius of an obstacle, then score heading to a lookahead point on the
    route, clearance, and a mild preference for straighter arcs."""
    best, best_score = -1, -1e9
    c, s = math.cos(th), math.sin(th)
    for j, pts in enumerate(fan):
        W = np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)
        d = plan.sample(plan.dist, W[:, 0], W[:, 1])
        if d.min() < R_BOT + 0.04:
            continue
        head = th + FAN_K[j] * FAN_LEN
        err = abs((math.atan2(target[1] - W[-1, 1], target[0] - W[-1, 0]) - head + math.pi)
                  % (2 * math.pi) - math.pi)
        score = -err + 0.3 * min(float(d.min()), 0.8) - 0.04 * abs(FAN_K[j])
        if score > best_score:
            best, best_score = j, score
    return best


def robot_timeline(plan, route):
    """Out along the route, dwell, turn in place, back, dwell, turn back.
    Returns keys [(t, x, y, heading_deg, chosen_arc)] and the loop period."""
    seg = np.hypot(*np.diff(route, axis=0).T)
    sd = np.concatenate([[0], np.cumsum(seg)])
    L = float(sd[-1])

    def at(s):
        s = min(max(s, 0.0), L)
        return float(np.interp(s, sd, route[:, 0])), float(np.interp(s, sd, route[:, 1]))

    def heading(s):
        a, b = at(s - 0.15), at(s + 0.15)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    ss = np.linspace(0, L, max(2, int(round(L / KF_DS))) + 1)
    th = np.unwrap([heading(s) for s in ss])
    fan = fan_local()
    T = trapezoid_time(L, L)
    keys = []

    def turn(t0, x, y, a0, a1, n=6):
        for i in range(1, n + 1):
            u = i / n
            keys.append((t0 + T_TURN * u, x, y, a0 + (a1 - a0) * (3 * u * u - 2 * u ** 3), -1))

    for i, s in enumerate(ss):                                    # out
        x, y = at(s)
        sel = choose_arc(plan, x, y, th[i], at(s + 1.0), fan) if 0 < i < len(ss) - 1 else -1
        keys.append((trapezoid_time(s, L), x, y, th[i], sel))
    t = T + T_DWELL
    gx, gy = at(L)
    keys.append((t, gx, gy, th[-1], -1))
    turn(t, gx, gy, th[-1], th[-1] + math.pi)
    t += T_TURN + T_DWELL
    for i in range(len(ss) - 1, -1, -1):                          # back
        x, y = at(ss[i])
        sel = (choose_arc(plan, x, y, th[i] + math.pi, at(ss[i] - 1.0), fan)
               if 0 < i < len(ss) - 1 else -1)
        keys.append((t + T - trapezoid_time(ss[i], L), x, y, th[i] + math.pi, sel))
    t += T + T_DWELL
    sx, sy = at(0)
    keys.append((t, sx, sy, th[0] + math.pi, -1))
    turn(t, sx, sy, th[0] + math.pi, th[0])
    period = t + T_TURN + T_DWELL
    keys.append((period, sx, sy, th[0], -1))            # explicit 100%: same as 0%
    # a little hysteresis so the chosen arc does not flicker between neighbours
    sel = [k[4] for k in keys]
    for i in range(1, len(sel) - 1):
        if sel[i - 1] == sel[i + 1] != sel[i] and sel[i] != -1 and sel[i - 1] != -1:
            sel[i] = sel[i - 1]
    keys = [(k[0], k[1], k[2], math.degrees(k[3]), s_) for k, s_ in zip(keys, sel)]

    # the reduced-motion frame: part way out along the route
    s0 = LAYOUT["robot_static"] * L
    x0, y0 = at(s0)
    th0 = float(np.interp(s0, ss, th))
    static = (x0, y0, math.degrees(th0), choose_arc(plan, x0, y0, th0, at(s0 + 1.0), fan))
    return keys, period, static, fan


def layer_robot(plan, route, keys, period, static, fan):
    """The robot, its rollout fan and its heat window: one group that drives the
    route. Inside it, a counter-animated group keeps the distance field fixed to
    the floor while the window (a static mask in the robot frame) moves over it."""
    x0, y0, x1, y1 = (v * PX for v in WIN)
    r = R_BOT * PX
    gray = "".join(pd(p * PX) for p in fan)
    amber = "".join(f'<path class="ha-arc ha-c{j}" d="{pd(p * PX)}"/>' for j, p in enumerate(fan))
    world = ('<g class="ha-rbi"><path class="ha-tint" d="M0 0H1600V1000H0z"/>'
             + "".join(f'<use href="#ha-ob" class="ha-hb ha-h{i}" style="stroke-width:{f(2 * d * PX, 0)}"/>'
                       for i, d in enumerate((0.9, 0.55, 0.3, 0.12)))
             + '<path class="ha-cells" d="M0 0H1600V1000H0z"/>'
             + '<use href="#ha-walls" class="ha-wl2"/><use href="#ha-furn" class="ha-fu2"/>'
             + '<use href="#ha-route" class="ha-rt2"/></g>')
    svg = (f'<g class="ha-rb"><g mask="url(#ha-hm)">{world}</g>'
           f'<path class="ha-fan" d="{gray}"/>{amber}'
           f'<rect class="ha-whl" x="-7" y="{f(-r - 3)}" width="14" height="6"/>'
           f'<rect class="ha-whl" x="-7" y="{f(r - 3)}" width="14" height="6"/>'
           f'<circle class="ha-body" r="{f(r)}"/><circle class="ha-lid" r="3.2"/>'
           f'<path class="ha-head" d="M0 0H{f(r)}"/></g>')
    mask = (f'<mask id="ha-hm" class="ha-mask" maskUnits="userSpaceOnUse" x="{f(x0)}" y="{f(y0)}" '
            f'width="{f(x1 - x0)}" height="{f(y1 - y0)}">'
            f'<rect x="{f(x0)}" y="{f(y0)}" width="{f(x1 - x0)}" height="{f(y1 - y0)}" fill="url(#ha-hg)"/></mask>'
            '<radialGradient id="ha-hg"><stop offset=".45" class="ha-m0"/><stop offset="1" class="ha-m1"/></radialGradient>')
    # keyframes: the robot frame, its exact inverse, and which rollout is chosen
    a, b = [], []
    for (t, x, y, d, _) in keys:
        p = pct(t, period)
        a.append(f"{p}{{transform:translate({f(x * PX)}px,{f(y * PX)}px) rotate({f(d)}deg)}}")
        b.append(f"{p}{{transform:rotate({f(-d)}deg) translate({f(-x * PX)}px,{f(-y * PX)}px)}}")
    css = f"@keyframes ha-rb{{{''.join(a)}}}@keyframes ha-rbi{{{''.join(b)}}}"
    for j in range(len(fan)):
        on = [1 if k[4] == j else 0 for k in keys]
        stops = []
        for i, v in enumerate(on):
            if i == 0 or i == len(on) - 1 or on[i - 1] != v or on[i + 1] != v:
                stops.append(f"{pct(keys[i][0], period)}{{opacity:{v}}}")
        css += f"@keyframes ha-c{j}{{{''.join(stops)}}}.ha-c{j}{{animation:ha-c{j} {f(period, 2)}s linear infinite}}"
    sx, sy, sd, ssel = static
    css += (f".ha-rb{{transform:translate({f(sx * PX)}px,{f(sy * PX)}px) rotate({f(sd)}deg);"
            f"animation:ha-rb {f(period, 2)}s linear infinite}}"
            f".ha-rbi{{transform:rotate({f(-sd)}deg) translate({f(-sx * PX)}px,{f(-sy * PX)}px);"
            f"animation:ha-rbi {f(period, 2)}s linear infinite}}"
            f".ha-arc{{opacity:0}}.ha-c{ssel}{{opacity:1}}")
    return svg, mask, css


# ==================================================================== arm
# A generic 6-DoF arm seen side-on, doing a planar pick-and-place: base yaw and
# both wrist rolls stay put, so shoulder, elbow and wrist pitch are the whole
# motion and the drawing is an exact nested-rotation rig.
SZ, L1, L2 = 0.34, 0.42, 0.40            # shoulder height, upper arm, forearm (m)
FLANGE, PALM, FING = 0.09, 0.04, 0.085   # wrist centre -> flange, palm depth, finger length
L3 = FLANGE + PALM + FING - 0.01         # wrist centre -> TCP, 1 cm inside the fingertips
FING_T = 0.016
CUBE = 0.07
A_X, B_X, Z_UP, Z_GRASP = 0.66, 0.38, 0.21, CUBE / 2 + 0.01
HOME = (0.48, 0.42)
G_OPEN, G_SHUT = 0.11, CUBE
DT, M_STEP, K_CHUNK = 0.55, 5, 8         # control period (s), re-plan every m, horizon k


def ik_planar(x, z, psi):
    """Closed-form planar IK, elbow up. Returns absolute link angles (rad)."""
    wx, wz = x - L3 * math.cos(psi), z - L3 * math.sin(psi)
    dx, dz = wx, wz - SZ
    c = max(-1.0, min(1.0, (dx * dx + dz * dz - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for b in (math.acos(c), -math.acos(c)):
        a1 = math.atan2(dz, dx) - math.atan2(L2 * math.sin(b), L1 + L2 * math.cos(b))
        sols.append((SZ + L1 * math.sin(a1), a1, a1 + b))
    _, p1, p2 = max(sols)
    return np.array([p1, p2 - p1, psi - p2])          # shoulder, elbow, wrist (relative)


def fk_planar(q):
    p1, p2 = q[0], q[0] + q[1]
    p3 = p2 + q[2]
    S = np.array([0.0, SZ])
    E = S + L1 * np.array([math.cos(p1), math.sin(p1)])
    W = E + L2 * np.array([math.cos(p2), math.sin(p2)])
    T = W + L3 * np.array([math.cos(p3), math.sin(p3)])
    return S, E, W, T


def arm_program():
    """Via-poses for A->B then B->A (the cube ends where it began, so the loop is
    seamless). Each entry: (duration, tcp (x, z) or None, finger gap)."""
    down = -math.pi / 2

    def phase(src, dst):
        return [(0.6, HOME, G_OPEN), (1.8, (src, Z_UP), G_OPEN), (1.2, (src, Z_GRASP), G_OPEN),
                (0.6, (src, Z_GRASP), G_SHUT), (0.3, (src, Z_GRASP), G_SHUT),
                (1.0, (src, Z_UP), G_SHUT), (1.6, (dst, Z_UP), G_SHUT), (1.0, (dst, Z_GRASP), G_SHUT),
                (0.6, (dst, Z_GRASP), G_OPEN), (0.3, (dst, Z_GRASP), G_OPEN),
                (0.8, (dst, Z_UP), G_OPEN), (1.2, HOME, G_OPEN)]
    prog = phase(A_X, B_X) + phase(B_X, A_X)
    keys, t = [(0.0, ik_planar(*HOME, down), G_OPEN)], 0.0
    for dur, tcp, gap in prog:
        t += dur
        keys.append((t, ik_planar(tcp[0], tcp[1], down), gap))
    return keys, t


def arm_state(keys, t):
    """Joint angles and gap at time t, eased exactly as the CSS animation is."""
    period = keys[-1][0]
    t %= period
    for (t0, q0, g0), (t1, q1, g1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = EASE((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return q0 + (q1 - q0) * u, g0 + (g1 - g0) * u
    return keys[-1][1], keys[-1][2]


def layer_arm(cx, cy, R):
    keys, period = arm_program()
    ts = np.linspace(0, period, 900)
    tcp = np.array([fk_planar(arm_state(keys, t)[0])[3] for t in ts])
    pts = [p for t in ts[::6] for p in fk_planar(arm_state(keys, t)[0])]
    pts += [np.array([x, z]) for x in (-0.12, 0.12) for z in (-0.02, 0.0)]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    ctr = (lo + hi) / 2
    s = 0.95 * R / np.max(np.hypot(*(pts - ctr).T))
    BX, BY = cx + s * ctr[0], cy + s * ctr[1] + 0.03 * R

    def img(p):
        return np.array([BX - s * p[0], BY - s * p[1]])

    q_ref, g_ref = keys[0][1], keys[0][2]
    S, E, W, T = (img(p) for p in fk_planar(q_ref))
    out = [f'<g class="ha-lp"><circle class="ha-lbg" cx="{f(cx)}" cy="{f(cy)}" r="{f(R)}"/>'
           '<g clip-path="url(#ha-lc)">']
    # elevation grid at 0.1 m, and the bench in section
    g = []
    for k in range(-12, 13):
        x = BX - s * 0.1 * k
        if abs(x - cx) < R:
            g.append(f"M{f(x)} {f(cy - R)}V{f(BY)}")
    for k in range(0, 16):
        y = BY - s * 0.1 * k
        if abs(y - cy) < R:
            g.append(f"M{f(cx - R)} {f(y)}H{f(cx + R)}")
    out.append(f'<path class="ha-eg" d="{"".join(g)}"/>')
    out.append(f'<path class="ha-bench" d="{rect_d(cx - R, BY, cx + R, BY + 0.05 * s)}"/>'
               f'<path class="ha-benchl" d="M{f(cx - R)} {f(BY)}H{f(cx + R)}"/>')
    # where the cube sits, both ends
    c = CUBE * s
    for X in (A_X, B_X):
        m = img((X, 0))
        out.append(f'<path class="ha-spot" d="M{f(m[0] - c * .8)} {f(m[1] + 4)}h{f(c * 1.6)}"/>')
    # the end-effector path over the whole loop, faint
    out.append(f'<path class="ha-ee" d="{pd(rdp(np.array([img(p) for p in tcp]), 0.35))}"/>')
    # action ticks: the waypoint each control step commands (receding horizon)
    t_static = keys[4][0]                      # fingers just closed on the cube at A
    ticks, tick_css = action_ticks(keys, period, img, t_static)
    out.append(ticks)
    # the arm at its home pose; each group turns about its joint. The same rig is
    # drawn twice: a pale copy offset up and right gives the links some depth.
    def capsule2(a, b, r):
        v = (b - a) / np.linalg.norm(b - a)
        n = np.array([-v[1], v[0]]) * r
        return (f"M{nums(a + n)}L{nums(b + n)}A{nums([r, r, '0', '0', _sweep(n, v), *(b - n)])}"
                f"L{nums(a - n)}A{nums([r, r, '0', '0', _sweep(-n, -v), *(a + n)])}z")

    def quad(ctr_, u, v):
        return pd([ctr_ - u - v, ctr_ + u - v, ctr_ + u + v, ctr_ - u + v], closed=True)

    t_dir = (T - W) / np.linalg.norm(T - W)            # tool axis (image), points down at home
    c_dir = np.array([-t_dir[1], t_dir[0]])            # finger closing direction
    base = img((0, 0))
    cube_c = T + t_dir * 0.01 * s

    def cube_d(m):
        return rect_d(m[0] - c / 2, m[1] - c / 2, m[0] + c / 2, m[1] + c / 2)

    def rig(front):
        def housing(p, r):
            d = f'<path class="ha-hs" d="{circle_d(p[0], p[1], r)}"/>'
            if front:
                d += (f'<path class="ha-hi" d="{circle_d(p[0], p[1], r * .52)}"/>'
                      f'<circle class="ha-hc" cx="{f(p[0])}" cy="{f(p[1])}" r="1.6"/>')
            return d

        def cl(a, b):
            return f'<path class="ha-cl" d="M{nums(a)}L{nums(b)}"/>' if front else ""
        fingers = []
        for name, sgn in (("ha-fa", 1), ("ha-fb", -1)):
            fc = W + t_dir * (FLANGE + PALM + FING / 2) * s + c_dir * sgn * (g_ref / 2 + FING_T / 2) * s
            fingers.append(f'<path class="ha-arm {name}" d="{quad(fc, c_dir * FING_T / 2 * s, t_dir * FING / 2 * s)}"/>')
        tool = (quad(W + t_dir * FLANGE / 2 * s, c_dir * .032 * s, t_dir * FLANGE / 2 * s)
                + quad(W + t_dir * (FLANGE + PALM / 2) * s, c_dir * .068 * s, t_dir * PALM / 2 * s))
        col_top = S[1]
        seam = base[1] - .13 * s
        body = (f'<path class="ha-arm" d="{rect_d(base[0] - .12 * s, base[1] - .035 * s, base[0] + .12 * s, base[1])}'
                f'{rect_d(base[0] - .075 * s, col_top, base[0] + .075 * s, base[1] - .035 * s)}"/>'
                + (f'<path class="ha-cl" d="M{nums([base[0] - .075 * s, seam])}H{f(base[0] + .075 * s)}"/>' if front else "")
                + f'<g class="ha-j1"><path class="ha-arm" d="{capsule2(S, E, .056 * s)}"/>{cl(S, E)}{housing(S, .088 * s)}'
                f'<g class="ha-j2"><path class="ha-arm" d="{capsule2(E, W, .046 * s)}"/>{cl(E, W)}{housing(E, .07 * s)}'
                f'<g class="ha-j3"><path class="ha-cube ha-kc" d="{cube_d(cube_c)}"/>'
                f'<path class="ha-arm" d="{tool}"/>{"".join(fingers)}{housing(W, .052 * s)}</g></g></g>')
        return body
    cubes = "".join(f'<path class="ha-cube {name}" d="{cube_d(img((X, CUBE / 2)))}"/>'
                    for name, X in (("ha-ka", A_X), ("ha-kb", B_X)))
    out.append(f'<g class="ha-back" transform="translate(6 -5)">{cubes}{rig(False)}</g>{cubes}{rig(True)}')
    out.append('</g>')
    out.append(f'<circle class="ha-loupe" cx="{f(cx)}" cy="{f(cy)}" r="{f(R)}"/>')
    tk = []
    for deg in range(0, 360, 5):
        l = 9 if deg % 30 == 0 else 4
        tk.append(pd([on_circle(cx, cy, R, deg), on_circle(cx, cy, R - l, deg)]))
    out.append(f'<path class="ha-rim" d="{"".join(tk)}"/></g>')

    # --- keyframes: joints (eased like the program), fingers, cubes
    css = []
    origin = {"ha-j1": S, "ha-j2": E, "ha-j3": W}
    for j, name in enumerate(("ha-j1", "ha-j2", "ha-j3")):
        stops = "".join(f"{pct(t, period)}{{transform:rotate({f(math.degrees(q[j] - q_ref[j]), 2)}deg)}}"
                        for t, q, _ in keys)
        css.append(f"@keyframes {name}{{{stops}}}")
        o = origin[name]
        css.append(f".{name}{{transform-box:view-box;transform-origin:{f(o[0])}px {f(o[1])}px;"
                   f"animation:{name} {f(period, 2)}s {EASE_CSS} infinite}}")
    for name, sgn in (("ha-fa", 1), ("ha-fb", -1)):
        stops = []
        for t, _, gap in keys:
            d = sgn * (gap - g_ref) / 2 * s
            stops.append(f"{pct(t, period)}{{transform:translate({f(c_dir[0] * d, 2)}px,{f(c_dir[1] * d, 2)}px)}}")
        css.append(f"@keyframes {name}{{{''.join(stops)}}}"
                   f".{name}{{animation:{name} {f(period, 2)}s {EASE_CSS} infinite}}")
    # which cube is visible: resting at A, carried, resting at B
    tk_ = [k[0] for k in keys]
    grab_a, drop_b = tk_[4], tk_[8]               # end of closing at A, start of opening at B
    grab_b, drop_a = tk_[16], tk_[20]
    e = 0.02

    def steps(name, segs):
        st = []
        for (t0, t1, v) in segs:
            st.append(f"{pct(t0, period)},{pct(t1, period)}{{opacity:{v}}}")
        return f"@keyframes {name}{{{''.join(st)}}}.{name}{{animation:{name} {f(period, 2)}s linear infinite}}"
    css.append(steps("ha-ka", [(0, grab_a, 1), (grab_a + e, drop_a - e, 0), (drop_a, period, 1)]))
    css.append(steps("ha-kb", [(0, drop_b - e, 0), (drop_b, grab_b, 1), (grab_b + e, period, 0)]))
    css.append(steps("ha-kc", [(0, grab_a - e, 0), (grab_a, drop_b, 1), (drop_b + e, grab_b - e, 0),
                               (grab_b, drop_a, 1), (drop_a + e, period, 0)]))
    # the reduced-motion frame
    qs, gs = arm_state(keys, t_static)
    for j, name in enumerate(("ha-j1", "ha-j2", "ha-j3")):
        css.append(f".{name}{{transform:rotate({f(math.degrees(qs[j] - q_ref[j]), 2)}deg)}}")
    for name, sgn in (("ha-fa", 1), ("ha-fb", -1)):
        d = sgn * (gs - g_ref) / 2 * s
        css.append(f".{name}{{transform:translate({f(c_dir[0] * d, 2)}px,{f(c_dir[1] * d, 2)}px)}}")
    css.append(".ha-kb,.ha-kc{opacity:0}")
    css.append(tick_css)
    return "".join(out), "".join(css), period, t_static


def action_ticks(keys, period, img, t_static):
    """Waypoint w (reached at w*DT) is first predicted by the earliest chunk that
    covers it (a chunk starts every m steps and spans k steps), flickers when a
    later chunk re-predicts it, and is consumed when the gripper gets there."""
    n = int(round(period / DT))
    T_inf = M_STEP * DT
    els, kf = [], {}
    for w in range(n):
        tw = w * DT
        j0 = math.ceil((w - K_CHUNK) / M_STEP)
        t_first = j0 * T_inf
        refresh = [j * T_inf for j in range(j0 + 1, j0 + 64) if j * T_inf < tw - 1e-9]
        phase = w % M_STEP
        q = img(fk_planar(arm_state(keys, tw)[0])[3])
        lag = (w // M_STEP) * T_inf
        live = any(t_first <= t_static + o < tw for o in (0, period, -period))
        style = f"animation-delay:{f(lag - period, 2)}s" if lag else ""
        if live:
            style += (";" if style else "") + "opacity:1"
        els.append(f'<rect class="ha-tk ha-p{phase}" x="{f(q[0] - 2.8)}" y="{f(q[1] - 2.8)}" width="5.6" '
                   f'height="5.6"' + (f' style="{style}"' if style else "") + "/>")
        if phase not in kf:
            st = {}

            def put(t, v):
                st[round((t % period) / period * 100, 2)] = v
            put(t_first, 0)
            put(t_first + 0.25, 1)
            for r in refresh:
                put(r, 1)
                put(r + 0.12, 0.3)
                put(r + 0.32, 1)
            put(tw, 1)
            put(tw + 0.4, 0)
            items = sorted(st.items())
            wrap = items[-1][1] if items[0][0] > 0 else items[0][1]
            body = "".join(f"{f(k_, 2)}%{{opacity:{v}}}" for k_, v in items)
            if items[0][0] > 0:
                body = f"0%{{opacity:{wrap}}}" + body
            if items[-1][0] < 100:
                body += f"100%{{opacity:{wrap}}}"
            kf[phase] = body
    css = ".ha-tk{opacity:0}" + "".join(
        f"@keyframes ha-p{p}{{{b}}}.ha-p{p}{{animation:ha-p{p} {f(period, 2)}s linear infinite}}"
        for p, b in sorted(kf.items()))
    return f'<g class="ha-tks">{"".join(els)}</g>', css


# ================================================================ policy
def layer_policy(x0, y0, period, t_static):
    """The policy as a block: observation tokens in, a stack of layers, and the
    action chunk out (k cells). Cells dim as their action is consumed; every m
    steps a new inference runs (a pulse through the layers) and the chunk is
    replaced before it runs out."""
    T_inf = M_STEP * DT
    tok = 10
    bars_x = x0 + 34
    cells_x = bars_x + 5 * 11 + 16
    w = cells_x + K_CHUNK * 13.5 - 3.5 + 13 - x0
    h = 62
    y_mid = y0 + h / 2
    els = [f'<path class="ha-mbg" d="{rect_d(x0, y0, x0 + w, y0 + h)}"/>']
    els.append('<path class="ha-tin" d="' + "".join(
        rect_d(x0 + 13, y_mid - 19 + i * 14, x0 + 13 + tok, y_mid - 19 + i * 14 + tok) for i in range(3)) + '"/>')
    els.append('<path class="ha-res" d="' + f"M{f(x0 + 13 + tok)} {f(y_mid)}H{f(cells_x - 4)}" + '"/>')
    for i in range(5):
        els.append(f'<path class="ha-bar ha-b{i}" d="{rect_d(bars_x + i * 11, y0 + 11, bars_x + i * 11 + 6, y0 + h - 11)}"/>')
    for c in range(K_CHUNK):
        x = cells_x + c * 13.5
        cls = f"ha-q{c}" if c < M_STEP - 1 else "ha-qx"
        els.append(f'<path class="ha-cell {cls}" d="{rect_d(x, y_mid - 5, x + 10, y_mid + 5)}"/>')
    els.append(f'<path class="ha-mct" d="{corner_ticks(x0, y0, x0 + w, y0 + h, 7)}"/>')
    els.append(f'<text class="ha-lbl" y="{f(y0 - 7)}"><tspan x="{f(x0 + 12)}">o<tspan class="ha-sub" dy="3">t</tspan></tspan>'
               f'<tspan x="{f(bars_x + 20)}" dy="-3">π</tspan>'
               f'<tspan x="{f(cells_x)}">a<tspan class="ha-sub" dy="3">t:t+k</tspan></tspan></text>')
    # animations, all on the inference period
    css = [f"@keyframes ha-bar{{0%,12%,100%{{fill:var(--ha-s)}}5%{{fill:var(--ha-a2)}}}}"]
    for i in range(5):
        d = (T_inf - 0.55 + 0.09 * i) % T_inf
        css.append(f".ha-b{i}{{animation:ha-bar {f(T_inf, 2)}s linear infinite;animation-delay:{f(d - T_inf, 2)}s}}")
    for c in range(M_STEP - 1):
        t_dim = (c + 1) * DT
        a = 100 * 0.06
        b = 100 * t_dim / T_inf
        css.append(f"@keyframes ha-q{c}{{0%{{opacity:.2}}{f(a, 2)}%,{f(b, 2)}%{{opacity:1}}"
                   f"{f(b + 6, 2)}%,100%{{opacity:.2}}}}.ha-q{c}{{animation:ha-q{c} {f(T_inf, 2)}s linear infinite}}")
    css.append(f"@keyframes ha-qx{{0%{{opacity:.45}}6%,100%{{opacity:1}}}}"
               f".ha-qx{{animation:ha-qx {f(T_inf, 2)}s linear infinite}}")
    # reduced-motion frame
    el = t_static % T_inf
    dim = [c for c in range(M_STEP - 1) if (c + 1) * DT <= el]
    if dim:
        css.append(",".join(f".ha-q{c}" for c in dim) + "{opacity:.2}")
    return "".join(els), "".join(css), (x0, y0, x0 + w, y0 + h)


# ============================================================ composition
ARM_BASE = (11.6, 6.18)         # the arm's place on the workshop bench (plan, metres)


def P(x, y):
    return np.array([x * PX, y * PX])


def layer_plan(plan, route):
    """The floor as a quiet architectural drawing: hatched walls, furniture in
    outline, door swings, and the planned route with a tick every metre."""
    furn = "".join(rect_d(*(np.array(o[1:]) * PX)) if o[0] == "rect" else
                   ellipse_d(o[1] * PX, o[2] * PX, o[3] * PX, o[3] * PX, 0) for o in plan.furniture)
    walls = "".join(rect_d(*(np.array(w) * PX)) for w in plan.walls)
    doors = []
    for (h, o, c) in plan.doors:
        hx, hy = P(*h)
        r = 0.9 * PX
        sw = "1" if o[0] * c[1] - o[1] * c[0] > 0 else "0"
        doors.append(f"M{nums([hx, hy])}l{nums([o[0] * r, o[1] * r])}"
                     f"A{nums([r, r, '0', '0', sw, hx + c[0] * r, hy + c[1] * r])}")
    pts = route * PX
    ticks = []
    for i in range(20, len(pts) - 1, 20):
        t = pts[i + 1] - pts[i - 1]
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t) * 4
        ticks.append(f"M{nums(pts[i] - n)}l{nums(2 * n)}")
    s, g = pts[0], pts[-1]
    # the arm's footprint on its bench: base disc, reach line, the two cube spots
    b = P(*ARM_BASE)
    a_, b_ = P(ARM_BASE[0] + A_X, ARM_BASE[1]), P(ARM_BASE[0] + B_X, ARM_BASE[1])
    foot = (f'<g class="ha-foot"><path d="{circle_d(b[0], b[1], 13)}M{nums(b + [13, 0])}H{f(a_[0])}"/>'
            f'<path d="{rect_d(a_[0] - 3.5, a_[1] - 3.5, a_[0] + 3.5, a_[1] + 3.5)}'
            f'{rect_d(b_[0] - 3.5, b_[1] - 3.5, b_[0] + 3.5, b_[1] + 3.5)}"/></g>')
    defs = (f'<path id="ha-walls" d="{walls}"/><path id="ha-furn" d="{furn}"/>'
            f'<path id="ha-route" d="{smooth_d(resample(pts, 22.0))}"/>')
    svg = (f'<g class="ha-plan"><use href="#ha-furn" class="ha-furn"/><use href="#ha-walls" class="ha-wall"/>'
           f'<path class="ha-door" d="{"".join(doors)}"/>{foot}'
           f'<use href="#ha-route" class="ha-route"/><path class="ha-rtick" d="{"".join(ticks)}"/>'
           f'<path class="ha-end" d="{circle_d(s[0], s[1], 4.5)}{circle_d(g[0], g[1], 4.5)}"/></g>')
    return svg, defs


def layer_io(L, mb):
    """Observations leave the detail view for the policy; action chunks come back."""
    x0, y0, x1, y1 = mb
    lx, ly, lr = L["loupe"]
    ym = (y0 + y1) / 2
    a = on_circle(lx, ly, lr, 124)
    b = on_circle(lx, ly, lr, 56)
    obs = cubic(a, (a[0] - 6, a[1] + 46), (x0 - 46, ym), (x0, ym))
    act = cubic((x1, ym), (x1 + 46, ym), (b[0] + 6, b[1] + 46), b)
    return f'<g class="ha-io"><path class="ha-st" d="{obs}"/><path class="ha-act" d="{act}"/></g>'


def build(L=LAYOUT):
    plan = build_plan()
    route = elastic_band(plan, plan_route(plan, L["start"], L["goal"], R_BOT))
    clear = float(plan.sample(plan.dist, route[:, 0], route[:, 1]).min())
    assert clear > 0.3, clear
    keys, period_r, static, fan = robot_timeline(plan, route)
    plan_svg, plan_defs = layer_plan(plan, route)
    robot_svg, mask, robot_css = layer_robot(plan, route, keys, period_r, static, fan)
    lx, ly, lr = L["loupe"]
    arm_svg, arm_css, period_a, t_static = layer_arm(lx, ly, lr)
    pol_svg, pol_css, mb = layer_policy(*L["model"], period_a, t_static)
    reach = math.hypot(max(abs(WIN[0]), WIN[2]), max(abs(WIN[1]), WIN[3])) + 0.9

    def near(x0, y0, x1, y1):            # within the window's reach of the route?
        cx_ = np.clip(route[:, 0], x0, x1)
        cy_ = np.clip(route[:, 1], y0, y1)
        return bool((np.hypot(route[:, 0] - cx_, route[:, 1] - cy_) < reach).any())
    obstacles = ("".join(f'<path d="{rect_d(*(np.array(w) * PX))}"/>' for w in plan.walls if near(*w))
                 + "".join(f'<path d="{rect_d(*(np.array(o[1:]) * PX))}"/>' if o[0] == "rect" else
                           f'<circle cx="{f(o[1] * PX)}" cy="{f(o[2] * PX)}" r="{f(o[3] * PX)}"/>'
                           for o in plan.furniture
                           if near(*(o[1:] if o[0] == "rect" else (o[1] - o[3], o[2] - o[3], o[1] + o[3], o[2] + o[3])))))
    defs = ('<defs><pattern id="ha-grid" width="50" height="50" patternUnits="userSpaceOnUse">'
            '<path class="ha-gl" d="M50 0V50H0"/></pattern>'
            '<pattern id="ha-hatch" width="5" height="5" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)"><path class="ha-hl" d="M0 0V5"/></pattern>'
            '<pattern id="ha-cellp" width="10" height="10" patternUnits="userSpaceOnUse">'
            '<path class="ha-cl2" d="M10 0V10H0"/></pattern>'
            f'<clipPath id="ha-lc"><circle cx="{f(lx)}" cy="{f(ly)}" r="{f(lr - 1)}"/></clipPath>'
            + mask + f'<g id="ha-ob">{obstacles}</g>' + plan_defs + '</defs>')
    body = ('<rect class="ha-grid" width="1600" height="1000"/>' + plan_svg + robot_svg
            + arm_svg + layer_io(L, mb) + pol_svg)
    css = STYLE + robot_css + arm_css + pol_css + PHONE + MOTION_OFF
    svg = (f'<svg class="ha-art" viewBox="0 0 {VB_W} {VB_H}" width="100%" height="100%" '
           f'preserveAspectRatio="xMaxYMin slice" focusable="false" aria-hidden="true" '
           f'xmlns="http://www.w3.org/2000/svg"><style>{css}</style>{defs}{body}</svg>')
    info = dict(route_m=round(float(np.sum(np.hypot(*np.diff(route, axis=0).T))), 2),
                clearance_m=round(clear, 3), robot_loop_s=round(period_r, 1), arm_loop_s=round(period_a, 1))
    return svg, info


STYLE = """
.ha-art{--ha-g:var(--ground,#fbfaf8);--ha-s:var(--surface,#fff);--ha-i:var(--ink,#16181c);--ha-i2:var(--ink-2,#545a66);--ha-i3:var(--ink-3,#7d838f);--ha-r:var(--rule,#e4e2dd);--ha-rs:var(--rule-strong,#cfccc5);--ha-a:var(--signal,#9a4415);--ha-a2:var(--signal-soft,#c86a2e);--ha-w:var(--wash,#f6e3d5);display:block}
.ha-grid{fill:url(#ha-grid)}.ha-gl{fill:none;stroke:var(--ha-r);stroke-width:1;opacity:.7}
.ha-furn{fill:none;stroke:var(--ha-rs);stroke-width:1}
.ha-wall{fill:url(#ha-hatch);stroke:var(--ha-rs);stroke-width:1}.ha-hl{stroke:var(--ha-rs);stroke-width:1}
.ha-door{fill:none;stroke:var(--ha-rs);stroke-width:.8;stroke-dasharray:3 3}
.ha-route{fill:none;stroke:var(--ha-a);stroke-width:1.6;stroke-linecap:round;opacity:.5}
.ha-rtick{stroke:var(--ha-a);stroke-width:1;opacity:.45}
.ha-end{fill:var(--ha-g);stroke:var(--ha-a);stroke-width:1.3;opacity:.75}
.ha-foot path{fill:none;stroke:var(--ha-i3);stroke-width:1}
.ha-mask{mask-type:alpha}.ha-m0{stop-color:var(--ha-i);stop-opacity:1}.ha-m1{stop-color:var(--ha-i);stop-opacity:0}
.ha-hb{fill:var(--ha-w);stroke:var(--ha-w);stroke-linejoin:round}
.ha-h0{opacity:.45}.ha-h1{opacity:.5}.ha-h2{opacity:.55}.ha-h3{fill:var(--ha-a2);stroke:var(--ha-a2);opacity:.4}
.ha-cells{fill:url(#ha-cellp)}.ha-cl2{fill:none;stroke:var(--ha-rs);stroke-width:.5}
.ha-wl2{fill:var(--ha-i3);stroke:var(--ha-i3);stroke-width:1}
.ha-fu2{fill:var(--ha-s);stroke:var(--ha-i3);stroke-width:1}
.ha-rt2{fill:none;stroke:var(--ha-a);stroke-width:2.4;stroke-linecap:round}
.ha-tint{fill:var(--ha-w);opacity:.55}
.ha-fan{fill:none;stroke:var(--ha-i2);stroke-width:.9;opacity:.7}
.ha-arc{fill:none;stroke:var(--ha-a);stroke-width:1.9;stroke-linecap:round}
.ha-whl,.ha-body{fill:var(--ha-s);stroke:var(--ha-i);stroke-width:1.4}.ha-lid{fill:var(--ha-i)}.ha-head{stroke:var(--ha-i);stroke-width:1.4}
.ha-lbg{fill:var(--ha-s)}.ha-loupe{fill:none;stroke:var(--ha-i2);stroke-width:1.2}.ha-rim{stroke:var(--ha-i3);stroke-width:1}
.ha-eg{fill:none;stroke:var(--ha-r);stroke-width:.8;opacity:.8}
.ha-bench{fill:url(#ha-hatch)}.ha-benchl{stroke:var(--ha-i2);stroke-width:1.4}
.ha-spot{stroke:var(--ha-i3);stroke-width:1}
.ha-ee{fill:none;stroke:var(--ha-i3);stroke-width:1;stroke-dasharray:2 3}
.ha-tk{fill:var(--ha-a)}
.ha-cube{fill:var(--ha-s);stroke:var(--ha-i);stroke-width:1.3}
.ha-arm{fill:var(--ha-s);stroke:var(--ha-i);stroke-width:1.4;stroke-linejoin:round}
.ha-cl{stroke:var(--ha-i3);stroke-width:.8;stroke-dasharray:7 2 1 2}
.ha-hs{fill:var(--ha-s);stroke:var(--ha-i);stroke-width:1.4}.ha-hi{fill:none;stroke:var(--ha-i);stroke-width:1}.ha-hc{fill:var(--ha-i)}
.ha-back .ha-arm,.ha-back .ha-hs,.ha-back .ha-cube{fill:var(--ha-r);stroke:var(--ha-i3);stroke-width:1}
.ha-mbg{fill:var(--ha-s);stroke:var(--ha-i3);stroke-width:.8}.ha-mct{fill:none;stroke:var(--ha-i);stroke-width:1.4}
.ha-tin{fill:var(--ha-s);stroke:var(--ha-i2);stroke-width:1}.ha-res{stroke:var(--ha-rs);stroke-width:1}
.ha-bar{fill:var(--ha-s);stroke:var(--ha-i2);stroke-width:1}
.ha-cell{fill:var(--ha-a)}
.ha-lbl{font:italic 13px var(--mono,ui-monospace,monospace);fill:var(--ha-i3)}.ha-sub{font-size:9px}
.ha-st{fill:none;stroke:var(--ha-i2);stroke-width:2.2;stroke-dasharray:2.2 6.8;animation:ha-flow 8s linear infinite}
.ha-act{fill:none;stroke:var(--ha-a);stroke-width:2.2;stroke-dasharray:7 6;animation:ha-flow2 8s linear infinite}
@keyframes ha-flow{to{stroke-dashoffset:-72}}@keyframes ha-flow2{to{stroke-dashoffset:-104}}
"""

# Portrait screens crop to the right-hand strip: lift the arm's view clear of the
# headline. On phones the robot's route is outside that strip or under the text,
# so its animation is stopped there to save battery.
PHONE = """
@media (max-aspect-ratio:3/4){.ha-lp{transform-box:view-box;transform-origin:1364px 334px;transform:translateY(-30px) scale(.9)}}
@media (max-width:600px){.ha-rb,.ha-rbi,.ha-arc{animation:none}}
"""

# The page's own reduced-motion rule only shortens durations; here every
# animation is removed, and the base styles above already describe a complete
# frame: robot part way along its route, arm with its fingers on the cube, the
# action horizon part consumed.
MOTION_OFF = """
@media (prefers-reduced-motion:reduce){.ha-art *{animation:none!important}}
"""


def main(argv):
    out = pathlib.Path(argv[1]) if len(argv) > 1 else OUT
    svg, info = build()
    out.write_text(svg, encoding="utf-8")
    print(f"{out}: {len(svg.encode())} bytes  {info}")


if __name__ == "__main__":
    main(sys.argv)
