#!/usr/bin/env python3
"""Generate a thumbnail figure for every project that has no photograph.

Standard library + numpy + Pillow only. Run from the repo root:

    python3 scripts/make_figures.py

Two rules this file exists to enforce:

1. **Nothing is invented as a result.** Where a figure can be computed from real
   data it is (the coverage decomposition uses his own obstacle file; the Bayes
   belief and the Kalman track are actually simulated). Everything else is drawn
   and captioned as a *schematic of the technique*, never as a measurement.
2. **Employer projects get illustrations of public algorithms only** — a distance
   field, a Kalman update, a training loop. Nothing depicts Miko's system, which
   is what `references/disclosure.md` forbids.

Pillow has no antialiasing, so everything is drawn at SS× and downsampled.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, SS = 1200, 800, 2
OUT = pathlib.Path("assets/media")

GROUND = (251, 250, 248)
INK = (22, 24, 28)
INK2 = (84, 90, 102)
INK3 = (125, 131, 143)
RULE = (228, 226, 221)
SIGNAL = (154, 68, 21)
SIGNAL_SOFT = (200, 106, 46)
WASH = (246, 227, 213)

F = "/usr/share/fonts/truetype/dejavu/"


def font(size, mono=True, bold=False):
    name = ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf") if mono \
        else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return ImageFont.truetype(F + name, size * SS)


def canvas(grid=True, step=50):
    im = Image.new("RGB", (W * SS, H * SS), GROUND)
    d = ImageDraw.Draw(im)
    if grid:
        for x in range(0, W, step):
            d.line([(x * SS, 0), (x * SS, H * SS)], fill=RULE, width=SS)
        for y in range(0, H, step):
            d.line([(0, y * SS), (W * SS, y * SS)], fill=RULE, width=SS)
    return im, d


def save(im, name):
    im = im.resize((W, H), Image.LANCZOS)
    p = OUT / name
    im.save(p, "JPEG", quality=86, optimize=True, progressive=True)
    print(f"  {name:34s} {p.stat().st_size // 1024:>4d} KB")


def label(d, x, y, text, size=17, col=INK3, bold=False, mono=True):
    d.text((x * SS, y * SS), text, font=font(size, mono, bold), fill=col)


def S(*vals):
    return tuple(v * SS for v in vals)


def arrow(d, p0, p1, col=INK2, w=3, head=13):
    d.line([S(*p0), S(*p1)], fill=col, width=w * SS)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (+1, -1):
        a = ang + s * 2.6
        d.line([S(*p1), S(p1[0] + head * math.cos(a), p1[1] + head * math.sin(a))],
               fill=col, width=w * SS)


def rounded(d, box, r, outline, width=3, fill=None):
    """Square corners: Pillow 7 has no rounded_rectangle, and the design plan
    calls for hairlines rather than rounded cards anyway."""
    d.rectangle([S(*box[:2]), S(*box[2:])], outline=outline, width=width * SS, fill=fill)


# --------------------------------------------------------------- 1. VLA loop

def fig_vla():
    """The data flywheel. A generic ML-ops cycle — no employer specifics."""
    im, d = canvas()
    boxes = [
        (150, 90, 520, 210, "teleoperation", "raw episodes"),
        (690, 90, 1060, 210, "curation", "trim - balance - QC"),
        (690, 560, 1060, 690, "evaluation", "sim or real, one path"),
        (150, 560, 520, 690, "fine-tuning", "policy checkpoint"),
    ]
    for (x0, y0, x1, y1, t, s) in boxes:
        rounded(d, (x0, y0, x1, y1), 8, INK, 3)
        label(d, x0 + 26, y0 + 30, t, 26, INK, bold=True)
        label(d, x0 + 26, y0 + 72, s, 16, INK3)

    arrow(d, (525, 150), (685, 150))
    arrow(d, (875, 215), (875, 555))
    arrow(d, (685, 625), (525, 625))
    arrow(d, (335, 555), (335, 215))

    d.ellipse([S(556, 355), S(644, 443)], outline=SIGNAL, width=3 * SS)
    label(d, 578, 384, "loop", 19, SIGNAL, bold=True)
    label(d, 150, 730, "Schematic: the data flywheel a VLA policy is trained inside.", 17, INK3)
    save(im, "fig-vla-loop.jpg")


# ------------------------------------------- 2. distance field + planned path

def fig_distance_field():
    """A real Euclidean distance transform over an occupancy grid, and a path
    actually searched over it with clearance folded into the edge cost — so the
    route provably does not pass through an obstacle. Public technique."""
    import heapq
    im, d = canvas(grid=False)
    nx, ny = 48, 30
    occ = np.zeros((ny, nx), bool)
    for (cx, cy, w, h) in [(8, 6, 7, 3), (22, 4, 3, 11), (33, 14, 10, 3),
                           (14, 18, 4, 8), (38, 3, 5, 4), (26, 22, 9, 3)]:
        occ[cy:cy + h, cx:cx + w] = True

    ys, xs = np.nonzero(occ)
    dist = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            dist[j, i] = 0.0 if occ[j, i] else float(np.min(np.hypot(xs - i, ys - j)))

    # Dijkstra with a clearance penalty: cheap in open space, expensive near walls
    start, goal = (28, 1), (3, 45)
    INF = float("inf")
    cost = np.full((ny, nx), INF)
    cost[start] = 0.0
    prev = {}
    pq = [(0.0, start)]
    while pq:
        c, (j, i) = heapq.heappop(pq)
        if c > cost[j, i]:
            continue
        if (j, i) == goal:
            break
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                nj, ni = j + dj, i + di
                if not (0 <= nj < ny and 0 <= ni < nx) or occ[nj, ni]:
                    continue
                step = math.hypot(dj, di)
                clearance = 6.0 / (dist[nj, ni] + 0.7)   # prefer open space
                nc = c + step + clearance
                if nc < cost[nj, ni]:
                    cost[nj, ni] = nc
                    prev[(nj, ni)] = (j, i)
                    heapq.heappush(pq, (nc, (nj, ni)))

    node, path = goal, [goal]
    while node in prev:
        node = prev[node]
        path.append(node)
    path.reverse()

    cw, ch, ox, oy = 23, 23, 60, 70
    dclip = 7.0
    for j in range(ny):
        for i in range(nx):
            x0, y0 = ox + i * cw, oy + j * ch
            if occ[j, i]:
                col = INK
            else:
                t = min(dist[j, i], dclip) / dclip
                col = tuple(int(GROUND[k] + (WASH[k] - GROUND[k]) * (1 - t))
                            for k in range(3))
            d.rectangle([S(x0, y0), S(x0 + cw - 1, y0 + ch - 1)], fill=col)

    pts = [(ox + i * cw + cw / 2, oy + j * ch + ch / 2) for (j, i) in path]
    d.line([S(*p) for p in pts], fill=SIGNAL, width=5 * SS, joint="curve")
    for p in (pts[0], pts[-1]):
        d.ellipse([S(p[0] - 9, p[1] - 9), S(p[0] + 9, p[1] + 9)], fill=SIGNAL)

    label(d, 60, 26, "distance-to-obstacle field", 20, INK2, bold=True)
    label(d, 60, 762, "Computed: a Euclidean distance transform, and a route searched "
                      "over it that prefers clearance.", 17, INK3)
    save(im, "fig-distance-field.jpg")


# ------------------------------------------- 3. Kalman track through a dropout

def fig_kalman():
    """A genuinely simulated constant-velocity filter, including a stretch where
    measurements stop and the estimate runs predict-only."""
    im, d = canvas()
    rng = np.random.default_rng(7)
    n = 120
    t = np.arange(n)
    true = np.stack([60 + t * 8.6, 420 + 150 * np.sin(t / 26)], 1)
    meas = true + rng.normal(0, 16, true.shape)
    gap = slice(58, 82)

    x = np.array([true[0, 0], true[0, 1], 8.0, 0.0])
    P = np.eye(4) * 40
    A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Q = np.diag([1.5, 1.5, 0.6, 0.6])
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    R = np.eye(2) * 260
    est = []
    for k in range(n):
        x = A @ x
        P = A @ P @ A.T + Q
        if not (gap.start <= k < gap.stop):
            y = meas[k] - Hm @ x
            Sm = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(Sm)
            x = x + K @ y
            P = (np.eye(4) - K @ Hm) @ P
        est.append(x[:2].copy())
    est = np.array(est)

    gx0 = 60 + gap.start * 8.6
    gx1 = 60 + (gap.stop - 1) * 8.6
    d.rectangle([S(gx0, 90), S(gx1, 700)], fill=(246, 240, 234))
    label(d, gx0 + 10, 104, "no detections", 16, SIGNAL_SOFT)

    d.line([S(*p) for p in true], fill=RULE, width=6 * SS, joint="curve")
    for k in range(n):
        if gap.start <= k < gap.stop:
            continue
        mx, my = meas[k]
        d.ellipse([S(mx - 5, my - 5), S(mx + 5, my + 5)], fill=INK3)
    d.line([S(*p) for p in est], fill=SIGNAL, width=5 * SS, joint="curve")

    label(d, 60, 34, "constant-velocity filter, predict-only through the gap", 20, INK2, bold=True)
    for i, (c, txt) in enumerate([(RULE, "true path"), (INK3, "detections"), (SIGNAL, "estimate")]):
        d.rectangle([S(60 + i * 210, 738), S(96 + i * 210, 744)], fill=c)
        label(d, 106 + i * 210, 728, txt, 16, INK3)
    save(im, "fig-kalman.jpg")


# ------------------------------------------------ 4. quadruped gait + friction

def fig_quadruped():
    im, d = canvas()
    legs = ["FL", "FR", "RL", "RR"]
    phase = [(0.0, 0.5), (0.5, 1.0), (0.5, 1.0), (0.0, 0.5)]
    x0, x1, top, rowh = 150, 760, 110, 74
    for i, (nm, (a, b)) in enumerate(zip(legs, phase)):
        y = top + i * rowh
        label(d, 78, y + 14, nm, 20, INK2, bold=True)
        d.line([S(x0, y + 26), S(x1, y + 26)], fill=RULE, width=2 * SS)
        for c in range(3):
            s = x0 + (c + a / 2) * (x1 - x0) / 3
            e = x0 + (c + b / 2) * (x1 - x0) / 3
            d.rectangle([S(s, y + 8), S(e, y + 44)], fill=INK)
    label(d, 150, 400, "stance", 17, INK3)
    d.rectangle([S(210, 396), S(246, 414)], fill=INK)
    label(d, 262, 400, "swing = gap", 17, INK3)
    label(d, 78, 60, "contact schedule", 20, INK2, bold=True)

    cx, cy = 960, 470
    d.line([S(cx - 190, cy), S(cx + 190, cy)], fill=INK, width=3 * SS)
    for s in (-1, 1):
        d.line([S(cx, cy - 250), S(cx + s * 150, cy)], fill=SIGNAL, width=4 * SS)
    d.line([S(cx, cy), S(cx, cy - 250)], fill=INK3, width=2 * SS)
    arrow(d, (cx, cy), (cx + 92, cy - 195), col=SIGNAL_SOFT, w=4)
    label(d, cx - 190, cy - 300, "friction cone", 20, INK2, bold=True)
    label(d, cx + 8, cy - 150, "f", 20, SIGNAL_SOFT, bold=True)
    label(d, 78, 748, "Schematic: the contact schedule and friction-cone constraints "
                      "the QP is solved against.", 17, INK3)
    save(im, "fig-quadruped.jpg")


# -------------------------------- 5. Morse cell decomposition, his actual data

def fig_coverage():
    """Computed from the project's own files/obs.txt — 11 circular obstacles.

    Cells are the free intervals of each vertical slab between consecutive
    critical x-coordinates; two cells are adjacent when they sit in neighbouring
    slabs and their y-intervals overlap. That is the actual decomposition, so the
    adjacency edges never cross an obstacle.
    """
    obs = [(0.1, -0.4, 0.2), (0.1, 0.4, 0.2), (-0.6, 0.4, 0.2), (-0.4, -0.4, 0.2),
           (0.5, 0.0, 0.3), (-0.2, 0.0, 0.1), (0.7, 0.7, 0.15), (-0.7, -0.7, 0.15),
           (0.7, -0.7, 0.15), (0.05, 0.0, 0.1), (-0.7, -0.1, 0.2)]
    im, d = canvas(grid=False)
    pad, size = 110, 600
    left = 300

    def T(x, y):
        return (left + (x + 1) / 2 * size, pad + (1 - y) / 2 * size)

    crit = sorted({round(cx + s * r, 5) for (cx, cy, r) in obs for s in (-1, 1)
                   if -1 < cx + s * r < 1})
    bounds = [-1.0] + crit + [1.0]

    # free y-intervals in each slab
    slabs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        xm = (a + b) / 2
        blocked = []
        for (cx, cy, r) in obs:
            dx = xm - cx
            if abs(dx) < r:
                dy = math.sqrt(r * r - dx * dx)
                blocked.append((cy - dy, cy + dy))
        blocked.sort()
        free, cur = [], -1.0
        for (s0, s1) in blocked:
            if s0 > cur:
                free.append((cur, min(s0, 1.0)))
            cur = max(cur, s1)
        if cur < 1.0:
            free.append((cur, 1.0))
        slabs.append((a, b, [f for f in free if f[1] - f[0] > 0.02]))

    # alternating wash so the slabs read as cells
    for i, (a, b, free) in enumerate(slabs):
        if i % 2:
            continue
        for (y0, y1) in free:
            d.rectangle([S(*T(a, y1)), S(*T(b, y0))], fill=(247, 241, 234))

    for a, b, free in slabs:
        for xb in (a, b):
            if -1 < xb < 1:
                d.line([S(*T(xb, 1)), S(*T(xb, -1))], fill=(226, 220, 210), width=2 * SS)

    for (cx, cy, r) in obs:
        d.ellipse([S(*T(cx - r, cy + r)), S(*T(cx + r, cy - r))], fill=INK)

    # adjacency: neighbouring slabs whose free intervals overlap
    centres = []
    for (a, b, free) in slabs:
        centres.append([((a + b) / 2, (y0 + y1) / 2, y0, y1) for (y0, y1) in free])
    nedge = 0
    for i in range(len(centres) - 1):
        for (ax, ay, a0, a1) in centres[i]:
            for (bx, by, b0, b1) in centres[i + 1]:
                if min(a1, b1) - max(a0, b0) > 0.01:
                    d.line([S(*T(ax, ay)), S(*T(bx, by))], fill=SIGNAL_SOFT, width=2 * SS)
                    nedge += 1
    ncell = 0
    for col in centres:
        for (cx, cy, _, _) in col:
            px, py = T(cx, cy)
            d.ellipse([S(px - 6, py - 6), S(px + 6, py + 6)], fill=SIGNAL)
            ncell += 1

    d.rectangle([S(*T(-1, 1)), S(*T(1, -1))], outline=INK2, width=3 * SS)
    label(d, left, 52, "exact cell decomposition by sweep line", 20, INK2, bold=True)
    label(d, left, pad + size + 34,
          f"Computed from the project's own obstacle file:", 17, INK3)
    label(d, left, pad + size + 60,
          f"{len(obs)} circles, {ncell} cells, {nedge} adjacency edges.", 17, INK3)
    save(im, "fig-coverage.jpg")


# ------------------------------------------------------- 6. drone descent

def fig_drone():
    im, d = canvas()
    gy = 640
    d.line([S(60, gy), S(1140, gy)], fill=INK, width=3 * SS)
    for x in range(70, 1140, 26):
        d.line([S(x, gy), S(x - 12, gy + 14)], fill=RULE, width=2 * SS)

    ts = np.linspace(0, 1, 90)
    px = 130 + ts * 760
    py = 150 + (gy - 190) * ts ** 2.4
    d.line([S(x, y) for x, y in zip(px, py)], fill=SIGNAL, width=5 * SS, joint="curve")
    for k in range(0, 90, 14):
        x, y = px[k], py[k]
        d.line([S(x - 26, y), S(x + 26, y)], fill=INK, width=3 * SS)
        for s in (-26, 26):
            d.ellipse([S(x + s - 9, y - 7), S(x + s + 9, y + 3)], outline=INK2, width=2 * SS)

    mx = 890
    d.rectangle([S(mx - 60, gy - 8), S(mx + 60, gy)], fill=INK)
    for i in range(3):
        for j in range(3):
            if (i + j) % 2 == 0:
                d.rectangle([S(mx - 45 + i * 30, gy - 46 + j * 13),
                             S(mx - 20 + i * 30, gy - 36 + j * 13)], fill=INK2)
    label(d, mx - 66, gy + 28, "marker", 17, INK3)
    label(d, 60, 60, "vision-guided descent onto a marked pad", 20, INK2, bold=True)
    label(d, 60, 748, "Schematic: the approach and landing profile.", 17, INK3)
    save(im, "fig-drone.jpg")


# ---------------------------------------- 7. 2-DoF arm workspace + search tree

def fig_manipulator():
    im, d = canvas(grid=False)
    cx, cy, l1, l2 = 350, 470, 150, 120
    d.ellipse([S(cx - (l1 + l2), cy - (l1 + l2)), S(cx + l1 + l2, cy + l1 + l2)],
              outline=RULE, width=3 * SS)
    d.ellipse([S(cx - abs(l1 - l2), cy - abs(l1 - l2)),
               S(cx + abs(l1 - l2), cy + abs(l1 - l2))], outline=RULE, width=3 * SS)

    for k, (a, b) in enumerate([(-1.15, 0.9), (-0.75, 0.55), (-0.35, 0.3), (0.05, 0.15)]):
        j = (cx + l1 * math.cos(a), cy + l1 * math.sin(a))
        e = (j[0] + l2 * math.cos(a + b), j[1] + l2 * math.sin(a + b))
        col = INK if k == 3 else (200, 198, 192)
        d.line([S(cx, cy), S(*j)], fill=col, width=7 * SS)
        d.line([S(*j), S(*e)], fill=col, width=7 * SS)
        d.ellipse([S(j[0] - 8, j[1] - 8), S(j[0] + 8, j[1] + 8)], fill=col)
    d.ellipse([S(cx - 12, cy - 12), S(cx + 12, cy + 12)], fill=INK)

    # config-space panel with a random tree
    bx, by, bw = 720, 200, 400
    d.rectangle([S(bx, by), S(bx + bw, by + bw)], outline=INK2, width=3 * SS)
    rng = np.random.default_rng(3)
    nodes = [(bx + 40, by + bw - 40)]
    for _ in range(130):
        t = (bx + rng.uniform(0, bw), by + rng.uniform(0, bw))
        near = min(nodes, key=lambda p: (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2)
        ang = math.atan2(t[1] - near[1], t[0] - near[0])
        new = (near[0] + 28 * math.cos(ang), near[1] + 28 * math.sin(ang))
        if bx < new[0] < bx + bw and by < new[1] < by + bw:
            d.line([S(*near), S(*new)], fill=(206, 203, 197), width=2 * SS)
            nodes.append(new)
    goal = (bx + bw - 50, by + 50)
    chain = min(nodes, key=lambda p: (p[0] - goal[0]) ** 2 + (p[1] - goal[1]) ** 2)
    d.line([S(bx + 40, by + bw - 40), S(*chain)], fill=SIGNAL, width=4 * SS)
    d.ellipse([S(goal[0] - 9, goal[1] - 9), S(goal[0] + 9, goal[1] + 9)], fill=SIGNAL)
    label(d, bx, by - 42, "configuration space", 20, INK2, bold=True)
    label(d, 150, 60, "planar 2-DoF workspace", 20, INK2, bold=True)
    label(d, 60, 748, "Schematic: reachable annulus, and sampling-based search over joint angles.",
          17, INK3)
    save(im, "fig-manipulator.jpg")


# ------------------------------------------- 8. Bug1 vs potential field

def fig_bug():
    im, d = canvas()
    start, goal = (130, 640), (1070, 180)
    ox, oy, orad = 600, 400, 150
    d.ellipse([S(ox - orad, oy - orad), S(ox + orad, oy + orad)], fill=INK)

    pts = [start, (420, 520)]
    for k in range(41):
        a = math.pi * 0.75 - k * (2 * math.pi / 40)
        pts.append((ox + (orad + 26) * math.cos(a), oy + (orad + 26) * math.sin(a)))
    pts += [(800, 300), goal]
    d.line([S(*p) for p in pts], fill=INK2, width=4 * SS, joint="curve")

    pf = [start, (330, 570), (470, 520), (540, 486), (566, 474)]
    d.line([S(*p) for p in pf], fill=SIGNAL, width=5 * SS, joint="curve")
    d.ellipse([S(560, 468), S(578, 486)], fill=SIGNAL)
    label(d, 470, 556, "local minimum", 17, SIGNAL)

    for p, t in ((start, "start"), (goal, "goal")):
        d.ellipse([S(p[0] - 11, p[1] - 11), S(p[0] + 11, p[1] + 11)], fill=INK)
        label(d, p[0] - 16, p[1] + 22, t, 17, INK3)
    label(d, 60, 60, "circumnavigate vs descend a potential", 20, INK2, bold=True)
    for i, (c, txt) in enumerate([(INK2, "boundary following"), (SIGNAL, "potential field")]):
        d.rectangle([S(60 + i * 330, 738), S(96 + i * 330, 744)], fill=c)
        label(d, 106 + i * 330, 728, txt, 16, INK3)
    save(im, "fig-bug.jpg")


# ----------------------------------------- 9. discrete Bayes filter, simulated

def fig_bayes():
    """A real recursive Bayes update over a 1-D corridor with three doors."""
    im, d = canvas(grid=False)
    n = 200
    doors = [40, 100, 158]
    prior = np.ones(n) / n

    def sense(bel, seen=True):
        lik = np.full(n, 0.2)
        for dd in doors:
            lik[max(0, dd - 6):dd + 6] = 0.8
        if not seen:
            lik = 1 - lik
        out = bel * lik
        return out / out.sum()

    def move(bel, k):
        out = np.roll(bel, k)
        ker = np.array([0.15, 0.7, 0.15])
        return np.convolve(out, ker, "same")

    steps = [("prior", prior)]
    b = sense(prior)
    steps.append(("sense door", b))
    b = move(b, 32)
    steps.append(("move", b))
    b = sense(b)
    steps.append(("sense door", b))

    left, right, top, rowh = 150, 1090, 150, 150
    for dd in doors:
        x = left + dd / n * (right - left)
        d.rectangle([S(x - 16, 92), S(x + 16, 116)], fill=INK)
    label(d, left, 60, "corridor with three doors", 20, INK2, bold=True)

    for i, (name, bel) in enumerate(steps):
        base = top + i * rowh + 96
        d.line([S(left, base), S(right, base)], fill=RULE, width=2 * SS)
        sc = 92 / max(bel.max(), 1e-9)
        pts = [(left + k / n * (right - left), base - bel[k] * sc) for k in range(n)]
        d.line([S(*p) for p in pts], fill=SIGNAL if i == len(steps) - 1 else INK2,
               width=4 * SS, joint="curve")
        label(d, 40, base - 22, name, 17, INK3)
    label(d, 150, 752, "Simulated: belief after each sense and move update.", 17, INK3)
    save(im, "fig-bayes.jpg")


# ----------------------------------------- 10. micromouse maze + flood fill

def fig_micromouse():
    im, d = canvas(grid=False)
    n = 12
    rng = np.random.default_rng(11)
    vwall = np.ones((n, n + 1), bool)
    hwall = np.ones((n + 1, n), bool)
    seen = np.zeros((n, n), bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack[-1]
        opts = []
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not seen[nr, nc]:
                opts.append((nr, nc, dr, dc))
        if not opts:
            stack.pop(); continue
        nr, nc, dr, dc = opts[rng.integers(len(opts))]
        if dr == 1: hwall[r + 1, c] = False
        elif dr == -1: hwall[r, c] = False
        elif dc == 1: vwall[r, c + 1] = False
        else: vwall[r, c] = False
        seen[nr, nc] = True
        stack.append((nr, nc))

    goal = (n // 2, n // 2)
    dist = np.full((n, n), 999)
    dist[goal] = 0
    q = [goal]
    while q:
        r, c = q.pop(0)
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            blocked = (hwall[r + 1, c] if dr == 1 else hwall[r, c] if dr == -1
                       else vwall[r, c + 1] if dc == 1 else vwall[r, c])
            if not blocked and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    cell, ox, oy = 52, 290, 110
    dmax = dist[dist < 999].max()
    for r in range(n):
        for c in range(n):
            t = dist[r, c] / dmax if dist[r, c] < 999 else 1
            col = tuple(int(GROUND[k] + (WASH[k] - GROUND[k]) * (1 - t)) for k in range(3))
            d.rectangle([S(ox + c * cell, oy + r * cell),
                         S(ox + (c + 1) * cell, oy + (r + 1) * cell)], fill=col)
    for r in range(n):
        for c in range(n + 1):
            if vwall[r, c]:
                d.line([S(ox + c * cell, oy + r * cell), S(ox + c * cell, oy + (r + 1) * cell)],
                       fill=INK, width=3 * SS)
    for r in range(n + 1):
        for c in range(n):
            if hwall[r, c]:
                d.line([S(ox + c * cell, oy + r * cell), S(ox + (c + 1) * cell, oy + r * cell)],
                       fill=INK, width=3 * SS)

    r, c = n - 1, 0
    path = [(r, c)]
    while dist[r, c] > 0 and len(path) < 400:
        best = None
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            blocked = (hwall[r + 1, c] if dr == 1 else hwall[r, c] if dr == -1
                       else vwall[r, c + 1] if dc == 1 else vwall[r, c])
            if not blocked and (best is None or dist[nr, nc] < dist[best[0], best[1]]):
                best = (nr, nc)
        if best is None: break
        r, c = best
        path.append((r, c))
    pts = [(ox + c * cell + cell / 2, oy + r * cell + cell / 2) for (r, c) in path]
    d.line([S(*p) for p in pts], fill=SIGNAL, width=5 * SS, joint="curve")

    label(d, 290, 56, "flood fill to the centre", 20, INK2, bold=True)
    label(d, 290, 756, "Simulated: cell distances to goal, and the route they induce.", 17, INK3)
    save(im, "fig-micromouse.jpg")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("generating project figures:")
    fig_vla()
    fig_distance_field()
    fig_kalman()
    fig_quadruped()
    fig_coverage()
    fig_drone()
    fig_manipulator()
    fig_bug()
    fig_bayes()
    fig_micromouse()
    print("done.")
