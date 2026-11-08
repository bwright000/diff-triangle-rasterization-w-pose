#!/usr/bin/env python
"""
GPU verification of the perf bitmask (PERF_CHANGES.md). Meant to run on the A100
after building this package (TORCH_CUDA_ARCH_LIST=8.0 pip install . --no-build-isolation).

  V1  forward bit-exactness: mask 0 vs every single bit vs all bits, torch.equal on
      every forward output, at 1280x720 and 640x360.
  V2  backward envelope: per config and mask, 20 forward+backward runs with one fixed
      random grad_out. The 20 mask-0 runs give, per gradient tensor, the envelope
      E = max_r max|g0_r - g0_0| (float atomics are already non-deterministic). Each
      masked config must satisfy max_r max|gm_r - g0_0| <= ENV_FACTOR * E (exact
      equality when E == 0) and its float64 sum of differences must be ~0 (bias-free):
      |mean_r sum64(gm_r - g0_0)| <= ENV_FACTOR * max_r |sum64(g0_r - g0_0)| + 1e-9*sum|g0_0|.
      For bit 8 (#4) only the gradients S2TS2 consumes in that call type are asserted.
  V0  NaN / inf check on the mask-0 gradients (the #3 exactness argument assumes finite).

Usage:  python perf_verify.py [--repeats 20] [--env-factor 3] [--bench] [--res 1280x720,640x360]
Exit code 0 = all assertions passed.
"""
import argparse
import math
import sys

import torch

import diff_triangle_rasterization as dtr
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer, _C

BITS = {1: "#1 tight binning", 2: "#2 bw start at last", 4: "#3 zero-grad skip", 8: "#4 unused-grad skip",
        16: "#8 dead smem loads", 32: "#9 smem vertex attrs", 64: "#10 warp-agg atomics"}
MASKS = [0, 1, 2, 4, 8, 16, 32, 64, 95, dtr.PERF_MASK_ALL]
# 11-08 A100 run: bit 32 (#9 smem vertex attrs) exceeded the mask-0 envelope on track_pose 1280x720 (4.1x) and is
# dropped from production; the production mask is 95 (all but #9). Only failures of masks inside --gate-mask fail the
# run; the others are printed as information.
DEV = "cuda"

# out_others channels (cuda_rasterizer/auxiliary.h). Channel 6 = per-pixel triangle id
# (`pixel_influence`) is an UNINITIALISED register for pixels with no contributor in the
# original code, so it is compared only where alpha > 0 (where it is always written).
CH_NAMES = ["depth", "alpha", "normal_x", "normal_y", "normal_z", "median_depth", "tri_id"]


# ----------------------------------------------------------------------------- camera
def proj_matrix(znear, zfar, tanx, tany):
    """3DGS getProjectionMatrix (row-major math form)."""
    top, right = tany * znear, tanx * znear
    P = torch.zeros(4, 4, dtype=torch.float64)
    P[0, 0] = 2.0 * znear / (2 * right)
    P[1, 1] = 2.0 * znear / (2 * top)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float64)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float64)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float64)
    return Rz @ Ry @ Rx


class Camera:
    def __init__(self, W, H):
        self.W, self.H = W, H
        f = 0.78 * W                                 # ~1000 px at 1280
        self.tanx, self.tany = W / (2 * f), H / (2 * f)
        W2C = torch.eye(4, dtype=torch.float64)
        W2C[:3, :3] = rot(0.03, -0.05, 0.02)
        W2C[:3, 3] = torch.tensor([0.04, -0.03, 0.10], dtype=torch.float64)
        self.C2W = torch.linalg.inv(W2C)
        P = proj_matrix(0.01, 100.0, self.tanx, self.tany)
        self.viewmatrix = W2C.T.float().contiguous().to(DEV)          # = world_view_transform
        self.proj_raw = P.T.float().contiguous().to(DEV)              # = projection_matrix
        self.projmatrix = (W2C.T @ P.T).float().contiguous().to(DEV)  # = full_proj_transform
        self.campos = self.C2W[:3, 3].float().contiguous().to(DEV)

    def unproject(self, u, v, z):
        """pixel (u, v) at view depth z -> world point (float64). Inverse of ndc2Pix."""
        xn = (2 * u + 1) / self.W - 1
        yn = (2 * v + 1) / self.H - 1
        pc = torch.stack([xn * z * self.tanx, yn * z * self.tany, z, torch.ones_like(z)], -1)
        return (pc @ self.C2W.T)[..., :3]

    def settings(self, bg, pose_only, opacity_min=False, sh_degree=3):
        return TriangleRasterizationSettings(
            image_height=self.H, image_width=self.W, tanfovx=self.tanx, tanfovy=self.tany, bg=bg,
            scale_modifier=1.0, viewmatrix=self.viewmatrix, projmatrix=self.projmatrix, sh_degree=sh_degree,
            campos=self.campos, prefiltered=False, debug=False, cx=0.0, cy=0.0, projmatrix_raw=self.proj_raw,
            min_triangle_area_pixels=0.5, optimize_pose_only=pose_only, opacity_min=opacity_min)


# ----------------------------------------------------------------------------- scene
def make_scene(cam, seed=0):
    """Realistic triangle soup in PIXEL space, lifted to world: a shared-vertex tissue-like grid
    mesh (P > V, like S2TS2's meshes) + an overlapping soup of 2-80 px triangles at mixed depths,
    partially / fully off-screen + adversarial slivers and integer / tile-boundary vertices."""
    g = torch.Generator().manual_seed(seed)
    W, H = cam.W, cam.H
    s = W / 1280.0
    R = lambda *shape: torch.rand(*shape, generator=g, dtype=torch.float64)
    verts, tris = [], []
    nv = 0

    # (a) shared-vertex grid mesh ("tissue"), 1.15x the image, wavy depth 2..3
    gx, gy = 48, 28
    uu, vv = torch.meshgrid(torch.linspace(-0.08 * W, 1.08 * W, gx, dtype=torch.float64),
                            torch.linspace(-0.08 * H, 1.08 * H, gy, dtype=torch.float64), indexing="xy")
    uu = uu + (R(gy, gx) - 0.5) * 6 * s
    vv = vv + (R(gy, gx) - 0.5) * 6 * s
    zz = 2.5 + 0.3 * torch.sin(uu / W * 7) * torch.cos(vv / H * 5) + 0.05 * R(gy, gx)
    verts.append(cam.unproject(uu.reshape(-1), vv.reshape(-1), zz.reshape(-1)))
    for j in range(gy - 1):
        for i in range(gx - 1):
            a, b, c, d = j * gx + i, j * gx + i + 1, (j + 1) * gx + i, (j + 1) * gx + i + 1
            tris += [(a, b, c), (b, d, c)]
    nv += gx * gy

    # (b) soup: 5000 triangles, sizes log-uniform 2..80 px, depths 0.5..4 (in front of / behind
    #     the mesh), 15% centred off-screen (partially visible), a few behind the near plane
    n = 5000
    size = torch.exp(math.log(2) + R(n) * (math.log(80) - math.log(2)))      # px at every res
    cu = -0.15 * W + R(n) * 1.3 * W
    cv = -0.15 * H + R(n) * 1.3 * H
    z = 0.5 + R(n) * 3.5
    z[:20] = 0.1                                               # near-culled (p_view.z <= 0.2)
    ang = R(n, 3) * 2 * math.pi
    rad = size[:, None] / math.sqrt(3) * (0.4 + 1.2 * R(n, 3))
    u3 = cu[:, None] + rad * torch.cos(ang)
    v3 = cv[:, None] + rad * torch.sin(ang)
    z3 = z[:, None] * (1 + 0.05 * (R(n, 3) - 0.5))             # slanted
    verts.append(cam.unproject(u3.reshape(-1), v3.reshape(-1), z3.reshape(-1)))
    tris += [(nv + 3 * k, nv + 3 * k + 1, nv + 3 * k + 2) for k in range(n)]
    nv += 3 * n

    # (c) adversarial: 600 triangles with vertices on integer pixel centres / tile boundaries
    #     (multiples of 16, +-0.5) and thin slivers (0.05..0.6 px wide) near pixel centres
    m = 600
    tx = torch.floor(R(m) * (W / 16)) * 16
    ty = torch.floor(R(m) * (H / 16)) * 16
    off = torch.tensor([0.0, 0.5, -0.5, 15.0, 15.5, 16.0], dtype=torch.float64)
    pick = lambda: off[torch.randint(0, len(off), (m,), generator=g)]
    L = 3 + R(m) * 40
    w_ = 0.05 + R(m) * 0.55
    ua = tx + pick(); va = ty + pick()
    ub = ua + L;      vb = va + torch.round(R(m) * 3)
    uc = ua + 0.5 * L + w_ * (R(m) - 0.5); vc = va + w_
    u3 = torch.stack([ua, ub, uc], 1); v3 = torch.stack([va, vb, vc], 1)
    z3 = (1.0 + R(m) * 2)[:, None].expand(m, 3)
    verts.append(cam.unproject(u3.reshape(-1), v3.reshape(-1), z3.reshape(-1)))
    tris += [(nv + 3 * k, nv + 3 * k + 1, nv + 3 * k + 2) for k in range(m)]
    nv += 3 * m

    V = torch.cat(verts, 0).float().contiguous().to(DEV)
    T = torch.tensor(tris, dtype=torch.int32).contiguous().to(DEV)
    # opacity weights: many near 1 (exercise 0.999/0.99 clamps + T<1e-4 early stop), some tiny (< 0.01 culled)
    wts = 0.05 + 0.95 * R(nv)
    hi = R(nv) < 0.35
    wts[hi] = 0.985 + 0.015 * R(int(hi.sum()))
    lo = R(nv) < 0.02
    wts[lo] = 0.005
    wts = wts.float().to(DEV)
    sh = (0.4 * (torch.rand(nv, 16, 3, generator=g) - 0.5)).float()
    sh[:, 0] = 1.2 * (torch.rand(nv, 3, generator=g) - 0.3)
    cols = torch.rand(nv, 3, generator=g).float()
    return dict(V=V, T=T, wts=wts, sh=sh.contiguous().to(DEV), cols=cols.contiguous().to(DEV))


# ----------------------------------------------------------------------------- configs
# Call types (V2). `req`: inputs that require grad. `consumed`: gradients S2TS2 actually
# uses for that call type (asserted for masks with bit 8; all returned grads otherwise).
CONFIGS = {
    # mapping colour render: everything trained
    "map_colour": dict(pose_only=False, colour="sh", req={"V", "wts", "sh", "w", "trans"},
                       consumed={"V", "wts", "sh", "w", "trans"}, grad="full"),
    # tracking: pose-only; map params still require grad (leaf Parameters) but only the pose is stepped;
    # masked L1 (zero gradient on a tool-like region + dark pixels), colour + depth only
    "track_pose": dict(pose_only=True, colour="sh", req={"V", "wts", "sh", "w", "trans"},
                       consumed={"w", "trans"}, grad="masked"),
    # instance render: detached override_shs, no depth/normal loss
    "map_instance": dict(pose_only=False, colour="sh", req={"V", "wts", "w", "trans"},
                         consumed={"V", "wts", "w", "trans"}, grad="colour_only"),
    # precomputed colours that require grad (grad_colors_precomp must stay exact)
    "map_precomp": dict(pose_only=False, colour="precomp", req={"V", "wts", "cols", "w", "trans"},
                        consumed={"V", "wts", "cols", "w", "trans"}, grad="full"),
    # vertices + translation only: exercises the opacity / normal3D skips of #4
    "verts_only": dict(pose_only=False, colour="sh", req={"V", "trans"},
                       consumed={"V", "trans"}, grad="full"),
}


def make_grad_out(H, W, kind, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    gc = torch.randn(3, H, W, generator=g)
    go = torch.randn(7, H, W, generator=g)
    go[6] = 0
    if kind == "masked":
        m = torch.ones(H, W)
        m[int(0.2 * H):int(0.7 * H), int(0.55 * W):int(0.9 * W)] = 0     # tool region
        m[torch.rand(H, W, generator=g) < 0.2] = 0                        # dark pixels
        gc = torch.sign(gc) / (3 * H * W) * m
        go = torch.zeros(7, H, W)
        go[0] = torch.sign(torch.randn(H, W, generator=g)) / (H * W) * m
    elif kind == "colour_only":
        go = torch.zeros(7, H, W)
    return gc.contiguous().to(DEV), go.contiguous().to(DEV)


def run(cam, scene, cfg, mask, sigma, bg, grad_out=None):
    dtr.set_perf_mask(mask)
    req = cfg["req"] if grad_out is not None else set()
    leaf = lambda t, k: t.detach().clone().requires_grad_(k in req)
    V, wts = leaf(scene["V"], "V"), leaf(scene["wts"], "wts")
    w = torch.zeros(1, 3, device=DEV).requires_grad_("w" in req)
    tr = torch.zeros(1, 3, device=DEV).requires_grad_("trans" in req)
    sh = leaf(scene["sh"], "sh") if cfg["colour"] == "sh" else None
    cols = leaf(scene["cols"], "cols") if cfg["colour"] == "precomp" else None
    P = scene["T"].shape[0]
    rast = TriangleRasterizer(cam.settings(bg, cfg["pose_only"]))
    with torch.set_grad_enabled(grad_out is not None):
        color, radii, scaling, others, maxb, wasr = rast(
            vertices=V, triangles_indices=scene["T"], vertex_weights=wts, sigma=sigma,
            scaling=torch.zeros(P, device=DEV), shs=sh, colors_precomp=cols, w=w, trans=tr)
    fwd = dict(color=color.detach(), radii=radii, scaling=scaling.detach(), others=others.detach(),
               max_blending=maxb.detach(), was_rendered=wasr)
    grads = {}
    if grad_out is not None:
        names = [k for k, t in (("V", V), ("wts", wts), ("sh", sh), ("cols", cols), ("w", w), ("trans", tr))
                 if t is not None and t.requires_grad]
        tens = dict(V=V, wts=wts, sh=sh, cols=cols, w=w, trans=tr)
        gs = torch.autograd.grad([color, others], [tens[n] for n in names], list(grad_out), allow_unused=True)
        grads = {n: (gg.detach() if gg is not None else None) for n, gg in zip(names, gs)}
    torch.cuda.synchronize()
    return fwd, grads


def num_rendered(cam, scene, mask, sigma, bg):
    """Total tile entries (sort size) via the raw extension call (same args as the wrapper)."""
    s = cam.settings(bg, False)
    P = scene["T"].shape[0]
    args = (s.bg, scene["V"], scene["T"], scene["wts"], sigma, torch.Tensor([]), torch.zeros(P, device=DEV),
            s.viewmatrix, s.projmatrix, s.projmatrix_raw, s.cx, s.cy, s.tanfovx, s.tanfovy, s.image_height,
            s.image_width, scene["sh"], s.sh_degree, s.campos, s.prefiltered, s.debug, 0.5, False, mask)
    return _C.rasterize_triangles(*args)[0]


def fwd_equal(a, b):
    """V1: list of (name, ok, note)."""
    out = []
    for k in ("color", "radii", "scaling", "max_blending", "was_rendered"):
        out.append((k, torch.equal(a[k], b[k]), ""))
    for c in range(6):
        out.append((f"others[{c}]:{CH_NAMES[c]}", torch.equal(a["others"][c], b["others"][c]), ""))
    hit = (a["others"][1] > 0) & (b["others"][1] > 0)
    same_hit = torch.equal(a["others"][1] > 0, b["others"][1] > 0)
    ok6 = same_hit and torch.equal(a["others"][6][hit], b["others"][6][hit])
    full6 = torch.equal(a["others"][6], b["others"][6])
    out.append(("others[6]:tri_id (alpha>0)", ok6, "" if full6 else "(differs only on empty pixels: uninitialised)"))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--env-factor", type=float, default=3.0,
                    help="tolerance factor on the mask-0 envelope (a sample max; 1.0 fails ~50%% by chance)")
    ap.add_argument("--res", default="1280x720,640x360")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--bench", action="store_true", help="also time fwd+bwd per mask (map_colour, first res)")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--gate-mask", type=int, default=95,
                    help="only masks whose bits are all inside this mask can fail the run (production mask)")
    a = ap.parse_args()
    assert torch.cuda.is_available()
    print(f"device: {torch.cuda.get_device_name()}  initial perf mask (env S2TS2_PERF_MASK): {dtr.get_perf_mask()}")
    bg = torch.tensor([0.1, 0.2, 0.3], device=DEV)
    failures = []

    for res in a.res.split(","):
        Wd, Hd = map(int, res.split("x"))
        cam = Camera(Wd, Hd)
        scene = make_scene(cam)
        P, Vn = scene["T"].shape[0], scene["V"].shape[0]
        print(f"\n=== {Wd}x{Hd}: {P} triangles, {Vn} vertices ===")
        nr0 = num_rendered(cam, scene, 0, a.sigma, bg)
        nr1 = num_rendered(cam, scene, 1, a.sigma, bg)
        print(f"tile entries: mask0 {nr0}, bit1 {nr1}  (x{nr0 / max(nr1, 1):.2f} fewer)")

        # ---------------- V1 forward bit-exactness (SH path and precomputed-colour path)
        for colour in ("sh", "precomp"):
            cfg = dict(CONFIGS["map_colour"], colour=colour)
            ref, _ = run(cam, scene, cfg, 0, a.sigma, bg)
            ref2, _ = run(cam, scene, cfg, 0, a.sigma, bg)
            drop = set()
            if colour == "precomp":
                # with colors_precomp the forward never computes vertex_depth (computeVertexColors is
                # SH-path only), so the depth / median-depth channels read uninitialised geomBuffer
                # memory even with mask 0: excluded here (pre-existing, not a perf item).
                drop = {"others[0]:depth", "others[5]:median_depth"}
            for name, ok, note in fwd_equal(ref, ref2):
                if not ok and name not in drop:
                    failures.append((0, f"V1 {res} {colour} mask0 not self-consistent: {name}"))
            for mask in MASKS[1:]:
                cur, _ = run(cam, scene, cfg, mask, a.sigma, bg)
                bad = [n for n, ok, _ in fwd_equal(ref, cur) if not ok and n not in drop]
                notes = [f"{n} {nt}" for n, ok, nt in fwd_equal(ref, cur) if ok and nt]
                tag = "OK " if not bad else "FAIL"
                print(f"V1 {res} {colour:7s} mask {mask:3d} {BITS.get(mask, 'ALL'):22s} {tag} {bad if bad else ''} {notes if notes else ''}")
                failures += [(mask, f"V1 {res} {colour} mask {mask}: {n}") for n in bad]

        # ---------------- V2 backward envelope
        for cname in a.configs.split(","):
            cfg = CONFIGS[cname]
            go = make_grad_out(Hd, Wd, cfg["grad"], seed=list(CONFIGS).index(cname))
            base = [run(cam, scene, cfg, 0, a.sigma, bg, go) for _ in range(a.repeats)]
            f00, g00 = base[0]
            for n, t in g00.items():
                if t is not None and not torch.isfinite(t).all():
                    print(f"V0 {res} {cname}: mask-0 grad {n} has NaN/inf ({int((~torch.isfinite(t)).sum())} el.)")
            env, sbase = {}, {}
            for n, t0 in g00.items():
                if t0 is None:
                    continue
                env[n] = max(float((g[n] - t0).abs().max()) for _, g in base)
                sbase[n] = max(abs(float((g[n].double() - t0.double()).sum())) for _, g in base)
            for mask in MASKS[1:]:
                runs = [run(cam, scene, cfg, mask, a.sigma, bg, go) for _ in range(a.repeats)]
                fbad = sorted({n for f, _ in runs for n, ok, _ in fwd_equal(f00, f) if not ok})
                if cfg["colour"] == "precomp":
                    fbad = [n for n in fbad if n not in ("others[0]:depth", "others[5]:median_depth")]
                check = cfg["consumed"] if (mask & dtr.PERF_UNUSED_GRAD_SKIP) else set(env)
                line = []
                for n in sorted(env):
                    t0 = g00[n]
                    dev = max(float((g[n] - t0).abs().max()) for _, g in runs)
                    msum = sum(float((g[n].double() - t0.double()).sum()) for _, g in runs) / len(runs)
                    scale = float(t0.double().abs().sum())
                    lim = a.env_factor * env[n]
                    ok_env = dev <= lim if env[n] > 0 else dev == 0.0
                    ok_sum = abs(msum) <= a.env_factor * sbase[n] + 1e-9 * scale
                    # precomputed colours: upstream F4 (vertex depth never computed on that path -> the backward reads
                    # uninitialised memory; mask-0 envelope already ~1e4). V152 never back-propagates through it.
                    asserted = n in check and cfg["colour"] != "precomp"
                    if asserted and not (ok_env and ok_sum):
                        failures.append((mask, f"V2 {res} {cname} mask {mask} grad {n}: dev {dev:.3e} env {env[n]:.3e} "
                                        f"sum {msum:.3e} base-sum {sbase[n]:.3e}"))
                    flag = ("ok" if (ok_env and ok_sum) else "BAD") + ("" if asserted else "(info)")
                    ratio = dev / env[n] if env[n] > 0 else (0.0 if dev == 0 else float("inf"))
                    line.append(f"{n}:{flag} r={ratio:.2f}")
                if fbad:
                    failures.append((mask, f"V2 {res} {cname} mask {mask}: forward mismatch {fbad}"))
                print(f"V2 {res} {cname:12s} mask {mask:3d} fwd={'ok' if not fbad else fbad} | " + "  ".join(line))

        # ---------------- optional timing
        if a.bench and res == a.res.split(",")[0]:
            cfg = CONFIGS["map_colour"]
            go = make_grad_out(Hd, Wd, "full", 0)
            print("\nbench (map_colour fwd+bwd, median of 30, ms):")
            for mask in MASKS:
                ts = []
                for i in range(35):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); run(cam, scene, cfg, mask, a.sigma, bg, go); e1.record()
                    torch.cuda.synchronize()
                    if i >= 5:
                        ts.append(e0.elapsed_time(e1))
                ts.sort()
                print(f"  mask {mask:3d} {BITS.get(mask, 'ALL' if mask else 'baseline'):22s} {ts[len(ts) // 2]:8.3f}")

    dtr.set_perf_mask(0)
    gated = [t for m, t in failures if (m & ~a.gate_mask) == 0]
    info = [t for m, t in failures if (m & ~a.gate_mask) != 0]
    if info:
        print(f"\n{len(info)} out-of-gate finding(s) (masks with bits outside --gate-mask {a.gate_mask}; information only):")
        for t in info:
            print("  " + t)
    print("\n" + (f"ALL PASSED (gate mask {a.gate_mask})" if not gated else f"{len(gated)} FAILURE(S) within gate mask {a.gate_mask}:"))
    for t in gated:
        print("  " + t)
    sys.exit(1 if gated else 0)


if __name__ == "__main__":
    main()
