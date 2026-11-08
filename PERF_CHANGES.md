# Perf changes (branch `s2ts2-perf`, base `a5eb504`)

These are the "Class A" items from the S2TS2 A100 kernel report (§2): #1, #2, #3, #4, #8, #9, #10.
Each one is behind a runtime bitmask. **Default mask = 0 = the original code path.**
None of them was compiled or run on this machine (no nvcc / GPU). GPU verification: `perf_verify.py`.

Not implemented on purpose: #5, #6, #7 (they reassociate float sums), #11 (Python side, S2TS2 tree),
anything that touches the alpha clamps (F2: forward 0.999 vs backward 0.99 stays as it is).

## Switching it on

| bit | item | constant (`cuda_rasterizer/config.h:38-44`, python `PERF_*`) |
|---|---|---|
| 1 | #1 tight tile binning | `PERF_TIGHT_BINNING` |
| 2 | #2 backward starts at the tile's max `last_contributor` | `PERF_BW_START_AT_LAST` |
| 4 | #3 zero-gradient pixel/tile skip and skip of `±0` atomics | `PERF_ZERO_GRAD_SKIP` |
| 8 | #4 unused-gradient skip (`ctx.needs_input_grad`) | `PERF_UNUSED_GRAD_SKIP` |
| 16 | #8 dead shared-memory loads off | `PERF_DEAD_SMEM` |
| 32 | #9 vertex attributes staged in shared memory | `PERF_SMEM_ATTR` |
| 64 | #10 warp-aggregated forward int atomics | `PERF_WARP_AGG_ATOMICS` |

- Env `S2TS2_PERF_MASK` (decimal or `0x..`) is read once when `diff_triangle_rasterization` is imported
  (`diff_triangle_rasterization/__init__.py:66`). After that, use `diff_triangle_rasterization.set_perf_mask(m)` /
  `get_perf_mask()`.
- The value is stored in a host-side static (`rasterize_points.cu:41-53`, bound in `ext.cpp:32-33`). It is **never**
  read by a kernel directly. Each `_RasterizeTriangles.forward` takes a snapshot of it (`__init__.py:137`), passes
  the snapshot to the forward, saves it in `ctx.perf_mask` (`:184`), and the matching backward reuses it (`:224`).
  So changing the mask between a forward and its backward is safe. Kernels receive it as a plain `int` argument.
- **Python API: unchanged for callers.** `TriangleRasterizer(...)(vertices=..., ...)` and `rasterize_triangles(...)`
  have the same signatures, so S2TS2's `mesh_splatting/triangle_renderer/__init__.py` needs no change. Only the
  internal extension calls changed: `_C.rasterize_triangles` gained a trailing `perf_mask`, and
  `_C.rasterize_triangles_backward` gained trailing `perf_mask, grad_flags`. Anything that calls `_C` directly must
  pass them. S2TS2 does not.
- The package name is unchanged (`diff_triangle_rasterization`). Building this clone into the env replaces the
  submodule build. Build for the A100 with `TORCH_CUDA_ARCH_LIST=8.0`. #10 needs sm_70 or newer; below that it falls
  back to plain atomics.

## What "mask 0 = original" means here
Every new code path sits in an `if (perf_mask & BIT)` branch. For #9 this is a template instantiation chosen at
launch, so the `false` kernel is the original one with 1-element staging arrays. With mask 0, the original loads,
arithmetic and atomics are executed in the original order. The SASS is not byte-identical, because the kernels have
extra arguments and untaken branches, so the register allocation can differ. The values computed are the same. The
static shared-memory footprint of the mask-0 kernels is unchanged, apart from one 4-byte `uint32_t` in the backward
(#2).

The A/B baseline is **this build with mask 0**, not a separate pristine build. A pristine build can also be compared
with the same `perf_verify.py` V1/V2 logic.

## Shared exactness facts used below
- **(E1) Contributor counts never leave the kernels.** fw writes `last_contributor` to `out_others[0]` and then
  overwrites it with `D` (`DEPTH_OFFSET == 0`). `n_contrib` and `median_contributor` live in `imgBuffer` and are only
  read by the backward of the same render, which walks the same sorted list. Renumbering list entries therefore
  cannot change any returned tensor, as long as the per-pixel sequence of *accepted* hits is unchanged.
- **(E2) The gradient accumulators start at `+0`** (`torch::zeros`, `rasterize_points.cu:206-222`). In round-to-nearest,
  `+0 + (−0) = +0` and `x + (−x) = +0`, so an accumulator never becomes `−0`. For every other `x` (including inf and
  NaN), `x + (±0) == x`. So leaving out addends that are exactly `±0` gives the same values for every interleaving
  of the remaining non-zero addends. Their order was already non-deterministic in the baseline, because of the float
  atomics. Pathological caveat: under FTZ, a non-zero accumulator whose sum lands in the subnormal range
  (|x| < 1.2e-38) is flushed to a signed zero, which can be `−0`. Then the skip can change the *sign of a zero*
  (`−0` vs `+0`). These compare equal and are `torch.equal`. Subnormal addends are flushed by the FTZ compare and by
  the float `atomicAdd` alike.

---

## #1 Tight tile binning (bit 1)
**Changed**
- `cuda_rasterizer/auxiliary.h:341-379`: new `tileMayContainTriangle(n, o, tx, ty)`.
- `cuda_rasterizer/forward.cu:477-490` (`preprocessCUDA`): `tiles_touched` = the number of bbox tiles that pass the test.
- `cuda_rasterizer/rasterizer_impl.cu:115-119` (`duplicateWithKeys`): the same test skips emitting a key for a tile.
- New `duplicateWithKeys` arguments `edge_normals`, `edge_offsets` (`geomState.normals/offsets`) and `perf_mask`.
- `radii`, `scaling`, `rect_min/max` and every other per-triangle output are unchanged. `radii` is returned as
  `visibility_filter` in S2TS2.

**Why it is output-identical.** The forward (`renderCUDA`) counts pixel centre `p` (integer coordinates) for an
entry only if `dist_k = n_k.x*px + n_k.y*py + o_k <= 0` in float for all three stored edges. Otherwise it
`continue`s, and the only side effect is `contributor++` (E1). The test drops a tile only if, for some edge `k`, the
**exact** edge function is greater than `bound = 1e-5*S + 1e-3` over every pixel centre of the whole 16×16 tile,
where `S = |n.x|*x_max + |n.y|*y_max + |o|`.

- The float evaluation error of the forward's `dist_k` is at most ≈ 3·2⁻²⁴·S (two products, two sums, with or
  without FMA contraction, in any order). The bound is more than 50× that, plus 1e-3 px of absolute slack (which also
  covers FTZ). So a dropped tile contains no pixel with `fl(dist_k) <= 0`.
- The minimum of a linear function over the tile's rectangle of pixel centres is at a corner. That corner value is
  computed in double. A float times an integer < 2¹³ is exact in double, and the sums round at 2⁻⁵³.
- NaN or inf edge data make the comparison false, so the tile is kept. In the forward, a NaN edge constrains nothing.
- The ±5 px bbox is kept as the outer superset. The test only removes tiles from it and never adds any.
- The kept entries of a tile are a subsequence of the original sequence in the same order. The radix sort is
  stable, and ties keep triangle-index order. The pixel's accepted-hit sequence is therefore unchanged, so colour,
  depth, alpha, normals, median depth, `max_blending` and `was_rendered` are bit-identical.

**Count = emit.** `preprocessCUDA` (count) and `duplicateWithKeys` (emit) call the same function on the same stored
floats and the same tile coordinates. It uses only `__dmul_rn` / `__dadd_rn`, `fabs` and comparisons. Contraction is
impossible, so the two always agree. A mismatch would corrupt the sort, so this is the item to watch.

**CPU cross-check (done here).** On the `perf_verify.py` scene, a float32 brute-force pixel test (both evaluation
orders) against the double tile test gave **0 violations**. Tile entries emitted / kept / truly needed:

| res | emitted | kept | needed |
|---|---|---|---|
| 1280×720 | 51558 | 27545 | 22164 |
| 640×360 | 38791 | 21366 | 16303 |

That is about 1.85× fewer entries. The report's sim found 2.1–2.4× for a different size mix.

**Verify:** V1 `torch.equal` on all forward outputs (bit 1 and all bits). V2 envelope. The `tile entries` line printed
by `perf_verify.py` shows how many entries were removed.

## #2 Backward walks only up to the tile's max `last_contributor` (bit 2)
**Changed**
- `cuda_rasterizer/backward.cu:948-958`: `rounds` is no longer `const`, and there is a new `range_end` and
  `__shared__ s_max_last_contributor`.
- `backward.cu:1045-1064`: block max-reduce (shared `atomicMax`) of `last_contributor` over the active pixels. Then
  `range_end = range.x + L`, `toDo = L`, `rounds = ceil(L/256)` and `contributor = L`.
- `backward.cu:1073,1075`: the loads use `range_end` (== `range.y` with mask 0).

**Why it is output-identical.** A pixel skips every entry whose relative index is `>= last_contributor`
(`contributor >= last_contributor → continue`, with no side effect). The entries at relative index `>= L`, where
`L` is the block max, are therefore skipped by every pixel. Starting the reverse walk at `range.x + L - 1` with
`contributor = L` keeps `contributor` equal to each entry's relative index after the decrement, so the
`median_contributor - 1` test is unchanged. `L` is clamped to `range.y - range.x`, which is a no-op for consistent
buffers. The branch condition is uniform (a kernel argument), so the `block.sync()` calls inside it are legal.

**Verify:** V2 envelope (bit 2 and all bits).

## #3 Zero-gradient pixel / tile skip and skip of exactly-zero atomics (bit 4)
**Changed**
- `cuda_rasterizer/backward.cu:30-41`: `atomicAddSkipZero(addr, v, skip_zero)`. It drops `v == 0` addends only when
  `skip_zero` is set; otherwise it is a plain `atomicAdd`.
- `backward.cu:939`: `zskip`.
- All 20 `atomicAdd` call sites in the render loop go through it. Per hit that is: colour 9, depth 3, median depth 3,
  `points2D` 6, `normal3D` 3, normals/offsets 3 per active edge, opacity 1.
- `backward.cu:1018-1043`: a pixel whose 10 dL inputs (colour C, depth, alpha, normal×3, median depth) are all `±0`
  is marked `done`. A block whose pixels are all `done` returns. This is `__syncthreads_count`, a uniform branch.

**Why it is output-identical.** Every addend of a pixel is a product of one of those dL inputs with *finite* factors:
- `alpha <= 0.99`, so `T` and `1/(1−alpha)` are finite.
- A hit only reaches the gradient code with `alpha >= 1/255`. That needs `Cx > 0`, which needs `phi_x < 0` strictly:
  `phi_x == 0` gives `__powf(0, sigma) = 0` for `sigma > 0`, and FTZ removes subnormal `phi_x`. So `sigma/phi_x` is
  finite.
- `invDen` is finite for triangles that passed the forward's inradius cull, since |denom| ≳ 1e-3 px².
- The colours, depths and `bg` are finite.

So every addend of an all-zero pixel is `±0`, and dropping it is exact by (E2). A NaN dL input is not `== 0`, so it
is not skipped. The per-atomic skip needs no finiteness argument at all: it drops only addends that are exactly 0.
S2TS2 uses `sigma = exp(·) > 0`.

**Precondition stated, not enforced:** `sigma > 0`, and finite colours, depths and `bg`. Under those, the baseline
has no `0·inf`.

**Verify:** V2 on `track_pose` (masked L1: tool region and dark pixels have zero gradient) and `map_instance`
(dL_depth ≡ 0). V0 prints any NaN/inf in the mask-0 gradients. If V0 prints anything, the #3 argument does not cover
that case.

## #4 Unused-gradient skip (bit 8)
**Changed**
- `diff_triangle_rasterization/__init__.py:69-91`: `_grad_flags(ctx, optimize_pose_only)` builds the flags from
  `ctx.needs_input_grad`.
- `cuda_rasterizer/config.h:46-51`: `GRAD_NEED_*`.
- `backward.cu:940-945`: `do_colors`, `do_opacity`, `do_normal3D`.
- `backward.cu:1240` (colour atomics, 9 per hit), `:1352` (`dL_dnormal3D`, 3 per hit), `:1385` (`dL_dopacity`, 1 per
  hit).

**Consumers, traced** (every read of each buffer in the package):
- `dL_dcolors` is returned as `grad_colors_precomp` (`rasterize_points.cu:267`, `__init__.py`). It is also read by
  `computeVertexColorGradients` → `computeColorFromSH`, which only writes `dL_dsh`. The SH-dir→position term is
  commented out (bw:229-240), and that kernel only runs when `!optimize_pose_only && colors_precomp == nullptr`
  (`rasterizer_impl.cu:508`). The `dL_dcolor` parameter of `BACKWARD::preprocess` is never read.
  → needed iff `needs_input_grad[5]` (`colors_precomp`), or `needs_input_grad[4]` (`sh`) and not pose-only.
  This follows the report's caveat: the skip is gated on `!needs_input_grad[colors_precomp]`.
- `dL_dopacity` is only read by `BACKWARD::preprocess` → `dL_dvertice_weights`, after the
  `if (optimize_pose_only) return`. → needed iff `needs_input_grad[2]` (`vertex_weights`) and not pose-only.
- `dL_dnormal3D` is only read by `BACKWARD::preprocess` → `dL_dtau[3..5]` (= `grad_w`), after the pose-only return.
  The `dL_dp*` values computed from it there are dead code. → needed iff `needs_input_grad[7]` (`w`) and not pose-only.

**Why it is output-identical for consumed outputs.** A skipped buffer stays all-zero. Its only consumers are outputs
for which autograd did not ask, or `grad_sh`, which is all-zero anyway in pose-only mode because its kernel is
skipped. Every buffer that has a consumer is computed by exactly the original code. The flags are ignored unless
bit 8 is set.

**S2TS2 call types (what bit 8 removes):**

| call type | what bit 8 skips |
|---|---|
| tracking (pose-only) | colour (sh/colors not needed), opacity and normal3D atomics: 13 of the ~34 per hit |
| instance render (`override_shs` detached) | colour atomics |
| mapping colour | nothing, if `w` (`cam_rot_delta`) requires grad. If it does not, normal3D atomics too. |

**Verify:** V2. For bit 8 it asserts only the consumed gradients of each call type, `CONFIGS[...]['consumed']` in
`perf_verify.py`:

| config | asserted gradients |
|---|---|
| `track_pose` | `w`, `trans` |
| `map_instance` | `V`, `wts`, `w`, `trans` |
| `verts_only` | `V`, `trans` |
| `map_colour`, `map_precomp` | all |

The other gradients are printed as `(info)`.

## #8 Dead shared-memory loads removed (bit 16)
**Changed**
- `cuda_rasterizer/forward.cu:599-604`: `collected_depths` and `collected_xy`.
- `cuda_rasterizer/backward.cu:1078-1087`: `collected_colors`, `collected_depths` and `collected_xy`.

These loads are skipped. The arrays stay declared, so the mask-0 smem footprint is unchanged.

**Why it is output-identical.** These arrays are written and never read. The only other mention is a comment at
bw "not collected_depths[j]". Removing them also removes F1: the backward `collected_colors` load reads the
**per-vertex** `colors` array with a triangle index (`coll_id*C + i`, `coll_id < P`, `P ≈ 2V`), which is out of
bounds.

**Verify:** V1 and V2. ptxas `-v`: nvcc may already have eliminated these stores, in which case this bit is a no-op
apart from the global loads.

## #9 Per-entry vertex attributes in shared memory (bit 32)
**Changed**
- `renderCUDA` gains `template <..., bool SMEM_ATTR>` in both `cuda_rasterizer/forward.cu:501` and
  `cuda_rasterizer/backward.cu:896`. The launch picks the instantiation from the mask
  (`forward.cu:810`, `backward.cu:1533`).
- When true, each batch also stages the entry's 3 vertex indices, 3 `vertex_depth` values and 3·C colours.
- The per-hit code reads them from shared memory instead of `triangles_indices`, `vertex_depth` and
  `features`/`colors` in global memory.

**Why it is output-identical.** The values are the same floats and ints, loaded from the same buffers. Nothing writes
those buffers during the kernel. The interpolation and gradient arithmetic is the *same source expression*: only the
load of `c0..c2` and `depth_vertex_*` is selected by a compile-time branch, and the arithmetic after it is shared.

**Cost.** Static shared memory per block:

| kernel | before | with #9 |
|---|---|---|
| forward | 25.6 KB | 41.0 KB |
| backward | 28.7 KB | 44.0 KB |

Both are below the 48 KB static limit. On the A100 (164 KB smem per SM) this lowers the possible blocks per SM from
6 to 4 (fwd) and from 5 to 3 (bwd), if smem is the limiter. **#9 can be slower if the kernels are occupancy-bound.**
Check `-Xptxas -v` and A/B it.

**Verify:** V1 and V2, the ptxas report, and timing (`perf_verify.py --bench`).

## #10 Warp-aggregated forward atomics (bit 64)
**Changed**
- `cuda_rasterizer/forward.cu:29-58`: `warpAggAtomicMaxInt` and `warpAggAtomicAddOneInt`.
- They are used at `forward.cu:673` (`max_blending` `atomicMax`) and `:751` (`was_rendered` `+1`).
- The lanes executing the call are grouped by triangle id with `cg::labeled_partition(cg::coalesced_threads(), j_id)`.
- One lane per group issues the atomic: for `max_blending`, with `cg::reduce(..., cg::greater<int>())` of the same
  int bit patterns; for `was_rendered`, with the group size.

**Why it is output-identical.** Integer max and integer add are exact, associative and commutative. The original
already applies `atomicMax` to the float's int bit pattern. Below sm_70 (and in the host pass) the helpers fall back
to plain atomics.

**Verify:** V1 (`max_blending` and `was_rendered` are `torch.equal`).

---

## Pre-existing issues found (not changed, mask-independent)
- **F1** (fixed only under bit 16): backward `collected_colors` reads out of bounds. See #8.
- **F2**: forward clamps alpha at 0.999, backward at 0.99. Unchanged by design.
- **F3**: `int pixel_influence;` (fw `renderCUDA`) is uninitialised for pixels with no accepted hit, and it is
  written to `out_others[6]` (S2TS2's per-pixel triangle id, `allmap[6]`). Its value on empty pixels is
  whatever the register held, so it can differ between builds, masks or launches. `perf_verify.py` compares channel 6
  only where alpha > 0, where it is always written.
- **F4**: `geomState.vertex_depth` is only computed on the SH path (`computeVertexColors` runs only when
  `colors_precomp == nullptr`). With precomputed colours, the depth and median-depth outputs, and the depth gradients,
  read uninitialised memory. S2TS2 only uses `override_color` under `no_grad` (`sem4d.py`). `perf_verify.py`
  excludes those two channels on the precomputed path.
- **F5**: the `debug=True` branch of `_RasterizeTriangles.backward` unpacks 5 results under the wrong names, and
  `grad_tau` is then undefined (NameError). Untouched.

## Not verifiable here (needs the A100)
- Compilation. This was re-read by hand, but never compiled. The CG `labeled_partition` / `reduce` and
  `__dmul_rn` / `__dadd_rn` usage targets CUDA ≥ 11.
- All V1/V2 assertions, the speed-ups, and the occupancy effect of #9.
- Whether nvcc already removed the #8 stores (ptxas smem bytes).

## Running the checks
```
TORCH_CUDA_ARCH_LIST=8.0 pip install . --no-build-isolation
python perf_verify.py                 # V0/V1/V2 at 1280x720 and 640x360, exit 0 = pass
python perf_verify.py --bench         # + fwd+bwd timing per mask (map_colour, 1280x720)
S2TS2_PERF_MASK=127 python ...        # S2TS2 end to end (V4 in the report)
```
V2 compares against the mask-0 envelope with a tolerance factor (`--env-factor`, default 3). The envelope is a
sample maximum, so a factor of 1.0 would fail about half the time by chance, even for an identical kernel. Each
reported `r=` is the ratio max-deviation / envelope.
