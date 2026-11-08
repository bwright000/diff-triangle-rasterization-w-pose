#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from typing import NamedTuple
import os
import torch.nn as nn
import torch
from . import _C

# ---------------------------------------------------------------------------
# Runtime performance bitmask (see PERF_CHANGES.md and cuda_rasterizer/config.h).
# 0 (default) = the original code path. Bits:
#   1 = #1 tight tile binning          2 = #2 backward starts at max last_contributor
#   4 = #3 zero-gradient pixel/tile skip 8 = #4 unused-gradient skip (ctx.needs_input_grad)
#  16 = #8 dead shared-memory loads off 32 = #9 vertex attributes in shared memory
#  64 = #10 warp-aggregated forward atomics
# The initial value is read once at import from the env var S2TS2_PERF_MASK
# (decimal or 0x.. hex). Each forward call snapshots the current mask and its
# backward reuses that snapshot, so changing the mask between a forward and
# its backward is safe.
# ---------------------------------------------------------------------------
PERF_TIGHT_BINNING = 1
PERF_BW_START_AT_LAST = 2
PERF_ZERO_GRAD_SKIP = 4
PERF_UNUSED_GRAD_SKIP = 8
PERF_DEAD_SMEM = 16
PERF_SMEM_ATTR = 32
PERF_WARP_AGG_ATOMICS = 64
PERF_MASK_ALL = 127

# grad_flags for #4 (must match cuda_rasterizer/config.h)
_GRAD_NEED_COLORS = 1
_GRAD_NEED_OPACITY = 2
_GRAD_NEED_NORMAL3D = 4


def set_perf_mask(mask):
    """Set the process-wide perf bitmask used by subsequent forward calls (0..127)."""
    mask = int(mask)
    if mask < 0 or mask > PERF_MASK_ALL:
        raise ValueError(f"perf mask must be in [0, {PERF_MASK_ALL}], got {mask}")
    _C.set_perf_mask(mask)


def get_perf_mask():
    return int(_C.get_perf_mask())


set_perf_mask(int(os.environ.get("S2TS2_PERF_MASK", "0"), 0))


def _grad_flags(ctx, optimize_pose_only):
    """Item #4: which render-kernel gradient buffers have a consumer.

    Indices of ctx.needs_input_grad follow _RasterizeTriangles.forward's inputs:
      0 vertices, 1 triangles_indices, 2 vertex_weights, 3 sigma, 4 sh,
      5 colors_precomp, 6 scaling, 7 w (-> dL_dtau[3:6]), 8 trans, 9 raster_settings.
    - dL_dcolors is returned as grad_colors_precomp and feeds dL_dsh, which is only
      computed when not optimize_pose_only (computeVertexColorGradients is skipped
      otherwise and grad_sh stays zero).
    - dL_dopacity only feeds dL_dvertice_weights, after the pose-only early return.
    - dL_dnormal3D only feeds dL_dtau[3:6] (= grad_w), after the pose-only early return.
    """
    needs = ctx.needs_input_grad
    flags = 0
    if needs[5] or (needs[4] and not optimize_pose_only):
        flags |= _GRAD_NEED_COLORS
    if needs[2] and not optimize_pose_only:
        flags |= _GRAD_NEED_OPACITY
    if needs[7] and not optimize_pose_only:
        flags |= _GRAD_NEED_NORMAL3D
    return flags

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_triangles(
    vertices,
    triangles_indices,
    vertex_weights,
    sigma,
    sh,
    colors_precomp,
    scaling,
    w,
    trans,
    raster_settings,
):
    return _RasterizeTriangles.apply(
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        sh,
        colors_precomp,
        scaling,
        w,
        trans,
        raster_settings,
    )

class _RasterizeTriangles(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        vertices,
        triangles_indices,
        vertex_weights,
        sigma,
        sh,
        colors_precomp,
        scaling,
        w,
        trans,
        raster_settings,
    ):

        # Snapshot the perf mask: the matching backward uses the same value.
        perf_mask = get_perf_mask()

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            colors_precomp,
            scaling,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.cx,
            raster_settings.cy,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            getattr(raster_settings, 'min_triangle_area_pixels', 0.5),
            getattr(raster_settings, 'opacity_min', False),
            perf_mask,
        )


        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending = _C.rasterize_triangles(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending = _C.rasterize_triangles(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.sigma = sigma
        ctx.perf_mask = perf_mask
        ctx.save_for_backward(vertices, triangles_indices, vertex_weights, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, scaling, depth, max_blending, was_rendered

    @staticmethod
    def backward(ctx, grad_out_color, _, __, grad_depth, _____, _______):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        sigma = ctx.sigma
        vertices, triangles_indices, vertex_weights, colors_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                vertices,
                triangles_indices,
                vertex_weights,
                sigma,
                radii, 
                colors_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.projmatrix_raw,
                raster_settings.cx,
                raster_settings.cy,
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                getattr(raster_settings, 'optimize_pose_only', False),
                getattr(raster_settings, 'opacity_min', False),
                ctx.perf_mask,
                _grad_flags(ctx, getattr(raster_settings, 'optimize_pose_only', False)))

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_vertices, grad_vertice_weights, grad_sigma, grad_colors_precomp, grad_sh = _C.rasterize_triangles_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_vertices, grad_vertice_weights, grad_colors_precomp, grad_sh, grad_tau = _C.rasterize_triangles_backward(*args)

        # grad_tau is (6,) with [translation(3), rotation(3)]
        dL_dtau = grad_tau.view(6)  # or grad_tau if already 1D
        grad_trans = dL_dtau[:3].view(1, -1)   # (1, 3) for cam_trans_delta
        grad_w    = dL_dtau[3:6].view(1, -1)   # (1, 3) for cam_rot_delta

        grads = (
            grad_vertices,  
            None,  # triangles_indices
            grad_vertice_weights, # needs to be changed later to vertex_weights
            None, # grad_sigma
            grad_sh,
            grad_colors_precomp,
            None,
            grad_w,
            grad_trans,
            None
        )

        return grads

class TriangleRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

    cx: float
    cy: float
    projmatrix_raw : torch.Tensor
    # Minimum screen-space triangle area (pixels²). Triangles below this are culled (removes slivers).
    min_triangle_area_pixels: float = 0.5
    # When True, backward only computes pose gradients (dL_dtau); skips vertex/color/normal grads for speed.
    optimize_pose_only: bool = False
    # Triangle opacity aggregation: False = mean of the three vertex weights,
    opacity_min: bool = False

class TriangleRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, vertices, triangles_indices, vertex_weights, sigma, scaling,  shs = None, colors_precomp = None, w = None, trans = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if w is None:
            w = torch.Tensor([])

        if trans is None:
            trans = torch.Tensor([])


        # Invoke C++/CUDA rasterization routine
        return rasterize_triangles(
            vertices,
            triangles_indices,
            vertex_weights,
            sigma,
            shs,
            colors_precomp,
            scaling,
            w,
            trans,
            raster_settings, 
        )


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)