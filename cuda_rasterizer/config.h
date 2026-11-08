/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#define MAX_NB_POINTS 3

// ---------------------------------------------------------------------------
// Runtime performance bitmask (S2TS2 A100 "Class A" items, see PERF_CHANGES.md).
// Passed into every kernel as a plain argument. mask == 0 runs the original
// code path; every optimisation lives in an `if (perf_mask & BIT)` branch.
// ---------------------------------------------------------------------------
#define PERF_TIGHT_BINNING     1   // #1  drop tiles no counted pixel can lie in
#define PERF_BW_START_AT_LAST  2   // #2  backward starts at the tile's max last_contributor
#define PERF_ZERO_GRAD_SKIP    4   // #3  zero-gradient pixel/tile skip + skip of +-0 atomics
#define PERF_UNUSED_GRAD_SKIP  8   // #4  skip atomics of gradients nobody consumes (grad_flags)
#define PERF_DEAD_SMEM         16  // #8  remove dead shared-memory loads
#define PERF_SMEM_ATTR         32  // #9  per-entry vertex attributes in shared memory
#define PERF_WARP_AGG_ATOMICS  64  // #10 warp-aggregated forward atomicMax / int atomicAdd
#define PERF_MASK_ALL          127

// grad_flags (item #4): which render-kernel gradient buffers are consumed downstream.
// Only read when (perf_mask & PERF_UNUSED_GRAD_SKIP); otherwise all are computed.
#define GRAD_NEED_COLORS       1   // dL_dcolors   (-> grad_colors_precomp, dL_dsh)
#define GRAD_NEED_OPACITY      2   // dL_dopacity  (-> grad_vertex_weights)
#define GRAD_NEED_NORMAL3D     4   // dL_dnormal3D (-> dL_dtau[3:6] = grad_w)
#define GRAD_NEED_ALL          7

#endif