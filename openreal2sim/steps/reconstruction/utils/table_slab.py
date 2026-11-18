#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometry helpers for building the table as a flat textured slab, used by
background_mesh_generation.py (bg_table_mode="slab").

A plane is fitted on the ground-mask pixels of the inpainted background depth, the mask
outline is projected onto it to give a footprint, and the footprint is extruded downward
into a slab whose top face is textured by the camera projection of the background image.
"""

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

# Set True to have build_table_slab() also measure the texture projection error and the
# closedness of the result. Off in the pipeline: it costs a mesh copy and a vertex weld.
SLAB_DIAGNOSTICS = False


# ──────────────────── camera ─────────────────────
def depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a depth map with intrinsics K into camera-frame points.

    Returns (H,W,3) for a (H,W) depth map; callers that want a flat list reshape it.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ii, jj = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    z = depth.astype(np.float64)
    x = (ii - cx) * z / fx
    y = (jj - cy) * z / fy
    return np.stack([x, y, z], -1)

def project_to_uv(pts: np.ndarray, K: np.ndarray, H: int, W: int,
                  clamp: bool = True) -> np.ndarray:
    """Project camera-frame points to the texture coordinates the per-pixel background mesh
    uses: u = col / (W-1), v = 1 - row / (H-1).

    With `clamp` the result is limited to [0,1]; without it the caller can tell which points
    fall outside the image.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.where(np.abs(pts[:, 2]) < 1e-9, 1e-9, pts[:, 2])
    u = (fx * pts[:, 0] / z + cx) / max(W - 1, 1)
    v = 1.0 - (fy * pts[:, 1] / z + cy) / max(H - 1, 1)
    uv = np.stack([u, v], -1)
    if not clamp:
        return uv
    uv[pts[:, 2] <= 0] = 0.0
    return np.clip(uv, 0.0, 1.0).astype(np.float32)

def pixels_to_plane_points(px: np.ndarray, K: np.ndarray,
                           normal: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    """Intersect the camera rays of the given pixels with the plane.
    Returns the 3D points and a validity flag (ray hits the plane in front of the camera).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dirs = np.stack([(px[:, 0] - cx) / fx, (px[:, 1] - cy) / fy, np.ones(len(px))], -1)
    denom = dirs @ normal
    valid = np.abs(denom) > 1e-9
    t = np.full(len(px), np.nan)
    t[valid] = -offset / denom[valid]
    valid &= np.isfinite(t) & (t > 0)
    return dirs * t[:, None], valid


# ──────────────────── plane ─────────────────────
def fit_plane_trimmed(pts: np.ndarray, dist_thresh: float = 0.01,
                      ransac_n: int = 10, num_iterations: int = 2000,
                      refine_iters: int = 3) -> tuple[np.ndarray, float, np.ndarray]:
    """Robust plane fit: the RANSAC fit the per-pixel path already uses, refined by a
    least squares fit on the inliers (trimmed least squares).

    Returns (normal, offset, inlier_mask) for the plane `normal . x + offset = 0`.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane_model, _ = pcd.segment_plane(distance_threshold=dist_thresh,
                                       ransac_n=ransac_n, num_iterations=num_iterations)
    a, b, c, d = plane_model
    normal = np.array([a, b, c], np.float64)
    nrm = np.linalg.norm(normal) + 1e-12
    normal, offset = normal / nrm, float(d) / nrm

    inliers = np.abs(pts @ normal + offset) <= dist_thresh
    for _ in range(refine_iters):
        if inliers.sum() < 3:
            break
        sel = pts[inliers]
        centroid = sel.mean(axis=0)
        _, _, vh = np.linalg.svd(sel - centroid, full_matrices=False)
        n_new = vh[-1]
        if np.dot(n_new, normal) < 0:
            n_new = -n_new
        normal = n_new / (np.linalg.norm(n_new) + 1e-12)
        offset = float(-normal @ centroid)
        inliers = np.abs(pts @ normal + offset) <= dist_thresh
    return normal, offset, inliers

def plane_rms(pts: np.ndarray, normal: np.ndarray, offset: float) -> float:
    r = pts @ normal + offset
    return float(np.sqrt(np.mean(r ** 2)))

def plane_frame(normal: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes."""
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, helper)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(normal, e1)
    e2 /= (np.linalg.norm(e2) + 1e-12)
    return e1, e2


# ──────────────────── mask and footprint ─────────────────────
def clean_ground_mask(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component with its holes filled."""
    m = mask.astype(np.uint8)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lab > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == biggest).astype(np.uint8)
    # fill holes: flood the complement from a border that is guaranteed to be outside,
    # everything in the complement that is not reached is a hole
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inv = (pad == 0).astype(np.uint8)
    n_lab, lab, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    outside = lab[0, 0]
    holes = (inv == 1) & (lab != outside)
    pad[holes] = 1
    return pad[1:-1, 1:-1].astype(bool)

def mask_outline(mask: np.ndarray, simplify_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Outline of a cleaned mask: the simplified pixel polygon and every boundary pixel.

    The dense outline is kept because it is what tells which footprint edges run along the
    image border; the simplified one is only the shape.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64)
    cnt = max(cnts, key=cv2.contourArea)
    dense = cnt.reshape(-1, 2).astype(np.float64)
    simple = cnt
    if simplify_px and simplify_px > 0:
        simple = cv2.approxPolyDP(cnt, float(simplify_px), True)
    return simple.reshape(-1, 2).astype(np.float64), dense


# ──────────────────── polygons ─────────────────────
def polygon_signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def drop_collinear(poly: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Remove polygon vertices that sit within `tol` metres of the segment between their
    neighbours. Nearly collinear vertices survive the pixel-space simplification but make
    ear clipping emit sliver triangles, which uniform subdivision then multiplies into
    thousands of zero-area faces with meaningless normals.
    """
    keep = list(range(len(poly)))
    changed = True
    while changed and len(keep) > 3:
        changed = False
        for k in range(len(keep)):
            a, b, c = poly[keep[k - 1]], poly[keep[k]], poly[keep[(k + 1) % len(keep)]]
            ac, ab = c - a, b - a
            L = np.linalg.norm(ac)
            dist = (abs(ac[0] * ab[1] - ac[1] * ab[0]) / L) if L > 1e-12 else np.linalg.norm(ab)
            if dist < tol:
                keep.pop(k)
                changed = True
                break
    return poly[keep]

def _edge_normals(poly: np.ndarray) -> np.ndarray:
    """Outward normal of every edge of a counter-clockwise polygon."""
    e = np.roll(poly, -1, axis=0) - poly
    n = np.stack([e[:, 1], -e[:, 0]], -1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab + 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

def border_edge_flags(poly: np.ndarray, src: np.ndarray, src_border: np.ndarray,
                      tol: float) -> np.ndarray:
    """Mark the footprint edges that run along the image border, by asking which outline
    points support each edge. Works for a hull or a rectangle, where the footprint
    vertices are not outline points themselves.
    """
    flags = np.zeros(len(poly), bool)
    for i in range(len(poly)):
        near = _seg_dist(src, poly[i], poly[(i + 1) % len(poly)]) < tol
        if near.sum() >= 3:
            flags[i] = src_border[near].mean() > 0.5
    return flags

def extend_border_edges(poly: np.ndarray, edge_on_border: np.ndarray, margin: float) -> np.ndarray:
    """Push the footprint edges that run along the image border outward by `margin`, so the
    slab reaches past the view frustum. Every other edge, in particular the real far edge
    of the table, stays exactly where the mask put it.
    """
    if margin <= 0 or not edge_on_border.any():
        return poly
    n = len(poly)
    nrm = _edge_normals(poly)
    out = []
    for i in range(n):
        prev_e, next_e = (i - 1) % n, i
        p = poly[i]
        if edge_on_border[prev_e] and edge_on_border[next_e]:
            # both sides move: mitre the corner so no notch is left between them
            a1, n1 = poly[prev_e] + margin * nrm[prev_e], nrm[prev_e]
            a2, n2 = p + margin * nrm[next_e], nrm[next_e]
            d1 = np.array([-n1[1], n1[0]])
            d2 = np.array([-n2[1], n2[0]])
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                out.append(p + margin * (n1 + n2) / (np.linalg.norm(n1 + n2) + 1e-12))
            else:
                s = ((a2 - a1)[0] * d2[1] - (a2 - a1)[1] * d2[0]) / den
                q = a1 + s * d1
                if np.linalg.norm(q - p) > 4 * margin:   # cap a degenerate mitre
                    q = p + margin * (n1 + n2) / (np.linalg.norm(n1 + n2) + 1e-12)
                out.append(q)
        elif edge_on_border[prev_e]:
            out.append(p + margin * nrm[prev_e])
            out.append(p)
        elif edge_on_border[next_e]:
            out.append(p)
            out.append(p + margin * nrm[next_e])
        else:
            out.append(p)
    return np.asarray(out, np.float64)

def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point in polygon test."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        dy = y2 - y1
        straddles = (y1 > y) != (y2 > y)
        xint = x1 + (y - y1) * (x2 - x1) / (dy if abs(dy) > 1e-18 else 1e-18)
        inside ^= straddles & (x < xint)
    return inside

def triangulate_polygon_2d(poly: np.ndarray) -> np.ndarray:
    """Ear clipping for a simple polygon given counter-clockwise. Falls back to a fan."""
    n = len(poly)
    if n < 3:
        return np.zeros((0, 3), np.int64)

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def inside(p, a, b, c):
        return area2(a, b, p) >= 0 and area2(b, c, p) >= 0 and area2(c, a, p) >= 0

    idx = list(range(n))
    faces = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        cut = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if area2(a, b, c) <= 1e-12:
                continue
            if any(inside(poly[m], a, b, c) for m in idx if m not in (i0, i1, i2)):
                continue
            faces.append([i0, i1, i2])
            idx.pop(k)
            cut = True
            break
        if not cut:
            break
    if len(idx) == 3:
        faces.append(list(idx))
    elif len(idx) > 3:
        print(f"[Warning] ear clipping stalled with {len(idx)} vertices left, using a fan")
        faces = [[0, i, i + 1] for i in range(1, n - 1)]
    return np.asarray(faces, np.int64)

def footprint_polygon(ground_mask: np.ndarray, K: np.ndarray, normal: np.ndarray,
                      offset: float, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                      mode: str = "hull", simplify_px: float = 4.0,
                      border_px: int = 2, margin: float = 0.3) -> tuple[np.ndarray, dict]:
    """Footprint of the table in plane coordinates (metres, basis e1/e2 about `origin`).

    The cleaned ground-mask outline is projected onto the plane, reduced to a convex hull /
    minimum-area rectangle / the simplified outline itself, and the edges that run along the
    image border are pushed out by `margin`.
    """
    H, W = ground_mask.shape
    mask = clean_ground_mask(ground_mask)
    px, dense_px = mask_outline(mask, simplify_px)
    diag = {"outline_px": int(len(px)), "outline_px_dense": int(len(dense_px)),
            "mask_px": int(mask.sum()), "mask_px_raw": int(ground_mask.sum())}
    if len(px) < 3:
        return np.zeros((0, 2)), diag

    def to_plane(p):
        on_border = ((p[:, 0] <= border_px) | (p[:, 0] >= W - 1 - border_px) |
                     (p[:, 1] <= border_px) | (p[:, 1] >= H - 1 - border_px))
        pts3, valid = pixels_to_plane_points(p, K, normal, offset)
        rel = pts3[valid] - origin
        return np.stack([rel @ e1, rel @ e2], -1), on_border[valid], int((~valid).sum())

    poly, _, dropped = to_plane(px)
    src, on_border, dropped_dense = to_plane(dense_px)
    diag["outline_dropped_no_hit"] = dropped
    diag["outline_dense_dropped_no_hit"] = dropped_dense
    if len(poly) < 3 or len(src) < 3:
        return np.zeros((0, 2)), diag
    diag["outline_on_border_dense"] = int(on_border.sum())

    # shape from the simplified outline (few vertices), border flags voted by the dense one
    if mode == "rect":
        poly = cv2.boxPoints(cv2.minAreaRect(poly.astype(np.float32))).astype(np.float64)
    elif mode == "hull":
        poly = cv2.convexHull(poly.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    elif mode != "outline":
        raise ValueError(f"unknown bg_table_footprint_mode: {mode}")
    if polygon_signed_area(poly) < 0:
        poly = poly[::-1]
    poly = drop_collinear(poly)
    area0 = abs(polygon_signed_area(poly))
    diag["area_before_margin"] = area0

    # only the edges that run along the image border are pushed out
    flags = border_edge_flags(poly, src, on_border, tol=0.05 * np.sqrt(max(area0, 1e-6)))
    diag["edges_on_border"] = f"{int(flags.sum())}/{len(flags)}"
    poly = extend_border_edges(poly, flags, margin)
    if polygon_signed_area(poly) < 0:
        poly = poly[::-1]
    poly = drop_collinear(poly)
    diag["footprint_vertices"] = int(len(poly))
    diag["footprint_area"] = abs(polygon_signed_area(poly))
    return poly, diag

def slab_shadow_mask(bg_depth: np.ndarray, K: np.ndarray, normal: np.ndarray, offset: float,
                     origin: np.ndarray, e1: np.ndarray, e2: np.ndarray, poly2: np.ndarray,
                     ground_mask: np.ndarray | None = None, eps: float = 0.01) -> np.ndarray:
    """Pixels the slab takes over from the per-pixel layer: the camera ray meets the slab
    footprint, and the pixel is either ground (the slab *is* that surface now) or sits on or
    behind the slab (it would be hidden).

    Catching the ground pixels explicitly matters where the real tabletop is not flat: those
    pixels are up to a few centimetres in front of the fitted plane and would otherwise stay
    as a bumpy sheet poking through the slab. Everything the slab does not cover is kept:
    the table's far edge, the walls, whatever is in front of the plane outside the mask.
    """
    H, W = bg_depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ii, jj = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    dirs = np.stack([(ii - cx) / fx, (jj - cy) / fy, np.ones_like(ii, np.float64)], -1)
    den = dirs @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(den) > 1e-9, -offset / den, np.nan)
    hit = dirs * t[..., None]
    rel = hit - origin
    p2 = np.stack([rel @ e1, rel @ e2], -1).reshape(-1, 2)
    inside = points_in_polygon(np.nan_to_num(p2), poly2).reshape(H, W)
    taken = bg_depth >= t - eps
    if ground_mask is not None:
        taken |= ground_mask
    return np.isfinite(t) & (t > 0) & inside & taken


# ──────────────────── meshing ─────────────────────
def boundary_edges(faces: np.ndarray, directed: bool = False) -> np.ndarray:
    """Edges used by a single face. With `directed` the winding is kept, so side walls can
    be built facing outward; otherwise the edges come back sorted."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    if not directed:
        return uniq[counts == 1]
    return e[counts[inv.reshape(-1)] == 1]

def subdivide_uniform(verts: np.ndarray, faces: np.ndarray,
                      max_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into k^2 similar triangles with the same k, so the result stays
    conforming (no T-junctions). k is chosen from the longest edge of the input.

    Subdivision is needed because a texture is pasted on the slab by per-vertex UVs: the
    pixel-to-plane map is a homography, and barycentric interpolation only approximates it.
    """
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    if len(faces) == 0 or max_edge is None or max_edge <= 0:
        return verts, faces
    fv = verts[faces]
    elen = np.linalg.norm(fv[:, [0, 1, 2]] - fv[:, [1, 2, 0]], axis=2)
    k = int(np.ceil(elen.max() / max_edge))
    k = max(1, min(k, 256))
    if k == 1:
        return verts, faces

    ij = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], np.int64)
    loc = -np.ones((k + 1, k + 1), np.int64)
    loc[ij[:, 0], ij[:, 1]] = np.arange(len(ij))
    w = np.stack([(k - ij[:, 0] - ij[:, 1]), ij[:, 0], ij[:, 1]], -1) / float(k)
    sub = []
    for i in range(k):
        for j in range(k - i):
            sub.append([loc[i, j], loc[i + 1, j], loc[i, j + 1]])
            if i + j < k - 1:
                sub.append([loc[i + 1, j], loc[i + 1, j + 1], loc[i, j + 1]])
    sub = np.asarray(sub, np.int64)

    pts = np.einsum('pk,fkd->fpd', w, fv).reshape(-1, 3)
    off = (np.arange(len(faces)) * len(ij))[:, None, None]
    new_faces = (sub[None] + off).reshape(-1, 3)
    # weld the duplicated edge vertices (coordinates agree to ~1e-15 by construction)
    _, first, inv = np.unique(np.round(pts, 9), axis=0, return_index=True, return_inverse=True)
    return pts[first], inv.reshape(-1)[new_faces]

def flat_color_uv(img: np.ndarray, mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """UV of the image pixel inside the mask whose colour is closest to the mask median,
    used as a single-texel flat colour for the slab sides, bottom and margin ring."""
    pix = img[mask]
    if len(pix) == 0:
        return np.array([0.5, 0.5], np.float32)
    med = np.median(pix.astype(np.float64), axis=0)
    rows, cols = np.nonzero(mask)
    k = int(np.argmin(np.abs(pix.astype(np.float64) - med).sum(axis=1)))
    return np.array([cols[k] / max(W - 1, 1), 1.0 - rows[k] / max(H - 1, 1)], np.float32)

def build_table_slab(poly2: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray,
                     normal: np.ndarray, K: np.ndarray, img_shape: tuple[int, int],
                     thickness: float, max_edge: float, flat_uv: np.ndarray,
                     img: Image.Image) -> tuple[trimesh.Trimesh, dict]:
    """Extrude the footprint downward into a textured slab.

    Top face: exactly planar, UVs from the camera projection, textured with the inpainted
    background image. The part of the top outside the image (the margin that reaches past
    the view frustum), the sides and the bottom are pinned to a single table-coloured
    texel, so nothing is covered by stretched border pixels.
    """
    H, W = img_shape
    faces2 = triangulate_polygon_2d(poly2)
    verts3 = origin + poly2[:, 0:1] * e1 + poly2[:, 1:2] * e2
    top_v, top_f = subdivide_uniform(verts3, faces2, max_edge)

    # orient the top face along the ground normal (up)
    tmp = trimesh.Trimesh(vertices=top_v, faces=top_f, process=False)
    if len(top_f) and np.mean(tmp.face_normals @ normal) < 0:
        top_f = top_f[:, ::-1]

    ring = boundary_edges(top_f, directed=True)
    n_top = len(top_v)
    down = -normal * thickness
    bot_v = top_v + down

    # sides get their own vertices so the flat colour does not bleed from the texture
    ring_ids = np.unique(ring)
    remap = -np.ones(n_top, np.int64)
    remap[ring_ids] = np.arange(len(ring_ids))
    side_top = top_v[ring_ids]
    side_bot = side_top + down

    v = np.vstack([top_v, bot_v, side_top, side_bot])
    o_bot, o_st, o_sb = n_top, 2 * n_top, 2 * n_top + len(ring_ids)
    f_top = top_f
    f_bot = top_f[:, ::-1] + o_bot
    a = remap[ring[:, 0]] + o_st
    b = remap[ring[:, 1]] + o_st
    a2 = remap[ring[:, 0]] + o_sb
    b2 = remap[ring[:, 1]] + o_sb
    # the boundary loop of the (up-facing) top runs counter-clockwise seen from above, so
    # this winding is the one that makes the side walls face outward
    f_side = np.vstack([np.stack([a, b2, b], -1), np.stack([a, a2, b2], -1)])

    uv = np.zeros((len(v), 2), np.float32)
    uv[:n_top] = project_to_uv(top_v, K, H, W)
    uv[n_top:] = flat_uv

    # the margin ring has no image behind it: give the faces that lie entirely outside the
    # image their own vertices on the flat texel instead of a stretch of the border pixels
    raw = project_to_uv(top_v, K, H, W, clamp=False)
    v_inside = ((raw >= 0) & (raw <= 1)).all(axis=1) & (top_v[:, 2] > 0)
    f_outside = ~v_inside[f_top].any(axis=1)
    if f_outside.any():
        used = np.unique(f_top[f_outside])
        dup = -np.ones(n_top, np.int64)
        dup[used] = len(v) + np.arange(len(used))
        v = np.vstack([v, top_v[used]])
        uv = np.vstack([uv, np.tile(flat_uv, (len(used), 1)).astype(np.float32)])
        f_top = f_top.copy()
        f_top[f_outside] = dup[f_top[f_outside]]

    f = np.vstack([f_top, f_bot, f_side]).astype(np.int64)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)

    fv = top_v[top_f] if len(top_f) else np.zeros((1, 3, 3))
    diag = {
        "top_vertices": int(n_top), "top_faces": int(len(top_f)),
        "slab_faces": int(len(f)), "side_quads": int(len(ring)),
        "top_max_edge_mm": float(1000 * np.linalg.norm(
            fv[:, [0, 1, 2]] - fv[:, [1, 2, 0]], axis=2).max()),
        "faces_on_flat_texel": int(f_outside.sum()),
    }
    if SLAB_DIAGNOSTICS:
        diag.update(slab_diagnostics(mesh, top_v, top_f, uv[:n_top], v_inside, K, H, W))
    return mesh, diag

def slab_diagnostics(mesh: trimesh.Trimesh, top_v: np.ndarray, top_f: np.ndarray,
                     uv_top: np.ndarray, v_inside: np.ndarray,
                     K: np.ndarray, H: int, W: int) -> dict:
    """Texture projection error and closedness of a slab, for verification runs."""
    if len(top_f):
        cen = top_v[top_f].mean(axis=1)
        err = np.linalg.norm((uv_top[top_f].mean(axis=1) - project_to_uv(cen, K, H, W))
                             * np.array([W - 1, H - 1]), axis=1)
        inside = v_inside[top_f].all(axis=1)
        areas = 0.5 * np.linalg.norm(np.cross(top_v[top_f][:, 1] - top_v[top_f][:, 0],
                                              top_v[top_f][:, 2] - top_v[top_f][:, 0]), axis=1)
        sliver = int((areas < 1e-3 * np.median(areas)).sum())
    else:
        err, inside, sliver = np.zeros(1), np.ones(1, bool), 0
    err_in = err[inside] if inside.any() else np.zeros(1)
    # the slab is closed, but the side, bottom and margin vertices are duplicated to carry
    # their own UVs, so closedness has to be checked with the texture seams welded
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    return {
        "faces_inside_image": int(inside.sum()), "top_sliver_faces": sliver,
        "uv_err_px_median": float(np.median(err_in)), "uv_err_px_max": float(err_in.max()),
        "winding_consistent": bool(welded.is_winding_consistent),
        "volume_l": float(1000 * welded.volume) if welded.is_watertight else None,
        "watertight_welded": bool(welded.is_watertight),
    }
