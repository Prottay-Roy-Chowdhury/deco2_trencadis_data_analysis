"""Automated broken-tile point-cloud processing pipeline.

The script preserves multiple selectable processing and color-grouping paths:
- adaptive two-stage RANSAC plane extraction;
- watershed and optional legacy point-cloud clustering paths;
- DBSCAN, reference-based, and GMM+BIC color grouping;
- unchanged fabrication-oriented XYZ export in metres.

Coordinate policy
-----------------
Input point clouds are stored in millimetres and converted once to metres at
load time. The conversion changes units only; it does not center, normalize,
rotate, or otherwise alter the world-coordinate frame.
"""

# Standard library
import json
import os
import random
from datetime import datetime

# Third-party libraries
import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# Project-local helpers
from helpers.session_manager import load_session

# ---------- Constants ----------
EPS_NUM = 1e-8

# =============================================================================
# Core point-cloud and plane-segmentation helpers
# =============================================================================

def preprocess_point_cloud(pcd, voxel_size=0.005, nb_neighbors=20, std_ratio=2.0):
    """
    Downsamples and denoises a point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input cloud.
    voxel_size : float
        Voxel size (m) for grid downsampling.
    nb_neighbors : int
        Number of neighbors for statistical outlier removal.
    std_ratio : float
        Standard deviation ratio threshold for outlier removal.

    Returns
    -------
    o3d.geometry.PointCloud
        Processed (downsampled + denoised) cloud.
    """
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

def detect_multiple_planes(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000,
                           min_points=1000, max_planes=5):
    """
    Iteratively segments up to `max_planes` planar regions from a point cloud.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input cloud to segment.
    distance_threshold : float
        RANSAC inlier distance threshold.
    ransac_n : int
        Number of points to sample for each RANSAC iteration.
    num_iterations : int
        Maximum RANSAC iterations.
    min_points : int
        Minimum inliers to accept a plane.
    max_planes : int
        Maximum number of planes to extract.

    Returns
    -------
    planes_for_cluster : list[o3d.geometry.PointCloud]
        Unpainted planes (preserve original colors) for later clustering.
    planes_for_display : list[o3d.geometry.PointCloud]
        Painted copies for visualization only.
    remaining : o3d.geometry.PointCloud
        Leftover (non-planar) points, unpainted.
    plane_models : list[list[float]]
        RANSAC plane equations [a, b, c, d] for ax + by + cz + d = 0.
    """
    planes_for_cluster = []
    planes_for_display = []
    plane_models = []
    remaining = pcd

    for plane_idx in range(max_planes):
        if len(remaining.points) < min_points:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold,
            ransac_n,
            num_iterations
        )

        if len(inliers) < min_points:
            break

        plane_model = [float(v) for v in plane_model]
        plane_models.append(plane_model)

        inlier_cloud = remaining.select_by_index(inliers)  # keep true colors
        planes_for_cluster.append(inlier_cloud)

        painted = o3d.geometry.PointCloud(inlier_cloud)    # painted copy for display
        painted.paint_uniform_color([random.random(), random.random(), random.random()])
        planes_for_display.append(painted)

        print(
            f"[plane-detect] plane {plane_idx}: "
            f"model={plane_model}, inliers={len(inliers)}"
        )

        remaining = remaining.select_by_index(inliers, invert=True)

    return planes_for_cluster, planes_for_display, remaining, plane_models

def detect_planes_with_multiscale_check(
    pcd,
    adaptive_threshold,
    ransac_n,
    num_iterations,
    min_points,
    max_planes,
    threshold_scales=(0.8, 1.0, 1.2),
    maximum_normal_difference_deg=3.0,
    maximum_position_difference=0.003
):
    """
    Runs multi-plane RANSAC at several nearby thresholds.

    The middle scale is returned as the main result, but the selected
    highest plane is compared across all scales.

    Returns
    -------
    main_planes_cluster
    main_planes_display
    main_remaining
    main_plane_models
    main_selected_index
    diagnostics
    """
    scale_results = []

    for scale in threshold_scales:
        threshold = float(
            adaptive_threshold * float(scale)
        )

        planes_cluster, planes_display, remaining, plane_models = (
            detect_multiple_planes(
                pcd,
                distance_threshold=threshold,
                ransac_n=int(ransac_n),
                num_iterations=int(num_iterations),
                min_points=int(min_points),
                max_planes=int(max_planes)
            )
        )

        if not planes_cluster:
            print(
                f"[multi-ransac] scale={scale:.2f}, "
                f"threshold={threshold:.6f}: no planes"
            )

            scale_results.append({
                "scale": float(scale),
                "threshold": threshold,
                "planes_cluster": planes_cluster,
                "planes_display": planes_display,
                "remaining": remaining,
                "plane_models": plane_models,
                "selected_index": None,
                "selected_model": None,
                "selected_position": None,
                "selected_points": 0
            })

            continue

        selected_index = select_highest_plane_index(
            planes_cluster,
            plane_models
        )

        selected_model = plane_models[
            selected_index
        ]

        selected_position = plane_position_at_centroid(
            planes_cluster[selected_index],
            selected_model
        )

        selected_points = len(
            planes_cluster[selected_index].points
        )

        print(
            f"[multi-ransac] scale={scale:.2f}, "
            f"threshold={threshold:.6f}, "
            f"planes={len(planes_cluster)}, "
            f"selected={selected_index}, "
            f"position={selected_position}, "
            f"points={selected_points}"
        )

        scale_results.append({
            "scale": float(scale),
            "threshold": threshold,
            "planes_cluster": planes_cluster,
            "planes_display": planes_display,
            "remaining": remaining,
            "plane_models": plane_models,
            "selected_index": int(selected_index),
            "selected_model": selected_model,
            "selected_position": selected_position,
            "selected_points": int(selected_points)
        })

    if not scale_results:
        raise RuntimeError(
            "Multi-scale RANSAC produced no results."
        )

    # Pick the scale closest to 1.0 as the main result.
    main_result = min(
        scale_results,
        key=lambda item: abs(item["scale"] - 1.0)
    )

    if main_result["selected_index"] is None:
        valid_results = [
            result
            for result in scale_results
            if result["selected_index"] is not None
        ]

        if not valid_results:
            raise RuntimeError(
                "No valid planes were found at any RANSAC scale."
            )

        main_result = min(
            valid_results,
            key=lambda item: abs(item["scale"] - 1.0)
        )

    reference_model = main_result["selected_model"]
    reference_position = main_result["selected_position"]

    comparisons = []
    stable_count = 0

    for result in scale_results:
        if result["selected_model"] is None:
            comparisons.append({
                "scale": result["scale"],
                "valid": False,
                "stable": False
            })
            continue

        normal_difference = plane_model_angle_degrees(
            reference_model,
            result["selected_model"]
        )

        if (
            reference_position is None
            or result["selected_position"] is None
        ):
            position_difference = None
            position_stable = False
        else:
            position_difference = abs(
                float(result["selected_position"])
                - float(reference_position)
            )

            position_stable = (
                position_difference
                <= float(maximum_position_difference)
            )

        normal_stable = (
            normal_difference
            <= float(maximum_normal_difference_deg)
        )

        stable = (
            normal_stable
            and position_stable
        )

        if stable:
            stable_count += 1

        comparisons.append({
            "scale": result["scale"],
            "threshold": result["threshold"],
            "valid": True,
            "normal_difference_deg": normal_difference,
            "position_difference": position_difference,
            "normal_stable": bool(normal_stable),
            "position_stable": bool(position_stable),
            "stable": bool(stable),
            "selected_points": result["selected_points"]
        })

    required_stable_count = max(
        2,
        int(np.ceil(len(threshold_scales) * 0.67))
    )

    is_stable = (
        stable_count >= required_stable_count
    )

    diagnostics = {
        "adaptive_threshold": float(adaptive_threshold),
        "threshold_scales": [
            float(value)
            for value in threshold_scales
        ],
        "main_scale": float(main_result["scale"]),
        "main_threshold": float(main_result["threshold"]),
        "stable_count": int(stable_count),
        "required_stable_count": int(required_stable_count),
        "is_stable": bool(is_stable),
        "comparisons": comparisons
    }

    if is_stable:
        print(
            f"[multi-ransac] stable: "
            f"{stable_count}/{len(threshold_scales)} "
            f"scales agree."
        )
    else:
        print(
            f"[multi-ransac][warn] unstable: "
            f"only {stable_count}/{len(threshold_scales)} "
            f"scales agree."
        )

    return (
        main_result["planes_cluster"],
        main_result["planes_display"],
        main_result["remaining"],
        main_result["plane_models"],
        main_result["selected_index"],
        diagnostics
    )


def estimate_histogram_peak_spread(
    values,
    search_min,
    search_max,
    bin_width,
    smoothing_sigma_bins=1.5,
    spread_window=0.004,
    minimum_half_width=0.0008,
    maximum_half_width=0.005,
    spread_multiplier=2.5
):
    """
    Detects the strongest histogram peak and estimates its spread.

    Returns
    -------
    peak_center : float
        Center of the strongest smoothed histogram peak.

    adaptive_half_width : float
        Estimated half-width around the peak.

    diagnostics : dict
        Histogram and spread information.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    values = values[
        (values >= float(search_min))
        & (values <= float(search_max))
    ]

    if values.size == 0:
        raise RuntimeError(
            "No values available inside the histogram search range."
        )

    bin_width = max(float(bin_width), EPS_NUM)

    bin_count = max(
        3,
        int(
            np.ceil(
                (float(search_max) - float(search_min))
                / bin_width
            )
        )
    )

    hist, edges = np.histogram(
        values,
        bins=bin_count,
        range=(
            float(search_min),
            float(search_max)
        )
    )

    hist_smooth = ndi.gaussian_filter1d(
        hist.astype(np.float64),
        sigma=float(smoothing_sigma_bins)
    )

    peak_bin = int(np.argmax(hist_smooth))

    peak_center = float(
        0.5 * (
            edges[peak_bin]
            + edges[peak_bin + 1]
        )
    )

    # Estimate spread only from values near the detected peak.
    local = values[
        np.abs(values - peak_center)
        <= float(spread_window)
    ]

    if local.size < 20:
        local = values

    local_median = float(np.median(local))

    mad = float(
        np.median(
            np.abs(local - local_median)
        )
    )

    robust_sigma = float(1.4826 * mad)

    # Also estimate peak width from the histogram's half maximum.
    peak_value = float(hist_smooth[peak_bin])
    half_maximum = peak_value * 0.5

    left = peak_bin
    while (
        left > 0
        and hist_smooth[left] >= half_maximum
    ):
        left -= 1

    right = peak_bin
    while (
        right < len(hist_smooth) - 1
        and hist_smooth[right] >= half_maximum
    ):
        right += 1

    fwhm = float(
        edges[min(right + 1, len(edges) - 1)]
        - edges[max(left, 0)]
    )

    histogram_half_width = 0.5 * fwhm

    robust_half_width = (
        float(spread_multiplier) * robust_sigma
        + 0.5 * bin_width
    )

    adaptive_half_width = float(
        np.clip(
            max(
                robust_half_width,
                histogram_half_width
            ),
            float(minimum_half_width),
            float(maximum_half_width)
        )
    )

    diagnostics = {
        "search_min": float(search_min),
        "search_max": float(search_max),
        "bin_width": float(bin_width),
        "value_count": int(values.size),
        "peak_bin": int(peak_bin),
        "peak_center": float(peak_center),
        "peak_count": float(peak_value),
        "local_count": int(local.size),
        "local_median": float(local_median),
        "mad": float(mad),
        "robust_sigma": float(robust_sigma),
        "fwhm": float(fwhm),
        "histogram_half_width": float(
            histogram_half_width
        ),
        "robust_half_width": float(
            robust_half_width
        ),
        "adaptive_half_width": float(
            adaptive_half_width
        )
    }

    return (
        peak_center,
        adaptive_half_width,
        diagnostics
    )


def cluster_colored_plane(pcd_plane, eps=0.02, min_samples=350, color_weight=1.7, position_weight=0.4):
    """
    Clusters a single plane using DBSCAN on [position*weight | color*weight] features.

    Parameters
    ----------
    pcd_plane : o3d.geometry.PointCloud
        Points belonging to one extracted plane.
    eps : float
        DBSCAN neighborhood radius (in feature space).
    min_samples : int
        Minimum samples to form a cluster.
    color_weight : float
        Weight applied to RGB features.
    position_weight : float
        Weight applied to XYZ positions.

    Returns
    -------
    clusters_disp : list[o3d.geometry.PointCloud]
        Painted clusters for on-screen display.
    clusters_orig : list[o3d.geometry.PointCloud]
        Unpainted clusters (true colors) aligned with clusters_disp.
    """
    points = np.asarray(pcd_plane.points) * position_weight
    colors = np.asarray(pcd_plane.colors) * color_weight
    features = np.hstack((points, colors)).astype(np.float32)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = db.labels_

    clusters_disp, clusters_orig = [], []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        indices = np.where(labels == cluster_id)[0]
        cluster_orig = pcd_plane.select_by_index(indices)
        clusters_orig.append(cluster_orig)

        cluster_disp = o3d.geometry.PointCloud(cluster_orig)
        cluster_disp.paint_uniform_color([random.random(), random.random(), random.random()])
        clusters_disp.append(cluster_disp)
    return clusters_disp, clusters_orig

# =============================================================================
# 2D hull geometry and fabrication export helpers
# =============================================================================

def _fit_plane_basis(points3d: np.ndarray):
    """
    Fits a local plane to 3D points via SVD and returns an orthonormal basis.

    Returns
    -------
    c : (3,) np.ndarray
        Centroid of points.
    u, v : (3,) np.ndarray
        In-plane orthonormal basis vectors.
    n : (3,) np.ndarray
        Plane normal (third singular vector).
    """
    c = points3d.mean(axis=0)
    P = points3d - c
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    u = Vt[0, :]
    v = Vt[1, :]
    n = Vt[2, :]
    return c, u, v, n

def _project_to_2d(points3d: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray):
    """
    Projects 3D points onto the local (u,v) plane coordinate system.

    Returns
    -------
    pts2 : (N,2) np.ndarray
        2D coordinates of points in the plane frame.
    """
    P = points3d - c
    x = P @ u
    y = P @ v
    return np.stack([x, y], axis=1)

def _monotone_chain_hull_indices(pts2: np.ndarray):
    """
    Computes indices of the 2D convex hull using the monotone chain algorithm.

    Parameters
    ----------
    pts2 : (N,2) np.ndarray
        2D points.

    Returns
    -------
    list[int]
        Indices (into pts2) of the hull polygon in order.
    """
    if len(pts2) < 3:
        return list(range(len(pts2)))
    order = np.lexsort((pts2[:, 1], pts2[:, 0]))
    pts = pts2[order]; idxs = order

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, lower_idx = [], []
    for p, i in zip(pts, idxs):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop(); lower_idx.pop()
        lower.append(p); lower_idx.append(i)

    upper, upper_idx = [], []
    for p, i in zip(reversed(pts), reversed(idxs)):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop(); upper_idx.pop()
        upper.append(p); upper_idx.append(i)

    hull_idx = lower_idx[:-1] + upper_idx[:-1]
    seen, out = set(), []
    for i in hull_idx:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def create_2d_hull_lineset_from_cluster(cluster: o3d.geometry.PointCloud):
    """
    Builds a LineSet of the 2D convex hull (projected in-plane) of a cluster.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud

    Returns
    -------
    o3d.geometry.LineSet | None
        LineSet for visualization or None if hull can't be formed.
    """
    pts = np.asarray(cluster.points)
    if pts.shape[0] < 3:
        return None
    c, u, v, _ = _fit_plane_basis(pts)
    pts2 = _project_to_2d(pts, c, u, v)
    hull_local_idx = _monotone_chain_hull_indices(pts2)
    if len(hull_local_idx) < 3:
        return None
    poly2 = pts2[hull_local_idx]
    poly3 = c + np.outer(poly2[:, 0], u) + np.outer(poly2[:, 1], v)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(poly3)
    lines = [[i, (i + 1) % len(poly3)] for i in range(len(poly3))]
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.paint_uniform_color([0, 0, 0])
    return ls

# ---- Extra helpers for hull-masked colors & centroids ----

def _polygon_centroid_2d(poly2: np.ndarray):
    """
    Computes centroid of a simple 2D polygon using the shoelace formula.

    Parameters
    ----------
    poly2 : (M,2) np.ndarray
        Polygon vertices in order.

    Returns
    -------
    (2,) np.ndarray
        Centroid in 2D.
    """
    x = poly2[:, 0]; y = poly2[:, 1]
    x1 = np.roll(x, -1); y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-12:
        return np.array([x.mean(), y.mean()], dtype=np.float32)
    Cx = (1.0 / (6.0 * A)) * np.sum((x + x1) * cross)
    Cy = (1.0 / (6.0 * A)) * np.sum((y + y1) * cross)
    return np.array([Cx, Cy], dtype=np.float32)

def _points_in_polygon(pts2: np.ndarray, poly2: np.ndarray):
    """
    Even–odd rule point-in-polygon test for a batch of 2D points.

    Parameters
    ----------
    pts2 : (N,2) np.ndarray
        Query points.
    poly2 : (M,2) np.ndarray
        Polygon vertices.

    Returns
    -------
    (N,) np.ndarray of bool
        Mask of points inside the polygon.
    """
    n = len(poly2)
    inside = np.zeros(len(pts2), dtype=bool)
    x = pts2[:, 0]; y = pts2[:, 1]
    px = poly2[:, 0]; py = poly2[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi)
        inside ^= cond
        j = i
    return inside

def get_hull_mask_and_centroid3d(cluster: o3d.geometry.PointCloud):
    """
    Computes a 2D-hull-based mask (in-plane) and its 3D centroid for a cluster.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud

    Returns
    -------
    mask : (N,) np.ndarray of bool
        Points inside the 2D hull.
    centroid3d : (3,) np.ndarray
        3D centroid of the polygon, mapped back to 3D.
    """
    pts = np.asarray(cluster.points)
    if len(pts) < 3:
        return np.ones(len(pts), dtype=bool), pts.mean(axis=0)
    c, u, v, _ = _fit_plane_basis(pts)
    pts2 = _project_to_2d(pts, c, u, v)
    hull_idx = _monotone_chain_hull_indices(pts2)
    if len(hull_idx) < 3:
        return np.ones(len(pts), dtype=bool), pts.mean(axis=0)
    poly2 = pts2[hull_idx]
    mask = _points_in_polygon(pts2, poly2)
    c2d = _polygon_centroid_2d(poly2)
    centroid3d = c + c2d[0] * u + c2d[1] * v
    return mask, centroid3d

# ---- Helper used by export (poly + plane frame + centroid)
def compute_hull_poly2_poly3(cluster: o3d.geometry.PointCloud):
    """
    Computes the 2D hull in plane coordinates and maps it back to 3D.

    Parameters
    ----------
    cluster : o3d.geometry.PointCloud

    Returns
    -------
    poly2 : (M,2) np.ndarray or None
        2D hull polygon in (u,v) coordinates.
    poly3 : (M,3) np.ndarray or None
        3D polygon points.
    c,u,v,n : (3,) np.ndarray
        Plane frame origin and basis.
    centroid3d : (3,) np.ndarray
        Centroid of the polygon in 3D.
    """
    pts = np.asarray(cluster.points)
    c, u, v, n = _fit_plane_basis(pts)
    if len(pts) < 3:
        return None, None, c, u, v, n, pts.mean(axis=0)
    pts2 = _project_to_2d(pts, c, u, v)
    hull_idx = _monotone_chain_hull_indices(pts2)
    if len(hull_idx) < 3:
        return None, None, c, u, v, n, pts.mean(axis=0)
    poly2 = pts2[hull_idx]
    poly3 = c + np.outer(poly2[:, 0], u) + np.outer(poly2[:, 1], v)
    c2d = _polygon_centroid_2d(poly2)
    centroid3d = c + c2d[0] * u + c2d[1] * v
    return poly2, poly3, c, u, v, n, centroid3d

# =============================================================================
# Shared color-space and DBSCAN grouping helpers
# =============================================================================

def _srgb_to_linear(c):
    """
    Converts sRGB components [0,1] to linear RGB.
    """
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def _rgb_to_lab(rgb):
    """
    Converts Nx3 sRGB array (0–1) to CIELAB (D65), returns Nx3 (L,a,b).
    """
    r, g, b = _srgb_to_linear(rgb[:, 0]), _srgb_to_linear(rgb[:, 1]), _srgb_to_linear(rgb[:, 2])
    RGB = np.stack([r, g, b], axis=1)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    XYZ = RGB @ M.T
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X, Y, Z = XYZ[:, 0] / Xn, XYZ[:, 1] / Yn, XYZ[:, 2] / Zn
    def f(t):
        delta = 6/29
        return np.where(t > delta**3, np.cbrt(t), t/(3*delta**2) + 4/29)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def _cluster_color_feature(cluster: o3d.geometry.PointCloud, mode: str, use_hull_mask: bool,
                          colors_override: np.ndarray | None = None):
    """
    Computes a representative color feature per cluster.

    If colors_override is provided, it must be Nx3 float in [0..1] (RGB).
    """
    if colors_override is not None and colors_override.size:
        cols = colors_override.astype(np.float32)
    else:
        cols = np.asarray(cluster.colors).astype(np.float32)

    if cols.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    mode_l = (mode or "Lab a*b* only").strip().lower()
    if mode_l.startswith("lab"):
        Lab = _rgb_to_lab(cols)
        return np.median(Lab[:, 1:3], axis=0).astype(np.float32)
    if mode_l.startswith("chrom"):
        norms = np.linalg.norm(cols, axis=1, keepdims=True)
        chrom = cols / (norms + EPS_NUM)
        return np.median(chrom, axis=0).astype(np.float32)
    return np.median(cols, axis=0).astype(np.float32)


def group_clusters_by_color(clusters_orig, mode="Lab a*b* only", eps_z=1.0, min_size=1,
                            use_hull_mask=False, centroids_out=None,
                            color_sampler=None):
    """
    Groups clusters by color feature using DBSCAN over z-scored features.

    Parameters
    ----------
    clusters_orig : list[o3d.geometry.PointCloud]
        Unpainted clusters (true colors).
    mode : str
        Color feature mode (see _cluster_color_feature).
    eps_z : float
        DBSCAN eps in z-score space.
    min_size : int
        Minimum cluster size for DBSCAN.
    use_hull_mask : bool
        If True, compute features using points inside the 2D hull.
    centroids_out : list
        If provided, filled with per-cluster 3D centroids from hull.

    Returns
    -------
    labels : (K,) np.ndarray of int
    group_to_color : dict[int, np.ndarray]
        Mean RGB per DBSCAN group (0–1).
    report : list[str]
        Human-readable summary lines.
    """
    if not clusters_orig:
        return np.array([]), {}, ["Color Grouping Report", "No clusters."]
    feats, reps_rgb, centroids = [], [], []
    for cl in clusters_orig:
                # centroid always from geometry (unchanged)
        _, centroid3d = get_hull_mask_and_centroid3d(cl)
        centroids.append(centroid3d)

        # colors: either from sampler (stitched image) or from point cloud
        cols_src = None
        if callable(color_sampler):
            cols_src = color_sampler(cl, use_hull_mask)

        if cols_src is not None and cols_src.size:
            cols_rep = cols_src
        else:
            cols_rep = np.asarray(cl.colors).astype(np.float32)

        reps_rgb.append(
            np.median(cols_rep, axis=0).astype(np.float32)
            if cols_rep.size else np.array([0.5, 0.5, 0.5], dtype=np.float32)
        )

        feats.append(_cluster_color_feature(cl, mode, use_hull_mask, colors_override=cols_src))
    F = np.stack(feats, axis=0).astype(np.float32)
    mu = F.mean(axis=0, keepdims=True); sigma = F.std(axis=0, keepdims=True) + EPS_NUM
    Fz = (F - mu) / sigma
    db = DBSCAN(eps=float(eps_z), min_samples=int(min_size)).fit(Fz)
    labels = db.labels_
    group_to_color = {}
    for g in sorted(set(labels)):
        if g == -1:
            continue
        idx = np.where(labels == g)[0]
        if len(idx) == 0:
            continue
        mean_rgb = np.mean(np.stack([reps_rgb[i] for i in idx], axis=0), axis=0)
        group_to_color[g] = np.clip(mean_rgb, 0.0, 1.0)
    report = []
    report.append("Color Grouping Report")
    report.append(f"Mode: {mode} | eps(z): {eps_z:.2f} | min_size: {min_size} | hull_mask: {use_hull_mask}")
    report.append(f"Total clusters: {len(clusters_orig)}")
    for g in sorted(set(labels)):
        if g == -1:
            continue
        idx = np.where(labels == g)[0].tolist()
        rgb = group_to_color[g]
        report.append(f"- Group {g}: count={len(idx)}  meanRGB=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})  members={idx}")
    noise_idx = np.where(labels == -1)[0].tolist()
    if noise_idx:
        report.append(f"- Noise (-1): count={len(noise_idx)}  members={noise_idx}")
    report.append("Per-cluster 2D-hull centroids (3D coords):")
    for i, c3 in enumerate(centroids):
        report.append(f"  cluster[{i}]: ({c3[0]:.4f}, {c3[1]:.4f}, {c3[2]:.4f})")
    if centroids_out is not None:
        centroids_out.clear(); centroids_out.extend(centroids)
    return labels, group_to_color, report


def validate_reference_color_groups(groups):
    """
    Validates reference colors received from GH/config.

    Expected structure:
    [
        {
            "group_id": 0,
            "name": "brown",
            "samples_rgb": [[181, 159, 133]],
            "max_distance": 12.0
        }
    ]
    """
    if groups is None:
        return []

    if not isinstance(groups, list):
        raise ValueError(
            "reference_color_groups must be a list."
        )

    validated = []
    used_ids = set()

    for index, item in enumerate(groups):
        if not isinstance(item, dict):
            raise ValueError(
                f"Reference color at index {index} "
                f"must be an object/dictionary."
            )

        if "group_id" not in item:
            raise ValueError(
                f"Reference color at index {index} "
                f"is missing group_id."
            )

        group_id = int(item["group_id"])

        if group_id in used_ids:
            raise ValueError(
                f"Duplicate reference group_id: "
                f"{group_id}"
            )

        used_ids.add(group_id)

        samples = item.get(
            "samples_rgb",
            []
        )

        if not isinstance(samples, list) or not samples:
            raise ValueError(
                f"Reference group {group_id} "
                f"requires at least one RGB sample."
            )

        validated_samples = []

        for sample_index, sample in enumerate(samples):
            if (
                not isinstance(sample, (list, tuple))
                or len(sample) != 3
            ):
                raise ValueError(
                    f"Reference group {group_id}, "
                    f"sample {sample_index} must contain "
                    f"exactly three RGB values."
                )

            rgb = [
                float(sample[0]),
                float(sample[1]),
                float(sample[2])
            ]

            if any(value < 0 for value in rgb):
                raise ValueError(
                    f"Reference group {group_id} "
                    f"contains a negative RGB value."
                )

            if any(value > 255 for value in rgb):
                raise ValueError(
                    f"Reference group {group_id} "
                    f"contains an RGB value above 255."
                )

            validated_samples.append(rgb)

        max_distance = float(
            item.get(
                "max_distance",
                12.0
            )
        )

        if max_distance <= 0:
            raise ValueError(
                f"Reference group {group_id} "
                f"max_distance must be positive."
            )

        validated.append({
            "group_id": group_id,
            "name": str(
                item.get(
                    "name",
                    f"group_{group_id}"
                )
            ),
            "samples_rgb": validated_samples,
            "max_distance": max_distance
        })

    return validated

# =============================================================================
# Runtime configuration and preview controls
# =============================================================================

SHOW_PREVIEW = True
# Set to None if you want each preview window to stay open until you close it.
# Set to a number like 2.0 for timed previews.
PREVIEW_TIME_SEC = None

DEFAULT_OUTPUT_INDEX = 1
DEFAULT_INPUT_KIND = "eye_to_base"  # options: "eye_to_base", "merged", "initial"

PROCESSING_CONFIG_TEMPLATE = "processing_config_{idx:02d}.json"
PROCESSING_REPORT_TEMPLATE = "processing_report_{idx:02d}.txt"
PROCESSING_OUTPUT_TEMPLATE = "processed_clusters_{idx:02d}.json"
PROCESSING_PARAMS_TEMPLATE = "processing_params_used_{idx:02d}.json"


# =============================================================================
# Visualization helpers
# =============================================================================

def show_preview(
    geometries,
    window_name="Preview",
    seconds=None
    ):
    """
    Shows an Open3D preview window after a processing step.

    If the caller does not explicitly provide `seconds`, the current
    runtime PREVIEW_TIME_SEC value is used.
    """
    if not SHOW_PREVIEW:
        return

    if geometries is None:
        return

    if not isinstance(geometries, list):
        geometries = [geometries]

    geometries = [
        geometry
        for geometry in geometries
        if geometry is not None
    ]

    if not geometries:
        return

    # Read the global value at call time, not function-definition time.
    if seconds is None:
        seconds = PREVIEW_TIME_SEC

    # None means manual close.
    if seconds is None:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name
        )
        return

    seconds = float(seconds)

    if seconds <= 0:
        return

    vis = o3d.visualization.Visualizer()

    try:
        vis.create_window(
            window_name=window_name,
            width=1280,
            height=720
        )

        for geometry in geometries:
            vis.add_geometry(geometry)

        import time

        end_time = time.time() + seconds

        while time.time() < end_time:
            if not vis.poll_events():
                break

            vis.update_renderer()
            time.sleep(0.01)

    finally:
        vis.destroy_window()


def _has_points(pc):
    return pc is not None and isinstance(pc, o3d.geometry.PointCloud) and len(pc.points) > 0


def make_grey_copy(pcd, color=(0.85, 0.85, 0.85)):
    if not _has_points(pcd):
        return None
    out = o3d.geometry.PointCloud(pcd)
    out.paint_uniform_color(list(color))
    return out

def show_cv_preview(
    image,
    window_name="Preview",
    seconds=None
    ):
    if not SHOW_PREVIEW:
        return

    if image is None:
        return

    # Read the updated runtime value.
    if seconds is None:
        seconds = PREVIEW_TIME_SEC

    cv2.imshow(window_name, image)

    # None means wait until the user closes/continues manually.
    if seconds is None:
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        return

    seconds = float(seconds)

    if seconds <= 0:
        cv2.destroyWindow(window_name)
        return

    cv2.waitKey(max(1, int(seconds * 1000)))
    cv2.destroyWindow(window_name)


def colorize_label_image(labels):
    labels = labels.astype(np.int32)
    H, W = labels.shape[:2]

    out = np.zeros((H, W, 3), dtype=np.uint8)

    for label_id in np.unique(labels):
        if label_id <= 0:
            continue

        rng = np.random.default_rng(int(label_id))
        color = rng.integers(40, 255, size=3, dtype=np.uint8)

        out[labels == label_id] = color

    return out


# =============================================================================
# Adaptive parameter estimation and watershed refinement
# =============================================================================

def estimate_median_spacing(pcd, sample_size=5000):
    """
    Estimates median nearest-neighbor spacing in meters.
    Uses a sampled cloud for speed on dense scans.
    """
    if not _has_points(pcd):
        return 0.005

    points = np.asarray(pcd.points)
    if len(points) < 3:
        return 0.005

    if len(points) > sample_size:
        idx = np.random.choice(len(points), sample_size, replace=False)
        sample = pcd.select_by_index(idx.tolist())
    else:
        sample = pcd

    distances = sample.compute_nearest_neighbor_distance()
    distances = np.asarray(distances, dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0]

    if distances.size == 0:
        return 0.005

    return float(np.median(distances))


def split_suspicious_labels_with_distance_watershed(
    labels_img,
    tile_mask,
    min_region_pixels=50
):
    refined = labels_img.copy().astype(np.int32)
    next_label = int(refined.max()) + 1

    # Multiple segmentation attempts
    variants = [
        {"sigma": 1.0, "min_distance": 5},
        {"sigma": 5.0, "min_distance": 20},
        {"sigma": 6.5, "min_distance": 25},
    ]

    for label_id in sorted(np.unique(labels_img)):
        if label_id <= 0:
            continue

        region = labels_img == label_id

        if int(region.sum()) < min_region_pixels:
            continue

        split_results = []

        # ----------------------------------
        # Try several watershed parameters
        # ----------------------------------
        for v in variants:

            distance = ndi.distance_transform_edt(region)

            distance_smooth = ndi.gaussian_filter(
                distance,
                sigma=v["sigma"]
            )

            coords = peak_local_max(
                distance_smooth,
                labels=region,
                min_distance=v["min_distance"],
                exclude_border=False
            )

            # no possible split
            if len(coords) <= 1:
                continue

            markers = np.zeros(
                region.shape,
                dtype=np.int32
            )

            for i, (r, c) in enumerate(coords, start=1):
                markers[r, c] = i

            markers = ndi.label(markers > 0)[0]

            sub_labels = watershed(
                -distance_smooth,
                markers,
                mask=region
            ).astype(np.int32)

            n_parts = int(sub_labels.max())

            if n_parts > 1:
                split_results.append(sub_labels)

        # ----------------------------------
        # Stability check
        # ----------------------------------
        if len(split_results) < 2:
            # only one method wanted split
            # probably false split
            continue

        # Use middle result
        accepted = split_results[len(split_results)//2]

        print(
            f"[stable split] label {label_id}: "
            f"{len(split_results)}/{len(variants)} variants agree"
        )

        refined[region] = 0

        for sub_id in sorted(np.unique(accepted)):
            if sub_id <= 0:
                continue

            sub_region = accepted == sub_id

            if int(sub_region.sum()) < min_region_pixels:
                continue

            refined[sub_region] = next_label
            next_label += 1

    return refined


def estimate_processing_params(raw_pcd):
    """
    Builds automatic starting parameters from the raw point cloud.
    These are meant to mimic the GUI defaults while adapting to point density.
    """
    if not _has_points(raw_pcd):
        raise ValueError("Cannot estimate parameters from an empty point cloud.")

    pts = np.asarray(raw_pcd.points)
    z = pts[:, 2]

    z_min = float(np.percentile(z, 0))
    z_max = float(np.percentile(z, 100))

    raw_spacing = estimate_median_spacing(raw_pcd)

    # Keep the same manual defaults you normally use in the GUI workflow.
    # raw_spacing is still saved in the params/report for reference.
    voxel_size = float(np.clip(raw_spacing * 1.0, 0.001, 0.02))
    plane_distance_threshold = float(np.clip(voxel_size * 3.0, 0.003, 0.05))
    dbscan_eps = float(np.clip(raw_spacing * 4.0, 0.005, 0.08))

    # Your old GUI default was 350. This scales with scan size but stays practical.
    dbscan_min_samples = int(np.clip(len(pts) * 0.002, 30, 150))

    min_plane_points = int(np.clip(len(pts) * 0.01, 500, 5000))

    return {
        "z_min": z_min,
        "z_max": z_max,
        "raw_spacing": raw_spacing,
        "voxel_size": .001,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_distance_threshold": .003,
        "ransac_n": 3,
        "ransac_iterations": 1000,
        "min_plane_points": min_plane_points,
        "max_planes": 5,
        "dbscan_eps": 0.08,
        "dbscan_min_samples": 30,
        "color_weight": 0.1,
        "position_weight": 8.0,
        "enable_2d_split_check": True,
        "split_min_frac": 0.15,
        "split_min_points": 200,
        "group_mode": "Lab a*b* only",
        "group_eps": 1.0,
        "group_min_size": 1,
        "group_use_hull_color": True,
        "watershed_min_cluster_points": 10,
        # "use_distance_markers": True,
        # "watershed_min_distance": 40,
        # "watershed_distance_sigma": 2.0,
        # "watershed_threshold_abs": 5,
        "touch_split_min_distance": 20,
        "touch_split_sigma": 2.0,
        "touch_split_threshold_abs": 3,
        "touch_split_min_region_pixels": 50,
        "use_rgb_height_edges": False,
        "rgb_edge_low": 20,
        "rgb_edge_high": 120,
        "height_edge_low": 10,
        "height_edge_high": 100,
        "edge_dilate_kernel": 3,
        "edge_dilate_iterations": 1,

        # ---- Two-stage plane reference workflow ----
        "reference_crop_above": 0.003,          # 2 mm above the initial top reference
        "reference_crop_below": 0.015,          # 15 mm below the initial top reference
        "base_parallel_tolerance_deg": 5.0,     # base must be approximately parallel
        "base_min_below_reference": 0.001,      # reject planes too close to top reference
        "second_plane_distance_threshold": 0.003,  # RANSAC threshold for second plane
        "second_min_plane_points": min_plane_points,
        "second_max_planes": 5,
        "tile_height_search_min": 0.002,        # search 2–10 mm above base
        "tile_height_search_max": 0.030,
        "tile_height_hist_bin": 0.00025,        # 0.25 mm histogram bins
        # "tile_top_band_half_width": 0.005,     # ±1.5 mm around detected tile-top peak

        "adaptive_ransac_enabled": True,

        "adaptive_ransac_coarse_threshold": 0.008,
        "adaptive_ransac_coarse_iterations": 500,
        "adaptive_ransac_residual_multiplier": 3.0,
        "adaptive_ransac_local_percentile": 30.0,
        "adaptive_ransac_min_threshold": 0.0015,
        "adaptive_ransac_max_threshold": 0.008,

        "multiscale_ransac_enabled": True,
        "multiscale_ransac_scales": [0.8, 1.0, 1.2],
        "multiscale_ransac_max_normal_difference_deg": 3.0,
        "multiscale_ransac_max_position_difference": 0.003,

        "second_adaptive_ransac_coarse_threshold": 0.008,
        "second_adaptive_ransac_coarse_iterations": 500,
        "second_adaptive_ransac_residual_multiplier": 3.0,
        "second_adaptive_ransac_local_percentile": 30.0,
        "second_adaptive_ransac_min_threshold": 0.0015,
        "second_adaptive_ransac_max_threshold": 0.008,

        "second_multiscale_ransac_scales": [0.8, 1.0, 1.2],
        "second_multiscale_max_normal_difference_deg": 3.0,
        "second_multiscale_max_position_difference": 0.003,

        # ---- Adaptive reference-to-base crop depth ----
        "reference_base_depth_search_min": 0.002,
        "reference_base_depth_search_max": 0.030,
        "reference_depth_hist_bin": 0.00025,
        "reference_depth_hist_smoothing": 1.5,

        "reference_depth_peak_spread_window": 0.004,
        "reference_depth_peak_spread_multiplier": 2.5,
        "reference_depth_peak_min_half_width": 0.0008,
        "reference_depth_peak_max_half_width": 0.005,

        "reference_crop_base_margin": 0.003,
        "reference_crop_min_below": 0.008,
        "reference_crop_max_below": 0.030,

        # Keep as fallback only
        "reference_crop_above": 0.003,
        "reference_crop_below": 0.015,

        # ---- Adaptive tile-top peak width ----
        "tile_height_search_min": 0.002,
        "tile_height_search_max": 0.030,
        "tile_height_hist_bin": 0.00025,
        "tile_height_hist_smoothing": 1.5,

        "tile_top_spread_window": 0.004,
        "tile_top_band_spread_multiplier": 2.5,
        "tile_top_band_min_half_width": 0.0008,
        "tile_top_band_max_half_width": 0.005,

        # ---- Reference-based color classification ----
        "use_reference_color_grouping": True,

        # Pixels removed from the cluster boundary before color sampling.
        "reference_color_hull_erosion_px": 3,

        # Reject darkest and brightest sampled pixels.
        "reference_color_low_percentile": 10.0,
        "reference_color_high_percentile": 90.0,

        # Minimum number of valid RGB pixels required for classification.
        "reference_color_min_pixels": 20,

        # Weighted Lab distance:
        # brightness has less influence than chromatic channels.
        "reference_color_lab_weights": [0.1, 1.0, 1.0],

        # The winning reference must beat the second-best by this amount.
        "reference_color_minimum_margin": 2.0,

        # References can contain one or many RGB samples.
        # RGB values may be either 0–255 or 0–1.
        "reference_color_groups": [],

        # ---- GMM + BIC automatic color grouping ----

        # Available options:
        # "dbscan"
        # "reference"
        # "gmm_bic"
        "color_grouping_method": "gmm_bic",

        # Operator-provided upper limit.
        # BIC may select fewer groups, but never more.
        "gmm_max_groups": 5,

        # Usually leave this at 1.
        "gmm_min_groups": 1,

        # GMM fitting settings.
        "gmm_covariance_type": "full",
        "gmm_n_init": 5,
        "gmm_max_iter": 300,
        "gmm_random_state": 0,
        "gmm_reg_covar": 1e-5,

        # Feature-channel weights before GMM.
        # Reduced L* weight makes brightness less influential.
        "gmm_lab_weights": [0.2, 1.0, 1.0],

        # Merge GMM components whose final mean Lab colors are too similar.
        # Set to 0 to disable post-GMM merging.
        "gmm_merge_distance": 6.0,

        # Distance weights used only during component merging.
        "gmm_merge_lab_weights": [0.1, 1.0, 1.0],

        # Optional uncertainty rejection.
        # Set 0 to keep every GMM assignment.
        "gmm_min_probability": 0.0,

        # Reuse the current image-space hull sampling settings.
        "gmm_color_min_pixels": 20,
    }


def load_config_override(config_path):
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_params(auto_params, override_params):
    params = dict(auto_params)
    for k, v in (override_params or {}).items():
        if v is not None:
            params[k] = v
    return params

def select_highest_plane_index(planes, plane_models):
    """
    Selects the highest detected plane using the stored RANSAC plane model.

    The model is [a, b, c, d] for:
        a*x + b*y + c*z + d = 0

    For each detected plane, this computes the plane's Z value at the
    centroid XY of that plane, then selects the maximum plane-Z.
    This avoids selecting based on the average Z of all inlier points.
    """
    if not planes:
        raise RuntimeError("No planes available for selection.")

    if not plane_models or len(plane_models) != len(planes):
        raise RuntimeError("Plane models are missing or do not match detected planes.")

    best_idx = 0
    best_z = -1e9

    for i, (plane, model) in enumerate(zip(planes, plane_models)):
        if not _has_points(plane):
            continue

        pts = np.asarray(plane.points)
        a, b, c, d = [float(v) for v in model]

        if abs(c) < EPS_NUM:
            print(f"[plane-select] plane {i}: skipped because c is near zero. model={model}")
            continue

        center = pts.mean(axis=0)
        x = float(center[0])
        y = float(center[1])
        z_plane = float(-(a * x + b * y + d) / c)

        print(
            f"[plane-select] plane {i}: "
            f"z_plane={z_plane:.6f}, "
            f"centroid_xy=({x:.6f}, {y:.6f}), "
            f"points={len(pts)}, model={model}"
        )

        if z_plane > best_z:
            best_z = z_plane
            best_idx = i

    print(f"[plane-select] selected highest plane index: {best_idx}, z_plane={best_z:.6f}")
    return best_idx

def normalize_plane_model(plane_model):
    """
    Returns a normalized plane model [a, b, c, d].

    The normal [a, b, c] will have unit length, so point-to-plane
    distances can be calculated directly as:

        distance = abs(a*x + b*y + c*z + d)
    """
    a, b, c, d = [float(v) for v in plane_model]

    norm = float(np.sqrt(a * a + b * b + c * c))
    if norm < EPS_NUM:
        raise RuntimeError("Invalid plane model: normal length is near zero.")

    return [
        a / norm,
        b / norm,
        c / norm,
        d / norm
    ]


def estimate_adaptive_plane_threshold(
    pcd,
    coarse_threshold=0.008,
    coarse_iterations=500,
    residual_multiplier=3.0,
    local_percentile=30.0,
    minimum_threshold=0.0015,
    maximum_threshold=0.008
):
    """
    Runs a provisional RANSAC plane fit and estimates the final RANSAC
    distance threshold from the robust residual spread near that plane.

    Returns
    -------
    adaptive_threshold : float
    provisional_model : list[float]
    diagnostics : dict
    """
    if not _has_points(pcd):
        raise RuntimeError(
            "Cannot estimate an adaptive plane threshold from an empty cloud."
        )

    provisional_model, provisional_inliers = pcd.segment_plane(
        distance_threshold=float(coarse_threshold),
        ransac_n=3,
        num_iterations=int(coarse_iterations)
    )

    provisional_model = normalize_plane_model(provisional_model)

    points = np.asarray(pcd.points, dtype=np.float64)
    a, b, c, d = provisional_model

    residuals = np.abs(
        points[:, 0] * a
        + points[:, 1] * b
        + points[:, 2] * c
        + d
    )

    residuals = residuals[np.isfinite(residuals)]

    if residuals.size == 0:
        raise RuntimeError(
            "Could not calculate point-to-plane residuals."
        )

    # Use only points relatively close to the provisional dominant plane.
    # This prevents unrelated planes and deep geometry from controlling
    # the scanner-noise estimate.
    local_limit = float(
        np.percentile(
            residuals,
            float(local_percentile)
        )
    )

    local_residuals = residuals[
        residuals <= local_limit
    ]

    if local_residuals.size < 100:
        local_residuals = residuals[
            residuals <= float(coarse_threshold)
        ]

    if local_residuals.size < 100:
        local_residuals = residuals

    median_residual = float(
        np.median(local_residuals)
    )

    mad = float(
        np.median(
            np.abs(
                local_residuals - median_residual
            )
        )
    )

    robust_sigma = float(1.4826 * mad)

    # Include the median offset as well as the residual spread.
    estimated_threshold = (
        median_residual
        + float(residual_multiplier) * robust_sigma
    )

    adaptive_threshold = float(
        np.clip(
            estimated_threshold,
            float(minimum_threshold),
            float(maximum_threshold)
        )
    )

    diagnostics = {
        "coarse_threshold": float(coarse_threshold),
        "coarse_inliers": int(len(provisional_inliers)),
        "local_percentile": float(local_percentile),
        "local_residual_count": int(local_residuals.size),
        "median_residual": median_residual,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "estimated_threshold": float(estimated_threshold),
        "adaptive_threshold": adaptive_threshold
    }

    print(
        "[auto-ransac] "
        f"coarse_inliers={len(provisional_inliers)}, "
        f"median={median_residual:.6f}, "
        f"MAD={mad:.6f}, "
        f"sigma={robust_sigma:.6f}, "
        f"threshold={adaptive_threshold:.6f}"
    )

    return (
        adaptive_threshold,
        provisional_model,
        diagnostics
    )


def plane_model_angle_degrees(model_a, model_b):
    """
    Returns the unsigned angle between two plane normals.

    Opposite normals represent the same plane orientation, so abs(dot)
    is used.
    """
    ma = normalize_plane_model(model_a)
    mb = normalize_plane_model(model_b)

    na = np.asarray(ma[:3], dtype=np.float64)
    nb = np.asarray(mb[:3], dtype=np.float64)

    dot = float(
        np.clip(
            abs(np.dot(na, nb)),
            0.0,
            1.0
        )
    )

    return float(
        np.degrees(
            np.arccos(dot)
        )
    )


def plane_position_at_centroid(plane, plane_model):
    """
    Evaluates the plane Z position at the plane point-cloud centroid XY.
    """
    if not _has_points(plane):
        return None

    model = normalize_plane_model(plane_model)
    a, b, c, d = model

    if abs(c) < EPS_NUM:
        return None

    centroid = np.asarray(
        plane.points,
        dtype=np.float64
    ).mean(axis=0)

    x = float(centroid[0])
    y = float(centroid[1])

    return float(
        -(a * x + b * y + d) / c
    )


# ---------- Automated Processor ----------

class AutomatedSegmentationProcessor:
    def __init__(
            self,
            pcd,
            export_directory,
            stitched_rgb_file=None,
            stitched_height_file=None,
            eye_to_base_rgb_file=None,
            eye_to_base_height_file=None,
            params=None,
            output_index=DEFAULT_OUTPUT_INDEX,
            input_kind=DEFAULT_INPUT_KIND):

        self.original_pcd = pcd
        self.export_dir = export_directory
        os.makedirs(self.export_dir, exist_ok=True)

        self.input_kind = (
            str(input_kind).strip().lower()
            if input_kind
            else DEFAULT_INPUT_KIND
        )

        self.stitched_rgb_file = stitched_rgb_file
        self.stitched_height_file = stitched_height_file
        self.eye_to_base_rgb_file = eye_to_base_rgb_file
        self.eye_to_base_height_file = eye_to_base_height_file
        self.output_index = output_index

        self.params = params or estimate_processing_params(pcd)

        self.filtered_pcd = None
        self.processed_pcd = None
        self.planes_cluster = []
        self.planes_display = []
        self.plane_models = []
        self.remaining_cloud = None
        self.clusters = []
        self.cluster_originals = []
        self.clusters_by_plane = {}
        self.cluster_disp_colors = []
        self.selected_plane_index = None

        # ---- Two-stage plane reference state ----
        self.reference_plane_index = None
        self.reference_plane_model = None
        self.reference_plane_normal = None
        self.reference_cropped_pcd = None
        self.base_plane_index = None
        self.base_plane_model = None
        self.base_plane_normal = None
        self.detected_tile_height = None
        self.tile_top_pcd = None

        self.hull_lines_2d = []
        self.cluster_hull_centroids = []
        self.color_group_labels = None
        self.color_group_mean_rgb = {}
        self.last_grouping_meta = None
        self.last_report_lines = []

        # ---- Stitched map state (RGB + height) ----
        self.stitched_rgb_bgr = None
        self.stitched_height_u16 = None
        self.stitched_height_preview = None

        # ---- Stitch mapping reconstructed from point cloud bounds ----
        self.stitch_res_mm = 1.0
        self.stitch_min_x_mm = None
        self.stitch_max_y_mm = None
        self.stitch_width = None
        self.stitch_height = None

        self.selected_plane_components_by_plane = {}
        self.selected_plane_num_components_by_plane = {}

        self.adaptive_plane_threshold = None
        self.adaptive_ransac_diagnostics = None
        self.multiscale_ransac_diagnostics = None

        self.second_adaptive_plane_threshold = None
        self.second_adaptive_ransac_diagnostics = None
        self.second_multiscale_ransac_diagnostics = None

        self.detected_reference_base_depth = None
        self.adaptive_reference_crop_below = None
        self.reference_depth_histogram_diagnostics = None

        self.adaptive_tile_top_band_half_width = None
        self.tile_top_peak_diagnostics = None

        # ---- Reference-based color grouping state ----
        self.reference_color_palette = []
        self.reference_color_cluster_rgb = []
        self.reference_color_cluster_lab = []
        self.reference_color_distances = []
        self.reference_color_assignments = None

        # ---- GMM + BIC grouping state ----
        self.gmm_selected_component_count = None
        self.gmm_bic_scores = {}
        self.gmm_initial_labels = None
        self.gmm_final_labels = None
        self.gmm_component_mean_lab = {}
        self.gmm_component_mean_rgb = {}
        self.gmm_cluster_probabilities = None

    # ---------- Stitched map helpers ----------

    # -------------------------------------------------------------------------
    # Image loading and point-cloud/image registration
    # -------------------------------------------------------------------------

    def load_stitched_maps(self):
        """
        Loads the RGB and height maps that correspond to the selected input kind.
        Also reconstructs image mapping from the input point-cloud bounds.
        """
        input_kind = str(
            getattr(self, "input_kind", DEFAULT_INPUT_KIND)
        ).strip().lower()

        if input_kind == "eye_to_base":
            rgb_file = self.eye_to_base_rgb_file
            height_file = self.eye_to_base_height_file
            source_name = "eye-to-base"

        else:
            rgb_file = self.stitched_rgb_file
            height_file = self.stitched_height_file
            source_name = "stitched"

        if not rgb_file or not os.path.exists(rgb_file):
            print(
                f"[warn] {source_name} RGB not found: "
                f"{rgb_file}"
            )
            return False

        if not height_file or not os.path.exists(height_file):
            print(
                f"[warn] {source_name} height not found: "
                f"{height_file}"
            )
            return False

        rgb = cv2.imread(
            rgb_file,
            cv2.IMREAD_COLOR
        )

        h16 = cv2.imread(
            height_file,
            cv2.IMREAD_UNCHANGED
        )

        if rgb is None:
            print(
                f"[warn] failed to read {source_name} RGB: "
                f"{rgb_file}"
            )
            return False

        if h16 is None:
            print(
                f"[warn] failed to read {source_name} height: "
                f"{height_file}"
            )
            return False

        # Common validation for both input kinds
        if h16.dtype != np.uint16:
            print(
                f"[warn] {source_name} height must be uint16. "
                f"got {h16.dtype}"
            )
            return False

        if (
            rgb.shape[0] != h16.shape[0]
            or rgb.shape[1] != h16.shape[1]
        ):
            print(
                f"[warn] {source_name} RGB/height shape mismatch: "
                f"rgb={rgb.shape}, height={h16.shape}"
            )
            return False

        # Preserve the existing downstream variable names
        self.stitched_rgb_bgr = rgb
        self.stitched_height_u16 = h16
        self.stitch_height, self.stitch_width = h16.shape[:2]

        nz = h16[h16 > 0]

        if nz.size:
            lo = int(nz.min())
            hi = int(nz.max())

            if hi > lo:
                h8 = (
                    (h16.astype(np.float32) - lo)
                    * (255.0 / (hi - lo))
                ).clip(0, 255).astype(np.uint8)
            else:
                h8 = (
                    (h16 > 0).astype(np.uint8) * 255
                )
        else:
            lo = 0
            hi = 0
            h8 = np.zeros_like(
                h16,
                dtype=np.uint8
            )

        self.stitched_height_preview = cv2.cvtColor(
            h8,
            cv2.COLOR_GRAY2BGR
        )

        pts_m = np.asarray(self.original_pcd.points)

        if pts_m.size == 0:
            print(
                "[warn] point cloud empty; "
                "cannot reconstruct image mapping."
            )
            return False

        X_mm = pts_m[:, 0] * 1000.0
        Y_mm = pts_m[:, 1] * 1000.0

        min_x = float(np.min(X_mm))
        min_y = float(np.min(Y_mm))
        max_x = float(np.max(X_mm))
        max_y = float(np.max(Y_mm))

        res = float(self.stitch_res_mm)

        expected_width = int(
            np.ceil((max_x - min_x) / res)
        ) + 1

        expected_height = int(
            np.ceil((max_y - min_y) / res)
        ) + 1

        if (
            expected_width != self.stitch_width
            or expected_height != self.stitch_height
        ):
            print(
                f"[warn] {source_name} dimensions mismatch "
                f"against point-cloud bounds: "
                f"expected=({expected_height}, {expected_width}), "
                f"actual=({self.stitch_height}, {self.stitch_width})"
            )

            # Keep your existing fallback mapping behaviour
            max_x = min_x + (
                self.stitch_width - 1
            ) * res

            min_y = max_y - (
                self.stitch_height - 1
            ) * res

        self.stitch_min_x_mm = min_x
        self.stitch_max_y_mm = max_y

        print(
            f"[debug] loaded map source: {source_name}"
        )
        print(
            f"[debug] RGB: {self.stitched_rgb_bgr.shape}"
        )
        print(
            f"[debug] height: {self.stitched_height_u16.shape}, "
            f"nonzero={int((h16 > 0).sum())}, "
            f"min={lo}, max={hi}"
        )
        print(
            f"[debug] mapping: "
            f"min_x_mm={self.stitch_min_x_mm:.3f}, "
            f"max_y_mm={self.stitch_max_y_mm:.3f}, "
            f"res_mm={res}"
        )

        return True

    def _project_xy_to_stitch_pixels(self, X_mm, Y_mm):
        res = float(self.stitch_res_mm)
        px = np.floor((X_mm - float(self.stitch_min_x_mm)) / res).astype(np.int32)
        py = np.floor((float(self.stitch_max_y_mm) - Y_mm) / res).astype(np.int32)
        return px, py

    def build_mask_from_plane(self, plane_idx):
        if self.stitched_rgb_bgr is None:
            return False
        if self.stitch_min_x_mm is None or self.stitch_max_y_mm is None:
            return False
        if not self.planes_cluster or plane_idx < 0 or plane_idx >= len(self.planes_cluster):
            return False

        plane = self.planes_cluster[plane_idx]
        if not _has_points(plane):
            return False

        pts_m = np.asarray(plane.points)
        X_mm = pts_m[:, 0] * 1000.0
        Y_mm = pts_m[:, 1] * 1000.0
        px, py = self._project_xy_to_stitch_pixels(X_mm, Y_mm)

        H = int(self.stitch_height)
        W = int(self.stitch_width)
        mask = np.zeros((H, W), dtype=np.uint8)

        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px = px[ok]
        py = py[ok]
        if px.size == 0:
            return False

        mask[py, px] = 255

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, k1, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)

        bin_mask = (mask > 0).astype(np.uint8)
        num, comp = cv2.connectedComponents(bin_mask, connectivity=8)

        self.selected_plane_components_by_plane[plane_idx] = comp.astype(np.int32)
        self.selected_plane_num_components_by_plane[plane_idx] = int(num - 1)

        print(f"[debug] plane {plane_idx}: 2D components={int(num - 1)} nonzero={int((mask > 0).sum())}")
        return True
    
    def apply_rgb_height_edges_to_tile_mask(self, tile_mask):
        """
        Uses RGB + height edges only inside the selected tile-layer mask.
        Returns a cut/refined tile mask.
        """
        if self.stitched_rgb_bgr is None or self.stitched_height_u16 is None:
            return tile_mask

        mask_bool = tile_mask.astype(bool)

        rgb = self.stitched_rgb_bgr.copy()
        h16 = self.stitched_height_u16.copy()

        # --- RGB edges ---
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray[~mask_bool] = 0

        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        rgb_edges = cv2.Canny(
            gray_blur,
            int(self.params.get("rgb_edge_low", 40)),
            int(self.params.get("rgb_edge_high", 120))
        )

        # --- Height edges ---
        h = h16.astype(np.float32)
        h[~mask_bool] = 0

        valid = h[mask_bool]
        if valid.size > 0:
            lo = np.percentile(valid, 2)
            hi = np.percentile(valid, 98)

            if hi > lo:
                h8 = ((h - lo) * (255.0 / (hi - lo))).clip(0, 255).astype(np.uint8)
            else:
                h8 = np.zeros_like(h16, dtype=np.uint8)
        else:
            h8 = np.zeros_like(h16, dtype=np.uint8)

        h8[~mask_bool] = 0
        h8_blur = cv2.GaussianBlur(h8, (5, 5), 0)

        height_edges = cv2.Canny(
            h8_blur,
            int(self.params.get("height_edge_low", 20)),
            int(self.params.get("height_edge_high", 80))
        )

        # --- combine edges only inside selected tile mask ---
        edge_mask = ((rgb_edges > 0) | (height_edges > 0)) & mask_bool

        # Slightly thicken edges so they cut bridges
        edge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                int(self.params.get("edge_dilate_kernel", 3)),
                int(self.params.get("edge_dilate_kernel", 3))
            )
        )

        edge_mask_u8 = edge_mask.astype(np.uint8)
        edge_mask_u8 = cv2.dilate(
            edge_mask_u8,
            edge_kernel,
            iterations=int(self.params.get("edge_dilate_iterations", 1))
        )

        # Cut edges from tile mask
        refined_mask = tile_mask.copy().astype(np.uint8)
        refined_mask[edge_mask_u8 > 0] = 0

        # Small cleanup
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, clean_kernel, iterations=1)

        show_cv_preview(rgb_edges, "Edge Cue 1 - RGB Edges")
        show_cv_preview(height_edges, "Edge Cue 2 - Height Edges")
        show_cv_preview(edge_mask_u8 * 255, "Edge Cue 3 - Combined Edge Mask")
        show_cv_preview(refined_mask * 255, "Edge Cue 4 - Refined Tile Mask")

        return refined_mask
    
    # -------------------------------------------------------------------------
    # Active adaptive two-stage plane workflow
    # -------------------------------------------------------------------------

    def detect_initial_planes_adaptive(self):
        if not _has_points(self.processed_pcd):
            raise RuntimeError(
                "No processed point cloud available."
            )

        if bool(
            self.params.get(
                "adaptive_ransac_enabled",
                True
            )
        ):
            (
                adaptive_threshold,
                provisional_model,
                adaptive_diagnostics
            ) = estimate_adaptive_plane_threshold(
                self.processed_pcd,
                coarse_threshold=float(
                    self.params.get(
                        "adaptive_ransac_coarse_threshold",
                        0.008
                    )
                ),
                coarse_iterations=int(
                    self.params.get(
                        "adaptive_ransac_coarse_iterations",
                        500
                    )
                ),
                residual_multiplier=float(
                    self.params.get(
                        "adaptive_ransac_residual_multiplier",
                        3.0
                    )
                ),
                local_percentile=float(
                    self.params.get(
                        "adaptive_ransac_local_percentile",
                        30.0
                    )
                ),
                minimum_threshold=float(
                    self.params.get(
                        "adaptive_ransac_min_threshold",
                        0.0015
                    )
                ),
                maximum_threshold=float(
                    self.params.get(
                        "adaptive_ransac_max_threshold",
                        0.008
                    )
                )
            )

            self.adaptive_plane_threshold = (
                adaptive_threshold
            )

            self.adaptive_ransac_diagnostics = (
                adaptive_diagnostics
            )

            self.adaptive_ransac_diagnostics[
                "provisional_model"
            ] = provisional_model

        else:
            self.adaptive_plane_threshold = float(
                self.params["plane_distance_threshold"]
            )

            self.adaptive_ransac_diagnostics = {
                "adaptive_enabled": False,
                "adaptive_threshold": (
                    self.adaptive_plane_threshold
                )
            }

        if bool(
            self.params.get(
                "multiscale_ransac_enabled",
                True
            )
        ):
            scales = self.params.get(
                "multiscale_ransac_scales",
                [0.8, 1.0, 1.2]
            )

            (
                self.planes_cluster,
                self.planes_display,
                self.remaining_cloud,
                self.plane_models,
                self.selected_plane_index,
                self.multiscale_ransac_diagnostics
            ) = detect_planes_with_multiscale_check(
                self.processed_pcd,
                adaptive_threshold=(
                    self.adaptive_plane_threshold
                ),
                ransac_n=int(
                    self.params["ransac_n"]
                ),
                num_iterations=int(
                    self.params[
                        "ransac_iterations"
                    ]
                ),
                min_points=int(
                    self.params[
                        "min_plane_points"
                    ]
                ),
                max_planes=int(
                    self.params["max_planes"]
                ),
                threshold_scales=tuple(
                    float(value)
                    for value in scales
                ),
                maximum_normal_difference_deg=float(
                    self.params.get(
                        "multiscale_ransac_max_normal_difference_deg",
                        3.0
                    )
                ),
                maximum_position_difference=float(
                    self.params.get(
                        "multiscale_ransac_max_position_difference",
                        0.003
                    )
                )
            )

        else:
            (
                self.planes_cluster,
                self.planes_display,
                self.remaining_cloud,
                self.plane_models
            ) = detect_multiple_planes(
                self.processed_pcd,
                distance_threshold=(
                    self.adaptive_plane_threshold
                ),
                ransac_n=int(
                    self.params["ransac_n"]
                ),
                num_iterations=int(
                    self.params[
                        "ransac_iterations"
                    ]
                ),
                min_points=int(
                    self.params[
                        "min_plane_points"
                    ]
                ),
                max_planes=int(
                    self.params["max_planes"]
                )
            )

            self.selected_plane_index = (
                select_highest_plane_index(
                    self.planes_cluster,
                    self.plane_models
                )
            )

            self.multiscale_ransac_diagnostics = {
                "multiscale_enabled": False
            }

        display = list(self.planes_display)

        grey = make_grey_copy(
            self.remaining_cloud
        )

        if grey is not None:
            display.append(grey)

        print(
            f"[step] Adaptive initial RANSAC: "
            f"threshold="
            f"{self.adaptive_plane_threshold:.6f}, "
            f"planes={len(self.planes_cluster)}, "
            f"selected={self.selected_plane_index}"
        )

        show_preview(
            display,
            "Initial Adaptive Multi-Scale RANSAC"
        )
    
    # -------------------------------------------------------------------------
    # Active image-mask and watershed tile segmentation
    # -------------------------------------------------------------------------

    def cluster_selected_plane_by_eroded_marker_watershed(self):
        if self.selected_plane_index is None:
            raise RuntimeError("No selected plane index available.")

        if self.stitched_rgb_bgr is None:
            raise RuntimeError("Image maps are not loaded.")

        selected_plane = self.planes_cluster[self.selected_plane_index]
        if not _has_points(selected_plane):
            raise RuntimeError("Selected plane is empty.")

        # Build the binary tile mask using your existing method
        ok = self.build_mask_from_plane(self.selected_plane_index)
        if not ok:
            raise RuntimeError("Could not build selected-plane image mask.")

        comp = self.selected_plane_components_by_plane[self.selected_plane_index]

        # Convert existing component map back to binary mask
        tile_mask = (comp > 0).astype(np.uint8)

        # Clean / create stable marker seeds
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        tile_mask_clean = cv2.morphologyEx(
            tile_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        tile_mask_clean = cv2.morphologyEx(
            tile_mask_clean,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        show_cv_preview(
            tile_mask_clean * 255,
            "Watershed 1 - Clean Tile Mask"
        )

        if bool(self.params.get("use_rgb_height_edges", True)):
            tile_mask_clean = self.apply_rgb_height_edges_to_tile_mask(tile_mask_clean)

        # Erode to break weak bridges and create safer seeds
        marker_mask = cv2.erode(
            tile_mask_clean,
            kernel,
            iterations=1
        )

        show_cv_preview(
            marker_mask * 255,
            "Watershed 2 - Eroded Marker Mask"
        )

        num_markers, markers = cv2.connectedComponents(
            marker_mask,
            connectivity=8
        )

        print(f"[watershed-marker] marker components={num_markers - 1}")

        # Watershed grows the eroded markers back inside the original tile mask
        distance = ndi.distance_transform_edt(tile_mask_clean.astype(bool))
        # distance_smooth = ndi.gaussian_filter(distance, sigma=2.0)

        # coords = peak_local_max(
        #     distance_smooth,
        #     labels=tile_mask_clean.astype(bool),
        #     min_distance=20,
        #     threshold_abs=3,
        #     exclude_border=False
        # )

        # markers = np.zeros(distance.shape, dtype=np.int32)

        # for i, (r, c) in enumerate(coords, start=1):
        #     markers[r, c] = i

        # markers = ndi.label(markers > 0)[0]

        # print(f"[watershed-marker] distance markers={int(markers.max())}")
        
        labels_img = watershed(
            -distance,
            markers,
            mask=tile_mask_clean.astype(bool)
        ).astype(np.int32)

        # labels_img = split_suspicious_labels_with_distance_watershed(
        #     labels_img,
        #     tile_mask_clean.astype(bool),
        #     min_distance=int(self.params.get("touch_split_min_distance", 20)),
        #     sigma=float(self.params.get("touch_split_sigma", 2.0)),
        #     threshold_abs=float(self.params.get("touch_split_threshold_abs", 3)),
        #     min_region_pixels=int(self.params.get("touch_split_min_region_pixels", 50))
        # )

        labels_img = split_suspicious_labels_with_distance_watershed(
            labels_img,
            tile_mask_clean.astype(bool),
            min_region_pixels=int(
                self.params.get("touch_split_min_region_pixels", 50)
            )
        )

        labels_vis = colorize_label_image(labels_img)

        show_cv_preview(
            labels_vis,
            "Watershed 3 - Final Labels"
        )

        print(f"[watershed-marker] final labels={int(labels_img.max())}")

        # Map selected-plane points to watershed labels
        pts_m = np.asarray(selected_plane.points)

        X_mm = pts_m[:, 0] * 1000.0
        Y_mm = pts_m[:, 1] * 1000.0

        px, py = self._project_xy_to_stitch_pixels(X_mm, Y_mm)

        H, W = labels_img.shape[:2]
        in_bounds = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        labels_for_points = np.full(len(pts_m), -1, dtype=np.int32)
        labels_for_points[in_bounds] = labels_img[py[in_bounds], px[in_bounds]]

        clusters_disp = []
        clusters_orig = []

        min_points = int(self.params.get("watershed_min_cluster_points", 50))

        for label_id in sorted(set(labels_for_points.tolist())):
            if label_id <= 0:
                continue

            idx = np.where(labels_for_points == label_id)[0]

            if len(idx) < min_points:
                print(
                    f"[skip small cluster] "
                    f"points={len(idx)}, "
                    f"threshold={min_points}"
                )
                continue

            cluster_orig = selected_plane.select_by_index(idx.tolist())

            cluster_disp = o3d.geometry.PointCloud(cluster_orig)
            cluster_disp.paint_uniform_color([
                random.random(),
                random.random(),
                random.random()
            ])

            clusters_orig.append(cluster_orig)
            clusters_disp.append(cluster_disp)

        self.clusters = clusters_disp
        self.cluster_originals = clusters_orig
        self.clusters_by_plane.clear()
        self.clusters_by_plane[self.selected_plane_index] = clusters_disp

        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()

        print(
            f"[step] Eroded-marker watershed clustering: "
            f"clusters={len(self.clusters)}"
        )

        show_preview(
            self.clusters,
            "5. Eroded-Marker Watershed Clusters"
        )

    def watershed_clusters_from_selected_plane(self):
        if self.selected_plane_index is None:
            raise RuntimeError("No selected plane index available.")

        if self.stitched_height_u16 is None:
            raise RuntimeError("Height image not loaded.")

        selected_plane = self.planes_cluster[self.selected_plane_index]
        if not _has_points(selected_plane):
            raise RuntimeError("Selected plane is empty.")

        # 1. Create image mask from selected plane points
        pts_m = np.asarray(selected_plane.points)
        X_mm = pts_m[:, 0] * 1000.0
        Y_mm = pts_m[:, 1] * 1000.0

        px, py = self._project_xy_to_stitch_pixels(X_mm, Y_mm)

        H = int(self.stitch_height)
        W = int(self.stitch_width)

        plane_mask = np.zeros((H, W), dtype=np.uint8)

        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not np.any(ok):
            raise RuntimeError("Selected plane does not project into image bounds.")

        plane_mask[py[ok], px[ok]] = 255

        # 2. Clean / close mask
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        plane_mask = cv2.dilate(plane_mask, k_open, iterations=1)
        plane_mask = cv2.morphologyEx(plane_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
        plane_mask = cv2.morphologyEx(plane_mask, cv2.MORPH_OPEN, k_open, iterations=1)

        binary = plane_mask > 0

        # 3. Distance transform
        distance = ndi.distance_transform_edt(binary)
        distance_smooth = ndi.gaussian_filter(distance, sigma=2.0)

        # 4. Find markers for watershed
        coords = peak_local_max(
            distance_smooth,
            labels=binary,
            min_distance=25,
            exclude_border=False
        )

        markers = np.zeros(distance.shape, dtype=np.int32)

        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        markers = ndi.label(markers > 0)[0]

        # 5. Watershed labels
        labels_img = watershed(
            -distance_smooth,
            markers,
            mask=binary
        ).astype(np.int32)

        num_labels = int(labels_img.max())
        print(f"[watershed] labels={num_labels}")

        # 6. Assign selected-plane points to image labels
        labels_for_points = np.full(len(pts_m), -1, dtype=np.int32)

        valid_idx = np.where(ok)[0]
        labels_for_points[valid_idx] = labels_img[py[ok], px[ok]]

        # 7. Convert image labels back to point cloud clusters
        clusters_orig = []
        clusters_disp = []

        min_points = int(self.params.get("watershed_min_cluster_points", 50))

        for label_id in sorted(set(labels_for_points.tolist())):
            if label_id <= 0:
                continue

            idx = np.where(labels_for_points == label_id)[0]

            if len(idx) < min_points:
                continue

            cluster_orig = selected_plane.select_by_index(idx.tolist())

            cluster_disp = o3d.geometry.PointCloud(cluster_orig)
            cluster_disp.paint_uniform_color([
                random.random(),
                random.random(),
                random.random()
            ])

            clusters_orig.append(cluster_orig)
            clusters_disp.append(cluster_disp)

        self.clusters = clusters_disp
        self.cluster_originals = clusters_orig
        self.clusters_by_plane.clear()
        self.clusters_by_plane[self.selected_plane_index] = clusters_disp

        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()

        print(f"[step] Watershed image clustering: clusters={len(self.clusters)}")

        show_preview(
            self.clusters,
            "5. Watershed Clusters From Image"
        )

    def sample_cluster_colors_from_stitched(self, cluster, use_hull_mask):
        if self.stitched_rgb_bgr is None:
            return np.zeros((0, 3), dtype=np.float32)
        if self.stitch_min_x_mm is None or self.stitch_max_y_mm is None:
            return np.zeros((0, 3), dtype=np.float32)
        if not _has_points(cluster):
            return np.zeros((0, 3), dtype=np.float32)

        pts_m = np.asarray(cluster.points)
        if pts_m.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if use_hull_mask:
            mask_in_hull, _ = get_hull_mask_and_centroid3d(cluster)
            if mask_in_hull.sum() >= 3:
                pts_m = pts_m[mask_in_hull]

        X_mm = pts_m[:, 0] * 1000.0
        Y_mm = pts_m[:, 1] * 1000.0
        px, py = self._project_xy_to_stitch_pixels(X_mm, Y_mm)

        H, W = self.stitched_rgb_bgr.shape[:2]
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not np.any(ok):
            return np.zeros((0, 3), dtype=np.float32)

        bgr = self.stitched_rgb_bgr[py[ok], px[ok], :].astype(np.float32)
        rgb = bgr[:, ::-1] / 255.0
        return rgb.astype(np.float32)

    # ---------- Main processing steps ----------

    # -------------------------------------------------------------------------
    # Retained alternative / legacy point-cloud pipeline options
    # -------------------------------------------------------------------------

    def apply_z_filter(self):
        z0 = float(self.params["z_min"])
        z1 = float(self.params["z_max"])
        if z0 > z1:
            z0, z1 = z1, z0

        Z = np.asarray(self.original_pcd.points)[:, 2]
        mask = (Z >= z0) & (Z <= z1)
        idx = np.nonzero(mask)[0].tolist()
        self.filtered_pcd = self.original_pcd.select_by_index(idx)

        print(f"[step] Z filter: z_min={z0:.5f}, z_max={z1:.5f}, points={len(self.filtered_pcd.points)}")
        show_preview(self.filtered_pcd, "1. After Z Filter")

    def preprocess(self):
        if not _has_points(self.filtered_pcd):
            raise RuntimeError("No filtered point cloud available.")

        self.processed_pcd = preprocess_point_cloud(
            self.filtered_pcd,
            voxel_size=float(self.params["voxel_size"]),
            nb_neighbors=int(self.params.get("nb_neighbors", 20)),
            std_ratio=float(self.params["std_ratio"])
        )

        print(f"[step] Preprocess: voxel={self.params['voxel_size']:.6f}, std={self.params['std_ratio']}, points={len(self.processed_pcd.points)}")
        show_preview(self.processed_pcd, "2. After Preprocess")

    def detect_planes(self):
        if not _has_points(self.processed_pcd):
            raise RuntimeError("No processed point cloud available.")

        self.planes_cluster, self.planes_display, self.remaining_cloud, self.plane_models = detect_multiple_planes(
            self.processed_pcd,
            distance_threshold=float(self.params["plane_distance_threshold"]),
            ransac_n=int(self.params["ransac_n"]),
            num_iterations=int(self.params["ransac_iterations"]),
            min_points=int(self.params["min_plane_points"]),
            max_planes=int(self.params["max_planes"])
        )

        display = list(self.planes_display)
        grey = make_grey_copy(self.remaining_cloud)
        if grey is not None:
            display.append(grey)

        print(f"[step] Detect planes: planes={len(self.planes_cluster)}, remaining={len(self.remaining_cloud.points) if self.remaining_cloud is not None else 0}")
        show_preview(display, "3. Detected Planes")

    def _normalized_plane_model(self, model, force_positive_global_z=False):
        """
        Returns a normalized plane model [a, b, c, d] and unit normal.

        If force_positive_global_z is True, the model is flipped when needed
        so the normal has a positive global-Z component. This keeps the first
        reference plane's signed-distance direction consistent.
        """
        a, b, c, d = [float(v) for v in model]
        normal = np.array([a, b, c], dtype=np.float64)
        norm = float(np.linalg.norm(normal))

        if norm < EPS_NUM:
            raise RuntimeError(f"Invalid plane model: {model}")

        normal /= norm
        d /= norm

        if force_positive_global_z and normal[2] < 0.0:
            normal = -normal
            d = -d

        normalized = [
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
            float(d)
        ]
        return normalized, normal


    def select_initial_top_reference(self):
        """
        Uses the existing highest-global-plane selector for the first RANSAC pass.
        The selected plane is only a directional/cropping reference.
        """
        if not self.planes_cluster or not self.plane_models:
            raise RuntimeError("Initial RANSAC planes are unavailable.")

        self.reference_plane_index = select_highest_plane_index(
            self.planes_cluster,
            self.plane_models
        )

        model, normal = self._normalized_plane_model(
            self.plane_models[self.reference_plane_index],
            force_positive_global_z=True
        )

        self.reference_plane_model = model
        self.reference_plane_normal = normal

        print(
            f"[reference-plane] selected initial top plane "
            f"index={self.reference_plane_index}, model={model}"
        )

        show_preview(
            self.planes_cluster[self.reference_plane_index],
            f"4. Initial Top Reference Plane {self.reference_plane_index}"
        )


    def crop_along_reference_normal(self):
        """
        Detects the dominant lower-plane peak relative to the initial top
        reference and uses it to set the crop depth automatically.
        """
        if not _has_points(self.processed_pcd):
            raise RuntimeError(
                "Processed point cloud is unavailable."
            )

        if self.reference_plane_model is None:
            raise RuntimeError(
                "Initial reference plane is unavailable."
            )

        a, b, c, d = self.reference_plane_model
        pts = np.asarray(
            self.processed_pcd.points,
            dtype=np.float64
        )

        signed = (
            pts[:, 0] * a
            + pts[:, 1] * b
            + pts[:, 2] * c
            + d
        )

        # Points below the top reference should have negative signed distance.
        # Convert them to positive depth values for histogram analysis.
        depth_below_reference = -signed

        search_min = float(
            self.params.get(
                "reference_base_depth_search_min",
                0.002
            )
        )

        search_max = float(
            self.params.get(
                "reference_base_depth_search_max",
                0.030
            )
        )

        bin_width = float(
            self.params.get(
                "reference_depth_hist_bin",
                0.00025
            )
        )

        try:
            (
                detected_base_depth,
                detected_peak_half_width,
                diagnostics
            ) = estimate_histogram_peak_spread(
                depth_below_reference,
                search_min=search_min,
                search_max=search_max,
                bin_width=bin_width,
                smoothing_sigma_bins=float(
                    self.params.get(
                        "reference_depth_hist_smoothing",
                        1.5
                    )
                ),
                spread_window=float(
                    self.params.get(
                        "reference_depth_peak_spread_window",
                        0.004
                    )
                ),
                minimum_half_width=float(
                    self.params.get(
                        "reference_depth_peak_min_half_width",
                        0.0008
                    )
                ),
                maximum_half_width=float(
                    self.params.get(
                        "reference_depth_peak_max_half_width",
                        0.005
                    )
                ),
                spread_multiplier=float(
                    self.params.get(
                        "reference_depth_peak_spread_multiplier",
                        2.5
                    )
                )
            )

            safety_margin = float(
                self.params.get(
                    "reference_crop_base_margin",
                    0.003
                )
            )

            # Keep slightly beyond the detected lower/base peak.
            below = (
                detected_base_depth
                + detected_peak_half_width
                + safety_margin
            )

            below = float(
                np.clip(
                    below,
                    float(
                        self.params.get(
                            "reference_crop_min_below",
                            0.008
                        )
                    ),
                    float(
                        self.params.get(
                            "reference_crop_max_below",
                            0.030
                        )
                    )
                )
            )

            self.detected_reference_base_depth = float(
                detected_base_depth
            )

            self.adaptive_reference_crop_below = float(
                below
            )

            self.reference_depth_histogram_diagnostics = (
                diagnostics
            )

        except Exception as exc:
            # Safe fallback to your previous fixed value.
            below = float(
                self.params.get(
                    "reference_crop_below",
                    0.015
                )
            )

            self.detected_reference_base_depth = None
            self.adaptive_reference_crop_below = below
            self.reference_depth_histogram_diagnostics = {
                "fallback": True,
                "error": str(exc),
                "crop_below": float(below)
            }

            print(
                f"[reference-crop][warn] adaptive depth detection "
                f"failed: {exc}. Using fallback={below:.6f} m."
            )

        above = float(
            self.params.get(
                "reference_crop_above",
                0.002
            )
        )

        keep = (
            (signed <= above)
            & (signed >= -below)
        )

        indices = np.where(keep)[0].tolist()

        self.reference_cropped_pcd = (
            self.processed_pcd.select_by_index(indices)
        )

        if not _has_points(self.reference_cropped_pcd):
            raise RuntimeError(
                "Reference-normal crop produced an empty cloud."
            )

        print(
            f"[reference-crop] above={above:.6f} m, "
            f"adaptive_below={below:.6f} m, "
            f"detected_base_depth="
            f"{self.detected_reference_base_depth}, "
            f"points={len(self.reference_cropped_pcd.points)}"
        )

        show_preview(
            self.reference_cropped_pcd,
            "5. Adaptive Crop Along Initial Plane Normal"
        )


    def detect_base_plane_from_cropped(self):
        """
        Runs a second adaptive multi-plane RANSAC pass on the cropped cloud and
        selects a large plane that is parallel to, and below, the initial
        top reference.
        """
        if not _has_points(self.reference_cropped_pcd):
            raise RuntimeError("Reference-cropped cloud is unavailable.")

        # ---------------------------------------------------------
        # 1. Estimate a new RANSAC threshold from the cropped cloud.
        # The cropped cloud can have a different residual/noise
        # distribution from the original processed cloud.
        # ---------------------------------------------------------
        (
            second_threshold,
            second_provisional_model,
            second_adaptive_diagnostics
        ) = estimate_adaptive_plane_threshold(
            self.reference_cropped_pcd,
            coarse_threshold=float(
                self.params.get(
                    "second_adaptive_ransac_coarse_threshold",
                    self.params.get(
                        "adaptive_ransac_coarse_threshold",
                        0.008
                    )
                )
            ),
            coarse_iterations=int(
                self.params.get(
                    "second_adaptive_ransac_coarse_iterations",
                    self.params.get(
                        "adaptive_ransac_coarse_iterations",
                        500
                    )
                )
            ),
            residual_multiplier=float(
                self.params.get(
                    "second_adaptive_ransac_residual_multiplier",
                    self.params.get(
                        "adaptive_ransac_residual_multiplier",
                        3.0
                    )
                )
            ),
            local_percentile=float(
                self.params.get(
                    "second_adaptive_ransac_local_percentile",
                    self.params.get(
                        "adaptive_ransac_local_percentile",
                        30.0
                    )
                )
            ),
            minimum_threshold=float(
                self.params.get(
                    "second_adaptive_ransac_min_threshold",
                    self.params.get(
                        "adaptive_ransac_min_threshold",
                        0.0015
                    )
                )
            ),
            maximum_threshold=float(
                self.params.get(
                    "second_adaptive_ransac_max_threshold",
                    self.params.get(
                        "adaptive_ransac_max_threshold",
                        0.008
                    )
                )
            )
        )

        self.second_adaptive_plane_threshold = float(second_threshold)
        self.second_adaptive_ransac_diagnostics = dict(
            second_adaptive_diagnostics
        )
        self.second_adaptive_ransac_diagnostics[
            "provisional_model"
        ] = second_provisional_model

        print(
            f"[base-ransac] adaptive threshold="
            f"{self.second_adaptive_plane_threshold:.6f} m"
        )

        # ---------------------------------------------------------
        # 2. Run nearby threshold scales and use the central result.
        # The multi-scale helper validates the highest plane, but for
        # this second stage we still perform the base selection below
        # using parallelism, offset, and point support.
        # ---------------------------------------------------------
        scales = self.params.get(
            "second_multiscale_ransac_scales",
            self.params.get(
                "multiscale_ransac_scales",
                [0.8, 1.0, 1.2]
            )
        )

        (
            planes,
            displays,
            remaining,
            models,
            multiscale_selected_index,
            second_multiscale_diagnostics
        ) = detect_planes_with_multiscale_check(
            self.reference_cropped_pcd,
            adaptive_threshold=self.second_adaptive_plane_threshold,
            ransac_n=int(self.params["ransac_n"]),
            num_iterations=int(self.params["ransac_iterations"]),
            min_points=int(
                self.params.get(
                    "second_min_plane_points",
                    self.params["min_plane_points"]
                )
            ),
            max_planes=int(
                self.params.get(
                    "second_max_planes",
                    self.params["max_planes"]
                )
            ),
            threshold_scales=tuple(
                float(value)
                for value in scales
            ),
            maximum_normal_difference_deg=float(
                self.params.get(
                    "second_multiscale_max_normal_difference_deg",
                    self.params.get(
                        "multiscale_ransac_max_normal_difference_deg",
                        3.0
                    )
                )
            ),
            maximum_position_difference=float(
                self.params.get(
                    "second_multiscale_max_position_difference",
                    self.params.get(
                        "multiscale_ransac_max_position_difference",
                        0.003
                    )
                )
            )
        )

        self.second_multiscale_ransac_diagnostics = (
            second_multiscale_diagnostics
        )

        display = list(displays)

        grey = make_grey_copy(remaining)
        if grey is not None:
            display.append(grey)

        print(
            f"[base-ransac] planes={len(planes)}, "
            f"remaining="
            f"{len(remaining.points) if remaining is not None else 0}, "
            f"multiscale_reference_index={multiscale_selected_index}"
        )

        show_preview(
            display,
            "6. Adaptive Second RANSAC Candidate Planes"
        )

        if not planes:
            raise RuntimeError(
                "Adaptive second RANSAC found no base-plane candidates."
            )

        # ---------------------------------------------------------
        # 3. Existing base selection logic remains unchanged.
        # ---------------------------------------------------------
        ref_model = self.reference_plane_model
        ref_normal = self.reference_plane_normal

        cos_tol = float(
            np.cos(
                np.deg2rad(
                    float(
                        self.params.get(
                            "base_parallel_tolerance_deg",
                            5.0
                        )
                    )
                )
            )
        )

        min_below = float(
            self.params.get(
                "base_min_below_reference",
                0.001
            )
        )

        candidates = []

        for i, (plane, model_raw) in enumerate(
            zip(planes, models)
        ):
            model, normal = self._normalized_plane_model(
                model_raw,
                force_positive_global_z=False
            )

            # Orient candidate normal to the same hemisphere as reference.
            if float(np.dot(normal, ref_normal)) < 0.0:
                normal = -normal
                model = [
                    float(-model[0]),
                    float(-model[1]),
                    float(-model[2]),
                    float(-model[3])
                ]

            parallel_score = float(
                np.dot(normal, ref_normal)
            )

            pts = np.asarray(plane.points)
            center = pts.mean(axis=0)

            # Signed location of candidate centroid relative to top reference.
            offset_from_reference = float(
                center[0] * ref_model[0]
                + center[1] * ref_model[1]
                + center[2] * ref_model[2]
                + ref_model[3]
            )

            point_count = int(len(pts))

            print(
                f"[base-candidate] plane={i}, "
                f"points={point_count}, "
                f"parallel={parallel_score:.6f}, "
                f"offset_from_reference="
                f"{offset_from_reference:.6f} m, "
                f"model={model}"
            )

            if parallel_score < cos_tol:
                print(
                    f"[base-candidate] rejected plane={i}: "
                    f"parallel score below tolerance."
                )
                continue

            if offset_from_reference > -min_below:
                print(
                    f"[base-candidate] rejected plane={i}: "
                    f"not sufficiently below reference."
                )
                continue

            candidates.append({
                "index": i,
                "point_count": point_count,
                "offset": offset_from_reference,
                "model": model,
                "normal": normal,
                "plane": plane
            })

        if not candidates:
            raise RuntimeError(
                "No adaptive second-pass plane was parallel to and below "
                "the initial top reference."
            )

        # The base should be the dominant continuous parallel plane.
        selected = max(
            candidates,
            key=lambda item: item["point_count"]
        )

        self.base_plane_index = int(selected["index"])
        self.base_plane_model = list(selected["model"])
        self.base_plane_normal = np.asarray(
            selected["normal"],
            dtype=np.float64
        )

        print(
            f"[base-select] selected plane="
            f"{self.base_plane_index}, "
            f"points={selected['point_count']}, "
            f"offset={selected['offset']:.6f} m, "
            f"adaptive_threshold="
            f"{self.second_adaptive_plane_threshold:.6f} m, "
            f"model={self.base_plane_model}"
        )

        show_preview(
            selected["plane"],
            f"7. Selected Base Plane {self.base_plane_index}"
        )


    def extract_tile_top_band_from_base(self):
        """
        Measures signed height above the selected base plane, detects the
        strongest tile-top height peak, and estimates the tile-top band width
        from the measured peak spread.
        """
        if not _has_points(self.reference_cropped_pcd):
            raise RuntimeError(
                "Reference-cropped cloud is unavailable."
            )

        if self.base_plane_model is None:
            raise RuntimeError(
                "Base plane model is unavailable."
            )

        a, b, c, d = self.base_plane_model

        pts = np.asarray(
            self.reference_cropped_pcd.points,
            dtype=np.float64
        )

        signed = (
            pts[:, 0] * a
            + pts[:, 1] * b
            + pts[:, 2] * c
            + d
        )

        # Ensure tile tops are on the positive side of the base.
        ref_plane_pts = np.asarray(
            self.planes_cluster[
                self.reference_plane_index
            ].points
        )

        ref_center = ref_plane_pts.mean(axis=0)

        ref_signed_to_base = float(
            ref_center[0] * a
            + ref_center[1] * b
            + ref_center[2] * c
            + d
        )

        if ref_signed_to_base < 0.0:
            signed = -signed

            self.base_plane_model = [
                -a,
                -b,
                -c,
                -d
            ]

            self.base_plane_normal = (
                -self.base_plane_normal
            )

            a, b, c, d = self.base_plane_model
            ref_signed_to_base = -ref_signed_to_base

        h_min = float(
            self.params.get(
                "tile_height_search_min",
                0.002
            )
        )

        h_max = float(
            self.params.get(
                "tile_height_search_max",
                0.030
            )
        )

        bin_width = float(
            self.params.get(
                "tile_height_hist_bin",
                0.00025
            )
        )

        (
            tile_height,
            adaptive_half_width,
            diagnostics
        ) = estimate_histogram_peak_spread(
            signed,
            search_min=h_min,
            search_max=h_max,
            bin_width=bin_width,
            smoothing_sigma_bins=float(
                self.params.get(
                    "tile_height_hist_smoothing",
                    1.5
                )
            ),
            spread_window=float(
                self.params.get(
                    "tile_top_spread_window",
                    0.004
                )
            ),
            minimum_half_width=float(
                self.params.get(
                    "tile_top_band_min_half_width",
                    0.0008
                )
            ),
            maximum_half_width=float(
                self.params.get(
                    "tile_top_band_max_half_width",
                    0.005
                )
            ),
            spread_multiplier=float(
                self.params.get(
                    "tile_top_band_spread_multiplier",
                    2.5
                )
            )
        )

        keep = (
            np.abs(signed - tile_height)
            <= adaptive_half_width
        )

        indices = np.where(keep)[0].tolist()

        self.tile_top_pcd = (
            self.reference_cropped_pcd.select_by_index(
                indices
            )
        )

        if not _has_points(self.tile_top_pcd):
            raise RuntimeError(
                "Detected adaptive tile-top band is empty."
            )

        self.detected_tile_height = float(
            tile_height
        )

        self.adaptive_tile_top_band_half_width = float(
            adaptive_half_width
        )

        self.tile_top_peak_diagnostics = diagnostics

        print(
            f"[tile-top-band] "
            f"detected_height={tile_height:.6f} m, "
            f"adaptive_band=±{adaptive_half_width:.6f} m, "
            f"peak_sigma="
            f"{diagnostics['robust_sigma']:.6f} m, "
            f"peak_fwhm={diagnostics['fwhm']:.6f} m, "
            f"points={len(self.tile_top_pcd.points)}, "
            f"reference_height_from_base="
            f"{ref_signed_to_base:.6f} m"
        )

        show_preview(
            self.tile_top_pcd,
            "8. Adaptive Signed-Distance Tile-Top Band"
        )

        # Preserve the existing downstream contract.
        self.planes_cluster = [
            self.tile_top_pcd
        ]

        self.planes_display = [
            o3d.geometry.PointCloud(
                self.tile_top_pcd
            )
        ]

        self.plane_models = [
            list(self.base_plane_model)
        ]

        self.remaining_cloud = None
        self.selected_plane_index = 0

    def cluster_planes_all(self):
        if not self.planes_cluster:
            raise RuntimeError("No planes available for clustering.")

        self.clusters = []
        self.cluster_originals = []
        self.clusters_by_plane.clear()

        for i, plane in enumerate(self.planes_cluster):
            clusters_disp, clusters_orig = cluster_colored_plane(
                plane,
                eps=float(self.params["dbscan_eps"]),
                min_samples=int(self.params["dbscan_min_samples"]),
                color_weight=float(self.params["color_weight"]),
                position_weight=float(self.params["position_weight"])
            )
            self.clusters_by_plane[i] = clusters_disp
            self.clusters.extend(clusters_disp)
            self.cluster_originals.extend(clusters_orig)
            print(f"[cluster] plane {i}: {len(clusters_orig)} clusters")

        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()

        display = list(self.clusters)
        grey = make_grey_copy(self.remaining_cloud)
        if grey is not None:
            display.append(grey)

        print(f"[step] Cluster all planes: total_clusters={len(self.clusters)}")
        show_preview(display, "4. DBSCAN Clusters")

    def cluster_selected_highest_plane(self):
        """
        Same clustering function as the GUI, but feeds only the automatically
        selected highest plane instead of clustering every detected plane.
        """
        if not self.planes_cluster:
            raise RuntimeError("No planes available for clustering.")

        self.selected_plane_index = select_highest_plane_index(
            self.planes_cluster,
            self.plane_models
        )
        selected_plane = self.planes_cluster[self.selected_plane_index]

        print(f"[step] Selected plane for clustering: {self.selected_plane_index}")
        show_preview(
            selected_plane,
            f"4. Selected Highest Plane {self.selected_plane_index}"
        )

        clusters_disp, clusters_orig = cluster_colored_plane(
            selected_plane,
            eps=float(self.params["dbscan_eps"]),
            min_samples=int(self.params["dbscan_min_samples"]),
            color_weight=float(self.params["color_weight"]),
            position_weight=float(self.params["position_weight"])
        )

        self.clusters = list(clusters_disp)
        self.cluster_originals = list(clusters_orig)
        self.clusters_by_plane.clear()
        self.clusters_by_plane[self.selected_plane_index] = list(clusters_disp)

        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()

        display = list(self.clusters)
        grey = make_grey_copy(self.remaining_cloud)
        if grey is not None:
            display.append(grey)

        print(
            f"[step] Cluster selected plane: plane={self.selected_plane_index}, "
            f"clusters={len(self.clusters)}"
        )
        show_preview(display, "5. DBSCAN Clusters From Selected Plane")

    def _capture_cluster_display_colors(self):
        self.cluster_disp_colors = []
        for cl in self.clusters:
            if len(cl.colors) > 0:
                self.cluster_disp_colors.append(np.asarray(cl.colors)[0].copy())
            else:
                self.cluster_disp_colors.append(np.array([0.5, 0.5, 0.5], dtype=np.float32))

    def _recompute_hull_centroids_for_all(self):
        self.cluster_hull_centroids = []
        for cl in self.clusters:
            _, c3 = get_hull_mask_and_centroid3d(cl)
            self.cluster_hull_centroids.append(c3)

    def split_dbscan_clusters_by_2d(self):
        if not bool(self.params.get("enable_2d_split_check", True)):
            print("[info] 2D split check disabled.")
            return

        if self.stitched_rgb_bgr is None:
            print("[info] stitched maps not available; skipping 2D split check.")
            return

        if self.selected_plane_index is not None:
            self.build_mask_from_plane(self.selected_plane_index)
        else:
            for plane_idx in range(len(self.planes_cluster)):
                self.build_mask_from_plane(plane_idx)

        if not self.selected_plane_components_by_plane:
            print("[warn] no 2D components available; skipping 2D split.")
            return

        new_disp = []
        new_orig = []
        min_frac = float(self.params.get("split_min_frac", 0.15))
        min_pts = int(self.params.get("split_min_points", 200))
        split_count = 0

        for cidx, (disp_cluster, orig_cluster) in enumerate(zip(self.clusters, self.cluster_originals)):
            pts_m = np.asarray(orig_cluster.points)
            if pts_m.size == 0:
                continue

            X_mm = pts_m[:, 0] * 1000.0
            Y_mm = pts_m[:, 1] * 1000.0
            px, py = self._project_xy_to_stitch_pixels(X_mm, Y_mm)

            # Use the component map that explains this cluster best.
            best_comp = None
            best_ok = None
            best_votes = 0
            for _, comp in self.selected_plane_components_by_plane.items():
                H, W = comp.shape[:2]
                ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                if not np.any(ok):
                    continue
                ids = comp[py[ok], px[ok]]
                votes = int((ids > 0).sum())
                if votes > best_votes:
                    best_votes = votes
                    best_comp = comp
                    best_ok = ok

            if best_comp is None or best_votes == 0:
                new_disp.append(disp_cluster)
                new_orig.append(orig_cluster)
                continue

            px_ok = px[best_ok]
            py_ok = py[best_ok]
            comp_ids_all = best_comp[py_ok, px_ok]
            comp_ids = comp_ids_all[comp_ids_all > 0]
            if comp_ids.size == 0:
                new_disp.append(disp_cluster)
                new_orig.append(orig_cluster)
                continue

            uniq, cnt = np.unique(comp_ids, return_counts=True)
            total = int(cnt.sum())
            keep = (cnt >= min_pts) & (cnt >= (min_frac * total))
            uniq_k = uniq[keep]
            cnt_k = cnt[keep]

            if uniq_k.size <= 1:
                new_disp.append(disp_cluster)
                new_orig.append(orig_cluster)
                continue

            split_count += 1
            comp_per_pt = np.zeros(len(pts_m), dtype=np.int32)
            comp_per_pt_idx = np.where(best_ok)[0]
            comp_per_pt[comp_per_pt_idx] = comp_ids_all

            for cid in uniq_k.tolist():
                sub_idx = np.where(comp_per_pt == cid)[0]
                if sub_idx.size == 0:
                    continue
                sub_orig = orig_cluster.select_by_index(sub_idx.tolist())
                sub_disp = o3d.geometry.PointCloud(sub_orig)
                sub_disp.paint_uniform_color([random.random(), random.random(), random.random()])
                new_orig.append(sub_orig)
                new_disp.append(sub_disp)

            print(f"[split] cluster[{cidx}] -> {int(uniq_k.size)} parts (2D comps: {uniq_k.tolist()} counts={cnt_k.tolist()})")

        if split_count == 0:
            print("[info] No clusters required splitting by 2D components.")
            return

        self.clusters = new_disp
        self.cluster_originals = new_orig
        self._capture_cluster_display_colors()
        self._recompute_hull_centroids_for_all()

        print(f"[step] 2D split applied: new_cluster_count={len(self.clusters)}")
        show_preview(self.clusters, "5. After 2D Split")

    # -------------------------------------------------------------------------
    # Shared hull generation and color-grouping dispatch targets
    # -------------------------------------------------------------------------

    def compute_convex_hulls_all_2d(self):
        self.hull_lines_2d = []
        for cl in self.cluster_originals:
            ls = create_2d_hull_lineset_from_cluster(cl)
            if ls is not None:
                self.hull_lines_2d.append(ls)

        display = list(self.clusters) + list(self.hull_lines_2d)
        print(f"[step] Hull 2D all: hulls={len(self.hull_lines_2d)}")
        show_preview(display, "6. 2D Hulls")

    # Color grouping option 1: legacy DBSCAN
    def group_clusters_color(self):
        if not self.cluster_originals:
            raise RuntimeError("No clusters available for color grouping.")

        centroids = []
        labels, group_colors, report_lines = group_clusters_by_color(
            self.cluster_originals,
            mode=str(self.params["group_mode"]),
            eps_z=float(self.params["group_eps"]),
            min_size=int(self.params["group_min_size"]),
            use_hull_mask=bool(self.params["group_use_hull_color"]),
            centroids_out=centroids,
            color_sampler=self.sample_cluster_colors_from_stitched
        )

        if labels.size == 0:
            self.last_report_lines = ["Color Grouping Report", "Color grouping produced no labels."]
            return

        for i, cl in enumerate(self.clusters):
            g = labels[i]
            if g == -1:
                continue
            rgb = group_colors.get(g, np.array([random.random(), random.random(), random.random()]))
            cl.paint_uniform_color(rgb.tolist())

        self.color_group_labels = labels
        self.color_group_mean_rgb = {int(k): group_colors[k].tolist() for k in group_colors}

        groups_summary = []
        for g in sorted(set(labels)):
            if g == -1:
                continue
            members = np.where(labels == g)[0].tolist()
            groups_summary.append({
                "group_id": int(g),
                "mean_rgb": self.color_group_mean_rgb[int(g)],
                "members": [int(x) for x in members]
            })

        self.last_grouping_meta = {
            "mode": str(self.params["group_mode"]),
            "eps_z": float(self.params["group_eps"]),
            "min_size": int(self.params["group_min_size"]),
            "use_hull_mask": bool(self.params["group_use_hull_color"]),
            "labels": labels.tolist(),
            "groups": groups_summary
        }

        self.cluster_hull_centroids = centroids
        self.last_report_lines = report_lines

        display = list(self.clusters) + list(self.hull_lines_2d)
        print(f"[step] Color grouping: groups={len(groups_summary)}, labels={labels.tolist()}")
        show_preview(display, "7. Color Grouping")

    # -------------------------------------------------------------------------
    # Color grouping option 2: reference-based classification
    # -------------------------------------------------------------------------

    def _normalize_reference_rgb(self, rgb):
        """
        Converts one RGB reference sample to float RGB in [0, 1].

        Accepts either:
            [181, 159, 133]
        or:
            [0.71, 0.62, 0.52]
        """
        value = np.asarray(
            rgb,
            dtype=np.float32
        ).reshape(3)

        if float(np.max(value)) > 1.0:
            value = value / 255.0

        return np.clip(
            value,
            0.0,
            1.0
        ).astype(np.float32)
    
    def prepare_reference_color_palette(self):
        """
        Converts the configured RGB reference samples to representative
        RGB and Lab centers.

        This does not modify the existing DBSCAN color grouping.
        """
        configured = self.params.get(
            "reference_color_groups",
            []
        )

        if not isinstance(configured, list):
            raise RuntimeError(
                "reference_color_groups must be a list."
            )

        palette = []
        used_ids = set()

        for item in configured:
            if not isinstance(item, dict):
                continue

            if "group_id" not in item:
                raise RuntimeError(
                    "Every reference color requires group_id."
                )

            group_id = int(item["group_id"])

            if group_id in used_ids:
                raise RuntimeError(
                    f"Duplicate reference group_id: {group_id}"
                )

            used_ids.add(group_id)

            samples_raw = item.get(
                "samples_rgb",
                []
            )

            if not samples_raw:
                raise RuntimeError(
                    f"Reference group {group_id} has no samples_rgb."
                )

            samples_rgb = np.stack(
                [
                    self._normalize_reference_rgb(sample)
                    for sample in samples_raw
                ],
                axis=0
            ).astype(np.float32)

            # Convert all reference samples to Lab.
            samples_lab = _rgb_to_lab(
                samples_rgb
            ).astype(np.float32)

            # Median is less sensitive to one bad reference sample.
            center_rgb = np.median(
                samples_rgb,
                axis=0
            ).astype(np.float32)

            center_lab = np.median(
                samples_lab,
                axis=0
            ).astype(np.float32)

            max_distance = float(
                item.get(
                    "max_distance",
                    12.0
                )
            )

            palette.append({
                "group_id": group_id,
                "name": str(
                    item.get(
                        "name",
                        f"group_{group_id}"
                    )
                ),
                "samples_rgb": samples_rgb,
                "samples_lab": samples_lab,
                "center_rgb": center_rgb,
                "center_lab": center_lab,
                "max_distance": max_distance
            })

            print(
                f"[reference-color] loaded group={group_id}, "
                f"name={palette[-1]['name']}, "
                f"samples={len(samples_rgb)}, "
                f"center_rgb={center_rgb.tolist()}, "
                f"max_distance={max_distance:.3f}"
            )

        if not palette:
            raise RuntimeError(
                "No valid reference colors were configured."
            )

        self.reference_color_palette = palette

    
    def build_eroded_cluster_hull_mask(self, cluster):
        """
        Projects a cluster into the loaded RGB image, creates a filled convex
        hull mask, then erodes the mask to avoid boundary/background pixels.

        Returns
        -------
        mask : uint8 image
            0 outside the valid sampling region and 255 inside.
        """
        if self.stitched_rgb_bgr is None:
            return None

        if (
            self.stitch_min_x_mm is None
            or self.stitch_max_y_mm is None
        ):
            return None

        if not _has_points(cluster):
            return None

        pts_m = np.asarray(
            cluster.points,
            dtype=np.float64
        )

        if len(pts_m) < 3:
            return None

        X_mm = pts_m[:, 0] * 1000.0
        Y_mm = pts_m[:, 1] * 1000.0

        px, py = self._project_xy_to_stitch_pixels(
            X_mm,
            Y_mm
        )

        height, width = (
            self.stitched_rgb_bgr.shape[:2]
        )

        valid = (
            (px >= 0)
            & (px < width)
            & (py >= 0)
            & (py < height)
        )

        if int(np.count_nonzero(valid)) < 3:
            return None

        pixel_points = np.column_stack(
            [
                px[valid],
                py[valid]
            ]
        ).astype(np.int32)

        # Remove duplicate projected pixels.
        pixel_points = np.unique(
            pixel_points,
            axis=0
        )

        if len(pixel_points) < 3:
            return None

        hull = cv2.convexHull(
            pixel_points.reshape(-1, 1, 2)
        )

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        cv2.fillConvexPoly(
            mask,
            hull,
            255
        )

        erosion_px = int(
            self.params.get(
                "reference_color_hull_erosion_px",
                3
            )
        )

        if erosion_px > 0:
            # Kernel must have an odd size.
            kernel_size = max(
                3,
                erosion_px * 2 + 1
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    kernel_size,
                    kernel_size
                )
            )

            eroded = cv2.erode(
                mask,
                kernel,
                iterations=1
            )

            # Do not allow erosion to completely delete a small valid tile.
            min_pixels = int(
                self.params.get(
                    "reference_color_min_pixels",
                    20
                )
            )

            if int(np.count_nonzero(eroded)) >= min_pixels:
                mask = eroded

        return mask
    

    def sample_reference_color_pixels(self, cluster):
        """
        Samples RGB pixels from the eroded image-space hull of one cluster.

        Returns float RGB values in [0, 1].
        """
        mask = self.build_eroded_cluster_hull_mask(
            cluster
        )

        if mask is None:
            return np.zeros(
                (0, 3),
                dtype=np.float32
            )

        valid = mask > 0

        if not np.any(valid):
            return np.zeros(
                (0, 3),
                dtype=np.float32
            )

        bgr = self.stitched_rgb_bgr[
            valid
        ].astype(np.float32)

        rgb = bgr[:, ::-1] / 255.0

        return np.clip(
            rgb,
            0.0,
            1.0
        ).astype(np.float32)
    

    def filter_reference_color_pixels(self, rgb_pixels):
        """
        Removes darkest and brightest sampled pixels according to Lab L*.

        The chromatic information remains unchanged; only likely shadow and
        highlight pixels are rejected.
        """
        rgb_pixels = np.asarray(
            rgb_pixels,
            dtype=np.float32
        )

        if (
            rgb_pixels.ndim != 2
            or rgb_pixels.shape[1] != 3
            or len(rgb_pixels) == 0
        ):
            return np.zeros(
                (0, 3),
                dtype=np.float32
            )

        lab_pixels = _rgb_to_lab(
            rgb_pixels
        ).astype(np.float32)

        lightness = lab_pixels[:, 0]

        low_percentile = float(
            self.params.get(
                "reference_color_low_percentile",
                10.0
            )
        )

        high_percentile = float(
            self.params.get(
                "reference_color_high_percentile",
                90.0
            )
        )

        low_percentile = float(
            np.clip(
                low_percentile,
                0.0,
                49.0
            )
        )

        high_percentile = float(
            np.clip(
                high_percentile,
                51.0,
                100.0
            )
        )

        low_value = float(
            np.percentile(
                lightness,
                low_percentile
            )
        )

        high_value = float(
            np.percentile(
                lightness,
                high_percentile
            )
        )

        keep = (
            (lightness >= low_value)
            & (lightness <= high_value)
        )

        filtered = rgb_pixels[keep]

        min_pixels = int(
            self.params.get(
                "reference_color_min_pixels",
                20
            )
        )

        # If filtering became too aggressive, use the original sample.
        if len(filtered) < min_pixels:
            filtered = rgb_pixels

        return filtered.astype(np.float32)
    

    def compute_reference_cluster_color(self, cluster):
        """
        Produces representative RGB and Lab values for one cluster.

        Workflow:
            eroded hull
            → image pixels
            → brightness outlier removal
            → median RGB
            → median Lab
        """
        rgb_pixels = self.sample_reference_color_pixels(
            cluster
        )

        min_pixels = int(
            self.params.get(
                "reference_color_min_pixels",
                20
            )
        )

        if len(rgb_pixels) < min_pixels:
            return None

        filtered_rgb = self.filter_reference_color_pixels(
            rgb_pixels
        )

        if len(filtered_rgb) < min_pixels:
            return None

        filtered_lab = _rgb_to_lab(
            filtered_rgb
        ).astype(np.float32)

        representative_rgb = np.median(
            filtered_rgb,
            axis=0
        ).astype(np.float32)

        representative_lab = np.median(
            filtered_lab,
            axis=0
        ).astype(np.float32)

        return {
            "rgb": representative_rgb,
            "lab": representative_lab,
            "raw_pixel_count": int(
                len(rgb_pixels)
            ),
            "filtered_pixel_count": int(
                len(filtered_rgb)
            )
        }
    

    def weighted_reference_lab_distance(
            self,
            cluster_lab,
            reference_lab
        ):
        """
        Weighted Euclidean distance in Lab space.

        A low L* weight reduces sensitivity to lighting differences.
        """
        cluster_lab = np.asarray(
            cluster_lab,
            dtype=np.float32
        ).reshape(3)

        reference_lab = np.asarray(
            reference_lab,
            dtype=np.float32
        ).reshape(3)

        weights = np.asarray(
            self.params.get(
                "reference_color_lab_weights",
                [0.1, 1.0, 1.0]
            ),
            dtype=np.float32
        ).reshape(3)

        weights = np.maximum(
            weights,
            0.0
        )

        difference = (
            cluster_lab - reference_lab
        )

        return float(
            np.sqrt(
                np.sum(
                    weights
                    * difference
                    * difference
                )
            )
        )
    

    def classify_cluster_against_reference_palette(
            self,
            cluster_lab
        ):
        """
        Finds the closest reference color and applies:

        1. maximum-distance acceptance;
        2. minimum separation from the second-best reference.

        Returns
        -------
        result : dict
        """
        if not self.reference_color_palette:
            raise RuntimeError(
                "Reference color palette is not prepared."
            )

        comparisons = []

        for reference in self.reference_color_palette:
            distance = self.weighted_reference_lab_distance(
                cluster_lab,
                reference["center_lab"]
            )

            comparisons.append({
                "group_id": int(
                    reference["group_id"]
                ),
                "name": reference["name"],
                "distance": float(distance),
                "max_distance": float(
                    reference["max_distance"]
                ),
                "center_rgb": reference[
                    "center_rgb"
                ]
            })

        comparisons.sort(
            key=lambda item: item["distance"]
        )

        best = comparisons[0]

        if len(comparisons) > 1:
            second = comparisons[1]
            margin = float(
                second["distance"]
                - best["distance"]
            )
        else:
            second = None
            margin = float("inf")

        minimum_margin = float(
            self.params.get(
                "reference_color_minimum_margin",
                2.0
            )
        )

        within_range = (
            best["distance"]
            <= best["max_distance"]
        )

        clear_winner = (
            margin >= minimum_margin
        )

        accepted = (
            within_range
            and clear_winner
        )

        return {
            "accepted": bool(accepted),
            "group_id": (
                int(best["group_id"])
                if accepted
                else -1
            ),
            "best_distance": float(
                best["distance"]
            ),
            "second_distance": (
                float(second["distance"])
                if second is not None
                else None
            ),
            "margin": float(margin),
            "max_distance": float(
                best["max_distance"]
            ),
            "reference_rgb": np.asarray(
                best["center_rgb"],
                dtype=np.float32
            ),
            "comparisons": comparisons
        }
    

    def group_clusters_by_reference_color(self):
        """
        Classifies every cluster against configured reference colors.

        Final accepted results are copied into the existing:
            self.color_group_labels
            self.color_group_mean_rgb

        This preserves the current cluster JSON export format.
        """
        configured_references = self.params.get(
            "reference_color_groups",
            []
        )

        if not configured_references:
            raise RuntimeError(
                "Reference color grouping is enabled, "
                "but no reference_color_groups were provided."
            )
        
        if not self.cluster_originals:
            raise RuntimeError(
                "No clusters available for reference color grouping."
            )

        if self.stitched_rgb_bgr is None:
            raise RuntimeError(
                "RGB image is not loaded."
            )

        self.prepare_reference_color_palette()

        cluster_count = len(
            self.cluster_originals
        )

        labels = np.full(
            cluster_count,
            -1,
            dtype=np.int32
        )

        representative_rgbs = [
            None
            for _ in range(cluster_count)
        ]

        representative_labs = [
            None
            for _ in range(cluster_count)
        ]

        classifications = [
            None
            for _ in range(cluster_count)
        ]

        for cluster_index, cluster in enumerate(
            self.cluster_originals
        ):
            color_data = (
                self.compute_reference_cluster_color(
                    cluster
                )
            )

            if color_data is None:
                print(
                    f"[reference-color] cluster={cluster_index}: "
                    f"insufficient valid image pixels → unknown"
                )
                continue

            representative_rgb = color_data["rgb"]
            representative_lab = color_data["lab"]

            classification = (
                self.classify_cluster_against_reference_palette(
                    representative_lab
                )
            )

            group_id = int(
                classification["group_id"]
            )

            labels[cluster_index] = group_id
            representative_rgbs[
                cluster_index
            ] = representative_rgb

            representative_labs[
                cluster_index
            ] = representative_lab

            classifications[
                cluster_index
            ] = classification

            print(
                f"[reference-color] cluster={cluster_index}, "
                f"rgb={representative_rgb.tolist()}, "
                f"lab={representative_lab.tolist()}, "
                f"group={group_id}, "
                f"distance="
                f"{classification['best_distance']:.3f}, "
                f"limit="
                f"{classification['max_distance']:.3f}, "
                f"margin={classification['margin']:.3f}, "
                f"pixels="
                f"{color_data['filtered_pixel_count']}"
            )

        # Calculate the exported group_mean_rgb from the actual clusters
        # assigned to each reference group. This matches the meaning of your
        # existing export better than exporting the reference RGB directly.
        group_mean_rgb = {}

        for reference in self.reference_color_palette:
            group_id = int(
                reference["group_id"]
            )

            member_indices = np.where(
                labels == group_id
            )[0]

            member_colors = [
                representative_rgbs[index]
                for index in member_indices
                if representative_rgbs[index] is not None
            ]

            if member_colors:
                mean_rgb = np.mean(
                    np.stack(
                        member_colors,
                        axis=0
                    ),
                    axis=0
                ).astype(np.float32)
            else:
                # No cluster assigned to this reference.
                # Do not add an unused group to the export dictionary.
                continue

            group_mean_rgb[group_id] = mean_rgb

        # Repaint display clusters according to assigned group.
        for cluster_index, display_cluster in enumerate(
            self.clusters
        ):
            group_id = int(
                labels[cluster_index]
            )

            if group_id == -1:
                # Unknown clusters remain in their existing display color.
                continue

            color = group_mean_rgb.get(
                group_id
            )

            if color is not None:
                display_cluster.paint_uniform_color(
                    color.tolist()
                )

        # Store separate reference-grouping diagnostics in memory only.
        self.reference_color_assignments = labels.copy()
        self.reference_color_cluster_rgb = (
            representative_rgbs
        )
        self.reference_color_cluster_lab = (
            representative_labs
        )
        self.reference_color_distances = (
            classifications
        )

        # Copy only the export-compatible final values.
        self.color_group_labels = labels

        self.color_group_mean_rgb = {
            int(group_id): color.tolist()
            for group_id, color
            in group_mean_rgb.items()
        }

        known_count = int(
            np.count_nonzero(labels >= 0)
        )

        unknown_count = int(
            np.count_nonzero(labels == -1)
        )

        print(
            f"[step] Reference color grouping: "
            f"known={known_count}, "
            f"unknown={unknown_count}, "
            f"groups={sorted(group_mean_rgb.keys())}"
        )

        show_preview(
            self.clusters,
            "7. Reference-Based Color Groups"
        )
    
    #------------------For GMM+BIC color grouping ---------------------------------------

    # -------------------------------------------------------------------------
    # Color grouping option 3: GMM + BIC with optional component merging
    # -------------------------------------------------------------------------

    def _weighted_lab_distance(
            self,
            lab_a,
            lab_b,
            weights
        ):
        """
        Weighted Euclidean distance between two Lab colors.
        """
        lab_a = np.asarray(
            lab_a,
            dtype=np.float64
        ).reshape(3)

        lab_b = np.asarray(
            lab_b,
            dtype=np.float64
        ).reshape(3)

        weights = np.asarray(
            weights,
            dtype=np.float64
        ).reshape(3)

        weights = np.maximum(
            weights,
            0.0
        )

        difference = lab_a - lab_b

        return float(
            np.sqrt(
                np.sum(
                    weights
                    * difference
                    * difference
                )
            )
        )
    

    def build_gmm_color_features(self):
        """
        Builds one robust color feature per tile cluster.

        Returns
        -------
        valid_indices : np.ndarray
            Original cluster indices that produced valid image colors.

        features : np.ndarray
            Weighted Lab features used by GMM.

        representative_rgbs : list
            RGB representative for every original cluster.

        representative_labs : list
            Lab representative for every original cluster.
        """
        if not self.cluster_originals:
            raise RuntimeError(
                "No clusters available for GMM color grouping."
            )

        if self.stitched_rgb_bgr is None:
            raise RuntimeError(
                "GMM color grouping requires a loaded RGB image."
            )

        cluster_count = len(
            self.cluster_originals
        )

        representative_rgbs = [
            None for _ in range(cluster_count)
        ]

        representative_labs = [
            None for _ in range(cluster_count)
        ]

        valid_indices = []
        raw_features = []

        min_pixels = int(
            self.params.get(
                "gmm_color_min_pixels",
                self.params.get(
                    "reference_color_min_pixels",
                    20
                )
            )
        )

        for cluster_index, cluster in enumerate(
            self.cluster_originals
        ):
            color_data = (
                self.compute_reference_cluster_color(
                    cluster
                )
            )

            if color_data is None:
                print(
                    f"[gmm-color] cluster={cluster_index}: "
                    f"no valid color sample"
                )
                continue

            if (
                int(color_data["filtered_pixel_count"])
                < min_pixels
            ):
                print(
                    f"[gmm-color] cluster={cluster_index}: "
                    f"only "
                    f"{color_data['filtered_pixel_count']} "
                    f"pixels; minimum={min_pixels}"
                )
                continue

            representative_rgb = np.asarray(
                color_data["rgb"],
                dtype=np.float64
            )

            representative_lab = np.asarray(
                color_data["lab"],
                dtype=np.float64
            )

            representative_rgbs[
                cluster_index
            ] = representative_rgb

            representative_labs[
                cluster_index
            ] = representative_lab

            valid_indices.append(
                cluster_index
            )

            raw_features.append(
                representative_lab
            )

        if not raw_features:
            raise RuntimeError(
                "No valid cluster colors were available for GMM."
            )

        valid_indices = np.asarray(
            valid_indices,
            dtype=np.int32
        )

        raw_features = np.stack(
            raw_features,
            axis=0
        ).astype(np.float64)

        weights = np.asarray(
            self.params.get(
                "gmm_lab_weights",
                [0.2, 1.0, 1.0]
            ),
            dtype=np.float64
        ).reshape(3)

        weights = np.maximum(
            weights,
            0.0
        )

        # Multiplying by sqrt(weight) makes Euclidean/Gaussian distance
        # equivalent to weighted squared Lab distance.
        feature_scale = np.sqrt(
            weights
        )

        features = (
            raw_features * feature_scale
        )

        return (
            valid_indices,
            features,
            representative_rgbs,
            representative_labs
        )
    

    def fit_best_gmm_by_bic(
            self,
            features
        ):
        """
        Fits GMM models from gmm_min_groups up to gmm_max_groups.

        Returns the model with the lowest BIC. The tested component count
        never exceeds the user-provided maximum or the number of samples.
        """
        features = np.asarray(
            features,
            dtype=np.float64
        )

        sample_count = int(
            len(features)
        )

        if sample_count == 0:
            raise RuntimeError(
                "Cannot fit GMM to an empty feature array."
            )

        requested_min = int(
            self.params.get(
                "gmm_min_groups",
                1
            )
        )

        requested_max = int(
            self.params.get(
                "gmm_max_groups",
                5
            )
        )

        requested_min = max(
            1,
            requested_min
        )

        requested_max = max(
            requested_min,
            requested_max
        )

        maximum_components = min(
            requested_max,
            sample_count
        )

        minimum_components = min(
            requested_min,
            maximum_components
        )

        best_model = None
        best_bic = float("inf")
        bic_scores = {}

        for component_count in range(
            minimum_components,
            maximum_components + 1
        ):
            try:
                model = GaussianMixture(
                    n_components=int(component_count),
                    covariance_type=str(
                        self.params.get(
                            "gmm_covariance_type",
                            "full"
                        )
                    ),
                    n_init=int(
                        self.params.get(
                            "gmm_n_init",
                            5
                        )
                    ),
                    max_iter=int(
                        self.params.get(
                            "gmm_max_iter",
                            300
                        )
                    ),
                    random_state=int(
                        self.params.get(
                            "gmm_random_state",
                            0
                        )
                    ),
                    reg_covar=float(
                        self.params.get(
                            "gmm_reg_covar",
                            1e-5
                        )
                    )
                )

                model.fit(
                    features
                )

                bic_value = float(
                    model.bic(features)
                )

                bic_scores[
                    int(component_count)
                ] = bic_value

                print(
                    f"[gmm-bic] components="
                    f"{component_count}, "
                    f"BIC={bic_value:.3f}, "
                    f"converged={model.converged_}"
                )

                if bic_value < best_bic:
                    best_bic = bic_value
                    best_model = model

            except Exception as exc:
                print(
                    f"[gmm-bic][warn] components="
                    f"{component_count} failed: {exc}"
                )

        if best_model is None:
            raise RuntimeError(
                "All GMM+BIC model fits failed."
            )

        self.gmm_bic_scores = {
            int(key): float(value)
            for key, value in bic_scores.items()
        }

        self.gmm_selected_component_count = int(
            best_model.n_components
        )

        print(
            f"[gmm-bic] selected components="
            f"{self.gmm_selected_component_count}, "
            f"maximum allowed={requested_max}, "
            f"BIC={best_bic:.3f}"
        )

        return best_model
    

    def compute_gmm_component_statistics(
            self,
            valid_indices,
            component_labels,
            representative_rgbs,
            representative_labs
        ):
        """
        Calculates component colors from actual assigned tile colors.
        """
        statistics = {}

        for component_id in sorted(
            set(
                int(value)
                for value in component_labels
            )
        ):
            local_members = np.where(
                component_labels
                == component_id
            )[0]

            if len(local_members) == 0:
                continue

            original_indices = valid_indices[
                local_members
            ]

            labs = np.stack(
                [
                    representative_labs[
                        int(index)
                    ]
                    for index in original_indices
                ],
                axis=0
            ).astype(np.float64)

            rgbs = np.stack(
                [
                    representative_rgbs[
                        int(index)
                    ]
                    for index in original_indices
                ],
                axis=0
            ).astype(np.float64)

            statistics[int(component_id)] = {
                "component_ids": [
                    int(component_id)
                ],
                "original_indices": [
                    int(index)
                    for index in original_indices
                ],
                "count": int(
                    len(original_indices)
                ),
                "mean_lab": np.mean(
                    labs,
                    axis=0
                ),
                "mean_rgb": np.clip(
                    np.mean(
                        rgbs,
                        axis=0
                    ),
                    0.0,
                    1.0
                )
            }

        return statistics
    

    def merge_similar_gmm_components(
            self,
            component_statistics
        ):
        """
        Agglomeratively merges the closest GMM component groups while their
        weighted mean-Lab distance is within gmm_merge_distance.

        GMM has already respected gmm_max_groups, so this stage can only
        reduce the number of groups.
        """
        groups = {
            int(group_id): {
                "component_ids": list(
                    data["component_ids"]
                ),
                "original_indices": list(
                    data["original_indices"]
                ),
                "count": int(
                    data["count"]
                ),
                "mean_lab": np.asarray(
                    data["mean_lab"],
                    dtype=np.float64
                ),
                "mean_rgb": np.asarray(
                    data["mean_rgb"],
                    dtype=np.float64
                )
            }
            for group_id, data
            in component_statistics.items()
        }

        merge_distance = float(
            self.params.get(
                "gmm_merge_distance",
                6.0
            )
        )

        merge_weights = self.params.get(
            "gmm_merge_lab_weights",
            [0.1, 1.0, 1.0]
        )

        # A non-positive value disables merging.
        if merge_distance <= 0.0:
            return groups

        next_group_id = (
            max(groups.keys()) + 1
            if groups
            else 0
        )

        while len(groups) > 1:
            group_ids = sorted(
                groups.keys()
            )

            closest_pair = None
            closest_distance = float("inf")

            for first_position in range(
                len(group_ids)
            ):
                for second_position in range(
                    first_position + 1,
                    len(group_ids)
                ):
                    group_a_id = group_ids[
                        first_position
                    ]

                    group_b_id = group_ids[
                        second_position
                    ]

                    distance = (
                        self._weighted_lab_distance(
                            groups[group_a_id][
                                "mean_lab"
                            ],
                            groups[group_b_id][
                                "mean_lab"
                            ],
                            merge_weights
                        )
                    )

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_pair = (
                            group_a_id,
                            group_b_id
                        )

            if (
                closest_pair is None
                or closest_distance
                > merge_distance
            ):
                break

            group_a_id, group_b_id = (
                closest_pair
            )

            group_a = groups.pop(
                group_a_id
            )

            group_b = groups.pop(
                group_b_id
            )

            count_a = int(
                group_a["count"]
            )

            count_b = int(
                group_b["count"]
            )

            total_count = (
                count_a + count_b
            )

            merged_lab = (
                group_a["mean_lab"] * count_a
                + group_b["mean_lab"] * count_b
            ) / max(
                total_count,
                1
            )

            merged_rgb = (
                group_a["mean_rgb"] * count_a
                + group_b["mean_rgb"] * count_b
            ) / max(
                total_count,
                1
            )

            groups[next_group_id] = {
                "component_ids": (
                    group_a["component_ids"]
                    + group_b["component_ids"]
                ),
                "original_indices": (
                    group_a["original_indices"]
                    + group_b["original_indices"]
                ),
                "count": int(
                    total_count
                ),
                "mean_lab": merged_lab,
                "mean_rgb": np.clip(
                    merged_rgb,
                    0.0,
                    1.0
                )
            }

            print(
                f"[gmm-merge] merged components "
                f"{group_a['component_ids']} + "
                f"{group_b['component_ids']}, "
                f"distance={closest_distance:.3f}, "
                f"new_count={total_count}"
            )

            next_group_id += 1

        return groups
    

    def finalize_gmm_group_labels(
            self,
            merged_groups,
            cluster_count
        ):
        """
        Sorts final groups deterministically and creates labels 0, 1, 2...
        """
        ordered_groups = sorted(
            merged_groups.values(),
            key=lambda group: (
                float(group["mean_lab"][0]),
                float(group["mean_lab"][1]),
                float(group["mean_lab"][2])
            )
        )

        final_labels = np.full(
            int(cluster_count),
            -1,
            dtype=np.int32
        )

        group_mean_rgb = {}
        group_mean_lab = {}

        for final_group_id, group in enumerate(
            ordered_groups
        ):
            member_indices = [
                int(index)
                for index
                in group["original_indices"]
            ]

            final_labels[
                member_indices
            ] = int(
                final_group_id
            )

            group_mean_rgb[
                int(final_group_id)
            ] = np.clip(
                np.asarray(
                    group["mean_rgb"],
                    dtype=np.float64
                ),
                0.0,
                1.0
            )

            group_mean_lab[
                int(final_group_id)
            ] = np.asarray(
                group["mean_lab"],
                dtype=np.float64
            )

            print(
                f"[gmm-final] group="
                f"{final_group_id}, "
                f"members={len(member_indices)}, "
                f"mean_lab="
                f"{group_mean_lab[final_group_id].tolist()}, "
                f"mean_rgb="
                f"{group_mean_rgb[final_group_id].tolist()}"
            )

        return (
            final_labels,
            group_mean_rgb,
            group_mean_lab
        )
    

    def group_clusters_color_gmm_bic(self):
        """
        Automatic unsupervised color grouping:

            robust tile colors
            → GMM models up to user maximum
            → BIC model selection
            → optional perceptual merging
            → deterministic final labels

        Existing DBSCAN and reference methods remain untouched.
        """
        if not self.cluster_originals:
            raise RuntimeError(
                "No clusters available for GMM+BIC grouping."
            )

        (
            valid_indices,
            features,
            representative_rgbs,
            representative_labs
        ) = self.build_gmm_color_features()

        model = self.fit_best_gmm_by_bic(
            features
        )

        component_labels = model.predict(
            features
        ).astype(np.int32)

        probabilities = model.predict_proba(
            features
        )

        maximum_probabilities = np.max(
            probabilities,
            axis=1
        )

        minimum_probability = float(
            self.params.get(
                "gmm_min_probability",
                0.0
            )
        )

        # Keep raw prediction for diagnostics.
        self.gmm_initial_labels = np.full(
            len(self.cluster_originals),
            -1,
            dtype=np.int32
        )

        self.gmm_initial_labels[
            valid_indices
        ] = component_labels

        self.gmm_cluster_probabilities = np.zeros(
            len(self.cluster_originals),
            dtype=np.float64
        )

        self.gmm_cluster_probabilities[
            valid_indices
        ] = maximum_probabilities

        if minimum_probability > 0.0:
            accepted_local = (
                maximum_probabilities
                >= minimum_probability
            )
        else:
            accepted_local = np.ones(
                len(valid_indices),
                dtype=bool
            )

        accepted_valid_indices = (
            valid_indices[
                accepted_local
            ]
        )

        accepted_component_labels = (
            component_labels[
                accepted_local
            ]
        )

        if len(accepted_valid_indices) == 0:
            raise RuntimeError(
                "All GMM assignments were rejected by "
                "gmm_min_probability."
            )

        component_statistics = (
            self.compute_gmm_component_statistics(
                accepted_valid_indices,
                accepted_component_labels,
                representative_rgbs,
                representative_labs
            )
        )

        merged_groups = (
            self.merge_similar_gmm_components(
                component_statistics
            )
        )

        (
            final_labels,
            group_mean_rgb,
            group_mean_lab
        ) = self.finalize_gmm_group_labels(
            merged_groups,
            len(self.cluster_originals)
        )

        # Repaint display clusters.
        for cluster_index, display_cluster in enumerate(
            self.clusters
        ):
            group_id = int(
                final_labels[
                    cluster_index
                ]
            )

            if group_id < 0:
                continue

            color = group_mean_rgb.get(
                group_id
            )

            if color is not None:
                display_cluster.paint_uniform_color(
                    color.tolist()
                )

        # Export-compatible state.
        self.color_group_labels = (
            final_labels
        )

        self.color_group_mean_rgb = {
            int(group_id): color.tolist()
            for group_id, color
            in group_mean_rgb.items()
        }

        # GMM-specific in-memory state.
        self.gmm_final_labels = (
            final_labels.copy()
        )

        self.gmm_component_mean_lab = {
            int(group_id): color.tolist()
            for group_id, color
            in group_mean_lab.items()
        }

        self.gmm_component_mean_rgb = {
            int(group_id): color.tolist()
            for group_id, color
            in group_mean_rgb.items()
        }

        final_group_ids = sorted(
            int(value)
            for value
            in np.unique(final_labels)
            if int(value) >= 0
        )

        unknown_indices = np.where(
            final_labels < 0
        )[0].tolist()

        groups_summary = []

        for group_id in final_group_ids:
            members = np.where(
                final_labels == group_id
            )[0].tolist()

            groups_summary.append({
                "group_id": int(group_id),
                "mean_rgb": (
                    self.color_group_mean_rgb[
                        group_id
                    ]
                ),
                "members": [
                    int(value)
                    for value in members
                ]
            })

        # Report only; the cluster JSON format remains unchanged.
        report_lines = [
            "GMM + BIC Color Grouping Report",
            (
                f"Maximum allowed groups: "
                f"{int(self.params.get('gmm_max_groups', 5))}"
            ),
            (
                f"BIC-selected components: "
                f"{self.gmm_selected_component_count}"
            ),
            (
                f"Final groups after merging: "
                f"{len(final_group_ids)}"
            ),
            (
                f"Merge distance: "
                f"{float(self.params.get('gmm_merge_distance', 6.0)):.3f}"
            ),
            (
                f"Minimum probability: "
                f"{minimum_probability:.3f}"
            ),
            f"BIC scores: {self.gmm_bic_scores}"
        ]

        for group in groups_summary:
            report_lines.append(
                f"- Group {group['group_id']}: "
                f"count={len(group['members'])}, "
                f"meanRGB="
                f"{group['mean_rgb']}, "
                f"members={group['members']}"
            )

        if unknown_indices:
            report_lines.append(
                f"- Unknown (-1): "
                f"count={len(unknown_indices)}, "
                f"members={unknown_indices}"
            )

        self.last_report_lines = (
            report_lines
        )

        self.last_grouping_meta = {
            "mode": "gmm_bic",
            "maximum_groups": int(
                self.params.get(
                    "gmm_max_groups",
                    5
                )
            ),
            "bic_selected_components": int(
                self.gmm_selected_component_count
            ),
            "final_group_count": int(
                len(final_group_ids)
            ),
            "labels": final_labels.tolist(),
            "groups": groups_summary
        }

        print(
            f"[step] GMM+BIC color grouping: "
            f"BIC components="
            f"{self.gmm_selected_component_count}, "
            f"final groups="
            f"{len(final_group_ids)}, "
            f"unknown={len(unknown_indices)}"
        )

        display = (
            list(self.clusters)
            + list(self.hull_lines_2d)
        )

        show_preview(
            display,
            "7. GMM + BIC Color Groups"
        )

    # -------------------------------------------------------------------------
    # Export and reporting
    # -------------------------------------------------------------------------

    def save_report_to_txt(self):
        lines = self.last_report_lines if self.last_report_lines else ["(empty report)"]
        path = os.path.join(
            self.export_dir,
            PROCESSING_REPORT_TEMPLATE.format(idx=self.output_index)
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[export] Saved report: {path}")
        return path

    def save_params_to_json(self):
        path = os.path.join(
            self.export_dir,
            PROCESSING_PARAMS_TEMPLATE.format(idx=self.output_index)
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, ensure_ascii=False, indent=2)
        print(f"[export] Saved parameters: {path}")
        return path

    def export_all_clusters_json(self):
        """
        Exports cluster geometry using the original legacy JSON structure.

        Processing parameters are intentionally excluded because they are
        saved separately by save_params_to_json().
        """
        if not self.clusters:
            raise RuntimeError(
                "No clusters to export. Run clustering first."
            )

        have_groups = (
            isinstance(
                self.color_group_labels,
                np.ndarray
            )
            and len(self.color_group_labels)
            == len(self.clusters)
        )

        # Preserve the original RANSAC plane index in the exported data.
        #
        # After tile-top extraction, self.selected_plane_index becomes 0
        # because planes_cluster is replaced with [self.tile_top_pcd].
        # Therefore, reference_plane_index is preferred so the exported
        # plane index remains equivalent to the old workflow.
        if self.reference_plane_index is not None:
            export_plane_index = int(
                self.reference_plane_index
            )

        elif self.selected_plane_index is not None:
            export_plane_index = int(
                self.selected_plane_index
            )

        else:
            export_plane_index = 0

        entries = []

        for cidx, disp_cluster in enumerate(
            self.clusters
        ):
            if cidx < len(self.cluster_originals):
                orig_cluster = (
                    self.cluster_originals[cidx]
                )
            else:
                orig_cluster = disp_cluster

            (
                poly2,
                poly3,
                origin,
                u,
                v,
                n,
                centroid3d
            ) = compute_hull_poly2_poly3(
                orig_cluster
            )

            entry = {
                "plane_index": export_plane_index,
                "cluster_index": int(cidx),
                "plane_frame": {
                    "origin": origin.tolist(),
                    "u": u.tolist(),
                    "v": v.tolist(),
                    "n": n.tolist()
                },
                "hull_3d": (
                    poly3.tolist()
                    if poly3 is not None
                    else []
                ),
                "centroid_3d": centroid3d.tolist()
            }

            if have_groups:
                group_id = int(
                    self.color_group_labels[cidx]
                )

                if group_id != -1:
                    mean_rgb = (
                        self.color_group_mean_rgb.get(
                            group_id,
                            None
                        )
                    )

                    # Convert NumPy arrays to ordinary lists
                    # so json.dump can serialize them.
                    if isinstance(mean_rgb, np.ndarray):
                        mean_rgb = mean_rgb.tolist()

                    entry["color_group"] = {
                        "group_id": group_id,
                        "group_mean_rgb": mean_rgb
                    }

                else:
                    entry["color_group"] = {
                        "group_id": -1
                    }

            entries.append(entry)

        # Keep exactly the original top-level structure.
        payload = {
            "units": "meters",
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(entries),
            "clusters": entries
        }

        path = os.path.join(
            self.export_dir,
            PROCESSING_OUTPUT_TEMPLATE.format(
                idx=self.output_index
            )
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"[export] Exported ALL clusters JSON: "
            f"{path}"
        )

        return path

    def run(self):
        print("\n[processing] Auto parameters:")
        print(json.dumps(self.params, indent=2))

        show_preview(
            self.original_pcd,
            "0. Raw Point Cloud"
        )

        self.load_stitched_maps()

        # 1. Preprocess the original cloud first. The old percentile/global-Z
        # filtering method remains in the script but is intentionally bypassed.
        self.filtered_pcd = self.original_pcd
        self.preprocess()

        # 2. Adaptive initial multi-plane RANSAC with multi-scale validation.
        # This sets:
        # - self.planes_cluster
        # - self.planes_display
        # - self.remaining_cloud
        # - self.plane_models
        # - self.selected_plane_index
        self.detect_initial_planes_adaptive()

        # Store/orient the selected initial top-reference model and normal
        # for the directional cropping step.
        self.select_initial_top_reference()

        # 3. Crop 2 mm above and 15 mm below along the reference normal.
        self.crop_along_reference_normal()

        # 4. Second RANSAC on the cropped cloud and select the parallel base.
        # This method should also use its own adaptive threshold and
        # multi-scale validation internally.
        self.detect_base_plane_from_cropped()

        # 5. Signed-distance height analysis and tile-top band extraction.
        self.extract_tile_top_band_from_base()

        print(
            f"[step] Selected tile-top band for watershed: "
            f"{self.selected_plane_index}"
        )

        # 6. Existing mask, watershed, hull, color grouping, and export flow.
        self.cluster_selected_plane_by_eroded_marker_watershed()
        # self.watershed_clusters_from_selected_plane()
        # self.split_dbscan_clusters_by_2d()

        self.compute_convex_hulls_all_2d()

        grouping_method = str(
            self.params.get(
                "color_grouping_method",
                "dbscan"
            )
        ).strip().lower()

        if grouping_method == "gmm_bic":
            self.group_clusters_color_gmm_bic()

        elif grouping_method == "reference":
            self.group_clusters_by_reference_color()

        elif grouping_method == "dbscan":
            self.group_clusters_color()

        else:
            raise ValueError(
                f"Unknown color_grouping_method: "
                f"{grouping_method}. "
                f"Expected 'dbscan', 'reference', "
                f"or 'gmm_bic'."
            )

        json_path = self.export_all_clusters_json()
        report_path = self.save_report_to_txt()
        params_path = self.save_params_to_json()

        return {
            "status": "ok",
            "output_index": int(self.output_index),
            "cluster_count": int(len(self.clusters)),
            "json_path": json_path,
            "report_path": report_path,
            "params_path": params_path,
        }


# =============================================================================
# Path resolution and runtime entry points
# =============================================================================

def resolve_processing_paths(paths, output_index=DEFAULT_OUTPUT_INDEX, input_kind=DEFAULT_INPUT_KIND):
    if input_kind == "eye_to_base":
        pcd_path = os.path.join(
            str(paths.merged_point_clouds),
            f"eye_to_base_point_cloud_{output_index:02d}.ply"
        )
    elif input_kind == "merged":
        pcd_path = os.path.join(
            str(paths.merged_point_clouds),
            f"merged{output_index:02d}.ply"
        )
    elif input_kind == "initial":
        pcd_path = os.path.join(
            str(paths.initial_point_clouds),
            f"point_cloud_{output_index:02d}.ply"
        )
    else:
        raise ValueError(f"Unknown input_kind: {input_kind}")

    stitched_rgb_path = os.path.join(
        str(paths.merged_images),
        f"stitched_rgb_{output_index:02d}.png"
    )

    stitched_height_path = os.path.join(
        str(paths.merged_depth_images),
        f"stitched_height_{output_index:02d}.png"
    )

    eye_to_base_rgb_path = os.path.join(
        str(paths.merged_images),
        f"eye_to_base_rgb_{output_index:02d}.png"
    )

    eye_to_base_height_path = os.path.join(
        str(paths.merged_depth_images),
        f"eye_to_base_height_{output_index:02d}.png"
    )

    return pcd_path, stitched_rgb_path, stitched_height_path, eye_to_base_rgb_path, eye_to_base_height_path, str(paths.exported_data)


def run_processing_from_config(config):
    global SHOW_PREVIEW, PREVIEW_TIME_SEC

    session_name = config.get("session") or None
    output_index = int(config.get("output_index", DEFAULT_OUTPUT_INDEX))
    input_kind = config.get("input_kind", DEFAULT_INPUT_KIND)
    show_preview_value = bool(config.get("show_preview", SHOW_PREVIEW))
    preview_time = config.get("preview_time_sec", PREVIEW_TIME_SEC)

    SHOW_PREVIEW = show_preview_value
    PREVIEW_TIME_SEC = preview_time

    paths = load_session(".", session_name)
    pcd_path, stitched_rgb_path, stitched_height_path, eye_to_base_rgb_path, eye_to_base_height_path, export_dir = resolve_processing_paths(
        paths,
        output_index=output_index,
        input_kind=input_kind
    )

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Input point cloud not found: {pcd_path}")

    pcd = o3d.io.read_point_cloud(pcd_path)
    if not _has_points(pcd):
        raise RuntimeError(f"Loaded point cloud is empty: {pcd_path}")
    
    # Unit conversion only: millimetres -> metres.
    # No centering, normalization, rotation, or coordinate-frame change occurs.
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / 1000.0)

    auto_params = estimate_processing_params(
        pcd
    )

    config_path = config.get(
        "processing_config_path"
    )

    if not config_path:
        config_path = os.path.join(
            export_dir,
            PROCESSING_CONFIG_TEMPLATE.format(
                idx=output_index
            )
        )

    override_params = load_config_override(
        config_path
    )

    # 1. Automatic defaults
    # 2. File-based overrides
    params = merge_params(
        auto_params,
        override_params
    )

    # 3. Generic runtime parameter dictionary
    inline_params = config.get(
        "params",
        {}
    )

    params = merge_params(
        params,
        inline_params
    )

    # 4. Explicit GH color inputs have highest priority
    runtime_color_keys = [
        # Grouping selector
        "color_grouping_method",

        # Existing reference grouping
        "use_reference_color_grouping",
        "reference_color_hull_erosion_px",
        "reference_color_low_percentile",
        "reference_color_high_percentile",
        "reference_color_min_pixels",
        "reference_color_lab_weights",
        "reference_color_minimum_margin",

        # GMM + BIC
        "gmm_max_groups",
        "gmm_min_groups",
        "gmm_covariance_type",
        "gmm_n_init",
        "gmm_max_iter",
        "gmm_random_state",
        "gmm_reg_covar",
        "gmm_lab_weights",
        "gmm_merge_distance",
        "gmm_merge_lab_weights",
        "gmm_min_probability",
        "gmm_color_min_pixels",
    ]

    for key in runtime_color_keys:
        if key in config and config[key] is not None:
            params[key] = config[key]

    reference_color_groups = config.get(
        "reference_color_groups",
        None
    )

    if reference_color_groups is not None:
        params["reference_color_groups"] = (
            validate_reference_color_groups(
                reference_color_groups
            )
        )

    processor = AutomatedSegmentationProcessor(
        pcd=pcd,
        export_directory=export_dir,
        stitched_rgb_file=stitched_rgb_path,
        stitched_height_file=stitched_height_path,
        eye_to_base_rgb_file=eye_to_base_rgb_path,
        eye_to_base_height_file=eye_to_base_height_path,
        params=params,
        output_index=output_index,
        input_kind=input_kind
    )

    result = processor.run()
    result.update({
        "session": paths.session_name,
        "input_kind": input_kind,
        "input_pcd": pcd_path,
        "stitched_rgb": stitched_rgb_path if os.path.exists(stitched_rgb_path) else None,
        "stitched_height": stitched_height_path if os.path.exists(stitched_height_path) else None,
        "eye_to_base_rgb": eye_to_base_rgb_path if os.path.exists(eye_to_base_rgb_path) else None,
        "eye_to_base_height": eye_to_base_height_path if os.path.exists(eye_to_base_height_path) else None,
    })
    return result


def main():
    session_name = input("Session name (blank = reuse last): ").strip() or None
    output_index_text = input(f"Output index (blank = {DEFAULT_OUTPUT_INDEX}): ").strip()
    output_index = int(output_index_text) if output_index_text else DEFAULT_OUTPUT_INDEX

    input_kind = input(f"Input kind [eye_to_base / merged / initial] (blank = {DEFAULT_INPUT_KIND}): ").strip() or DEFAULT_INPUT_KIND

    preview_text = input("Preview time seconds (blank = close manually, 0 = no preview): ").strip()
    if preview_text == "0":
        show_preview_value = False
        preview_time = None
    elif preview_text:
        show_preview_value = True
        preview_time = float(preview_text)
    else:
        show_preview_value = True
        preview_time = None

    grouping_method = input(
        "Color grouping method "
        "[gmm_bic / dbscan / reference] "
        "(blank = gmm_bic): "
    ).strip().lower() or "gmm_bic"

    reference_color_groups = []

    if grouping_method == "gmm_bic":
        max_groups_text = input(
            "Maximum color groups "
            "(blank = 5): "
        ).strip()

        gmm_max_groups = (
            int(max_groups_text)
            if max_groups_text
            else 5
        )

        if gmm_max_groups < 1:
            raise ValueError(
                "Maximum color groups must be at least 1."
            )

        merge_distance_text = input(
            "GMM merge distance "
            "(blank = 6.0, 0 = no merging): "
        ).strip()

        gmm_merge_distance = (
            float(merge_distance_text)
            if merge_distance_text
            else 6.0
        )

    elif grouping_method == "reference":
        reference_colors_text = input(
            "Reference color groups JSON: "
        ).strip()

        if not reference_colors_text:
            raise ValueError(
                "Reference grouping requires reference JSON."
            )

        try:
            reference_color_groups = (
                validate_reference_color_groups(
                    json.loads(
                        reference_colors_text
                    )
                )
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid reference color JSON: {exc}"
            )

        gmm_max_groups = 5
        gmm_merge_distance = 6.0

    elif grouping_method == "dbscan":
        gmm_max_groups = 5
        gmm_merge_distance = 6.0

    else:
        raise ValueError(
            "Color grouping method must be "
            "'gmm_bic', 'dbscan', or 'reference'."
        )

    result = run_processing_from_config({
        "session": session_name,
        "output_index": output_index,
        "input_kind": input_kind,
        "show_preview": show_preview_value,
        "preview_time_sec": preview_time,
        "color_grouping_method": (
            grouping_method
        ),

        "gmm_max_groups": (
            gmm_max_groups
        ),

        "gmm_merge_distance": (
            gmm_merge_distance
        ),

        "reference_color_groups": (
            reference_color_groups
        )
    })

    print("\n[processing] Finished:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()