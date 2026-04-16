"""
Correctness and benchmark tests for vectorized implementations of:
  - render_depth_map
  - depth_map_to_3d_points
  - extract_ground_plane
  - compute_occupancy_grid_2d

Each reference function is a verbatim copy of the original for-loop implementation.
"""

import json
import time
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Reference (original loop-based) implementations
# ---------------------------------------------------------------------------

def _reference_render_depth_map(positions, colors, resolution=(1280, 720), focal_length=1000, max_depth=100):
    import cv2
    width, height = resolution
    depth_map = np.full((height, width), max_depth, dtype=np.float32)
    color_map = np.zeros((height, width, 3), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)

    for pos, col in zip(positions, colors):
        x, y, z = pos
        if z <= 0 or z > max_depth:
            continue
        u = int(focal_length * x / z + width / 2)
        v = int(height / 2 - focal_length * y / z)
        if 0 <= u < width and 0 <= v < height:
            if z < depth_map[v, u]:
                depth_map[v, u] = z
            color_map[v, u] += col
            count_map[v, u] += 1

    mask = count_map > 0
    color_map[mask] = color_map[mask] / count_map[mask, np.newaxis]

    depth_normalized = np.clip(depth_map / max_depth, 0, 1)
    depth_colored = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    invalid_mask = depth_map >= max_depth
    depth_colored[invalid_mask] = [0, 0, 0]

    return depth_map, depth_colored


def _reference_depth_map_to_3d_points(depth_map, depth_colored, focal_length, width, height):
    step = max(1, min(width, height) // 200)
    depth_points = []
    depth_colors = []
    depth_max = depth_map.max()

    for v in range(0, height, step):
        for u in range(0, width, step):
            z = depth_map[v, u]
            if z >= depth_max * 0.99 or z <= 0:
                continue
            x = (u - width / 2) * z / focal_length
            y = (height / 2 - v) * z / focal_length
            depth_points.append([x, y, z])
            color = depth_colored[v, u] / 255.0
            depth_colors.append(color)

    if len(depth_points) == 0:
        return np.array([]), np.array([])
    return np.array(depth_points), np.array(depth_colors)


def _reference_extract_ground_plane(positions, height_threshold=0.3, grid_size=5.0):
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_z, max_z = positions[:, 2].min(), positions[:, 2].max()
    nx = int((max_x - min_x) / grid_size) + 1
    nz = int((max_z - min_z) / grid_size) + 1
    height_map = np.full((nx, nz), np.inf)

    for pos in positions:
        i = int((pos[0] - min_x) / grid_size)
        j = int((pos[2] - min_z) / grid_size)
        if 0 <= i < nx and 0 <= j < nz:
            height_map[i, j] = min(height_map[i, j], pos[1])

    ground_points = []
    for i in range(nx):
        for j in range(nz):
            if height_map[i, j] != np.inf:
                x = min_x + i * grid_size
                z = min_z + j * grid_size
                y = height_map[i, j]
                ground_points.append([x, y, z])

    ground_points = np.array(ground_points) if ground_points else np.empty((0, 3))
    median_y = np.median(positions[:, 1]) if len(positions) > 0 else 0.0
    is_ground = np.abs(positions[:, 1] - median_y) < height_threshold

    return ground_points, is_ground, height_map, (min_x, max_x, min_z, max_z, grid_size)


def _reference_compute_occupancy_grid_2d(positions, ground_info, resolution=1.0, obstacle_height=0.5):
    min_x, max_x, min_z, max_z, _ = ground_info
    nx = int((max_x - min_x) / resolution) + 1
    nz = int((max_z - min_z) / resolution) + 1
    occupancy_grid = np.zeros((nx, nz))
    ground_level = positions[:, 1].min()

    for pos in positions:
        if pos[1] < (ground_level + obstacle_height):
            continue
        i = int((pos[0] - min_x) / resolution)
        j = int((pos[2] - min_z) / resolution)
        if 0 <= i < nx and 0 <= j < nz:
            occupancy_grid[i, j] = 1

    return occupancy_grid, (min_x, max_x, min_z, max_z, resolution)


# ---------------------------------------------------------------------------
# Import the vectorized implementations (bypass utils/__init__.py which
# pulls in heavy deps like PIL/torch that may not be installed in the
# test environment).
# ---------------------------------------------------------------------------

import importlib.util

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_depth_mod = _import_module_from_file(
    "depth_rendering",
    os.path.join(_project_root, "utils", "depth_rendering.py"))
_nav_mod = _import_module_from_file(
    "navigation",
    os.path.join(_project_root, "utils", "navigation.py"))

render_depth_map = _depth_mod.render_depth_map
depth_map_to_3d_points = _depth_mod.depth_map_to_3d_points
extract_ground_plane = _nav_mod.extract_ground_plane
compute_occupancy_grid_2d = _nav_mod.compute_occupancy_grid_2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positions(rng, n):
    """Generate random 3D positions with z mostly positive (camera-facing)."""
    pos = rng.standard_normal((n, 3)).astype(np.float64)
    # Shift z to be mostly positive so points are in front of camera
    pos[:, 2] = np.abs(pos[:, 2]) * 20 + 0.1  # z in [0.1, ~60]
    # Spread x, y a bit
    pos[:, 0] *= 5
    pos[:, 1] *= 5
    return pos


def _make_colors(rng, n):
    """Generate random RGB colors in [0,1]."""
    return rng.uniform(0, 1, (n, 3)).astype(np.float64)


def _time_fn(fn, *args, repeats=3, **kwargs):
    """Return median wall-clock time in ms over `repeats` runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.median(times), result


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_render_depth_map_correctness():
    rng = np.random.default_rng(0)
    for n in [0, 100, 10_000]:
        if n == 0:
            pos = np.empty((0, 3))
            col = np.empty((0, 3))
        else:
            pos = _make_positions(rng, n)
            col = _make_colors(rng, n)

        ref_dm, ref_dc = _reference_render_depth_map(pos, col)
        new_dm, new_dc = render_depth_map(pos, col)

        np.testing.assert_allclose(new_dm, ref_dm, atol=1e-5,
                                   err_msg=f"depth_map mismatch at N={n}")
        np.testing.assert_array_equal(new_dc, ref_dc,
                                      err_msg=f"depth_colored mismatch at N={n}")
    print("  [PASS] render_depth_map correctness")


def test_depth_map_to_3d_points_correctness():
    import cv2
    rng = np.random.default_rng(42)
    width, height = 640, 480
    focal_length = 500

    # Create a synthetic depth map with some valid values
    depth_map = np.full((height, width), 100.0, dtype=np.float32)
    # Sprinkle some valid depth values
    n_valid = 5000
    vs = rng.integers(0, height, n_valid)
    us = rng.integers(0, width, n_valid)
    depth_map[vs, us] = rng.uniform(0.5, 50.0, n_valid).astype(np.float32)

    # Create a colored version
    depth_normalized = np.clip(depth_map / 100.0, 0, 1)
    depth_colored = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    ref_pts, ref_cols = _reference_depth_map_to_3d_points(
        depth_map, depth_colored, focal_length, width, height)
    new_pts, new_cols = depth_map_to_3d_points(
        depth_map, depth_colored, focal_length, width, height)

    assert ref_pts.shape == new_pts.shape, (
        f"Shape mismatch: ref={ref_pts.shape} new={new_pts.shape}")
    if ref_pts.size > 0:
        np.testing.assert_allclose(new_pts, ref_pts, atol=1e-5,
                                   err_msg="depth_map_to_3d_points points mismatch")
        np.testing.assert_allclose(new_cols, ref_cols, atol=1e-5,
                                   err_msg="depth_map_to_3d_points colors mismatch")

    # Test empty depth map
    empty_dm = np.full((height, width), 100.0, dtype=np.float32)
    empty_dc = np.zeros((height, width, 3), dtype=np.uint8)
    ref_e_pts, ref_e_cols = _reference_depth_map_to_3d_points(
        empty_dm, empty_dc, focal_length, width, height)
    new_e_pts, new_e_cols = depth_map_to_3d_points(
        empty_dm, empty_dc, focal_length, width, height)
    assert ref_e_pts.shape == new_e_pts.shape
    assert ref_e_cols.shape == new_e_cols.shape

    print("  [PASS] depth_map_to_3d_points correctness")


def test_extract_ground_plane_correctness():
    rng = np.random.default_rng(0)
    for n in [100, 10_000]:
        pos = _make_positions(rng, n)

        ref_gp, ref_ig, ref_hm, ref_gi = _reference_extract_ground_plane(pos)
        new_gp, new_ig, new_hm, new_gi = extract_ground_plane(pos)

        # height_map must match exactly (both use min)
        np.testing.assert_array_equal(new_hm, ref_hm,
                                      err_msg=f"height_map mismatch at N={n}")
        # is_ground must match exactly
        np.testing.assert_array_equal(new_ig, ref_ig,
                                      err_msg=f"is_ground mismatch at N={n}")
        # ground_info must match
        assert ref_gi == new_gi, f"ground_info mismatch at N={n}"
        # ground_points: same set of points, but order may differ
        # Sort both by (x, z, y) for comparison
        if ref_gp.size > 0:
            ref_sorted = ref_gp[np.lexsort((ref_gp[:, 1], ref_gp[:, 2], ref_gp[:, 0]))]
            new_sorted = new_gp[np.lexsort((new_gp[:, 1], new_gp[:, 2], new_gp[:, 0]))]
            np.testing.assert_allclose(new_sorted, ref_sorted, atol=1e-5,
                                       err_msg=f"ground_points mismatch at N={n}")

    print("  [PASS] extract_ground_plane correctness")


def test_compute_occupancy_grid_2d_correctness():
    rng = np.random.default_rng(0)
    for n in [100, 10_000]:
        pos = _make_positions(rng, n)
        ground_info = (pos[:, 0].min(), pos[:, 0].max(),
                       pos[:, 2].min(), pos[:, 2].max(), 5.0)

        ref_og, ref_info = _reference_compute_occupancy_grid_2d(pos, ground_info)
        new_og, new_info = compute_occupancy_grid_2d(pos, ground_info)

        np.testing.assert_array_equal(new_og, ref_og,
                                      err_msg=f"occupancy_grid mismatch at N={n}")
        assert ref_info == new_info, f"grid_info mismatch at N={n}"

    print("  [PASS] compute_occupancy_grid_2d correctness")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmarks():
    """Run benchmarks for all 4 functions at N=10k, 100k, 1M and print table."""
    import cv2

    sizes = [10_000, 100_000, 1_000_000]
    results = {}

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS (median of 3 runs, ms)")
    print("=" * 80)

    # --- render_depth_map ---
    print("\n--- render_depth_map ---")
    print(f"{'N':>12s}  {'ref (ms)':>12s}  {'vec (ms)':>12s}  {'speedup':>10s}")
    results["render_depth_map"] = {}
    for n in sizes:
        rng = np.random.default_rng(0)
        pos = _make_positions(rng, n)
        col = _make_colors(rng, n)

        # Only run reference for <= 100k (too slow at 1M)
        if n <= 100_000:
            ref_ms, _ = _time_fn(_reference_render_depth_map, pos, col, repeats=3)
        else:
            # Estimate from 100k
            ref_ms = results["render_depth_map"]["100000"]["ref_ms"] * (n / 100_000)

        new_ms, _ = _time_fn(render_depth_map, pos, col, repeats=3)
        speedup = ref_ms / new_ms if new_ms > 0 else float("inf")
        est = " (est)" if n > 100_000 else ""
        print(f"{n:>12,d}  {ref_ms:>12.1f}{est:5s}  {new_ms:>12.1f}       {speedup:>7.1f}x")
        results["render_depth_map"][str(n)] = {
            "ref_ms": round(ref_ms, 2),
            "vec_ms": round(new_ms, 2),
            "speedup": round(speedup, 2),
            "ref_estimated": n > 100_000,
        }

    # --- depth_map_to_3d_points ---
    print("\n--- depth_map_to_3d_points ---")
    print(f"{'pixels':>12s}  {'ref (ms)':>12s}  {'vec (ms)':>12s}  {'speedup':>10s}")
    results["depth_map_to_3d_points"] = {}
    # Use various image sizes to vary point count
    image_sizes = [(640, 480), (1280, 720), (2560, 1440)]
    for width, height in image_sizes:
        rng = np.random.default_rng(42)
        depth_map = np.full((height, width), 100.0, dtype=np.float32)
        n_valid = min(width * height // 4, 200_000)
        vs = rng.integers(0, height, n_valid)
        us = rng.integers(0, width, n_valid)
        depth_map[vs, us] = rng.uniform(0.5, 50.0, n_valid).astype(np.float32)
        depth_normalized = np.clip(depth_map / 100.0, 0, 1)
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        ref_ms, _ = _time_fn(_reference_depth_map_to_3d_points,
                             depth_map, depth_colored, 500, width, height, repeats=3)
        new_ms, _ = _time_fn(depth_map_to_3d_points,
                             depth_map, depth_colored, 500, width, height, repeats=3)
        speedup = ref_ms / new_ms if new_ms > 0 else float("inf")
        label = f"{width}x{height}"
        print(f"{label:>12s}  {ref_ms:>12.1f}       {new_ms:>12.1f}       {speedup:>7.1f}x")
        results["depth_map_to_3d_points"][label] = {
            "ref_ms": round(ref_ms, 2),
            "vec_ms": round(new_ms, 2),
            "speedup": round(speedup, 2),
        }

    # --- extract_ground_plane ---
    print("\n--- extract_ground_plane ---")
    print(f"{'N':>12s}  {'ref (ms)':>12s}  {'vec (ms)':>12s}  {'speedup':>10s}")
    results["extract_ground_plane"] = {}
    for n in sizes:
        rng = np.random.default_rng(0)
        pos = _make_positions(rng, n)

        if n <= 100_000:
            ref_ms, _ = _time_fn(_reference_extract_ground_plane, pos, repeats=3)
        else:
            ref_ms = results["extract_ground_plane"]["100000"]["ref_ms"] * (n / 100_000)

        new_ms, _ = _time_fn(extract_ground_plane, pos, repeats=3)
        speedup = ref_ms / new_ms if new_ms > 0 else float("inf")
        est = " (est)" if n > 100_000 else ""
        print(f"{n:>12,d}  {ref_ms:>12.1f}{est:5s}  {new_ms:>12.1f}       {speedup:>7.1f}x")
        results["extract_ground_plane"][str(n)] = {
            "ref_ms": round(ref_ms, 2),
            "vec_ms": round(new_ms, 2),
            "speedup": round(speedup, 2),
            "ref_estimated": n > 100_000,
        }

    # --- compute_occupancy_grid_2d ---
    print("\n--- compute_occupancy_grid_2d ---")
    print(f"{'N':>12s}  {'ref (ms)':>12s}  {'vec (ms)':>12s}  {'speedup':>10s}")
    results["compute_occupancy_grid_2d"] = {}
    for n in sizes:
        rng = np.random.default_rng(0)
        pos = _make_positions(rng, n)
        ground_info = (pos[:, 0].min(), pos[:, 0].max(),
                       pos[:, 2].min(), pos[:, 2].max(), 5.0)

        if n <= 100_000:
            ref_ms, _ = _time_fn(_reference_compute_occupancy_grid_2d,
                                 pos, ground_info, repeats=3)
        else:
            ref_ms = results["compute_occupancy_grid_2d"]["100000"]["ref_ms"] * (n / 100_000)

        new_ms, _ = _time_fn(compute_occupancy_grid_2d, pos, ground_info, repeats=3)
        speedup = ref_ms / new_ms if new_ms > 0 else float("inf")
        est = " (est)" if n > 100_000 else ""
        print(f"{n:>12,d}  {ref_ms:>12.1f}{est:5s}  {new_ms:>12.1f}       {speedup:>7.1f}x")
        results["compute_occupancy_grid_2d"][str(n)] = {
            "ref_ms": round(ref_ms, 2),
            "vec_ms": round(new_ms, 2),
            "speedup": round(speedup, 2),
            "ref_estimated": n > 100_000,
        }

    # Write JSON
    bench_path = os.path.join(os.path.dirname(__file__), "vectorization_bench.json")
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results written to {bench_path}")

    return results


# ---------------------------------------------------------------------------
# Main / pytest entry
# ---------------------------------------------------------------------------

def test_all_correctness():
    """Single pytest entry that runs all correctness checks."""
    test_render_depth_map_correctness()
    test_depth_map_to_3d_points_correctness()
    test_extract_ground_plane_correctness()
    test_compute_occupancy_grid_2d_correctness()


if __name__ == "__main__":
    print("Running correctness tests...")
    test_all_correctness()
    print("\nAll correctness tests passed!\n")
    run_benchmarks()
