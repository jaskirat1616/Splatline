#!/usr/bin/env python3
"""
Splatline pipeline benchmark harness.

Times each pipeline stage (depth rendering, navigation, pathfinding) across
synthetic or real Gaussian data and reports median/p95 per-frame latencies.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --n-frames 10 --point-counts 100000,500000
    python -m scripts.benchmark --real-ply /path/to/plys
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

BENCH_DIR = Path("/tmp/splatline_bench")


def _make_synthetic_frame(idx: int, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Generate a synthetic gradient-noise frame (H, W, 3) uint8."""
    rng = np.random.RandomState(idx)
    # Horizontal gradient + per-pixel noise
    gradient = np.linspace(0, 255, width, dtype=np.float32)
    img = np.tile(gradient, (height, 1))
    img = np.stack([img, img * 0.7, img * 0.3], axis=-1)
    noise = rng.randint(0, 30, size=(height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_synthetic_gaussians(n_points: int, seed: int = 0) -> dict:
    """
    Build a synthetic Gaussian data dict matching the schema returned by
    ``utils.frame_processing.load_gaussian_data``.
    """
    rng = np.random.RandomState(seed)
    # Positions spread in a 50x50x50 cube centred at (0, 0, 25) so most
    # points have positive z (in front of camera).
    positions = rng.randn(n_points, 3).astype(np.float32)
    positions[:, 0] *= 15.0  # x spread
    positions[:, 1] *= 5.0   # y spread (height)
    positions[:, 2] = np.abs(positions[:, 2]) * 15.0 + 5.0  # z in [5, ~50]

    colors = rng.rand(n_points, 3).astype(np.float32)
    scales = np.abs(rng.randn(n_points, 3).astype(np.float32)) * 0.1
    opacities = rng.rand(n_points).astype(np.float32) * 0.8 + 0.2

    metadata = SimpleNamespace(
        focal_length_px=1000.0,
        color_space="linearRGB",
    )

    return {
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "opacities": opacities,
        "metadata": metadata,
    }


def ensure_synthetic_data(
    n_frames: int,
    point_counts: List[int],
    bench_dir: Path,
) -> Tuple[List[np.ndarray], Dict[int, List[dict]]]:
    """Return (frames_list, {N: [gauss_dicts]})."""
    frames = [_make_synthetic_frame(i) for i in range(n_frames)]

    gauss_by_n: Dict[int, List[dict]] = {}
    for n in point_counts:
        gauss_by_n[n] = [
            _make_synthetic_gaussians(n, seed=1000 * n + i) for i in range(n_frames)
        ]

    return frames, gauss_by_n


# ---------------------------------------------------------------------------
# Real PLY loading (optional)
# ---------------------------------------------------------------------------

def load_real_plys(ply_dir: Path, n_frames: int):
    """Load up to n_frames PLY files using the project loader."""
    # Import lazily so we only depend on SHARP when --real-ply is given.
    from utils.frame_processing import load_gaussian_data

    ply_files = sorted(ply_dir.glob("*.ply"))[:n_frames]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in {ply_dir}")

    data_list = []
    for p in ply_files:
        d = load_gaussian_data(p)
        if d is None:
            raise RuntimeError(f"Failed to load {p}")
        data_list.append(d)
    return data_list


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_call(fn, *args, **kwargs):
    """Run fn(*args) once and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return result, elapsed


def _bench_stage(fn, args_per_frame: list, warmup: int = 1) -> List[float]:
    """
    Benchmark *fn* across frames.

    ``args_per_frame`` is a list of (args_tuple, kwargs_dict) pairs.
    Returns list of elapsed_ms (one per frame, excluding warmup).
    """
    # Warmup: run the first frame once and discard
    for _ in range(warmup):
        a, kw = args_per_frame[0]
        fn(*a, **kw)

    times = []
    for a, kw in args_per_frame:
        _, elapsed = _time_call(fn, *a, **kw)
        times.append(elapsed)
    return times


def stats(times: List[float]) -> dict:
    """Compute median, p95, fps from a list of ms timings."""
    arr = np.array(times)
    med = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    fps = 1000.0 / med if med > 0 else float("inf")
    return {"median_ms": round(med, 2), "p95_ms": round(p95, 2), "fps": round(fps, 2)}


# ---------------------------------------------------------------------------
# Pathfinding grid helper
# ---------------------------------------------------------------------------

def _make_pathfinding_grid(size: int = 200, obstacle_frac: float = 0.2, seed: int = 42):
    """
    Create a random occupancy grid with a guaranteed path from (0,0) to
    (size-1, size-1). We carve a corridor after random fill to guarantee
    connectivity.
    """
    rng = np.random.RandomState(seed)
    grid = (rng.rand(size, size) < obstacle_frac).astype(np.float64)
    # Carve a corridor along row 0 and column size-1
    grid[0, :] = 0
    grid[:, -1] = 0
    grid[0, 0] = 0
    grid[-1, -1] = 0
    return grid


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    n_frames: int,
    point_counts: List[int],
    output_path: Path,
    real_ply_dir: Optional[Path],
):
    # Import the specific submodules directly via importlib so we do NOT
    # trigger utils/__init__.py (which eagerly pulls in frame_processing ->
    # PIL/torch/SHARP, none of which are needed for the benchmark).
    import importlib.util

    _project_root = Path(__file__).resolve().parent.parent

    def _load_module(name: str, rel_path: str):
        spec = importlib.util.spec_from_file_location(name, _project_root / rel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _depth_mod = _load_module("utils.depth_rendering", "utils/depth_rendering.py")
    _nav_mod = _load_module("utils.navigation", "utils/navigation.py")
    _pf_mod = _load_module("utils.pathfinding", "utils/pathfinding.py")

    render_depth_map = _depth_mod.render_depth_map
    depth_map_to_3d_points = _depth_mod.depth_map_to_3d_points
    extract_ground_plane = _nav_mod.extract_ground_plane
    detect_obstacles = _nav_mod.detect_obstacles
    compute_occupancy_grid_2d = _nav_mod.compute_occupancy_grid_2d
    find_free_paths = _pf_mod.find_free_paths

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    print("Preparing synthetic data ...")
    frames, gauss_by_n = ensure_synthetic_data(n_frames, point_counts, BENCH_DIR)

    if real_ply_dir is not None:
        print(f"Loading real PLYs from {real_ply_dir} ...")
        real_data = load_real_plys(real_ply_dir, n_frames)
        real_n = real_data[0]["positions"].shape[0]
        gauss_by_n[real_n] = real_data
        if real_n not in point_counts:
            point_counts = point_counts + [real_n]

    # Pathfinding grids (one per frame, same size for fairness)
    pf_grids = [_make_pathfinding_grid(200, seed=i) for i in range(n_frames)]

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    all_results: Dict[str, dict] = {}  # stage -> {N -> stats}

    for n in point_counts:
        data_list = gauss_by_n[n]
        n_label = f"{n}"
        print(f"\n--- N = {n:,} points ({n_frames} frames) ---")

        # 1. render_depth_map
        stage = "render_depth_map"
        args = [
            ((d["positions"], d["colors"]), {"resolution": (1920, 1080), "focal_length": 1000})
            for d in data_list
        ]
        times = _bench_stage(render_depth_map, args)
        s = stats(times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

        # Capture depth maps for the next stage
        depth_results = [render_depth_map(d["positions"], d["colors"], (1920, 1080), 1000) for d in data_list]

        # 2. depth_map_to_3d_points
        stage = "depth_map_to_3d_points"
        args = [
            ((dm, dc, 1000, 1920, 1080), {})
            for dm, dc in depth_results
        ]
        times = _bench_stage(depth_map_to_3d_points, args)
        s = stats(times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

        # 3. extract_ground_plane
        stage = "extract_ground_plane"
        args = [
            ((d["positions"],), {})
            for d in data_list
        ]
        times = _bench_stage(extract_ground_plane, args)
        s = stats(times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

        # Capture ground info for later stages
        ground_results = [extract_ground_plane(d["positions"]) for d in data_list]

        # 4. detect_obstacles
        stage = "detect_obstacles"
        args = [
            ((d["positions"], gr[1]), {})
            for d, gr in zip(data_list, ground_results)
        ]
        times = _bench_stage(detect_obstacles, args)
        s = stats(times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

        # 5. compute_occupancy_grid_2d
        stage = "compute_occupancy_grid_2d"
        args = [
            ((d["positions"], gr[3]), {})
            for d, gr in zip(data_list, ground_results)
        ]
        times = _bench_stage(compute_occupancy_grid_2d, args)
        s = stats(times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

        # 6. find_free_paths (200x200 grid, independent of N)
        stage = "find_free_paths"
        args = [
            ((g, (0, 0), (199, 199)), {})
            for g in pf_grids
        ]
        times = _bench_stage(find_free_paths, args)
        s = stats(times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

        # 7. Total pipeline (excl PLY load + ML)
        stage = "total_pipeline"
        total_times: List[float] = []
        for i, d in enumerate(data_list):
            if i == 0:
                # Extra warmup iteration for total pipeline
                _run_pipeline(d, pf_grids[0], render_depth_map, depth_map_to_3d_points,
                              extract_ground_plane, detect_obstacles,
                              compute_occupancy_grid_2d, find_free_paths)

            _, elapsed = _time_call(
                _run_pipeline, d, pf_grids[i],
                render_depth_map, depth_map_to_3d_points,
                extract_ground_plane, detect_obstacles,
                compute_occupancy_grid_2d, find_free_paths,
            )
            total_times.append(elapsed)

        s = stats(total_times)
        all_results.setdefault(stage, {})[n_label] = {**s, "times": total_times}
        print(f"  {stage}: median={s['median_ms']:.1f} ms  p95={s['p95_ms']:.1f} ms  fps={s['fps']:.1f}")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    git_sha = _git_sha()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build JSON-safe results (drop raw times lists for the summary; keep
    # them in the full dump).
    json_results = {}
    for stage, by_n in all_results.items():
        json_results[stage] = {}
        for n_label, vals in by_n.items():
            json_results[stage][n_label] = {
                "median_ms": vals["median_ms"],
                "p95_ms": vals["p95_ms"],
                "fps": vals["fps"],
                "times_ms": [round(t, 3) for t in vals["times"]],
            }

    payload = {
        "git_sha": git_sha,
        "timestamp": timestamp,
        "n_frames": n_frames,
        "point_counts": point_counts,
        "results": json_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON results written to {output_path}")

    # Pretty table
    _print_table(all_results, point_counts)

    # One-line summary for the largest N
    largest = str(max(point_counts))
    total_med = all_results["total_pipeline"][largest]["median_ms"]
    print(f"\nTotal per-frame pipeline cost (excl PLY+ML): {total_med:.1f} ms median")


def _run_pipeline(data, pf_grid, render_depth_map, depth_map_to_3d_points,
                  extract_ground_plane, detect_obstacles,
                  compute_occupancy_grid_2d, find_free_paths):
    """Run the full per-frame pipeline (minus PLY load) and return None."""
    positions = data["positions"]
    colors = data["colors"]

    dm, dc = render_depth_map(positions, colors, (1920, 1080), 1000)
    depth_map_to_3d_points(dm, dc, 1000, 1920, 1080)
    _, ground_mask, _, ground_info = extract_ground_plane(positions)
    detect_obstacles(positions, ground_mask)
    compute_occupancy_grid_2d(positions, ground_info)
    find_free_paths(pf_grid, (0, 0), (199, 199))


def _print_table(all_results, point_counts):
    """Print a formatted ASCII table to stdout."""
    stages = [
        "render_depth_map",
        "depth_map_to_3d_points",
        "extract_ground_plane",
        "detect_obstacles",
        "compute_occupancy_grid_2d",
        "find_free_paths",
        "total_pipeline",
    ]
    header = f"{'stage':<30} {'N':>12} {'median_ms':>12} {'p95_ms':>12} {'fps':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for stage in stages:
        by_n = all_results.get(stage, {})
        for n in point_counts:
            n_label = str(n)
            if n_label in by_n:
                v = by_n[n_label]
                n_fmt = f"{n:,}"
                print(
                    f"{stage:<30} {n_fmt:>12} {v['median_ms']:>12.1f} {v['p95_ms']:>12.1f} {v['fps']:>10.1f}"
                )
        # Blank line between stages
        print()
    print(sep)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Splatline pipeline benchmark harness",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=20,
        help="Number of synthetic frames to benchmark (default: 20)",
    )
    parser.add_argument(
        "--point-counts",
        type=str,
        default="100000,500000,1000000",
        help="Comma-separated point counts (default: 100000,500000,1000000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/splatline_bench/results.json",
        help="Path for JSON output (default: /tmp/splatline_bench/results.json)",
    )
    parser.add_argument(
        "--real-ply",
        type=str,
        default=None,
        help="If provided, load real PLY files from this directory",
    )
    args = parser.parse_args()

    point_counts = [int(x.strip()) for x in args.point_counts.split(",")]
    output_path = Path(args.output)
    real_ply_dir = Path(args.real_ply) if args.real_ply else None

    run_benchmark(
        n_frames=args.n_frames,
        point_counts=point_counts,
        output_path=output_path,
        real_ply_dir=real_ply_dir,
    )


if __name__ == "__main__":
    main()
