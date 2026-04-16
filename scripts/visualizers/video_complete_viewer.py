#!/usr/bin/env python3
"""
Complete Video Viewer - Shows EVERYTHING together in Rerun!

Displays simultaneously:
1. Original video frames (2D)
2. Depth maps (2D colored visualization)
3. Occupancy grid (2D top-down)
4. 3D point cloud (full scene)
5. Navigation data (ground + obstacles)

Performance features:
- Parallel frame preprocessing across CPU cores
- Disk cache (.splatline_cache.npz) to skip recomputation on re-runs
- Point budget downsampling for responsive Rerun streaming
- Ordered streaming: user can scrub as soon as initial frames arrive
"""

import argparse
import hashlib
import os
import sys
import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# NOTE: rerun and utils.visualization are imported lazily inside the functions
# that need them so that --help, tests, and worker subprocesses (which never
# touch Rerun) work without rerun installed.


# ---------------------------------------------------------------------------
# Config hash -- used to decide if a cache entry is still valid
# ---------------------------------------------------------------------------

def _config_hash(obstacle_height: float, resolution: float, point_budget: int) -> str:
    """Deterministic hash of the processing parameters that affect cache validity."""
    payload = f"{obstacle_height:.6f}|{resolution:.6f}|{point_budget}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

# Keys stored in every .npz cache file
_CACHE_ARRAY_KEYS = [
    "depth_map", "depth_colored", "occupancy_grid",
    "ground_points", "obstacle_points",
    "depth_3d_points", "depth_3d_colors",
    "positions_downsampled", "colors_downsampled", "scales_downsampled",
]

_CACHE_SCALAR_KEYS = [
    "focal_length", "depth_width", "depth_height",
    "mean_pos", "std_pos",
    "depth_mean_pos", "depth_std_pos",
]


def _cache_path_for(ply_path: Path, cache_dir: Path | None) -> Path:
    """Return the .splatline_cache.npz path for a given PLY."""
    base_dir = cache_dir if cache_dir is not None else ply_path.parent
    return base_dir / f"{ply_path.stem}.splatline_cache.npz"


def _try_load_cache(ply_path: Path, cache_dir: Path | None,
                    cfg_hash: str) -> dict | None:
    """Load a cache file if it exists, is newer than the PLY, and the config
    hash matches.  Returns the dict of arrays/scalars or None."""
    cp = _cache_path_for(ply_path, cache_dir)
    if not cp.exists():
        return None
    # mtime check
    if cp.stat().st_mtime <= ply_path.stat().st_mtime:
        return None
    try:
        npz = np.load(cp, allow_pickle=False)
        stored_hash = str(npz["config_hash"])
        if stored_hash != cfg_hash:
            return None
        result = {}
        for k in _CACHE_ARRAY_KEYS + _CACHE_SCALAR_KEYS:
            result[k] = npz[k]
        return result
    except Exception:
        return None


def _write_cache(ply_path: Path, cache_dir: Path | None,
                 cfg_hash: str, data: dict) -> None:
    """Persist processed frame data to a .splatline_cache.npz file."""
    cp = _cache_path_for(ply_path, cache_dir)
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"config_hash": np.array(cfg_hash)}
        for k in _CACHE_ARRAY_KEYS + _CACHE_SCALAR_KEYS:
            save_dict[k] = data[k]
        np.savez(cp, **save_dict)
    except Exception as exc:
        warnings.warn(f"Failed to write cache for {ply_path.name}: {exc}")


# ---------------------------------------------------------------------------
# Point-budget downsampling
# ---------------------------------------------------------------------------

def _downsample(positions, colors, scales, budget: int):
    """Return (positions, colors, scales) downsampled to at most *budget* points."""
    n = len(positions)
    if n <= budget:
        return positions, colors, scales
    indices = np.random.choice(n, size=budget, replace=False)
    return positions[indices], colors[indices], scales[indices]


# ---------------------------------------------------------------------------
# Worker function (top-level, pickleable)
# ---------------------------------------------------------------------------

def _preprocess_frame(args: tuple) -> dict | None:
    """Process one frame completely: load PLY -> compute derived data -> cache.

    Must be a **top-level** function so ``ProcessPoolExecutor`` can pickle it.

    Parameters
    ----------
    args : tuple
        (idx, ply_path_str, frames_dir_str, obstacle_height, resolution,
         point_budget, cfg_hash, use_cache, cache_dir_str)

    Returns
    -------
    dict or None
        Processed frame payload (numpy arrays + scalars) with an ``"_idx"``
        key for ordering, or ``None`` on failure.
    """
    (idx, ply_path_str, frames_dir_str, obstacle_height, resolution,
     point_budget, cfg_hash, use_cache, cache_dir_str) = args

    ply_path = Path(ply_path_str)
    frames_dir = Path(frames_dir_str) if frames_dir_str else None
    cache_dir = Path(cache_dir_str) if cache_dir_str else None

    # Ensure project root is on sys.path inside the worker
    _project_root = ply_path.parent
    # Walk up to find utils -- project_root is two levels above scripts/visualizers
    # but we derive it from the PLY location generically by checking for utils dir
    for candidate in [
        Path(__file__).parent.parent.parent,
        Path.cwd(),
    ]:
        if (candidate / "utils").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

    # --- cache hit path ---
    if use_cache:
        cached = _try_load_cache(ply_path, cache_dir, cfg_hash)
        if cached is not None:
            # We still need the video frame (it's not cached -- it's cheap to load)
            video_frame = _load_video_frame_for(ply_path, frames_dir)
            cached["video_frame"] = video_frame
            cached["_idx"] = idx
            cached["_cache_hit"] = True
            return cached

    # --- cache miss: full processing ---
    try:
        from utils.frame_processing import process_frame_complete
        from utils.depth_rendering import depth_map_to_3d_points

        frame_idx = int(ply_path.stem.split('_')[-1]) if 'frame_' in ply_path.stem else idx

        data = process_frame_complete(
            ply_path, frame_idx, frames_dir,
            obstacle_height, resolution
        )
        if data is None:
            return None

        # Compute depth 3D points (moved here so it is parallelized + cached)
        depth_3d_points, depth_3d_colors = depth_map_to_3d_points(
            data["depth_map"],
            data["depth_colored"],
            data["focal_length"],
            data["depth_width"],
            data["depth_height"],
        )

        # Downsample scene cloud
        pos_ds, col_ds, scl_ds = _downsample(
            data["positions"], data["colors"], data["scales"], point_budget
        )

        # Downsample depth 3D cloud
        if len(depth_3d_points) > point_budget:
            d_indices = np.random.choice(len(depth_3d_points), size=point_budget, replace=False)
            depth_3d_points = depth_3d_points[d_indices]
            depth_3d_colors = depth_3d_colors[d_indices]

        # Stats for camera transform (constant per frame, cached)
        mean_pos = pos_ds.mean(axis=0) if len(pos_ds) > 0 else np.zeros(3)
        std_pos = pos_ds.std(axis=0) if len(pos_ds) > 0 else np.ones(3)
        depth_mean_pos = depth_3d_points.mean(axis=0) if len(depth_3d_points) > 0 else np.zeros(3)
        depth_std_pos = depth_3d_points.std(axis=0) if len(depth_3d_points) > 0 else np.ones(3)

        result = {
            "depth_map": data["depth_map"],
            "depth_colored": data["depth_colored"],
            "occupancy_grid": data["occupancy_grid"],
            "ground_points": data["ground_points"],
            "obstacle_points": data["obstacle_points"],
            "depth_3d_points": depth_3d_points,
            "depth_3d_colors": depth_3d_colors,
            "positions_downsampled": pos_ds,
            "colors_downsampled": col_ds,
            "scales_downsampled": scl_ds,
            "focal_length": np.array(data["focal_length"]),
            "depth_width": np.array(data["depth_width"]),
            "depth_height": np.array(data["depth_height"]),
            "mean_pos": mean_pos,
            "std_pos": std_pos,
            "depth_mean_pos": depth_mean_pos,
            "depth_std_pos": depth_std_pos,
            "video_frame": data["video_frame"],
            "_idx": idx,
            "_cache_hit": False,
        }

        # Write cache
        if use_cache:
            _write_cache(ply_path, cache_dir, cfg_hash, result)

        return result

    except Exception as exc:
        warnings.warn(f"Worker failed on {ply_path.name}: {exc}")
        return None


def _load_video_frame_for(ply_path: Path, frames_dir: Path | None):
    """Load the video frame matching *ply_path* (cheap, not cached)."""
    if frames_dir is None or not frames_dir.exists():
        return None
    frame_path = frames_dir / f"{ply_path.stem}.png"
    if frame_path.exists():
        try:
            from PIL import Image
            return np.array(Image.open(frame_path))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Main visualisation entry point
# ---------------------------------------------------------------------------

def visualize_complete_video(input_dir: Path, max_frames: int = None,
                            resolution: float = 0.5,
                            obstacle_height: float = 0.5,
                            skip_frames: int = 1,
                            size_multiplier: float = 1.0,
                            point_budget: int = 500_000,
                            no_cache: bool = False,
                            cache_dir: str | None = None,
                            workers: int | None = None):
    """
    Complete visualization with everything!
    """
    import rerun as rr
    from utils.visualization import setup_complete_viewer_blueprint

    # Find all PLY files
    ply_files = sorted(list(input_dir.glob("*.ply")))

    if len(ply_files) == 0:
        print(f"No PLY files found in {input_dir}")
        return 1

    if max_frames:
        ply_files = ply_files[:max_frames]

    if skip_frames > 1:
        ply_files = ply_files[::skip_frames]

    num_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)

    print("=" * 80)
    print("COMPLETE VIDEO VIEWER - Everything in One Place!")
    print("=" * 80)
    print(f"\n  Input: {input_dir}")
    print(f"  Frames: {len(ply_files)}")
    print(f"  Settings:")
    print(f"   - Grid resolution: {resolution}m")
    print(f"   - Obstacle height: {obstacle_height}m")
    print(f"   - Point size: {size_multiplier}x")
    print(f"   - Point budget: {point_budget:,}")
    print(f"   - Workers: {num_workers}")
    print(f"   - Cache: {'disabled' if no_cache else 'enabled'}")
    if cache_dir:
        print(f"   - Cache dir: {cache_dir}")
    if skip_frames > 1:
        print(f"   - Skip: Every {skip_frames} frames")

    # Check for frames
    frames_dir = input_dir.parent / "frames"
    has_frames = frames_dir.exists()

    if has_frames:
        print(f"  Found video frames: {frames_dir}")
    else:
        print(f"  No video frames found (will generate from 3D)")

    # Initialize Rerun -- main process only
    print("\n  Launching Rerun viewer...")
    rr.init("Complete Video Viewer - All Data", spawn=True)

    # Set up coordinate system for proper camera controls
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Set up comprehensive blueprint with TWO 3D views side by side
    blueprint = setup_complete_viewer_blueprint()
    rr.send_blueprint(blueprint)

    print("\n  Processing frames in parallel...")
    print("=" * 80)

    cfg_hash = _config_hash(obstacle_height, resolution, point_budget)
    use_cache = not no_cache
    cache_dir_str = cache_dir if cache_dir else None
    frames_dir_str = str(frames_dir) if frames_dir.exists() else None

    # Build task arguments
    task_args = []
    for idx, ply_path in enumerate(ply_files):
        task_args.append((
            idx,
            str(ply_path),
            frames_dir_str,
            obstacle_height,
            resolution,
            point_budget,
            cfg_hash,
            use_cache,
            cache_dir_str,
        ))

    # Parallel processing with ordered streaming to Rerun
    next_to_emit = 0
    pending: dict[int, dict] = {}
    cache_hits = 0
    cache_misses = 0
    failed = 0

    pbar = tqdm(total=len(ply_files), desc="Loading complete data")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {}
        for ta in task_args:
            fut = executor.submit(_preprocess_frame, ta)
            future_to_idx[fut] = ta[0]  # idx

        for future in as_completed(future_to_idx):
            frame_result = future.result()
            if frame_result is None:
                # Find which idx failed and mark it
                fidx = future_to_idx[future]
                pending[fidx] = None  # sentinel
                failed += 1
                pbar.update(1)
            else:
                fidx = frame_result["_idx"]
                if frame_result.get("_cache_hit"):
                    cache_hits += 1
                else:
                    cache_misses += 1
                pending[fidx] = frame_result
                pbar.update(1)

            # Emit as many consecutive frames as possible
            while next_to_emit in pending:
                payload = pending.pop(next_to_emit)
                if payload is not None:
                    _emit_frame_to_rerun(
                        payload, next_to_emit, ply_files[next_to_emit],
                        size_multiplier
                    )
                next_to_emit += 1

    # Drain any remaining (shouldn't happen, but just in case)
    while next_to_emit < len(ply_files):
        payload = pending.pop(next_to_emit, None)
        if payload is not None:
            _emit_frame_to_rerun(payload, next_to_emit, ply_files[next_to_emit],
                                 size_multiplier)
        next_to_emit += 1

    pbar.close()

    print("\n" + "=" * 80)
    print("COMPLETE VIEWER READY!")
    print(f"  Cache hits: {cache_hits}  |  Cache misses: {cache_misses}  |  Failed: {failed}")
    print("=" * 80)

    _print_viewer_help()

    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        return 0


# ---------------------------------------------------------------------------
# Rerun logging for a single frame (main process only)
# ---------------------------------------------------------------------------

def _emit_frame_to_rerun(data: dict, idx: int, ply_path: Path,
                         size_multiplier: float) -> None:
    """Log one processed frame to Rerun. Called from main process only."""
    import rerun as rr
    from utils.visualization import log_camera_transform, create_occupancy_grid_image

    frame_idx = int(ply_path.stem.split('_')[-1]) if 'frame_' in ply_path.stem else idx

    rr.set_time_sequence("frame", frame_idx)

    # 1. Video frame
    video_frame = data.get("video_frame")
    if video_frame is not None:
        rr.log("video/frame", rr.Image(video_frame))

    # 2. Colored depth map
    rr.log("depth/colored", rr.Image(data["depth_colored"]))

    # 3. Full colored 3D point cloud (scene) -- downsampled
    positions = data["positions_downsampled"]
    colors = data["colors_downsampled"]
    scales = data["scales_downsampled"]

    if len(positions) > 0:
        rr.log(
            "world/scene/points",
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=np.mean(scales, axis=1) * 0.3 * size_multiplier
            )
        )

    # 4. Depth map as 3D point cloud -- downsampled
    depth_3d_points = data["depth_3d_points"]
    depth_3d_colors = data["depth_3d_colors"]

    if len(depth_3d_points) > 0:
        rr.log(
            "world/depth/points",
            rr.Points3D(
                positions=depth_3d_points,
                colors=depth_3d_colors,
                radii=[1.0] * len(depth_3d_points)
            )
        )

    # 5. Occupancy grid
    grid_viz = create_occupancy_grid_image(data["occupancy_grid"])
    rr.log("grid/occupancy", rr.Image(grid_viz))

    # 6. Camera transforms
    mean_pos = data["mean_pos"]
    std_pos = data["std_pos"]
    if np.any(std_pos > 0):
        log_camera_transform("world/camera", mean_pos, std_pos)

    depth_mean_pos = data.get("depth_mean_pos")
    depth_std_pos = data.get("depth_std_pos")
    if depth_mean_pos is not None and len(depth_3d_points) > 0:
        log_camera_transform("world/depth/camera", depth_mean_pos, depth_std_pos)


# ---------------------------------------------------------------------------
# CLI help text (unchanged visual output)
# ---------------------------------------------------------------------------

def _print_viewer_help():
    print("\n  What You're Seeing:")
    print("\n  3D Scene Window (Top Left):")
    print("     FULL COLORED POINT CLOUD")
    print("     Shows original colors from video")
    print("     Your complete 3D reconstruction")
    print("     Rotate/pan/zoom to explore")

    print("\n  3D Depth Map Window (Top Right):")
    print("     DEPTH MAP AS 3D POINT CLOUD")
    print("     Colored by depth (blue=near, red=far)")
    print("     Rotate/pan/zoom to explore")

    print("\n  Original Video (Bottom Left):")
    print("     Your source video frames")
    print("     Compare with 3D reconstructions")

    print("\n  Depth Map (Bottom Middle):")
    print("     Blue/Purple = Near objects")
    print("     Red/Yellow = Far objects")
    print("     Generated from 3D points")

    print("\n  Occupancy Grid (Bottom Right):")
    print("     Green = Free space (robot can go)")
    print("     Red = Occupied (obstacles)")
    print("     Bird's eye / top-down view")

    print("\n  3D VIEW CONTROLS (BOTH WINDOWS):")
    print("  " + "-" * 55)
    print("  Timeline (bottom) -> Scrub through frames")
    print("")
    print("  ROTATE:  Left click + drag")
    print("  PAN:     Right click + drag (primary)")
    print("           OR Middle mouse + drag")
    print("           OR Shift + Left click + drag")
    print("  ZOOM:    Mouse wheel / Trackpad scroll")
    print("  RESET:   Double click anywhere")
    print("  " + "-" * 55)
    print("\n  Both 3D windows are INDEPENDENT -- pan/zoom/rotate each separately.")

    print("\n  Tips:")
    print("  - Use timeline to see how scene changes frame-by-frame")
    print("  - Compare video with colored 3D point cloud")
    print("  - Watch depth map to understand distance")
    print("  - Occupancy grid shows robot's bird's eye view")

    print("\nPress Ctrl+C to exit.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Complete video viewer - everything in one place!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View your high-quality video with everything
  python video_complete_viewer.py -i output_13588904_3840_2160_30fps/gaussians/ --max-frames 30

  # All frames
  python video_complete_viewer.py -i output_13588904_3840_2160_30fps/gaussians/

  # Faster preview (every 5th frame)
  python video_complete_viewer.py -i output_grok_video_full/gaussians/ --skip 5

  # Larger point sizes for better visibility
  python video_complete_viewer.py -i output_video/gaussians/ --size 2.0 --max-frames 20

  # Fine navigation grid
  python video_complete_viewer.py -i output_video/gaussians/ --resolution 0.3 --max-frames 30

  # Fast re-open with cache (second run is near-instant)
  python video_complete_viewer.py -i output_video/gaussians/ --max-frames 30

  # Reduce point cloud density for slower machines
  python video_complete_viewer.py -i output_video/gaussians/ --point-budget 200000
        """
    )

    parser.add_argument("-i", "--input", type=Path, required=True,
                       help="Directory containing PLY files (gaussians)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process")
    parser.add_argument("--skip", type=int, default=1,
                       help="Process every Nth frame (default: 1)")
    parser.add_argument("--resolution", type=float, default=0.5,
                       help="Occupancy grid resolution (default: 0.5m)")
    parser.add_argument("--obstacle-height", type=float, default=0.5,
                       help="Obstacle height threshold (default: 0.5m)")
    parser.add_argument("--size", type=float, default=1.0,
                       help="Point size multiplier (default: 1.0)")

    # New performance flags
    parser.add_argument("--point-budget", type=int, default=500_000,
                       help="Max points per cloud before downsampling (default: 500000)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable reading and writing of .splatline_cache.npz files")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory for cache files (default: alongside PLYs)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel worker processes (default: cpu_count - 1)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist")
        return 1

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory")
        return 1

    # Clamp workers
    if args.workers is not None:
        args.workers = max(1, args.workers)

    return visualize_complete_video(
        args.input,
        args.max_frames,
        args.resolution,
        args.obstacle_height,
        args.skip,
        args.size,
        point_budget=args.point_budget,
        no_cache=args.no_cache,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    import sys
    sys.exit(main())
