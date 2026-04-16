#!/usr/bin/env python3
"""
High-Quality Video to 3D Converter - Process ALL frames without skipping.

This script extracts every frame from a video and converts each to a 3D
Gaussian Splat PLY file.  It batches SHARP inference for higher GPU
utilisation and overlaps CPU-side PLY saving with the next GPU batch.
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Tuple, Optional

import cv2
import torch
import numpy as np
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_all_frames(video_path: Path, output_dir: Path, frame_skip: int = 1) -> int:
    """
    Extract frames from video with optional skipping.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save frames
        frame_skip: Extract every Nth frame (1 = all frames, 2 = every other, etc.)

    Returns:
        Number of frames extracted
    """
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Extracting frames from: {video_path.name}")
    print(f"    Output directory: {frames_dir}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_output_frames = (total_frames + frame_skip - 1) // frame_skip

    print(f"    Video info: {total_frames} frames @ {fps:.2f} FPS")

    if frame_skip > 1:
        print(f"    Frame skip: Every {frame_skip} frame(s)")
        print(f"    Will extract: ~{expected_output_frames} frames\n")
    else:
        print(f"    Extracting all {total_frames} frames\n")

    print(f"    Extracting frames...")

    video_frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save every Nth frame
        if video_frame_count % frame_skip == 0:
            # Save frame as PNG (lossless)
            frame_path = frames_dir / f"frame_{saved_frame_count:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved_frame_count += 1

            if saved_frame_count % 10 == 0 or saved_frame_count == 1:
                print(f"    Extracted: {saved_frame_count} frames "
                      f"(video frame {video_frame_count}/{total_frames})",
                      end="\r")

        video_frame_count += 1

    cap.release()
    print(f"\n    Extracted {saved_frame_count} frames from "
          f"{video_frame_count} video frames")

    return saved_frame_count


# ---------------------------------------------------------------------------
# Batched SHARP inference helpers
# ---------------------------------------------------------------------------

# Default SHARP internal resolution. The model was trained at 1536x1536;
# smaller shapes run faster with modest quality loss (compute scales as H*W).
# 1024x1024 is ~2.25x faster than 1536x1536 and usually visually acceptable.
DEFAULT_INTERNAL_SHAPE = (1024, 1024)


def _preprocess_image(
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    internal_shape=DEFAULT_INTERNAL_SHAPE,
):
    """
    Preprocess a single image for SHARP inference.

    Returns:
        (image_resized, disparity_factor, height, width)
    """
    image_pt = (
        torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1)
        / 255.0
    )
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    return image_resized, disparity_factor, height, width


def _build_intrinsics(
    f_px: float,
    height: int,
    width: int,
    device: torch.device,
    internal_shape=DEFAULT_INTERNAL_SHAPE,
):
    """Build and return (K, K_resized) intrinsic matrices for a single frame."""
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    return intrinsics, intrinsics_resized


def _autocast_ctx(device_type: str, use_fp16: bool):
    """
    Return a context manager enabling fp16 autocast on GPU devices.
    Falls back to nullcontext on CPU or when disabled.
    """
    from contextlib import nullcontext
    if use_fp16 and device_type in ("cuda", "mps"):
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def _batched_predict(
    predictor,
    images_resized: torch.Tensor,
    disparity_factors: torch.Tensor,
    use_fp16: bool = True,
):
    """
    Run the SHARP predictor on a batch of images, optionally in fp16 autocast.
    """
    device_type = images_resized.device.type
    with _autocast_ctx(device_type, use_fp16):
        return predictor(images_resized, disparity_factors)


def _unproject_single(
    gaussians_ndc_single,
    intrinsics_resized,
    device,
    internal_shape=DEFAULT_INTERNAL_SHAPE,
):
    """Unproject a single frame's NDC gaussians to metric space."""
    from sharp.utils.gaussians import unproject_gaussians

    return unproject_gaussians(
        gaussians_ndc_single,
        torch.eye(4).to(device),
        intrinsics_resized,
        internal_shape,
    )


# ---------------------------------------------------------------------------
# PLY save function (runs in worker thread)
# ---------------------------------------------------------------------------

def _save_ply_in_thread(save_ply_fn, gaussians_data, f_px, hw_tuple,
                        output_path):
    """
    Save a single PLY file.  Designed to run in a ThreadPoolExecutor.

    Using threads (not processes) avoids pickling issues with Gaussians3D
    objects and the sharp package.  PLY saving is I/O-bound (numpy
    serialisation + disk write) and releases the GIL during the heavy
    parts, so threads provide effective parallelism here.
    """
    save_ply_fn(gaussians_data, f_px, hw_tuple, output_path)


# ---------------------------------------------------------------------------
# Core conversion function
# ---------------------------------------------------------------------------

def convert_frames_to_3d(
    frames_dir: Path,
    output_dir: Path,
    device: str = "default",
    batch_size: int = 4,
    internal_shape=DEFAULT_INTERNAL_SHAPE,
    use_fp16: bool = True,
):
    """
    Convert all frames to 3D Gaussian Splats using SHARP Python API.

    Batches GPU inference and overlaps CPU-side PLY writing with the next
    GPU batch for higher throughput.

    Args:
        frames_dir: Directory containing frame images
        output_dir: Directory to save PLY files
        device: Device to use ('default', 'cuda', 'mps', 'cpu')
        batch_size: Number of images per GPU batch (auto-falls back to 1 on OOM)
        internal_shape: SHARP internal resolution (square). Smaller = faster
            at small quality cost. Default 1024; model native is 1536.
        use_fp16: Run predictor under fp16 autocast on GPU devices. ~1.5-2x
            speedup on MPS/CUDA with negligible quality loss.
    """
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply

    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Converting frames to 3D Gaussian Splats ---")
    print(f"    Input: {frames_dir}")
    print(f"    Output: {gaussians_dir}")
    print(f"    Device: {device}")
    print(f"    Batch size: {batch_size}")
    print(f"    Internal shape: {internal_shape[0]}x{internal_shape[1]}")
    print(f"    Precision: {'fp16 (autocast)' if use_fp16 else 'fp32'}")

    # Auto-detect device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"    Using CUDA (GPU)")
        elif torch.mps.is_available():
            device = "mps"
            print(f"    Using MPS (Apple Silicon GPU)")
        else:
            device = "cpu"
            print(f"    Using CPU (will be slower)")

    device_obj = torch.device(device)

    # Load model
    print(f"\n    Loading SHARP model...")
    DEFAULT_MODEL_URL = (
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    )

    try:
        print(f"    Downloading model from: {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True
        )

        gaussian_predictor = create_predictor(PredictorParams())
        gaussian_predictor.load_state_dict(state_dict)
        gaussian_predictor.eval()
        gaussian_predictor.to(device_obj)
        print(f"    Model loaded successfully")
    except Exception as e:
        print(f"\n    Error loading model: {e}")
        return False

    # Gather image paths
    extensions = io.get_supported_image_extensions()
    image_paths: List[Path] = []
    for ext in extensions:
        image_paths.extend(list(frames_dir.glob(f"*{ext}")))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"    No images found in {frames_dir}")
        return False

    total_frames = len(image_paths)
    print(f"\n    Processing {total_frames} frames "
          f"(batch_size={batch_size})...")

    # ------------------------------------------------------------------
    # Main loop: batched predict + overlapped PLY save
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    processed = 0
    save_futures: List[Future] = []

    # We use a ThreadPoolExecutor to overlap PLY saves with GPU work.
    # Threads (not processes) avoid pickling issues with Gaussians3D
    # objects.  PLY saving is I/O-bound and releases the GIL during
    # numpy serialisation and disk writes, so threads provide effective
    # parallelism.  max_workers=2 keeps I/O busy without contention.
    with ThreadPoolExecutor(max_workers=2) as save_pool:
        batch_start = 0
        while batch_start < total_frames:
            batch_end = min(batch_start + batch_size, total_frames)
            batch_paths = image_paths[batch_start:batch_end]
            current_batch_size = len(batch_paths)

            # -- Stage 1: Load + preprocess all images in this batch -------
            images_resized_list = []
            disp_factors_list = []
            frame_meta = []  # (f_px, height, width, output_path) per image

            for img_path in batch_paths:
                try:
                    image, _, f_px = io.load_rgb(img_path)
                    height, width = image.shape[:2]

                    img_resized, disp_factor, h, w = _preprocess_image(
                        image, f_px, device_obj, internal_shape
                    )
                    images_resized_list.append(img_resized)
                    disp_factors_list.append(disp_factor)
                    frame_meta.append((
                        f_px, h, w,
                        gaussians_dir / f"{img_path.stem}.ply",
                    ))
                except Exception as e:
                    print(f"\n    [{batch_start + len(frame_meta) + 1}/"
                          f"{total_frames}] Error loading {img_path.name}: {e}")
                    continue

            if not images_resized_list:
                batch_start = batch_end
                continue

            # -- Stage 2: Batched GPU inference ----------------------------
            images_batch = torch.cat(images_resized_list, dim=0)
            disp_batch = torch.cat(disp_factors_list, dim=0)

            try:
                gaussians_ndc_batch = _batched_predict(
                    gaussian_predictor, images_batch, disp_batch, use_fp16
                )
            except (RuntimeError,) as oom_err:
                # Auto-fallback: retry one-by-one on OOM / MPS error
                if current_batch_size > 1:
                    oom_msg = str(oom_err)
                    if ("out of memory" in oom_msg.lower()
                            or "mps" in oom_msg.lower()):
                        print(f"\n    OOM with batch_size="
                              f"{current_batch_size}"
                              f" -- falling back to batch_size=1")
                        batch_size = 1
                        # Retry this batch one-by-one
                        for i, (img_r, df) in enumerate(
                            zip(images_resized_list, disp_factors_list)
                        ):
                            f_px_i, h_i, w_i, out_path_i = frame_meta[i]
                            try:
                                g_ndc = _batched_predict(
                                    gaussian_predictor, img_r, df, use_fp16
                                )
                                _, K_resized_i = _build_intrinsics(
                                    f_px_i, h_i, w_i, device_obj, internal_shape
                                )
                                gaussians_i = _unproject_single(
                                    g_ndc, K_resized_i, device_obj, internal_shape
                                )
                                gaussians_cpu = gaussians_i.to(torch.device("cpu"))
                                fut = save_pool.submit(
                                    _save_ply_in_thread,
                                    save_ply,
                                    gaussians_cpu,
                                    f_px_i,
                                    (h_i, w_i),
                                    out_path_i,
                                )
                                save_futures.append(fut)
                                processed += 1
                                print(f"    [{processed}/{total_frames}] "
                                      f"{out_path_i.name} (b=1)",
                                      end="\r")
                            except Exception as inner_e:
                                print(f"\n    Error on fallback for "
                                      f"{out_path_i.name}: {inner_e}")
                        batch_start = batch_end
                        continue
                raise

            # -- Stage 3: Per-frame unproject + async PLY save -------------
            for i in range(len(frame_meta)):
                f_px_i, h_i, w_i, out_path_i = frame_meta[i]

                _, K_resized_i = _build_intrinsics(
                    f_px_i, h_i, w_i, device_obj, internal_shape
                )

                # Gaussians3D is a NamedTuple: subscripting returns a FIELD,
                # not a batch slice. Reconstruct a per-frame Gaussians3D by
                # slicing each tensor along dim 0 (the batch dim). Using
                # i:i+1 preserves the [1, N, ...] shape the unproject path
                # expects.
                g_ndc_i = type(gaussians_ndc_batch)(
                    mean_vectors=gaussians_ndc_batch.mean_vectors[i:i + 1],
                    singular_values=gaussians_ndc_batch.singular_values[i:i + 1],
                    quaternions=gaussians_ndc_batch.quaternions[i:i + 1],
                    colors=gaussians_ndc_batch.colors[i:i + 1],
                    opacities=gaussians_ndc_batch.opacities[i:i + 1],
                )

                gaussians_i = _unproject_single(
                    g_ndc_i, K_resized_i, device_obj, internal_shape
                )

                # Move tensors to CPU before submitting to the save thread.
                # This releases GPU memory sooner so the next batch can
                # start without waiting.
                gaussians_cpu = gaussians_i.to(torch.device("cpu"))

                fut = save_pool.submit(
                    _save_ply_in_thread,
                    save_ply,
                    gaussians_cpu,
                    f_px_i,
                    (h_i, w_i),
                    out_path_i,
                )
                save_futures.append(fut)
                processed += 1
                print(f"    [{processed}/{total_frames}] "
                      f"{out_path_i.name}", end="\r")

            batch_start = batch_end

        # -- Wait for all outstanding PLY saves to finish ------------------
        errors = 0
        for fut in save_futures:
            try:
                fut.result()
            except Exception as e:
                errors += 1
                print(f"\n    PLY save error: {e}")

    t_elapsed = time.perf_counter() - t_start
    fps = processed / t_elapsed if t_elapsed > 0 else float("inf")

    print(f"\n\n    Done: {processed} frames in {t_elapsed:.1f}s "
          f"({fps:.2f} frames/sec)")
    if errors:
        print(f"    {errors} PLY save error(s) occurred.")
    print(f"    Successfully converted frames to 3D!")
    return True


# ---------------------------------------------------------------------------
# Legacy single-frame prediction (kept for reference / direct usage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_image_direct(
    predictor, image: np.ndarray, f_px: float, device: torch.device
):
    """
    Predict Gaussians from a single image using SHARP.

    This is the original per-frame path kept for backward compatibility.
    The batched path in ``convert_frames_to_3d`` is preferred.

    Args:
        predictor: SHARP predictor model
        image: Input image as numpy array [H, W, 3]
        f_px: Focal length in pixels
        device: Torch device

    Returns:
        Gaussians3D object
    """
    from sharp.utils.gaussians import unproject_gaussians

    image_resized, disp_factor, height, width = _preprocess_image(
        image, f_px, device, DEFAULT_INTERNAL_SHAPE
    )
    gaussians_ndc = predictor(image_resized, disp_factor)

    _, K_resized = _build_intrinsics(
        f_px, height, width, device, DEFAULT_INTERNAL_SHAPE
    )

    gaussians = unproject_gaussians(
        gaussians_ndc,
        torch.eye(4).to(device),
        K_resized,
        DEFAULT_INTERNAL_SHAPE,
    )

    return gaussians


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_to_3d_high_quality.py",
        description=(
            "Convert video frames to high-quality 3D Gaussian Splats.\n\n"
            "Supports batched GPU inference and overlapped PLY writing\n"
            "for significantly higher throughput."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments (preserve backward compatibility with the old
    # ``script video_file [device] [skip]`` interface).
    p.add_argument(
        "video_file",
        metavar="VIDEO",
        help="Path to your video file (mp4, mov, avi, etc.)",
    )
    p.add_argument(
        "device",
        nargs="?",
        default="default",
        help=(
            "Compute device: 'cuda', 'mps', 'cpu', or 'default' "
            "(auto-detect).  [default: default]"
        ),
    )
    p.add_argument(
        "skip",
        nargs="?",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = all frames).",
    )

    # New optional flag
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        dest="batch_size",
        help=(
            "Number of images per GPU batch.  Higher = better GPU "
            "utilisation but more VRAM.  Auto-falls back to 1 on OOM.  "
            "[default: 4]"
        ),
    )
    p.add_argument(
        "--internal-shape",
        type=int,
        default=1024,
        metavar="N",
        dest="internal_shape",
        help=(
            "SHARP internal square resolution (e.g. 1024 or 1536). "
            "Smaller is faster (~(1536/N)^2 speedup) with modest quality "
            "loss.  The model was trained at 1536.  [default: 1024]"
        ),
    )
    p.add_argument(
        "--fp32",
        action="store_true",
        dest="fp32",
        help=(
            "Disable fp16 autocast during predictor forward pass. "
            "Slower but slightly higher fidelity.  GPU devices only."
        ),
    )

    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("HIGH-QUALITY VIDEO TO 3D CONVERTER")
    print("Convert video frames to 3D Gaussian Splats")
    print("=" * 70)

    parser = _build_parser()
    args = parser.parse_args()

    video_path = Path(args.video_file)
    device = args.device
    frame_skip = args.skip
    batch_size = args.batch_size
    internal_shape = (args.internal_shape, args.internal_shape)
    use_fp16 = not args.fp32

    if frame_skip < 1:
        print("Error: Frame skip must be >= 1")
        sys.exit(1)

    if batch_size < 1:
        print("Error: --batch-size must be >= 1")
        sys.exit(1)

    if args.internal_shape < 64 or args.internal_shape > 4096:
        print("Error: --internal-shape must be in [64, 4096]")
        sys.exit(1)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(f"output_{video_path.stem}")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n    Input video: {video_path}")
    print(f"    Output directory: {output_dir}")
    print(f"    Device: {device}")
    print(f"    Frame skip: Every {frame_skip} frame(s)")
    print(f"    Batch size: {batch_size}")

    # Step 1: Extract frames
    try:
        num_frames = extract_all_frames(video_path, output_dir, frame_skip)
        if num_frames == 0:
            print("Error: No frames extracted")
            sys.exit(1)
    except Exception as e:
        print(f"Error extracting frames: {e}")
        sys.exit(1)

    # Step 2: Convert frames to 3D (batched + overlapped)
    frames_dir = output_dir / "frames"
    success = convert_frames_to_3d(
        frames_dir,
        output_dir,
        device,
        batch_size=batch_size,
        internal_shape=internal_shape,
        use_fp16=use_fp16,
    )

    if not success:
        print("\n    Conversion failed")
        sys.exit(1)

    # Summary
    gaussians_dir = output_dir / "gaussians"
    ply_files = list(gaussians_dir.glob("*.ply"))

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\n    Summary:")
    print(f"    Extracted frames: {num_frames}")
    print(f"    PLY files created: {len(ply_files)}")
    if frame_skip > 1:
        print(f"    Frame skip: Every {frame_skip} frame(s)")
    print(f"\n    Outputs:")
    print(f"    Frames: {output_dir / 'frames'}")
    print(f"    3D Splats: {output_dir / 'gaussians'}")

    print(f"\n    Next steps - Visualize your 3D video:")
    print(f"\n    Option 1 - Rerun viewer (recommended):")
    print(f"    python visualize_with_rerun.py -i {gaussians_dir}/")
    print(f"\n    Option 2 - Web viewer:")
    print(f"    python start_3d_viewer.py {gaussians_dir}/")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
