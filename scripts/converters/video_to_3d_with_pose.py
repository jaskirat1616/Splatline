#!/usr/bin/env python3
"""
Video to 3D with Pose Detection (Splatline)
===========================================
Converts video frames to 3D Gaussian Splats (via Apple ML-SHARP) AND runs
3D human pose detection on each frame.

Viewer layout (3 panels)
------------------------
  Left   – 3D scene view : Gaussian point cloud only
  Centre – 3D pose view  : pose skeleton only (clean, dark background)
  Right  – 2D video view : original video frame

How the 3D pose works
---------------------
MediaPipe Pose gives 33 body landmarks in 2D image space (normalized x,y).
For each landmark we find the k=30 nearest Gaussians in projected image
coordinates and take the median depth, then back-project with the pinhole
camera model.  The skeleton sits *inside* the 3D scene.

Usage
-----
  # Convert video then visualise:
  python video_to_3d_with_pose.py <video.mp4> [--device mps] [--skip 2] [--size 2.0]

  # Use an already-converted output directory (skips SHARP):
  python video_to_3d_with_pose.py --gaussians-dir <path> --frames-dir <path> [--size 2.0]

  # --size scales the joint sphere and bone tube radii (default: 1.0)

Requirements
------------
  pip install mediapipe rerun-sdk opencv-python numpy
  (plus the Apple ML-SHARP package for 3D conversion)
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# MediaPipe skeleton definition
# ---------------------------------------------------------------------------

# COCO-17 skeleton used by YOLO pose
# Each tuple is (joint_a, joint_b) forming one bone
# Indices: 0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear
#          5=l_shoulder 6=r_shoulder 7=l_elbow 8=r_elbow
#          9=l_wrist 10=r_wrist 11=l_hip 12=r_hip
#          13=l_knee 14=r_knee 15=l_ankle 16=r_ankle
POSE_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Shoulders
    (5, 6),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Torso sides
    (5, 11), (6, 12),
    # Hips
    (11, 12),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

NUM_KEYPOINTS = 17   # COCO-17

# RGB colours per joint group
_FACE_COLOR  = [210, 210, 255]   # lavender
_ARM_COLOR   = [ 80, 160, 255]   # blue
_LEG_COLOR   = [ 80, 230, 140]   # green
_TORSO_COLOR = [255, 200,  80]   # amber

_JOINT_COLOR_MAP = {
    **{i: _FACE_COLOR  for i in [0, 1, 2, 3, 4]},
    **{i: _ARM_COLOR   for i in [5, 6, 7, 8, 9, 10]},
    **{i: _TORSO_COLOR for i in [11, 12]},
    **{i: _LEG_COLOR   for i in [13, 14, 15, 16]},
}

# Bone colours per person (up to 4 people, cycles if more)
_PERSON_BONE_COLORS = [
    [0,   220, 200],   # cyan    – person 0
    [255, 160,   0],   # orange  – person 1
    [220,  60, 220],   # magenta – person 2
    [100, 255, 100],   # green   – person 3
]
_BONE_COLOR = _PERSON_BONE_COLORS[0]   # default single-person fallback

MAX_POSES = 4   # maximum simultaneous people to detect


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, output_dir: Path, frame_skip: int = 1):
    """
    Extract PNG frames from a video file.

    Returns
    -------
    (fps, num_frames_extracted)
    """
    frames_dir = output_dir / "frames"
    existing = sorted(frames_dir.glob("*.png"))
    if existing:
        print(f"  Found {len(existing)} existing frames – skipping extraction.")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        return fps, len(existing)

    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_raw   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected    = (total_raw + frame_skip - 1) // frame_skip

    print(f"  Video: {total_raw} frames @ {fps:.1f} FPS  |  "
          f"extracting every {frame_skip} frame(s) => ~{expected} frames")

    vid_idx   = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if vid_idx % frame_skip == 0:
            cv2.imwrite(str(frames_dir / f"frame_{saved_idx:06d}.png"), frame)
            saved_idx += 1
            if saved_idx % 25 == 0:
                print(f"  Extracted {saved_idx} / ~{expected} ...", end="\r")
        vid_idx += 1

    cap.release()
    print(f"\n  Extracted {saved_idx} frames.")
    return fps, saved_idx


# ---------------------------------------------------------------------------
# SHARP 3-D Gaussian conversion
# ---------------------------------------------------------------------------

def run_sharp_conversion(frames_dir: Path, output_dir: Path, device: str = "default"):
    """
    Convert extracted PNG frames to 3D Gaussian Splat PLY files using SHARP.
    Skips frames whose PLY already exists.
    """
    import torch
    import torch.nn.functional as F
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io
    from sharp.utils.gaussians import save_ply, unproject_gaussians

    gaussians_dir = output_dir / "gaussians"
    gaussians_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device_obj = torch.device(device)
    print(f"  Device: {device}")

    # Load model once
    MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    print("  Loading SHARP model (first run downloads ~1 GB)…")
    state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, progress=True)
    predictor  = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device_obj)
    print("  Model loaded.")

    image_paths = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    print(f"  Converting {len(image_paths)} frames…")

    for idx, img_path in enumerate(image_paths):
        out_ply = gaussians_dir / f"{img_path.stem}.ply"
        if out_ply.exists():
            continue

        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name}", end="  ")

        image, _, f_px = io.load_rgb(img_path)
        height, width  = image.shape[:2]

        with torch.no_grad():
            internal_shape = (1536, 1536)
            img_pt = (torch.from_numpy(image.copy()).float()
                      .to(device_obj).permute(2, 0, 1) / 255.0)
            disp_factor = torch.tensor([f_px / width]).float().to(device_obj)

            img_resized  = F.interpolate(img_pt[None], size=internal_shape,
                                         mode="bilinear", align_corners=True)
            gaussians_ndc = predictor(img_resized, disp_factor)

            K = torch.tensor([
                [f_px,    0, width / 2,  0],
                [   0, f_px, height / 2, 0],
                [   0,    0,          1, 0],
                [   0,    0,          0, 1],
            ]).float().to(device_obj)
            K_resized    = K.clone()
            K_resized[0] *= internal_shape[0] / width
            K_resized[1] *= internal_shape[1] / height

            gaussians = unproject_gaussians(
                gaussians_ndc, torch.eye(4).to(device_obj), K_resized, internal_shape
            )

        save_ply(gaussians, f_px, (height, width), out_ply)
        print("done")

    return gaussians_dir


# ---------------------------------------------------------------------------
# Pose detection helpers
# ---------------------------------------------------------------------------

# EMA smoothing factor: 0 = no smoothing, 1 = never update.
# 0.35 gives fluid motion while still following fast tennis swings.
_SMOOTH_ALPHA = 0.35


def _enhance_frame_for_detection(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Improve frame contrast before sending to MediaPipe.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    luminance channel.  Helps in outdoor scenes, harsh shadows, and
    motion-blurred frames where low contrast causes missed detections.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _smooth_joints(
    current: np.ndarray,
    previous,
    alpha: float = _SMOOTH_ALPHA,
) -> np.ndarray:
    """
    Exponential Moving Average between current and previous joint positions.

    current  : (33, N) array for this frame
    previous : (33, N) array from last frame, or None on first detection
    alpha    : weight on current frame (1 = no smoothing, 0 = frozen)
    """
    if previous is None or previous.shape != current.shape:
        return current
    return (alpha * current + (1.0 - alpha) * previous).astype(np.float32)


def setup_pose_detector():
    """
    Load YOLOv8x-pose-p6 – the highest-accuracy YOLO pose model.
    The model (~150 MB) is downloaded automatically by ultralytics on first run
    and cached in ~/.cache/ultralytics/.

    Returns the loaded YOLO model.
    """
    from ultralytics import YOLO
    model = YOLO("yolov8x-pose-p6.pt")
    print("  Pose detector: YOLOv8x-pose-p6  (multi-person, high accuracy)")
    return model


def detect_poses(frame_bgr: np.ndarray, model, img_w: int, img_h: int):
    """
    Detect all people in one BGR frame with YOLO pose + BoT-SORT tracking.

    Tracking with persist=True maintains consistent person IDs across frames
    so the EMA smoother stays matched to the right person.

    Returns
    -------
    list of (lm2d, None) tuples – one per detected person.
      lm2d : (17, 4) float  [x_norm, y_norm, 0, confidence]
             Normalised to 0-1 so downstream code is format-agnostic.
    """
    results = model.predict(
        frame_bgr,
        verbose=False,
        conf=0.25,
        iou=0.45,
        classes=0,        # person class only
    )

    poses = []
    if not results or results[0].keypoints is None:
        return poses

    kps = results[0].keypoints.data  # (N, 17, 3) tensor [x_px, y_px, conf]
    for person_kps in kps:
        arr = person_kps.cpu().numpy().astype(np.float32)   # (17, 3)
        lm2d = np.zeros((17, 4), dtype=np.float32)
        lm2d[:, 0] = arr[:, 0] / max(img_w, 1)   # normalise x → 0-1
        lm2d[:, 1] = arr[:, 1] / max(img_h, 1)   # normalise y → 0-1
        lm2d[:, 3] = arr[:, 2]                    # confidence as visibility
        poses.append((lm2d, None))                # no world landmarks from YOLO
    return poses


def image_joints_for_display(
    landmarks_2d: np.ndarray,
    img_w: int,
    img_h: int,
    gaussian_positions=None,
    f_px: float = 0.0,
    spread: float = 6.0,
) -> np.ndarray:
    """
    Build 3D display joints for the pose-only view from 2D YOLO keypoints.

    When gaussian_positions + f_px are supplied the function depth-lifts each
    joint using the nearest Gaussians in image space, giving a real 3D skeleton
    with per-limb depth variation.  Otherwise it falls back to a flat (Z=0)
    metric-scaled silhouette.

    Coordinate convention: Y-up (Rerun default), X-right, Z-forward.

    Parameters
    ----------
    landmarks_2d        : (17, 4) [x_norm, y_norm, 0, conf]
    img_w, img_h        : frame dimensions
    gaussian_positions  : (N, 3) Gaussian means in camera space, or None
    f_px                : focal length in pixels (required for depth lifting)
    spread              : horizontal display range in metres (default ±3 m)
    """
    TORSO_IDS = [5, 6, 11, 12]   # COCO: l/r shoulder + l/r hip
    cx, cy = img_w / 2.0, img_h / 2.0

    conf  = landmarks_2d[:, 3]
    kp_px = landmarks_2d[:, :2] * np.array([img_w, img_h], dtype=np.float32)

    # Torso centre (normalised image space) – used for X-offset separation
    vis_torso = [landmarks_2d[j] for j in TORSO_IDS if conf[j] > 0.2]
    torso_x_n = float(np.mean([lm[0] for lm in vis_torso])) if vis_torso else 0.5
    torso_y_n = float(np.mean([lm[1] for lm in vis_torso])) if vis_torso else 0.5

    # ------------------------------------------------------------------
    # Path A: depth-lift using Gaussian depth field (proper 3D)
    # ------------------------------------------------------------------
    if gaussian_positions is not None and f_px > 0 and len(gaussian_positions) >= 5:
        valid = gaussian_positions[:, 2] > 0.05
        pts   = gaussian_positions[valid]

        if len(pts) >= 5:
            g_u = pts[:, 0] / pts[:, 2] * f_px + cx
            g_v = pts[:, 1] / pts[:, 2] * f_px + cy
            g_d = pts[:, 2]
            k   = min(20, len(g_d))

            joints_cam = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            for i in range(NUM_KEYPOINTS):
                px, py  = kp_px[i]
                dist2   = (g_u - px) ** 2 + (g_v - py) ** 2
                nearest = np.argpartition(dist2, k)[:k]
                depth   = float(np.median(g_d[nearest]))
                joints_cam[i] = [
                    (px - cx) / f_px * depth,
                    (py - cy) / f_px * depth,
                    depth,
                ]

            # Centre on torso
            torso_ids_vis = [i for i in TORSO_IDS if conf[i] > 0.2]
            torso_cam = (joints_cam[torso_ids_vis].mean(axis=0)
                         if torso_ids_vis else joints_cam.mean(axis=0))
            joints_rel = joints_cam - torso_cam

            # Camera Y-down → display Y-up
            joints_rel[:, 1] *= -1.0

            # Horizontal separation
            joints_rel[:, 0] += (torso_x_n - 0.5) * spread
            return joints_rel

    # ------------------------------------------------------------------
    # Path B: flat metric-scaled silhouette (no depth info)
    # ------------------------------------------------------------------
    torso_px = np.array([torso_x_n * img_w, torso_y_n * img_h], dtype=np.float32)

    nose_vis  = conf[0] > 0.3
    ankle_vis = [j for j in [15, 16] if conf[j] > 0.3]
    if nose_vis and ankle_vis:
        ankle_y   = float(np.mean([kp_px[j, 1] for j in ankle_vis]))
        height_px = max(abs(ankle_y - kp_px[0, 1]), 30.0)
    else:
        height_px = img_h * 0.65

    scale  = 1.7 / height_px
    joints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for i in range(NUM_KEYPOINTS):
        joints[i] = [
             (kp_px[i, 0] - torso_px[0]) * scale,
            -(kp_px[i, 1] - torso_px[1]) * scale,   # flip Y → up
            0.0,
        ]
    joints[:, 0] += (torso_x_n - 0.5) * spread
    return joints


def _estimate_body_depth(
    landmarks_2d: np.ndarray,
    gaussian_positions: np.ndarray,
    f_px: float,
    img_w: int,
    img_h: int,
) -> float:
    """
    Estimate the depth of the person's torso in camera space by sampling
    Gaussians inside a box around the torso centre (hips + shoulders).

    Uses the 15th-percentile depth in that region so we land on the *surface*
    of the person rather than objects behind them.
    """
    cx, cy = img_w / 2.0, img_h / 2.0

    # Torso landmarks: COCO l/r shoulders (5,6) + l/r hips (11,12)
    torso_ids = [5, 6, 11, 12]
    vis_lms   = [landmarks_2d[j] for j in torso_ids if landmarks_2d[j, 3] > 0.1]

    if vis_lms:
        torso_u = float(np.mean([lm[0] for lm in vis_lms])) * img_w
        torso_v = float(np.mean([lm[1] for lm in vis_lms])) * img_h
        # Bounding box of the torso in image pixels
        u_min = min(lm[0] for lm in vis_lms) * img_w
        u_max = max(lm[0] for lm in vis_lms) * img_w
        v_min = min(lm[1] for lm in vis_lms) * img_h
        v_max = max(lm[1] for lm in vis_lms) * img_h
        # Expand by 50 % for safety
        pad_u = max((u_max - u_min) * 0.5, img_w * 0.05)
        pad_v = max((v_max - v_min) * 0.5, img_h * 0.05)
    else:
        torso_u, torso_v = cx, cy
        pad_u = img_w * 0.2
        pad_v = img_h * 0.2

    valid = gaussian_positions[:, 2] > 0.05
    pts   = gaussian_positions[valid]

    if len(pts) < 5:
        return 2.0  # fallback

    # Project Gaussians to image plane
    g_u = pts[:, 0] / pts[:, 2] * f_px + cx
    g_v = pts[:, 1] / pts[:, 2] * f_px + cy
    g_d = pts[:, 2]

    in_box = (
        (g_u >= torso_u - pad_u) & (g_u <= torso_u + pad_u) &
        (g_v >= torso_v - pad_v) & (g_v <= torso_v + pad_v)
    )

    region_d = g_d[in_box] if in_box.sum() >= 5 else g_d

    # 15th percentile → near the front surface of the person
    return float(np.percentile(region_d, 15))


def backproject_to_3d(
    landmarks_2d: np.ndarray,
    world_landmarks,
    gaussian_positions: np.ndarray,
    f_px: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Place MediaPipe world landmarks into camera space.

    Strategy
    --------
    1. Estimate the person's scene depth D by sampling Gaussians near the torso.
    2. Back-project the torso centre (image space) → 3D camera-space anchor.
    3. MediaPipe world_landmarks are metric and body-centred (origin = hip midpoint).
       Their axes are:  +X right, +Y *up*, +Z toward camera.
       Camera space is: +X right, +Y *down*, +Z into scene.
       → flip Y to convert.
    4. Translate the flipped world skeleton so its hip origin lands at the anchor.

    If world_landmarks are unavailable, fall back to single-depth lifting.
    """
    cx, cy = img_w / 2.0, img_h / 2.0

    # Step 1 – body depth
    depth = _estimate_body_depth(landmarks_2d, gaussian_positions, f_px, img_w, img_h)

    # Step 2 – torso centre → 3D anchor  (COCO-17 indices)
    torso_ids = [5, 6, 11, 12]
    vis_lms   = [landmarks_2d[j] for j in torso_ids if landmarks_2d[j, 3] > 0.1]
    if vis_lms:
        tu = float(np.mean([lm[0] for lm in vis_lms])) * img_w
        tv = float(np.mean([lm[1] for lm in vis_lms])) * img_h
    else:
        tu, tv = cx, cy

    anchor = np.array([
        (tu - cx) / f_px * depth,
        (tv - cy) / f_px * depth,
        depth,
    ], dtype=np.float32)

    # Step 3 + 4 – place world skeleton at anchor
    if world_landmarks is not None:
        joints_3d = world_landmarks.copy().astype(np.float32)
        joints_3d[:, 1] *= -1.0   # flip Y: MediaPipe up → camera down
        # world origin is the hip midpoint; compute it in world coords
        hip_origin = (world_landmarks[23] + world_landmarks[24]) / 2.0
        hip_origin[1] *= -1.0
        # translate so hip_origin lands at anchor
        joints_3d += (anchor - hip_origin)
    else:
        # Fallback: project every joint at the same estimated depth
        joints_3d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks_2d):
            px, py = lm[0] * img_w, lm[1] * img_h
            joints_3d[i] = [
                (px - cx) / f_px * depth,
                (py - cy) / f_px * depth,
                depth,
            ]

    return joints_3d


def draw_pose_overlay(frame_bgr: np.ndarray, all_poses: list) -> np.ndarray:
    """
    Draw skeletons for every detected person onto a copy of frame_bgr.
    Each person gets a different bone colour.

    all_poses : list of (landmarks_2d, world_landmarks) tuples
    """
    if not all_poses:
        return frame_bgr.copy()

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for p_idx, (landmarks_2d, _) in enumerate(all_poses):
        if landmarks_2d is None:
            continue
        rgb_bone = _PERSON_BONE_COLORS[p_idx % len(_PERSON_BONE_COLORS)]
        bgr_bone = (rgb_bone[2], rgb_bone[1], rgb_bone[0])

        # Bones
        for (a, b) in POSE_CONNECTIONS:
            if landmarks_2d[a, 3] > 0.1 and landmarks_2d[b, 3] > 0.1:
                pt1 = (int(landmarks_2d[a, 0] * w), int(landmarks_2d[a, 1] * h))
                pt2 = (int(landmarks_2d[b, 0] * w), int(landmarks_2d[b, 1] * h))
                cv2.line(out, pt1, pt2, bgr_bone, 2, cv2.LINE_AA)

        # Joints
        for i, lm in enumerate(landmarks_2d):
            if lm[3] > 0.1:
                px = int(lm[0] * w)
                py = int(lm[1] * h)
                rgb_j = _JOINT_COLOR_MAP.get(i, [255, 255, 255])
                bgr_j = (rgb_j[2], rgb_j[1], rgb_j[0])
                cv2.circle(out, (px, py), 5, bgr_j, -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def build_bone_segments(joints_3d: np.ndarray, landmarks_2d) -> list:
    """
    Return a list of [start_3d, end_3d] pairs for visible bones only.
    """
    segments = []
    for (a, b) in POSE_CONNECTIONS:
        vis_a = landmarks_2d[a, 3] if landmarks_2d is not None else 1.0
        vis_b = landmarks_2d[b, 3] if landmarks_2d is not None else 1.0
        if vis_a > 0.1 and vis_b > 0.1:
            segments.append([joints_3d[a].tolist(), joints_3d[b].tolist()])
    return segments


# ---------------------------------------------------------------------------
# Rerun visualisation
# ---------------------------------------------------------------------------

def _linear_to_srgb_u8(colors_linear: np.ndarray, metadata=None) -> np.ndarray:
    """
    Convert linearRGB float (0-1) to sRGB uint8 (0-255).

    SHARP stores colors in linear RGB.  Without this conversion everything
    looks dark because monitors expect gamma-corrected sRGB.

    Uses the official IEC 61966-2-1 piecewise formula.
    Falls back to a simple power-2.2 gamma if the input is already uint8 or
    out of the expected 0-1 range.
    """
    # Try the SHARP color-space util first (most accurate)
    try:
        from sharp.utils import color_space as cs_utils
        import torch
        if getattr(metadata, "color_space", "linearRGB") == "linearRGB":
            c = torch.from_numpy(colors_linear.astype(np.float32))
            srgb = cs_utils.linearRGB2sRGB(c).numpy()
            return (srgb * 255).clip(0, 255).astype(np.uint8)
    except Exception:
        pass

    # Fallback: manual piecewise sRGB gamma
    c = colors_linear.astype(np.float32).clip(0.0, 1.0)
    srgb = np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(c, 1.0 / 2.4) - 0.055,
    )
    return (srgb * 255).clip(0, 255).astype(np.uint8)


def _get_f_px(metadata, img_w: int) -> float:
    """Try to extract focal length from PLY metadata, fall back to estimate."""
    if metadata is None:
        return float(img_w) * 0.85

    # metadata might be a dict or a namespace-like object
    if isinstance(metadata, dict):
        for key in ("f_px", "focal_length", "focal"):
            if key in metadata:
                return float(metadata[key])

    for attr in ("f_px", "focal_length", "focal"):
        if hasattr(metadata, attr):
            val = getattr(metadata, attr)
            if val is not None:
                return float(val)

    return float(img_w) * 0.85


def visualize_with_rerun(
    frames_dir: Path,
    gaussians_dir: Path,
    video_fps: float = 30.0,
    size: float = 1.0,
):
    """
    Core visualisation loop – three Rerun panels:

      Left   (world)      – 3D Gaussian point cloud only (no skeleton)
      Centre (pose_only)  – pose skeleton only, clean view
      Right  (camera)     – original 2D video frame

    The ``size`` multiplier scales joint sphere and bone tube radii.
    """
    from sharp.utils.gaussians import load_ply

    ply_files   = sorted(gaussians_dir.glob("*.ply"))
    frame_files = sorted(frames_dir.glob("*.png"))

    if not ply_files:
        print("No PLY files found – run conversion first.")
        return

    n = min(len(ply_files), len(frame_files))
    print(f"\nVisualising {n} frames  |  fps={video_fps:.1f}  |  size={size}")

    # Scaled radii
    joint_radius = 0.025 * size
    bone_radius  = 0.007 * size

    # ------------------------------------------------------------------
    # Rerun setup
    # ------------------------------------------------------------------
    rr.init("Video 3D + Pose Detection", spawn=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D Scene",
                origin="/world",
            ),
            rrb.Spatial3DView(
                name="3D Pose",
                origin="/pose_only",
            ),
            rrb.Spatial2DView(
                name="Video + Pose",
                origin="/camera/frame",
            ),
            column_shares=[2, 1, 1],
        ),
        # Keep side panels open so you can easily click/navigate views
        collapse_panels=False,
    )
    rr.send_blueprint(blueprint)

    # Tell Rerun the coordinate convention used by SHARP (world/gaussians):
    #   Right (+X), Down (+Y), Forward (+Z) – standard camera / OpenCV space.
    # pose_only uses MediaPipe world coords (Y-up) so we leave it at default.
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ------------------------------------------------------------------
    # Pose detector – YOLOv8x-pose-p6  (auto-downloads on first run)
    # ------------------------------------------------------------------
    print("Setting up pose detector…")
    yolo_model = setup_pose_detector()

    # Per-person smoothing state  { person_idx: {'lm2d': array, 'world': array} }
    smooth_state: dict = {}

    # ------------------------------------------------------------------
    # Per-frame loop
    # ------------------------------------------------------------------
    for i in range(n):
        ply_path   = ply_files[i]
        frame_path = frame_files[i]

        print(f"  [{i+1:>4}/{n}]  {ply_path.name}", end="  ")

        # --- Load frame image ----------------------------------------
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            print("(skipped – frame missing)")
            continue
        img_h, img_w = frame_bgr.shape[:2]

        # --- Load Gaussians ------------------------------------------
        try:
            gaussians, metadata = load_ply(ply_path)
            positions  = gaussians.mean_vectors.cpu().numpy().squeeze()
            colors     = gaussians.colors.cpu().numpy().squeeze()
            scales     = gaussians.singular_values.cpu().numpy().squeeze()
            opacities  = gaussians.opacities.cpu().numpy().squeeze()
            f_px       = _get_f_px(metadata, img_w)
        except Exception as exc:
            print(f"(skipped – gaussian error: {exc})")
            continue

        # Drop near-transparent splats
        opacity_mask = opacities > 0.05
        positions = positions[opacity_mask]
        colors    = colors[opacity_mask]
        scales    = scales[opacity_mask]

        colors_u8 = _linear_to_srgb_u8(colors, metadata)

        # --- Multi-person pose detection (YOLO + BoT-SORT tracking) -------
        enhanced_bgr = _enhance_frame_for_detection(frame_bgr)
        all_poses    = detect_poses(enhanced_bgr, yolo_model, img_w, img_h)

        # EMA temporal smoothing per person (keyed by detection order / track id)
        smoothed_poses = []
        for p_idx, (lm2d, lm_world) in enumerate(all_poses):
            prev    = smooth_state.get(p_idx, {})
            s_lm2d  = _smooth_joints(lm2d, prev.get("lm2d"), _SMOOTH_ALPHA)
            smooth_state[p_idx] = {"lm2d": s_lm2d, "world": None}
            smoothed_poses.append((s_lm2d, None))

        for p_idx in list(smooth_state.keys()):
            if p_idx >= len(all_poses):
                del smooth_state[p_idx]

        all_poses = smoothed_poses

        # --- Rerun logging -------------------------------------------
        rr.set_time_sequence("frame", i)
        rr.set_time_seconds("time", i / video_fps)

        # Panel 1 – 3D Gaussian cloud
        rr.log(
            "world/gaussians",
            rr.Points3D(
                positions=positions,
                colors=colors_u8,
                radii=np.mean(scales, axis=1) * 0.5,
            ),
        )

        # Panel 2 – 3D pose (one entity per person for independent colour)
        active_paths = set()
        for p_idx, (lm2d, lm_world) in enumerate(all_poses):
            bone_color  = _PERSON_BONE_COLORS[p_idx % len(_PERSON_BONE_COLORS)]
            j_path      = f"pose_only/person_{p_idx}/joints"
            s_path      = f"pose_only/person_{p_idx}/skeleton"
            active_paths.update([j_path, s_path])

            # Display joints: depth-lifted via Gaussian cloud → real 3D positions
            display_joints = image_joints_for_display(
                lm2d, img_w, img_h,
                gaussian_positions=positions,
                f_px=f_px,
            )

            vis_mask      = lm2d[:, 3] > 0.1
            vis_joints    = display_joints[vis_mask]
            vis_joint_rgb = np.array(
                [_JOINT_COLOR_MAP.get(j, [255, 255, 255])
                 for j in np.where(vis_mask)[0]],
                dtype=np.uint8,
            )
            segments = build_bone_segments(display_joints, lm2d)

            if len(vis_joints) > 0:
                rr.log(j_path, rr.Points3D(
                    positions=vis_joints,
                    colors=vis_joint_rgb,
                    radii=joint_radius,
                ))
            if segments:
                rr.log(s_path, rr.LineStrips3D(
                    strips=[[s[0], s[1]] for s in segments],
                    colors=bone_color,
                    radii=bone_radius,
                ))

        # Clear slots that had a person last frame but not this one
        for p_idx in range(len(all_poses), MAX_POSES):
            rr.log(f"pose_only/person_{p_idx}/joints",   rr.Points3D(positions=np.zeros((0, 3))))
            rr.log(f"pose_only/person_{p_idx}/skeleton", rr.LineStrips3D(strips=[]))

        # Panel 3 – 2D video frame with all skeletons drawn on top
        overlay_bgr = draw_pose_overlay(frame_bgr, all_poses)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        rr.log("camera/frame", rr.Image(overlay_rgb))

        print(f"{len(all_poses)} person(s)")

    # YOLO model has no explicit close method

    print(f"\nAll {n} frames loaded into Rerun.")
    print("\nControls:")
    print("  3D views : left-drag = rotate | right-drag = pan | scroll = zoom")
    print("  Timeline : drag the slider at the bottom to step through frames")
    print("\nPress Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting.")


def _empty_pose():
    """Clear all per-person pose slots in the pose-only view."""
    for p_idx in range(MAX_POSES):
        rr.log(f"pose_only/person_{p_idx}/joints",   rr.Points3D(positions=np.zeros((0, 3))))
        rr.log(f"pose_only/person_{p_idx}/skeleton", rr.LineStrips3D(strips=[]))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_to_3d_with_pose.py",
        description=(
            "Convert a video to 3D Gaussian Splats and detect human pose in 3D.\n\n"
            "Viewer layout:\n"
            "  Left   – 3D scene (gaussians + pose skeleton)\n"
            "  Centre – 3D pose only (skeleton on black background)\n"
            "  Right  – original 2D video frame"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Two modes via mutually exclusive group of positional/optional ---
    src = p.add_argument_group("source  (pick one mode)")
    src.add_argument(
        "video",
        nargs="?",
        metavar="VIDEO",
        help="Path to a video file (mp4 / mov / avi …). "
             "Frames are extracted and converted to 3D automatically.",
    )
    src.add_argument(
        "--gaussians-dir",
        metavar="PATH",
        help="Path to an existing gaussians/ directory (skips SHARP conversion).",
    )
    src.add_argument(
        "--frames-dir",
        metavar="PATH",
        help="Path to an existing frames/ directory (required with --gaussians-dir).",
    )

    # --- Conversion options ---
    conv = p.add_argument_group("conversion options  (video mode only)")
    conv.add_argument(
        "--device",
        default="default",
        choices=["default", "cuda", "mps", "cpu"],
        help="Compute device for SHARP (default: auto-detect).",
    )
    conv.add_argument(
        "--skip",
        type=int,
        default=1,
        metavar="N",
        help="Extract every N-th frame (default: 1 = every frame).",
    )

    # --- Visualisation options ---
    viz = p.add_argument_group("visualisation options")
    viz.add_argument(
        "--size",
        type=float,
        default=1.0,
        metavar="SCALE",
        help="Scale joint sphere and bone tube radii (default: 1.0, try 2.0 for larger).",
    )

    return p


def main():
    print("\n" + "=" * 70)
    print("VIDEO TO 3D WITH POSE DETECTION")
    print("=" * 70)

    parser = _build_parser()
    args   = parser.parse_args()

    # ----------------------------------------------------------------
    # Validate: need exactly one source mode
    # ----------------------------------------------------------------
    have_video   = args.video is not None
    have_preconv = args.gaussians_dir is not None

    if not have_video and not have_preconv:
        parser.print_help()
        sys.exit(1)

    if have_preconv and not args.frames_dir:
        parser.error("--frames-dir is required when --gaussians-dir is used.")

    if args.skip < 1:
        parser.error("--skip must be >= 1.")

    # ----------------------------------------------------------------
    # Mode A: pre-converted data
    # ----------------------------------------------------------------
    if have_preconv:
        gaussians_dir = Path(args.gaussians_dir)
        frames_dir    = Path(args.frames_dir)

        for d, label in [(gaussians_dir, "--gaussians-dir"), (frames_dir, "--frames-dir")]:
            if not d.exists():
                print(f"Error: {label} path does not exist: {d}")
                sys.exit(1)

        print(f"Gaussians : {gaussians_dir}")
        print(f"Frames    : {frames_dir}")
        print(f"Size      : {args.size}")
        visualize_with_rerun(frames_dir, gaussians_dir, size=args.size)
        return

    # ----------------------------------------------------------------
    # Mode B: video file (extract + convert + visualise)
    # ----------------------------------------------------------------
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(f"output_{video_path.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input      : {video_path}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {args.device}")
    print(f"Frame skip : {args.skip}")
    print(f"Size       : {args.size}")
    print()

    # Step 1 – extract frames
    print("[ Step 1 / 3 ]  Extracting frames…")
    video_fps, num_frames = extract_frames(video_path, output_dir, args.skip)
    frames_dir = output_dir / "frames"

    if num_frames == 0:
        print("Error: no frames extracted.")
        sys.exit(1)

    # Step 2 – 3D conversion (skip if PLYs already exist)
    gaussians_dir = output_dir / "gaussians"
    existing_ply  = list(gaussians_dir.glob("*.ply")) if gaussians_dir.exists() else []

    if not existing_ply:
        print("\n[ Step 2 / 3 ]  Converting frames to 3D Gaussian Splats…")
        try:
            run_sharp_conversion(frames_dir, output_dir, args.device)
        except Exception as exc:
            print(f"\nSHARP conversion failed: {exc}")
            sys.exit(1)
    else:
        print(f"\n[ Step 2 / 3 ]  Found {len(existing_ply)} PLY files – skipping conversion.")

    # Step 3 – visualise
    print("\n[ Step 3 / 3 ]  Detecting poses and launching Rerun viewer…")
    visualize_with_rerun(frames_dir, gaussians_dir, video_fps, size=args.size)


if __name__ == "__main__":
    main()
