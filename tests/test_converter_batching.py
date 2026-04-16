#!/usr/bin/env python3
"""
Unit tests for batched SHARP inference in video_to_3d_high_quality.py.

These tests mock the SHARP predictor, io utilities, and save_ply so that
no model weights or GPU are required.  The goal is to validate:

  1. Batching logic: N frames with batch_size B produces ceil(N/B) batches.
  2. Uneven tail: 7 frames at batch_size=4 processes batches of [4, 3].
  3. OOM fallback: when the predictor raises RuntimeError (simulating OOM)
     on a multi-frame batch, the code retries frame-by-frame.
  4. PLY save overlap: save_ply is called exactly once per frame and
     receives the correct (f_px, (height, width), path) arguments.
  5. Timing output: the function prints frames/sec statistics.
"""

import sys
import types
import tempfile
from pathlib import Path
from unittest import mock
from collections import namedtuple

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Lightweight stand-ins for the sharp package
# ---------------------------------------------------------------------------

class FakeGaussians3D:
    """
    Mimics the Gaussians3D dataclass returned by the SHARP predictor.

    Supports indexing by ``[i]`` to extract single-frame results from a
    batch, just like the real Gaussians3D (which stores tensors with a
    leading batch dimension).
    """

    def __init__(self, batch_size: int = 1, num_points: int = 128):
        self.batch_size = batch_size
        self.num_points = num_points
        # Representative tensor shapes: [B, P, C]
        self.mean_vectors = torch.randn(batch_size, num_points, 3)
        self.colors = torch.rand(batch_size, num_points, 3)
        self.singular_values = torch.rand(batch_size, num_points, 3) * 0.01
        self.opacities = torch.rand(batch_size, num_points, 1)
        # Extra fields the real object may carry
        self._fields = [
            "mean_vectors", "colors", "singular_values", "opacities",
        ]

    def __getitem__(self, idx):
        """Return a single-frame FakeGaussians3D."""
        g = FakeGaussians3D(batch_size=1, num_points=self.num_points)
        g.mean_vectors = self.mean_vectors[idx : idx + 1]
        g.colors = self.colors[idx : idx + 1]
        g.singular_values = self.singular_values[idx : idx + 1]
        g.opacities = self.opacities[idx : idx + 1]
        return g

    def cpu(self):
        """Move all tensors to CPU (no-op since we are already on CPU)."""
        g = FakeGaussians3D(
            batch_size=self.batch_size, num_points=self.num_points
        )
        g.mean_vectors = self.mean_vectors.cpu()
        g.colors = self.colors.cpu()
        g.singular_values = self.singular_values.cpu()
        g.opacities = self.opacities.cpu()
        return g


class FakePredictor(torch.nn.Module):
    """
    A mock predictor that accepts [B, 3, 1536, 1536] images and [B]
    disparity factors, returning a FakeGaussians3D with the correct
    batch size.
    """

    def __init__(self, fail_above_batch: int = 0):
        """
        Args:
            fail_above_batch: if > 0, raise RuntimeError (simulating OOM)
                when the batch dimension exceeds this value.
        """
        super().__init__()
        self.fail_above_batch = fail_above_batch
        self.call_log: list = []  # (batch_size,) for each call

    def forward(self, images: torch.Tensor, disp_factors: torch.Tensor):
        B = images.shape[0]
        if self.fail_above_batch and B > self.fail_above_batch:
            raise RuntimeError(
                f"Simulated out of memory: batch {B} exceeds limit "
                f"{self.fail_above_batch}"
            )
        self.call_log.append(B)
        return FakeGaussians3D(batch_size=B, num_points=64)


# ---------------------------------------------------------------------------
# Helpers to install the mock ``sharp`` package into sys.modules
# ---------------------------------------------------------------------------

def _make_sharp_modules(predictor_instance=None):
    """
    Build a fake ``sharp`` package tree and insert it into ``sys.modules``
    so that ``from sharp.models import ...`` works without the real package.

    Returns the predictor instance used.
    """
    if predictor_instance is None:
        predictor_instance = FakePredictor()

    PredictorParams = namedtuple("PredictorParams", [])

    def create_predictor(params):
        return predictor_instance

    # -- sharp.models --
    sharp_models = types.ModuleType("sharp.models")
    sharp_models.PredictorParams = PredictorParams
    sharp_models.create_predictor = create_predictor

    # -- sharp.utils.io --
    sharp_utils_io = types.ModuleType("sharp.utils.io")

    def get_supported_image_extensions():
        return [".png", ".jpg", ".jpeg"]

    def load_rgb(path):
        """Return a fake (H, W, 3) image, no alpha, and a focal length."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        return img, None, 500.0  # image, alpha, f_px

    sharp_utils_io.get_supported_image_extensions = get_supported_image_extensions
    sharp_utils_io.load_rgb = load_rgb

    # -- sharp.utils.gaussians --
    sharp_utils_gaussians = types.ModuleType("sharp.utils.gaussians")

    _save_log = []

    def save_ply(gaussians, f_px, hw, path):
        """Record the call instead of writing to disk."""
        _save_log.append({
            "f_px": f_px,
            "hw": hw,
            "path": str(path),
        })

    def unproject_gaussians(g_ndc, extrinsics, intrinsics, shape):
        """Identity unproject: return the input unchanged."""
        return g_ndc

    sharp_utils_gaussians.save_ply = save_ply
    sharp_utils_gaussians.unproject_gaussians = unproject_gaussians
    sharp_utils_gaussians._save_log = _save_log

    # -- sharp.utils --
    sharp_utils = types.ModuleType("sharp.utils")
    sharp_utils.io = sharp_utils_io
    sharp_utils.gaussians = sharp_utils_gaussians

    # -- sharp (root) --
    sharp = types.ModuleType("sharp")
    sharp.models = sharp_models
    sharp.utils = sharp_utils

    # Register in sys.modules
    for name, mod in [
        ("sharp", sharp),
        ("sharp.models", sharp_models),
        ("sharp.utils", sharp_utils),
        ("sharp.utils.io", sharp_utils_io),
        ("sharp.utils.gaussians", sharp_utils_gaussians),
    ]:
        sys.modules[name] = mod

    return predictor_instance, _save_log


def _cleanup_sharp_modules():
    """Remove fake sharp modules from sys.modules."""
    to_remove = [k for k in sys.modules if k == "sharp" or k.startswith("sharp.")]
    for k in to_remove:
        del sys.modules[k]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_modules():
    """Ensure sharp modules are cleaned up after each test."""
    yield
    _cleanup_sharp_modules()


@pytest.fixture()
def frames_dir(tmp_path):
    """Create a temporary directory with 7 fake PNG frame files."""
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(7):
        # Write a minimal valid PNG-like file (just needs to exist for glob)
        p = d / f"frame_{i:06d}.png"
        # Create a small real image so cv2.imread won't fail if needed
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(p), img)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBatchedConversion:
    """Validate the batched conversion pipeline with mocked SHARP."""

    def test_basic_batching_7_frames_batch4(self, tmp_path, frames_dir):
        """7 frames with batch_size=4 should produce 2 batches: [4, 3]."""
        predictor, save_log = _make_sharp_modules()

        # Patch torch.hub so it doesn't try to download anything
        with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
            # Import AFTER sharp modules are installed
            # We need to reload the module since it may have been cached
            import importlib
            if "scripts.converters.video_to_3d_high_quality" in sys.modules:
                del sys.modules["scripts.converters.video_to_3d_high_quality"]

            # Add the project root to the path so we can import
            project_root = str(
                Path(__file__).resolve().parent.parent
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # We need to import the convert function.  Since SHARP is mocked
            # we also need to patch `create_predictor` to return our fake
            # and skip the state_dict load.
            from scripts.converters.video_to_3d_high_quality import (
                convert_frames_to_3d,
            )

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            result = convert_frames_to_3d(
                frames_dir, output_dir, device="cpu", batch_size=4,
            )

        assert result is True

        # The predictor should have been called for 2 batches: 4 + 3
        assert predictor.call_log == [4, 3], (
            f"Expected batches [4, 3], got {predictor.call_log}"
        )

        # save_ply should have been called 7 times (once per frame)
        assert len(save_log) == 7, (
            f"Expected 7 PLY saves, got {len(save_log)}"
        )

        # Verify all 7 frame stems appear in the save paths
        saved_stems = sorted(
            Path(entry["path"]).stem for entry in save_log
        )
        expected_stems = sorted(f"frame_{i:06d}" for i in range(7))
        assert saved_stems == expected_stems

    def test_batch_size_1(self, tmp_path, frames_dir):
        """batch_size=1 should process each frame individually."""
        predictor, save_log = _make_sharp_modules()

        with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
            if "scripts.converters.video_to_3d_high_quality" in sys.modules:
                del sys.modules["scripts.converters.video_to_3d_high_quality"]

            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from scripts.converters.video_to_3d_high_quality import (
                convert_frames_to_3d,
            )

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            result = convert_frames_to_3d(
                frames_dir, output_dir, device="cpu", batch_size=1,
            )

        assert result is True
        assert predictor.call_log == [1] * 7
        assert len(save_log) == 7

    def test_batch_size_larger_than_frames(self, tmp_path, frames_dir):
        """batch_size > num_frames should process all frames in one batch."""
        predictor, save_log = _make_sharp_modules()

        with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
            if "scripts.converters.video_to_3d_high_quality" in sys.modules:
                del sys.modules["scripts.converters.video_to_3d_high_quality"]

            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from scripts.converters.video_to_3d_high_quality import (
                convert_frames_to_3d,
            )

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            result = convert_frames_to_3d(
                frames_dir, output_dir, device="cpu", batch_size=16,
            )

        assert result is True
        # All 7 frames in a single batch
        assert predictor.call_log == [7]
        assert len(save_log) == 7

    def test_oom_fallback(self, tmp_path, frames_dir):
        """
        When the predictor OOMs on batch_size > 1, the code should
        automatically fall back to batch_size=1 and still process all
        frames.
        """
        # Predictor fails on any batch > 1
        fake_pred = FakePredictor(fail_above_batch=1)
        _, save_log = _make_sharp_modules(predictor_instance=fake_pred)

        with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
            if "scripts.converters.video_to_3d_high_quality" in sys.modules:
                del sys.modules["scripts.converters.video_to_3d_high_quality"]

            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from scripts.converters.video_to_3d_high_quality import (
                convert_frames_to_3d,
            )

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            result = convert_frames_to_3d(
                frames_dir, output_dir, device="cpu", batch_size=4,
            )

        assert result is True
        # All 7 frames should have been saved despite the initial OOM
        assert len(save_log) == 7

        # The batch=4 call raised before recording itself, so call_log
        # contains only the successful B=1 calls: 4 retries from the
        # first batch + 3 from the remaining frames = 7 total.
        assert all(b == 1 for b in fake_pred.call_log), (
            f"Expected all B=1 calls after fallback, got {fake_pred.call_log}"
        )
        assert len(fake_pred.call_log) == 7, (
            f"Expected 7 successful calls, got {len(fake_pred.call_log)}"
        )

    def test_save_receives_correct_metadata(self, tmp_path, frames_dir):
        """Verify each save_ply call receives the correct f_px and (h, w)."""
        _, save_log = _make_sharp_modules()

        with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
            if "scripts.converters.video_to_3d_high_quality" in sys.modules:
                del sys.modules["scripts.converters.video_to_3d_high_quality"]

            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from scripts.converters.video_to_3d_high_quality import (
                convert_frames_to_3d,
            )

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            convert_frames_to_3d(
                frames_dir, output_dir, device="cpu", batch_size=4,
            )

        for entry in save_log:
            # Our mock load_rgb returns 480x640 images with f_px=500.0
            assert entry["f_px"] == 500.0
            assert entry["hw"] == (480, 640)

    def test_timing_output(self, tmp_path, frames_dir, capsys):
        """Verify that timing statistics are printed."""
        _make_sharp_modules()

        with mock.patch("torch.hub.load_state_dict_from_url", return_value={}):
            if "scripts.converters.video_to_3d_high_quality" in sys.modules:
                del sys.modules["scripts.converters.video_to_3d_high_quality"]

            project_root = str(Path(__file__).resolve().parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from scripts.converters.video_to_3d_high_quality import (
                convert_frames_to_3d,
            )

            output_dir = tmp_path / "output"
            output_dir.mkdir()

            convert_frames_to_3d(
                frames_dir, output_dir, device="cpu", batch_size=4,
            )

        captured = capsys.readouterr()
        assert "frames/sec" in captured.out or "frames/s" in captured.out


class TestPreprocessing:
    """Test the preprocessing helper in isolation."""

    def test_preprocess_output_shape(self):
        """_preprocess_image should produce [1, 3, 1536, 1536] tensor."""
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from scripts.converters.video_to_3d_high_quality import (
            _preprocess_image,
        )

        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        device = torch.device("cpu")

        img_resized, disp_factor, h, w = _preprocess_image(
            image, 500.0, device
        )

        assert img_resized.shape == (1, 3, 1536, 1536)
        assert disp_factor.shape == (1,)
        assert h == 480
        assert w == 640
        # disparity_factor = f_px / width = 500 / 640
        expected_disp = 500.0 / 640.0
        assert abs(disp_factor.item() - expected_disp) < 1e-5

    def test_build_intrinsics(self):
        """_build_intrinsics should produce correct K and K_resized."""
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from scripts.converters.video_to_3d_high_quality import (
            _build_intrinsics,
            INTERNAL_SHAPE,
        )

        device = torch.device("cpu")
        f_px, h, w = 500.0, 480, 640
        K, K_resized = _build_intrinsics(f_px, h, w, device)

        assert K.shape == (4, 4)
        assert K[0, 0].item() == f_px
        assert K[1, 1].item() == f_px
        assert K[0, 2].item() == w / 2
        assert K[1, 2].item() == h / 2

        # K_resized scales rows 0 and 1
        assert abs(
            K_resized[0, 0].item() - f_px * INTERNAL_SHAPE[0] / w
        ) < 1e-3


class TestArgparse:
    """Verify the CLI argument parser preserves backward compatibility."""

    def test_positional_args(self):
        """Old-style positional args should still parse correctly."""
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from scripts.converters.video_to_3d_high_quality import _build_parser

        parser = _build_parser()

        # Minimal: just the video file
        args = parser.parse_args(["myvideo.mp4"])
        assert args.video_file == "myvideo.mp4"
        assert args.device == "default"
        assert args.skip == 1
        assert args.batch_size == 4

        # All three positional args
        args = parser.parse_args(["myvideo.mp4", "mps", "5"])
        assert args.video_file == "myvideo.mp4"
        assert args.device == "mps"
        assert args.skip == 5
        assert args.batch_size == 4

    def test_batch_size_flag(self):
        """--batch-size flag should override the default."""
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from scripts.converters.video_to_3d_high_quality import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["myvideo.mp4", "--batch-size", "8"])
        assert args.batch_size == 8

    def test_all_args_combined(self):
        """Positional + --batch-size together."""
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from scripts.converters.video_to_3d_high_quality import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "myvideo.mp4", "cuda", "3", "--batch-size", "16",
        ])
        assert args.video_file == "myvideo.mp4"
        assert args.device == "cuda"
        assert args.skip == 3
        assert args.batch_size == 16
