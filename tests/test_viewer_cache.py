#!/usr/bin/env python3
"""
Tests for the video_complete_viewer caching, downsampling, and worker logic.

Exercises the cache write/read cycle with a monkeypatched
``process_frame_complete`` so we never need real PLY files or GPU libs.
"""

import os
import sys
import tempfile
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Ensure project root is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Pre-populate sys.modules with stubs for ALL heavy dependencies so that
# importing any module in the project tree never triggers a real import of
# PIL, torch, rerun, sharp, cv2, etc.
#
# We must stub the *utils* sub-packages too because utils/__init__.py eagerly
# re-exports from them, and that would pull in sharp / torch / PIL.
# ---------------------------------------------------------------------------

_STUB_MODULES = [
    # Third-party libs that aren't installed in the test env
    "PIL", "PIL.Image",
    "torch",
    "rerun",
    "cv2",
    "sharp", "sharp.utils", "sharp.utils.gaussians",
    "sharp.utils.color_space",
    # Project-internal modules that have heavy transitive deps
    "utils",
    "utils.frame_processing",
    "utils.depth_rendering",
    "utils.navigation",
    "utils.visualization",
    "utils.pathfinding",
    "utils.config",
    "utils.io_utils",
    "utils.geometry",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        # Mark package-like stubs with __path__ so sub-import resolution works
        if "." not in _mod_name or _mod_name.count(".") == 0:
            _m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _m

# Provide dummy callables on the stubs that the worker function imports.
_fp_mod = sys.modules["utils.frame_processing"]
_fp_mod.process_frame_complete = None  # type: ignore[attr-defined]  # will be patched per-test

_dr_mod = sys.modules["utils.depth_rendering"]
_dr_mod.depth_map_to_3d_points = None  # type: ignore[attr-defined]

# Now importing the viewer module will skip heavy deps and find the stubs.
from scripts.visualizers.video_complete_viewer import (
    _cache_path_for,
    _config_hash,
    _downsample,
    _preprocess_frame,
    _try_load_cache,
    _write_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canned_process_result(frame_idx: int = 0, n_points: int = 200) -> dict:
    """Return a dict that mimics ``process_frame_complete`` output."""
    rng = np.random.RandomState(frame_idx)
    positions = rng.randn(n_points, 3).astype(np.float32)
    colors = rng.rand(n_points, 3).astype(np.float32)
    scales = np.abs(rng.randn(n_points, 3)).astype(np.float32)
    return {
        "frame": frame_idx,
        "positions": positions,
        "colors": colors,
        "scales": scales,
        "ground_points": positions[:10],
        "obstacle_points": positions[10:20],
        "ground_level": 0.0,
        "occupancy_grid": rng.randint(0, 2, size=(50, 50)).astype(np.int32),
        "depth_map": rng.rand(72, 128).astype(np.float32),
        "depth_colored": (rng.rand(72, 128, 3) * 255).astype(np.uint8),
        "video_frame": (rng.rand(72, 128, 3) * 255).astype(np.uint8),
        "focal_length": 500.0,
        "depth_width": 128,
        "depth_height": 72,
    }


def _canned_depth_3d(depth_map, depth_colored, focal_length, width, height):
    """Fake ``depth_map_to_3d_points`` that returns small deterministic arrays."""
    n = 30
    rng = np.random.RandomState(42)
    return rng.randn(n, 3).astype(np.float32), rng.rand(n, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: _config_hash
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_deterministic(self):
        h1 = _config_hash(0.5, 0.5, 500_000)
        h2 = _config_hash(0.5, 0.5, 500_000)
        assert h1 == h2

    def test_changes_with_params(self):
        h1 = _config_hash(0.5, 0.5, 500_000)
        h2 = _config_hash(0.6, 0.5, 500_000)
        h3 = _config_hash(0.5, 0.3, 500_000)
        h4 = _config_hash(0.5, 0.5, 100_000)
        assert len({h1, h2, h3, h4}) == 4


# ---------------------------------------------------------------------------
# Tests: _downsample
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_noop_when_under_budget(self):
        pos = np.ones((100, 3))
        col = np.ones((100, 3))
        scl = np.ones((100, 3))
        p, c, s = _downsample(pos, col, scl, budget=200)
        assert len(p) == 100

    def test_reduces_to_budget(self):
        n = 1_000
        pos = np.random.randn(n, 3)
        col = np.random.rand(n, 3)
        scl = np.random.rand(n, 3)
        p, c, s = _downsample(pos, col, scl, budget=100)
        assert len(p) == 100
        assert len(c) == 100
        assert len(s) == 100

    def test_consistency(self):
        """Same indices applied to all three arrays."""
        n = 500
        pos = np.arange(n * 3).reshape(n, 3).astype(float)
        col = pos * 2
        scl = pos * 3
        np.random.seed(99)
        p, c, s = _downsample(pos, col, scl, budget=50)
        np.testing.assert_array_equal(c, p * 2)
        np.testing.assert_array_equal(s, p * 3)


# ---------------------------------------------------------------------------
# Tests: cache write / read cycle
# ---------------------------------------------------------------------------

class TestCacheWriteRead:
    def _make_fake_ply(self, tmpdir: Path, name: str = "frame_000001.ply") -> Path:
        ply = tmpdir / name
        ply.write_text("fake ply content")
        return ply

    def test_roundtrip(self, tmp_path):
        ply = self._make_fake_ply(tmp_path)
        cfg = _config_hash(0.5, 0.5, 500_000)

        data = {
            "depth_map": np.random.rand(10, 10).astype(np.float32),
            "depth_colored": np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8),
            "occupancy_grid": np.random.randint(0, 2, (20, 20)).astype(np.int32),
            "ground_points": np.random.randn(5, 3).astype(np.float32),
            "obstacle_points": np.random.randn(3, 3).astype(np.float32),
            "depth_3d_points": np.random.randn(15, 3).astype(np.float32),
            "depth_3d_colors": np.random.rand(15, 3).astype(np.float32),
            "positions_downsampled": np.random.randn(50, 3).astype(np.float32),
            "colors_downsampled": np.random.rand(50, 3).astype(np.float32),
            "scales_downsampled": np.random.rand(50, 3).astype(np.float32),
            "focal_length": np.array(500.0),
            "depth_width": np.array(128),
            "depth_height": np.array(72),
            "mean_pos": np.array([1.0, 2.0, 3.0]),
            "std_pos": np.array([0.5, 0.5, 0.5]),
            "depth_mean_pos": np.array([0.0, 0.0, 5.0]),
            "depth_std_pos": np.array([1.0, 1.0, 1.0]),
        }

        _write_cache(ply, None, cfg, data)

        cache_file = _cache_path_for(ply, None)
        assert cache_file.exists()
        assert cache_file.name == "frame_000001.splatline_cache.npz"

        loaded = _try_load_cache(ply, None, cfg)
        assert loaded is not None

        for key in data:
            np.testing.assert_array_almost_equal(loaded[key], data[key], decimal=5,
                                                  err_msg=f"Mismatch in key '{key}'")

    def test_stale_cache_rejected(self, tmp_path):
        ply = self._make_fake_ply(tmp_path)
        cfg = _config_hash(0.5, 0.5, 500_000)

        data = {k: np.zeros(1) for k in
                ["depth_map", "depth_colored", "occupancy_grid",
                 "ground_points", "obstacle_points",
                 "depth_3d_points", "depth_3d_colors",
                 "positions_downsampled", "colors_downsampled", "scales_downsampled",
                 "focal_length", "depth_width", "depth_height",
                 "mean_pos", "std_pos", "depth_mean_pos", "depth_std_pos"]}

        _write_cache(ply, None, cfg, data)

        time.sleep(0.05)
        ply.write_text("updated ply content")

        loaded = _try_load_cache(ply, None, cfg)
        assert loaded is None

    def test_wrong_config_rejected(self, tmp_path):
        ply = self._make_fake_ply(tmp_path)
        cfg_old = _config_hash(0.5, 0.5, 500_000)
        cfg_new = _config_hash(0.5, 0.5, 100_000)

        data = {k: np.zeros(1) for k in
                ["depth_map", "depth_colored", "occupancy_grid",
                 "ground_points", "obstacle_points",
                 "depth_3d_points", "depth_3d_colors",
                 "positions_downsampled", "colors_downsampled", "scales_downsampled",
                 "focal_length", "depth_width", "depth_height",
                 "mean_pos", "std_pos", "depth_mean_pos", "depth_std_pos"]}

        _write_cache(ply, None, cfg_old, data)
        loaded = _try_load_cache(ply, None, cfg_new)
        assert loaded is None

    def test_custom_cache_dir(self, tmp_path):
        ply_dir = tmp_path / "plys"
        ply_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        ply = self._make_fake_ply(ply_dir)
        cfg = _config_hash(0.5, 0.5, 500_000)

        data = {k: np.zeros(1) for k in
                ["depth_map", "depth_colored", "occupancy_grid",
                 "ground_points", "obstacle_points",
                 "depth_3d_points", "depth_3d_colors",
                 "positions_downsampled", "colors_downsampled", "scales_downsampled",
                 "focal_length", "depth_width", "depth_height",
                 "mean_pos", "std_pos", "depth_mean_pos", "depth_std_pos"]}

        _write_cache(ply, cache_dir, cfg, data)

        assert (cache_dir / "frame_000001.splatline_cache.npz").exists()
        assert not (ply_dir / "frame_000001.splatline_cache.npz").exists()

        loaded = _try_load_cache(ply, cache_dir, cfg)
        assert loaded is not None


# ---------------------------------------------------------------------------
# Tests: _preprocess_frame with monkeypatched imports
# ---------------------------------------------------------------------------

class TestPreprocessFrame:
    """Test the worker function with mocked heavy dependencies.

    We patch the stub modules in sys.modules directly (they were inserted
    at the top of this file) so that when ``_preprocess_frame`` does
    ``from utils.frame_processing import process_frame_complete`` it
    picks up our fakes.
    """

    def _make_ply(self, tmpdir: Path, idx: int) -> Path:
        ply = tmpdir / f"frame_{idx:06d}.ply"
        ply.write_text("fake")
        return ply

    def _patch_worker_deps(self, canned_result, depth_fn=_canned_depth_3d):
        """Return a context manager that patches the stub modules."""
        fp_mod = sys.modules["utils.frame_processing"]
        dr_mod = sys.modules["utils.depth_rendering"]

        class _Ctx:
            def __enter__(self_ctx):
                self_ctx._orig_pfc = getattr(fp_mod, "process_frame_complete", None)
                self_ctx._orig_d3d = getattr(dr_mod, "depth_map_to_3d_points", None)
                if callable(canned_result):
                    fp_mod.process_frame_complete = canned_result
                else:
                    fp_mod.process_frame_complete = lambda *a, **kw: canned_result
                dr_mod.depth_map_to_3d_points = depth_fn
                return self_ctx

            def __exit__(self_ctx, *exc):
                fp_mod.process_frame_complete = self_ctx._orig_pfc
                dr_mod.depth_map_to_3d_points = self_ctx._orig_d3d

        return _Ctx()

    def test_cache_miss_then_hit(self, tmp_path):
        """First call computes; second call hits cache and produces identical output."""
        ply = self._make_ply(tmp_path, 42)
        cfg = _config_hash(0.5, 0.5, 500_000)
        canned = _canned_process_result(42, n_points=200)

        with self._patch_worker_deps(canned):
            args = (
                0, str(ply), None, 0.5, 0.5,
                500_000, cfg, True, None,
            )

            # --- Cache miss ---
            result1 = _preprocess_frame(args)
            assert result1 is not None
            assert result1["_cache_hit"] is False

            cache_file = _cache_path_for(ply, None)
            assert cache_file.exists()

            # --- Cache hit ---
            result2 = _preprocess_frame(args)
            assert result2 is not None
            assert result2["_cache_hit"] is True

            for key in ["depth_map", "depth_colored", "occupancy_grid",
                        "positions_downsampled", "colors_downsampled",
                        "scales_downsampled", "depth_3d_points", "depth_3d_colors",
                        "mean_pos", "std_pos"]:
                np.testing.assert_array_almost_equal(
                    result1[key], result2[key], decimal=5,
                    err_msg=f"Cache mismatch on '{key}'"
                )

    def test_no_cache_flag(self, tmp_path):
        """With use_cache=False, no cache file should be written."""
        ply = self._make_ply(tmp_path, 7)
        cfg = _config_hash(0.5, 0.5, 500_000)
        canned = _canned_process_result(7)

        with self._patch_worker_deps(canned):
            args = (0, str(ply), None, 0.5, 0.5, 500_000, cfg, False, None)
            result = _preprocess_frame(args)
            assert result is not None

        cache_file = _cache_path_for(ply, None)
        assert not cache_file.exists()

    def test_worker_handles_none_from_processing(self, tmp_path):
        """If process_frame_complete returns None, worker returns None gracefully."""
        ply = self._make_ply(tmp_path, 99)
        cfg = _config_hash(0.5, 0.5, 500_000)

        with self._patch_worker_deps(None):
            args = (0, str(ply), None, 0.5, 0.5, 500_000, cfg, False, None)
            result = _preprocess_frame(args)
            assert result is None

    def test_worker_handles_exception(self, tmp_path):
        """If process_frame_complete raises, worker returns None with a warning."""
        ply = self._make_ply(tmp_path, 99)
        cfg = _config_hash(0.5, 0.5, 500_000)

        def _boom(*a, **kw):
            raise RuntimeError("corrupt PLY")

        with self._patch_worker_deps(_boom):
            args = (0, str(ply), None, 0.5, 0.5, 500_000, cfg, False, None)
            with pytest.warns(UserWarning, match="Worker failed"):
                result = _preprocess_frame(args)
            assert result is None


# ---------------------------------------------------------------------------
# Tests: point budget downsampling in worker
# ---------------------------------------------------------------------------

class TestPointBudgetInWorker:
    def test_large_cloud_gets_downsampled(self, tmp_path):
        ply = tmp_path / "frame_000001.ply"
        ply.write_text("fake")
        cfg = _config_hash(0.5, 0.5, 50)
        canned = _canned_process_result(1, n_points=500)

        fp_mod = sys.modules["utils.frame_processing"]
        dr_mod = sys.modules["utils.depth_rendering"]
        orig_pfc = getattr(fp_mod, "process_frame_complete", None)
        orig_d3d = getattr(dr_mod, "depth_map_to_3d_points", None)
        try:
            fp_mod.process_frame_complete = lambda *a, **kw: canned
            dr_mod.depth_map_to_3d_points = _canned_depth_3d

            args = (0, str(ply), None, 0.5, 0.5, 50, cfg, False, None)
            result = _preprocess_frame(args)
        finally:
            fp_mod.process_frame_complete = orig_pfc
            dr_mod.depth_map_to_3d_points = orig_d3d

        assert result is not None
        assert len(result["positions_downsampled"]) == 50
        assert len(result["colors_downsampled"]) == 50
        assert len(result["scales_downsampled"]) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
