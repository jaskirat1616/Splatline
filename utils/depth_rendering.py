"""
Depth map rendering utilities for 3D Gaussian Splatting.
"""

import numpy as np
import cv2


def render_depth_map(positions, colors, resolution=(1280, 720), focal_length=1000, max_depth=100):
    """
    Render depth map and colored depth visualization from 3D points.
    
    Args:
        positions: Array of 3D positions (N, 3)
        colors: Array of RGB colors (N, 3)
        resolution: Tuple of (width, height)
        focal_length: Camera focal length in pixels
        max_depth: Maximum depth value
    
    Returns:
        depth_map: Grayscale depth map (H, W)
        depth_colored: Colored depth visualization (H, W, 3)
    """
    width, height = resolution

    # Initialize buffers
    depth_map = np.full((height, width), max_depth, dtype=np.float32)
    color_map = np.zeros((height, width, 3), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.int32)

    # Vectorized projection of all points to image plane
    positions = np.asarray(positions)
    colors = np.asarray(colors)

    if positions.size == 0:
        # No points to process — skip to averaging
        pass
    else:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        # Mask: skip points behind camera or too far
        valid = (z > 0) & (z <= max_depth)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        col = colors[valid]

        # Project to image coordinates
        # int() truncates toward zero in Python, which matches np.trunc().astype(int)
        u_f = focal_length * x / z + width / 2
        v_f = height / 2 - focal_length * y / z
        u = np.trunc(u_f).astype(np.intp)
        v = np.trunc(v_f).astype(np.intp)

        # Bounds mask
        bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[bounds]
        v = v[bounds]
        z = z[bounds]
        col = col[bounds]

        # Depth test: np.minimum.at for per-pixel closest depth
        np.minimum.at(depth_map, (v, u), z)

        # Accumulate color (ALL valid in-bounds points, regardless of depth test — matches original)
        np.add.at(color_map, (v, u), col.astype(np.float32))
        np.add.at(count_map, (v, u), 1)
    
    # Average colors
    mask = count_map > 0
    color_map[mask] = color_map[mask] / count_map[mask, np.newaxis]
    
    # Create colored depth visualization
    depth_normalized = np.clip(depth_map / max_depth, 0, 1)
    depth_colored = cv2.applyColorMap(
        (depth_normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Mark invalid regions as black
    invalid_mask = depth_map >= max_depth
    depth_colored[invalid_mask] = [0, 0, 0]
    
    # No flip needed - depth map should match video frame orientation directly
    return depth_map, depth_colored


def depth_map_to_3d_points(depth_map, depth_colored, focal_length, width, height):
    """
    Convert depth map to 3D point cloud for visualization.
    
    Args:
        depth_map: Depth map (H, W)
        depth_colored: Colored depth visualization (H, W, 3)
        focal_length: Camera focal length in pixels
        width: Image width
        height: Image height
    
    Returns:
        depth_points: 3D points (N, 3)
        depth_colors: RGB colors (N, 3)
    """
    # Sample points from depth map (every Nth pixel for performance)
    step = max(1, min(width, height) // 200)  # Adjust for performance

    # Get valid depth range
    depth_max = depth_map.max()

    # Build sampled grid with meshgrid
    v_coords = np.arange(0, height, step)
    u_coords = np.arange(0, width, step)
    uu, vv = np.meshgrid(u_coords, v_coords)  # uu shape (len_v, len_u), same for vv
    zz = depth_map[vv, uu]

    # Mask invalid depth
    valid = (zz < depth_max * 0.99) & (zz > 0)

    uu_valid = uu[valid]
    vv_valid = vv[valid]
    zz_valid = zz[valid]

    if zz_valid.size == 0:
        return np.array([]), np.array([])

    # Convert pixel to 3D coordinates
    x = (uu_valid - width / 2) * zz_valid / focal_length
    y = (height / 2 - vv_valid) * zz_valid / focal_length
    depth_points = np.column_stack([x, y, zz_valid])

    # Get colors from depth colored image, normalized to [0,1]
    depth_colors = depth_colored[vv_valid, uu_valid].astype(np.float64) / 255.0

    return depth_points, depth_colors

