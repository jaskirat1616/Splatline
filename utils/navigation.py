"""
Navigation utilities for obstacle detection and occupancy grid generation.
"""

import numpy as np
from scipy.spatial import KDTree


def extract_ground_plane(positions, height_threshold=0.3, grid_size=5.0):
    """
    Detect ground plane for navigation.
    
    Args:
        positions: Array of 3D positions (N, 3)
        height_threshold: Threshold for ground detection
        grid_size: Size of grid cells for ground plane estimation
    
    Returns:
        ground_points: Array of ground plane points (M, 3)
        is_ground: Boolean array indicating ground points (N,)
        height_map: 2D height map (nx, nz)
        ground_info: Tuple of (min_x, max_x, min_z, max_z, grid_size)
    """
    # Create 2D grid (XZ plane)
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_z, max_z = positions[:, 2].min(), positions[:, 2].max()

    # Grid resolution
    nx = int((max_x - min_x) / grid_size) + 1
    nz = int((max_z - min_z) / grid_size) + 1

    # Find lowest points in each grid cell (likely ground) — vectorized
    height_map = np.full((nx, nz), np.inf)

    # Compute all grid indices at once (int() truncates toward zero = np.trunc then cast)
    i_all = np.trunc((positions[:, 0] - min_x) / grid_size).astype(np.intp)
    j_all = np.trunc((positions[:, 2] - min_z) / grid_size).astype(np.intp)

    # Bounds mask
    valid = (i_all >= 0) & (i_all < nx) & (j_all >= 0) & (j_all < nz)
    i_v = i_all[valid]
    j_v = j_all[valid]
    y_v = positions[valid, 1]

    # Scatter minimum height per cell
    np.minimum.at(height_map, (i_v, j_v), y_v)

    # Create ground plane mesh — vectorized
    finite_mask = np.isfinite(height_map)
    ii, jj = np.nonzero(finite_mask)
    if ii.size > 0:
        gx = min_x + ii.astype(np.float64) * grid_size
        gz = min_z + jj.astype(np.float64) * grid_size
        gy = height_map[ii, jj]
        ground_points = np.column_stack([gx, gy, gz])
    else:
        ground_points = np.empty((0, 3))
    
    # Identify points near ground
    median_y = np.median(positions[:, 1]) if len(positions) > 0 else 0.0
    is_ground = np.abs(positions[:, 1] - median_y) < height_threshold
    
    return ground_points, is_ground, height_map, (min_x, max_x, min_z, max_z, grid_size)


def detect_obstacles(positions, ground_mask, min_height=0.5, cluster_dist=2.0):
    """
    Detect obstacles (non-ground points above certain height).
    
    Args:
        positions: Array of 3D positions (N, 3)
        ground_mask: Boolean array indicating ground points (N,)
        min_height: Minimum height above ground for obstacle
        cluster_dist: Distance threshold for clustering obstacles
    
    Returns:
        obstacle_points: Array of obstacle points (M, 3)
        clusters: List of obstacle cluster bounding boxes
    """
    # Filter to non-ground points
    obstacle_points = positions[~ground_mask]
    
    if len(obstacle_points) == 0:
        return obstacle_points, []
    
    # Filter by height (ignore very low points)
    ground_level = np.median(positions[ground_mask, 1]) if ground_mask.sum() > 0 else positions[:, 1].min()
    height_mask = (obstacle_points[:, 1] - ground_level) > min_height
    obstacle_points = obstacle_points[height_mask]
    
    if len(obstacle_points) == 0:
        return obstacle_points, []
    
    # Simple clustering by proximity
    tree = KDTree(obstacle_points)
    clusters = []
    visited = np.zeros(len(obstacle_points), dtype=bool)
    
    for i in range(len(obstacle_points)):
        if visited[i]:
            continue
        
        # Find nearby points
        indices = tree.query_ball_point(obstacle_points[i], cluster_dist)
        visited[indices] = True
        
        cluster_points = obstacle_points[indices]
        
        # Calculate bounding box
        bbox = {
            'min': cluster_points.min(axis=0).tolist(),
            'max': cluster_points.max(axis=0).tolist(),
            'center': cluster_points.mean(axis=0).tolist(),
            'size': (cluster_points.max(axis=0) - cluster_points.min(axis=0)).tolist(),
            'num_points': len(cluster_points)
        }
        
        clusters.append(bbox)
    
    return obstacle_points, clusters


def compute_occupancy_grid_2d(positions, ground_info, resolution=1.0, obstacle_height=0.5):
    """
    Create 2D occupancy grid (bird's eye view) for navigation.
    
    Args:
        positions: Array of 3D positions (N, 3)
        ground_info: Tuple of (min_x, max_x, min_z, max_z, grid_size)
        resolution: Resolution of occupancy grid in meters
        obstacle_height: Minimum height for obstacle detection
    
    Returns:
        occupancy_grid: 2D array where 0=free, 1=occupied (nx, nz)
        grid_info: Tuple of (min_x, max_x, min_z, max_z, resolution)
    """
    min_x, max_x, min_z, max_z, _ = ground_info
    
    # Grid dimensions
    nx = int((max_x - min_x) / resolution) + 1
    nz = int((max_z - min_z) / resolution) + 1
    
    occupancy_grid = np.zeros((nx, nz))

    # Mark occupied cells — vectorized
    ground_level = positions[:, 1].min()

    # Height filter: skip low points (on ground)
    above = positions[:, 1] >= (ground_level + obstacle_height)
    pos_above = positions[above]

    if pos_above.size > 0:
        i_all = np.trunc((pos_above[:, 0] - min_x) / resolution).astype(np.intp)
        j_all = np.trunc((pos_above[:, 2] - min_z) / resolution).astype(np.intp)

        # Bounds mask
        valid = (i_all >= 0) & (i_all < nx) & (j_all >= 0) & (j_all < nz)
        occupancy_grid[i_all[valid], j_all[valid]] = 1

    return occupancy_grid, (min_x, max_x, min_z, max_z, resolution)

