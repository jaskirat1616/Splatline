#!/usr/bin/env python3
"""
Visualize 3D Gaussian Splats with 360° panoramic background.

This creates an immersive environment by placing your 3D Gaussians
inside a 360° photo/image sphere.
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
from PIL import Image
import urllib.request


def create_360_background(image_path: Optional[Path] = None, radius: float = 50.0,
                         rotation_offset: float = 0.0, vertical_offset: float = 0.0):
    """
    Create 360° panoramic background sphere.

    Args:
        image_path: Path to equirectangular 360° image. If None, downloads sample.
        radius: Radius of the background sphere
        rotation_offset: Horizontal rotation offset in degrees (0-360)
        vertical_offset: Vertical offset in degrees (-90 to 90)
    """

    # Load or download 360° image
    if image_path is None or not image_path.exists():
        print("📥 No 360° image provided, creating procedural environment...")
        print("💡 TIP: Download a 360° panorama and use --bg /path/to/image.jpg")
        print("   Free 360° images: https://polyhaven.com/hdris (download as JPG)")
        print("")
        create_gradient_sphere(radius)
        return

    # Load the 360° image
    try:
        # Check if it's an EXR file (HDR)
        if str(image_path).lower().endswith('.exr'):
            print("📸 Loading HDR/EXR panorama...")
            try:
                import OpenEXR
                import Imath

                exr_file = OpenEXR.InputFile(str(image_path))
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1

                # Read RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                channels = ['R', 'G', 'B']
                channel_data = [exr_file.channel(c, FLOAT) for c in channels]

                # Convert to numpy arrays
                pano_array = np.zeros((height, width, 3), dtype=np.float32)
                for i, data in enumerate(channel_data):
                    channel = np.frombuffer(data, dtype=np.float32)
                    pano_array[:, :, i] = channel.reshape(height, width)

                # Tone map HDR to LDR (simple Reinhard)
                pano_array = pano_array / (1.0 + pano_array)

                # Increase brightness for better visibility
                pano_array = np.power(pano_array, 0.7)  # Gamma correction

                print(f"✅ Loaded EXR panorama: {pano_array.shape}")

            except ImportError:
                print("⚠️  OpenEXR not installed. Installing...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'OpenEXR'])
                print("✅ Installed OpenEXR. Please run the script again.")
                return
        else:
            # Regular image (JPG, PNG)
            pano_image = Image.open(image_path)
            pano_array = np.array(pano_image)

            # Normalize to 0-1
            if pano_array.dtype == np.uint8:
                pano_array = pano_array / 255.0

            print(f"✅ Loaded 360° image: {pano_array.shape}")

    except Exception as e:
        print(f"❌ Error loading image: {e}")
        print(f"   Using procedural environment instead...")
        create_gradient_sphere(radius)
        return

    # Create sphere mesh with texture mapping
    # Using UV sphere with equirectangular texture mapping

    resolution = 400  # Ultra high resolution for seamless background

    positions = []
    colors = []

    # Generate sphere points
    for i in range(resolution):
        for j in range(resolution):
            # Spherical coordinates
            theta = (i / resolution) * 2 * np.pi  # 0 to 2π (longitude)
            phi = (j / resolution) * np.pi  # 0 to π (latitude)

            # Convert to Cartesian
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            positions.append([x, y, z])

            # Sample color from panorama image
            # Map theta (0-2π) to image width
            # Map phi (0-π) to image height
            img_x = int((theta / (2 * np.pi)) * pano_array.shape[1]) % pano_array.shape[1]
            img_y = int((phi / np.pi) * pano_array.shape[0]) % pano_array.shape[0]

            # Get RGB color from image
            pixel_color = pano_array[img_y, img_x]

            # Already normalized to 0-1
            color = pixel_color

            # Handle different array shapes
            if len(color.shape) == 0 or len(color) < 3:
                color = [color, color, color] if len(color.shape) == 0 else list(color) + [0] * (3 - len(color))

            colors.append(color[:3])  # RGB only

    # Log the 360° background sphere
    # Calculate optimal point size based on resolution to eliminate gaps
    point_size = (2 * np.pi * radius) / resolution * 1.2  # Slightly overlapping

    rr.log(
        "environment/360_background",
        rr.Points3D(
            positions=np.array(positions),
            colors=np.array(colors),
            radii=[point_size] * len(positions)  # Auto-sized to prevent gaps
        ),
        static=True
    )

    print(f"✅ Created 360° background with {len(positions)} points")


def create_gradient_sphere(radius: float = 50.0):
    """Fallback: Create beautiful procedural environment."""

    print("🎨 Creating procedural environment...")

    resolution = 100
    positions = []
    colors = []

    for i in range(resolution):
        for j in range(resolution):
            theta = (i / resolution) * 2 * np.pi
            phi = (j / resolution) * np.pi

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)

            positions.append([x, y, z])

            # Beautiful gradient: sunset/space theme
            # Top = deep space purple/blue
            # Horizon = orange/pink
            # Bottom = dark blue

            height_norm = (z / radius + 1.0) / 2.0  # 0 to 1

            # Create horizon glow effect
            horizon_distance = abs(height_norm - 0.5) * 2  # 0 at horizon, 1 at poles

            if height_norm > 0.5:  # Upper hemisphere - space
                # Deep blue to purple
                r = 0.1 + (1 - horizon_distance) * 0.4
                g = 0.05 + (1 - horizon_distance) * 0.3
                b = 0.2 + (1 - horizon_distance) * 0.4
            else:  # Lower hemisphere
                # Dark blue
                r = 0.05 + (1 - horizon_distance) * 0.3
                g = 0.08 + (1 - horizon_distance) * 0.35
                b = 0.15 + (1 - horizon_distance) * 0.4

            # Add horizon glow (orange/pink)
            if horizon_distance < 0.3:
                glow = (0.3 - horizon_distance) / 0.3
                r += glow * 0.5
                g += glow * 0.3
                b += glow * 0.1

            colors.append([min(r, 1.0), min(g, 1.0), min(b, 1.0)])

    rr.log(
        "environment/360_background",
        rr.Points3D(
            positions=np.array(positions),
            colors=np.array(colors),
            radii=[1.2] * len(positions)
        ),
        static=True
    )

    # Add stars
    num_stars = 300
    star_positions = []
    star_colors = []

    np.random.seed(42)
    for _ in range(num_stars):
        # Only in upper hemisphere (space part)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi * 0.6)  # Upper 60%

        r = radius * 0.95
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        star_positions.append([x, y, z])

        # White/yellow stars
        brightness = np.random.uniform(0.7, 1.0)
        star_colors.append([brightness, brightness, brightness * 0.9])

    rr.log(
        "environment/stars",
        rr.Points3D(
            positions=np.array(star_positions),
            colors=np.array(star_colors),
            radii=[0.3] * num_stars
        ),
        static=True
    )

    print("✅ Created procedural sunset/space environment with stars")


def visualize_with_360(input_path: Path, background_image: Optional[Path] = None,
                       max_frames: int = None, size_multiplier: float = 1.0):
    """Visualize Gaussians with 360° background."""

    # Find PLY files
    if input_path.is_file():
        ply_files = [input_path]
    else:
        ply_files = sorted(list(input_path.glob("*.ply")))
        if max_frames:
            ply_files = ply_files[:max_frames]

    if len(ply_files) == 0:
        print(f"❌ No PLY files found")
        return 1

    print(f"\n{'='*80}")
    print(f"🌍 360° IMMERSIVE GAUSSIAN VIEWER")
    print(f"{'='*80}\n")
    print(f"📁 Input: {input_path}")
    print(f"📊 Frames: {len(ply_files)}")
    if background_image:
        print(f"🖼️  Background: {background_image}")
    print()

    # Initialize Rerun
    rr.init("360° Gaussian Splat Viewer", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Create 360° background (large radius so it surrounds everything)
    create_360_background(background_image, radius=200.0)

    # Setup blueprint
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Spatial3DView(
            name="🌍 360° Immersive View",
            origin="world",
            background=[0.0, 0.0, 0.0],  # Black background
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    # Process frames
    print("🔄 Loading Gaussian splats...")

    for idx, ply_path in enumerate(ply_files):
        rr.set_time_sequence("frame", idx)

        # Load Gaussian data
        gaussians, metadata = load_ply(ply_path)

        positions = gaussians.mean_vectors.cpu().numpy().squeeze()
        colors = gaussians.colors.cpu().numpy().squeeze()
        scales = gaussians.singular_values.cpu().numpy().squeeze()
        opacities = gaussians.opacities.cpu().numpy().squeeze()

        # Convert colors if needed
        if metadata.color_space == "linearRGB":
            import torch
            colors_torch = torch.from_numpy(colors).float()
            colors = cs_utils.linearRGB2sRGB(colors_torch).numpy()

        # Filter by opacity
        opacity_mask = opacities > 0.1
        positions = positions[opacity_mask]
        colors = colors[opacity_mask]
        scales = scales[opacity_mask]

        # Log Gaussians
        rr.log(
            "world/gaussians",
            rr.Points3D(
                positions=positions,
                colors=colors,
                radii=np.mean(scales, axis=1) * 0.3 * size_multiplier
            )
        )

        print(f"  Frame {idx+1}/{len(ply_files)}: {len(positions):,} points")

    print(f"\n{'='*80}")
    print(f"✨ 360° VIEWER READY!")
    print(f"{'='*80}\n")
    print("🎮 CONTROLS:")
    print("  🔄 Rotate:  Left click + drag")
    print("  ↔️  Pan:     Right click + drag")
    print("  🔍 Zoom:    Mouse wheel")
    print("  ⏯️  Timeline: Use bottom slider\n")
    print("💡 TIP: Rotate around to see the full 360° environment!\n")

    # Keep running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Gaussians with 360° panoramic background",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use sample 360° background (auto-downloaded)
  python visualize_with_360_background.py -i gaussians/

  # Use your own 360° panorama
  python visualize_with_360_background.py -i gaussians/ --bg my_360_photo.jpg

  # Single frame with custom background
  python visualize_with_360_background.py -i frame_000000.ply --bg nature_360.jpg

Supported 360° image formats:
  - Equirectangular (most common 360° photo format)
  - JPG, PNG
  - Any resolution (will be sampled)

Where to get 360° images:
  - Your own 360° camera photos
  - Wikimedia Commons 360° category
  - Free stock photo sites with 360° filter
  - Google Street View downloads
        """
    )

    parser.add_argument('-i', '--input', type=Path, required=True,
                       help='PLY file or directory with PLY files')
    parser.add_argument('--bg', '--background', type=Path, dest='background',
                       help='Path to 360° equirectangular image (optional)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to load')
    parser.add_argument('--size', type=float, default=1.0,
                       help='Point size multiplier (default: 1.0)')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Error: {args.input} does not exist")
        return 1

    return visualize_with_360(args.input, args.background, args.max_frames, args.size)


if __name__ == "__main__":
    import sys
    sys.exit(main())
