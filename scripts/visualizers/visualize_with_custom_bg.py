#!/usr/bin/env python3
"""
Visualize Gaussian Splats with custom background image in Rerun.
Sets the viewport background to your image - no 3D sphere!
"""

import argparse
from pathlib import Path
import numpy as np
import rerun as rr
from sharp.utils.gaussians import load_ply
from sharp.utils import color_space as cs_utils
from PIL import Image
import torch


def visualize_with_background(input_path: Path, background_image: Path = None,
                              max_frames: int = None, size_multiplier: float = 1.0):
    """Visualize Gaussians with custom background."""

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
    print(f"🎨 GAUSSIAN VIEWER WITH CUSTOM BACKGROUND")
    print(f"{'='*80}\n")
    print(f"📁 Input: {input_path}")
    print(f"📊 Frames: {len(ply_files)}")
    if background_image:
        print(f"🖼️  Background: {background_image}")
    print()

    # Initialize Rerun
    rr.init("Gaussian Splat Viewer", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Load background image if provided
    bg_color = None
    if background_image and background_image.exists():
        try:
            # Check if EXR file
            if str(background_image).lower().endswith('.exr'):
                print("📸 Loading EXR/HDR background...")
                import OpenEXR
                import Imath

                exr_file = OpenEXR.InputFile(str(background_image))
                header = exr_file.header()
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1

                # Read RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                channels = ['R', 'G', 'B']
                channel_data = [exr_file.channel(c, FLOAT) for c in channels]

                # Convert to numpy
                img_array = np.zeros((height, width, 3), dtype=np.float32)
                for i, data in enumerate(channel_data):
                    channel = np.frombuffer(data, dtype=np.float32)
                    img_array[:, :, i] = channel.reshape(height, width)

                # Tone map HDR to LDR
                img_array = img_array / (1.0 + img_array)
                img_array = np.power(np.clip(img_array, 0, 1), 0.7)

                # Get average color
                avg_color = np.mean(img_array, axis=(0, 1))
                bg_color = avg_color.tolist()

                # Convert to uint8 for display
                img_display = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

            else:
                # Regular image (JPG, PNG)
                from PIL import Image
                img = Image.open(background_image)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_array = np.array(img)
                avg_color = np.mean(img_array, axis=(0, 1)) / 255.0
                bg_color = avg_color.tolist()
                img_display = img_array

            print(f"✅ Loaded background image: {width}x{height}")
            print(f"✅ Background color: RGB({bg_color[0]:.2f}, {bg_color[1]:.2f}, {bg_color[2]:.2f})")

            # Log the image as a 2D view
            rr.log("background_image", rr.Image(img_display), static=True)

        except ImportError:
            print("⚠️  OpenEXR not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'OpenEXR'])
            print("✅ Installed. Please run again.")
            return 1
        except Exception as e:
            print(f"⚠️  Could not load background image: {e}")
            bg_color = [0.1, 0.1, 0.15]  # Dark blue fallback
    else:
        bg_color = [0.1, 0.1, 0.15]  # Dark blue default

    # Setup blueprint with background color
    blueprint = rr.blueprint.Blueprint(
        rr.blueprint.Horizontal(
            rr.blueprint.Spatial3DView(
                name="3D View",
                origin="world",
                background=bg_color,  # Set background color here!
            ),
            rr.blueprint.Spatial2DView(
                name="Background Image",
                origin="background_image"
            ) if background_image else None,
            column_shares=[3, 1] if background_image else [1]
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
    print(f"✨ VIEWER READY!")
    print(f"{'='*80}\n")
    print("🎮 The 3D viewport background is set to your image's average color")
    print("🖼️  The full image is shown in the side panel")
    print("\n")

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
        description="Visualize Gaussians with custom background"
    )

    parser.add_argument('-i', '--input', type=Path, required=True,
                       help='PLY file or directory with PLY files')
    parser.add_argument('--bg', type=Path,
                       help='Background image (JPG, PNG, EXR)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to load')
    parser.add_argument('--size', type=float, default=1.0,
                       help='Point size multiplier')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ Error: {args.input} does not exist")
        return 1

    return visualize_with_background(args.input, args.bg, args.max_frames, args.size)


if __name__ == "__main__":
    import sys
    sys.exit(main())
