# Apple ML-SHARP Rerun

Convert 2D videos and photos into interactive 3D scenes using ML-SHARP and Rerun. Explore your videos in 3D space with depth maps, navigation tools, and creative effects.

## Demo Video

![Demo Video Preview](docs/assets/demo_preview.gif)

**[Click here to download the full demo video](docs/assets/demo_video.mov)** | [View thumbnail](docs/assets/demo_thumbnail.jpg)

---

## 🎯 For Non-Technical Users - Quick Start

**New to coding? No problem!** Follow these steps to turn your videos and photos into 3D scenes.

### What You Need:
1. A computer (Mac, Windows, or Linux)
2. Python installed (download from [python.org](https://www.python.org/downloads/))
3. Your video file (MP4, MOV, AVI) or photo (JPG, PNG)

### Step-by-Step Guide:

#### **Convert a Video to 3D (Recommended)**

1. **Open Terminal (Mac/Linux) or Command Prompt (Windows)**
2. **Navigate to this folder** (where you downloaded this project)
   ```bash
   cd /path/to/Apple-ml-sharp-rerun
   ```
3. **Install required software** (run these commands):
   ```bash
   pip install rerun-sdk numpy pillow opencv-python scipy torch tqdm
   pip install sharp
   ```
4. **Convert your video to 3D:**
   ```bash
   python scripts/converters/video_to_3d_high_quality.py your_video.mp4 mps
   ```
   *Replace `your_video.mp4` with your actual video filename*

5. **Wait for processing** - This takes a few minutes depending on video length. The ML-SHARP model (~2.5GB) downloads automatically on first use.

6. **View your 3D scene:**
   ```bash
   python scripts/visualizers/video_complete_viewer.py -i output_your_video/gaussians/
   ```

#### **View an Existing 3D File (.ply)**

If you already have a `.ply` file:

1. **Open Terminal/Command Prompt**
2. **Navigate to this folder**
3. **View the 3D file:**
   ```bash
   python scripts/visualizers/visualize_with_rerun.py -i path/to/your/file.ply
   ```

### Controls in the 3D Viewer:

- **Left Click + Drag**: Rotate the view
- **Right Click + Drag**: Pan/move the view
- **Scroll Wheel**: Zoom in/out
- **Double Click**: Reset view

### Tips:
- Start with short videos (10-30 seconds) for faster processing
- Make sure your video has good lighting and clear objects
- The first run downloads the ML-SHARP model (~2.5GB)

### Need Help?
Check the [Troubleshooting](#-troubleshooting) section below.

---

## 🚀 Setup Guide

### Prerequisites

1. **Python 3.8 or higher** - Check with: `python --version` or `python3 --version`
2. **pip** (Python package manager) - Usually comes with Python
3. **Git** (optional, for cloning the repository)

### Installation Steps

#### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install rerun-sdk numpy pillow opencv-python scipy torch tqdm
```

#### Step 2: Install ML-SHARP

**ML-SHARP (Sparse Hierarchical Attention-based Radiance Prediction)** is Apple's model for converting 2D images/videos into 3D Gaussian Splatting scenes. Official repository: [apple/ml-sharp](https://github.com/apple/ml-sharp)

##### Option A: Install via pip (Recommended)

```bash
pip install sharp
```

The ML-SHARP model weights (~2.5GB) will be downloaded automatically when you run the video conversion script. The model is hosted at:
- Model URL: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`

##### Option B: Install from Source (GitHub)

If you want to install from the official GitHub repository:

1. Clone the ML-SHARP repository:
   ```bash
   git clone https://github.com/apple/ml-sharp.git
   cd ml-sharp
   ```

2. Create a conda environment (recommended by ML-SHARP):
   ```bash
   conda create -n sharp python=3.13
   conda activate sharp
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package:
   ```bash
   pip install -e .
   ```

**Note:** The ML-SHARP Python package (`sharp`) provides:
- `sharp.models` - Model definitions and predictor creation (`PredictorParams`, `create_predictor`)
- `sharp.utils.gaussians` - Gaussian Splatting utilities (`load_ply`, `save_ply`, `unproject_gaussians`)
- `sharp.utils.io` - Image I/O utilities
- `sharp.utils.color_space` - Color space conversions

**ML-SHARP CLI:** You can also use the official ML-SHARP CLI:
```bash
# Convert images to 3D Gaussian Splats
sharp predict -i /path/to/input/images -o /path/to/output/gaussians

# Test installation
sharp --help
```

#### Step 3: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import rerun; import numpy; import torch; import sharp; print('✓ All dependencies installed!')"
```

You should see: `✓ All dependencies installed!`

#### Step 4: Test with Sample Data

If you have sample `.ply` files in `output_test/`, try visualizing one:

```bash
python scripts/visualizers/visualize_with_rerun.py -i output_test/IMG_4707.ply --size 2.0
```

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster processing
  - **CUDA** (NVIDIA GPUs on Windows/Linux)
  - **MPS** (Apple Silicon Macs - M1, M2, M3, etc.)
  - **CPU**: Works but will be slower
- **Disk Space**: At least 5GB free space for models and outputs

### Device Selection

When converting videos, the script automatically detects the device:
- **CUDA**: NVIDIA GPUs (fastest)
- **MPS**: Apple Silicon GPUs (fast)
- **CPU**: Fallback (slower)

You can also specify manually:
```bash
python scripts/converters/video_to_3d_high_quality.py video.mp4 cuda  # For NVIDIA GPU
python scripts/converters/video_to_3d_high_quality.py video.mp4 mps   # For Apple Silicon
python scripts/converters/video_to_3d_high_quality.py video.mp4 cpu   # For CPU only
```

---

## 📖 Usage Examples

### 1. Convert 2D Video to 3D

**High Quality (Recommended):**
```bash
python scripts/converters/video_to_3d_high_quality.py your_video.mp4 mps
```

This creates an `output_your_video/` directory with:
- `frames/` - Extracted video frames (PNG)
- `gaussians/` - 3D Gaussian Splat files (PLY) - one per frame

**Process Every Nth Frame (Faster):**
```bash
# Process every 2nd frame (2x faster)
python scripts/converters/video_to_3d_high_quality.py video.mp4 mps 2

# Process every 5th frame (5x faster)
python scripts/converters/video_to_3d_high_quality.py video.mp4 mps 5
```

**Video to 3D + human pose (YOLO):**

Converts each frame with ML-SHARP, runs multi-person pose estimation (YOLOv8x-pose), and opens a Rerun layout with three panels: 3D Gaussian scene, 3D skeletons (depth from the splat cloud), and the video with pose overlay.

```bash
pip install ultralytics
python scripts/converters/video_to_3d_with_pose.py your_video.mp4 --size 4.0
```

Use existing frames and PLYs without re-running SHARP:

```bash
python scripts/converters/video_to_3d_with_pose.py \
  --gaussians-dir output_your_video/gaussians \
  --frames-dir output_your_video/frames \
  --size 4.0
```

- `--device` / `--skip` apply when converting from a video (same as `video_to_3d_high_quality.py`).
- `--size` scales joint and bone thickness in 3D.
- YOLO weights download automatically on first run; use the same Python environment for `pip install ultralytics` as for `python3`.

**Standard Quality:**
```bash
python scripts/converters/video_to_3d.py your_video.mp4
```

**Quick Preview:**
```bash
python scripts/converters/video_to_3d_simple.py your_video.mp4
```

### 2. View 3D Video/Scene

**Single PLY File:**
```bash
python scripts/visualizers/visualize_with_rerun.py -i output_test/IMG_4707.ply --size 2.0
```

**Complete 3D Video Viewer:**
Shows original video, depth maps, 3D point cloud, and navigation data side by side:
```bash
python scripts/visualizers/video_complete_viewer.py \
    -i output_your_video/gaussians/ \
    --max-frames 30 \
    --size 2.0
```

**Options:**
- `-i, --input`: Directory containing PLY files (gaussians folder)
- `--max-frames`: Maximum frames to process (default: all)
- `--skip`: Process every Nth frame (default: 1)
- `--resolution`: Occupancy grid resolution in meters (default: 0.5)
- `--obstacle-height`: Obstacle height threshold in meters (default: 0.5)
- `--size`: Point size multiplier (default: 1.0)

**Video Navigation Analysis:**
```bash
python scripts/visualizers/video_navigation.py \
    -i output_your_video/gaussians/ \
    --max-frames 30
```

### 3. Build Navigation Map

Extract navigation data from 3D scenes:
```bash
python scripts/navigation/build_navigation_map.py \
    -i output_test/IMG_4707.ply \
    --resolution 0.5
```

**With Path Planning:**
```bash
python scripts/navigation/build_navigation_map.py \
    -i output_test/IMG_4707.ply \
    --resolution 0.5 \
    --plan-path \
    --start 0 0 \
    --goal 50 50 \
    -o navigation_map.json
```

### 4. Apply Effects

**Depth-based Fog Effect:**
```bash
python scripts/creative/apply_depth_effects.py -i scene.ply --effect fog
```

**Create Camera Path (Orbit):**
```bash
python scripts/creative/create_camera_path.py -i scene.ply --path orbit
```

---

## 📁 Project Structure

```
Apple-ml-sharp-rerun/
├── scripts/
│   ├── converters/            # 2D video to 3D conversion
│   │   ├── video_to_3d_high_quality.py  ⭐ Recommended
│   │   ├── video_to_3d_with_pose.py     # SHARP + YOLO pose + Rerun
│   │   ├── video_to_3d.py
│   │   └── video_to_3d_simple.py
│   ├── visualizers/           # 3D visualization viewers
│   │   ├── video_complete_viewer.py     ⭐ Complete viewer with dual windows
│   │   ├── video_navigation.py
│   │   ├── visualize_with_rerun.py
│   │   ├── visualize_with_360_background.py
│   │   └── visualize_with_custom_bg.py
│   ├── navigation/            # Navigation & SLAM tools
│   │   ├── build_navigation_map.py
│   │   ├── extract_slam_data.py
│   │   └── demo_navigation.py
│   └── creative/              # Creative effects
│       ├── apply_depth_effects.py
│       ├── compose_3d_scenes.py
│       └── create_camera_path.py
├── utils/                      # Reusable utility modules
│   ├── depth_rendering.py     # Depth map rendering
│   ├── frame_processing.py    # Frame processing
│   ├── navigation.py          # Navigation algorithms
│   ├── pathfinding.py         # Pathfinding
│   ├── visualization.py       # Viewer setup
│   ├── config.py              # Configuration
│   ├── io_utils.py            # File I/O
│   └── geometry.py            # 3D geometry
├── examples/                   # Example scripts
├── tests/                      # Test scripts
├── configs/                    # Configuration files
└── data/                       # Sample data
```

---

## 🎮 3D Viewer Controls

**Rotate View:**
- Left click + drag

**Pan/Move View:**
- Right click + drag (primary)
- Middle mouse + drag
- Shift + Left click + drag

**Zoom:**
- Mouse wheel / Trackpad scroll

**Reset View:**
- Double click anywhere

**Tips:**
- Both 3D windows in the complete viewer work independently
- Use the timeline at the bottom to scrub through video frames
- You can pan, zoom, and rotate each window separately

---

## 🛠️ Utility Modules

The `utils/` module provides reusable components:

- **`depth_rendering.py`**: Render depth maps from 3D points
- **`frame_processing.py`**: Load and process PLY files
- **`navigation.py`**: Ground detection, obstacle detection, occupancy grids
- **`pathfinding.py`**: A* pathfinding algorithm
- **`visualization.py`**: Set up Rerun viewers
- **`config.py`**: Configuration classes
- **`io_utils.py`**: File I/O helpers
- **`geometry.py`**: 3D transformations

**Usage Example:**
```python
from utils import (
    load_gaussian_data,
    render_depth_map,
    extract_ground_plane,
    setup_complete_viewer_blueprint
)

# Load PLY file
data = load_gaussian_data("scene.ply")

# Render depth map
depth_map, depth_colored = render_depth_map(
    data['positions'],
    data['colors'],
    resolution=(1280, 720)
)
```

See `examples/` directory for complete examples.

---

## 📊 Output Structure

When converting videos, the output structure is:

```
output_<video_name>/
├── frames/          # Extracted video frames (PNG)
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── gaussians/       # 3D Gaussian Splat files (PLY)
│   ├── frame_000000.ply
│   ├── frame_000001.ply
│   └── ...
└── json/            # Metadata (optional)
    └── ...
```

---

## 🔧 Configuration

Default configurations are in `utils/config.py`:

- **`ViewerConfig`**: Point sizes, opacity thresholds, rotations
- **`NavigationConfig`**: Obstacle heights, grid resolution
- **`DepthConfig`**: Depth rendering settings
- **`ConversionConfig`**: Video conversion settings

You can customize these:
```python
from utils import ViewerConfig, NavigationConfig

viewer_cfg = ViewerConfig(point_size_multiplier=2.0, opacity_threshold=0.2)
nav_cfg = NavigationConfig(obstacle_height=0.7, grid_resolution=0.3)
```

---

## 🆘 Troubleshooting

### Common Issues

#### **Import Errors**

**Problem:** `ModuleNotFoundError: No module named 'rerun'` or similar

**Solution:**
```bash
pip install -r requirements.txt
pip install sharp
```

#### **ML-SHARP Not Found**

**Problem:** `ModuleNotFoundError: No module named 'sharp'`

**Solution:**
```bash
pip install sharp
```

If that doesn't work, install from the official GitHub repository:
```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
pip install -r requirements.txt
pip install -e .
```

Or use conda (recommended by ML-SHARP):
```bash
git clone https://github.com/apple/ml-sharp.git
cd ml-sharp
conda create -n sharp python=3.13
conda activate sharp
pip install -r requirements.txt
pip install -e .
```

#### **Model Download Issues**

**Problem:** Model download fails or is slow

**Solution:**
- Check your internet connection
- The model is ~2.5GB, ensure you have enough disk space
- Model URL: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
- You can manually download and place it in a cache directory

#### **GPU Not Detected**

**Problem:** CUDA/MPS errors or GPU not detected

**Solution:**
- **NVIDIA GPU (CUDA):**
  - Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
  - Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- **Apple Silicon (MPS):**
  - MPS is automatically available on Apple Silicon Macs
  - Use `mps` device: `python scripts/converters/video_to_3d_high_quality.py video.mp4 mps`
- **CPU Fallback:**
  - Use `cpu`: `python scripts/converters/video_to_3d_high_quality.py video.mp4 cpu`
  - Note: CPU is much slower

#### **File Not Found**

**Problem:** `FileNotFoundError` when running scripts

**Solution:**
- Run scripts from the project root directory
- Use absolute paths if relative paths don't work
- Check that input files exist

#### **Memory Errors**

**Problem:** Out of memory during processing

**Solution:**
- Process fewer frames: `--max-frames 10`
- Use lower resolution videos
- Close other applications
- Use CPU instead of GPU if GPU memory is limited

#### **Python Version Issues**

**Problem:** Scripts don't work with your Python version

**Solution:**
- Make sure you have Python 3.8 or higher: `python --version`
- Use `python3` instead of `python` if needed
- Consider using a virtual environment

#### **Rerun Version Mismatch - Blank Viewer**

**Problem:** Rerun viewer is blank/nothing displays, with errors like:
- "Rerun Viewer: v0.23.1 vs Rerun SDK: v0.27.0"
- "dropping LogMsg due to failed decode"
- "transport error"

**Solution:**
This is caused by version mismatch. The viewer cannot decode messages from a newer SDK.

**Option 1: Downgrade SDK to match viewer (Recommended)**
```bash
pip install rerun-sdk==0.23.1
```
This matches your viewer version (v0.23.1) and should fix the blank viewer.

**Option 2: Update viewer to match SDK**
Follow the error message to update the viewer to v0.27.0, or:
```bash
# Using cargo (if you have Rust installed)
cargo binstall --force rerun-cli@0.27.0

# Or download from: https://github.com/rerun-io/rerun/releases/0.27.0/
```

**Note:** The decode errors mean the viewer can't display data - this must be fixed for visualization to work.

---

## 📝 Notes

- The ML-SHARP library (`sharp`) is required for 2D-to-3D video conversion
- The ML-SHARP model weights (~2.5GB) download automatically on first use
- Output directories are created automatically
- PLY files should be in ML-SHARP Gaussian Splatting format
- Rerun viewer runs independently - close the window to exit

---

## 🙏 Credits & Acknowledgments

This project uses these open-source technologies:

### Core Technologies

- **[Rerun](https://github.com/rerun-io/rerun)** - Visualize Everything Fast
  - SDK for logging, storing, querying, and visualizing multimodal data
  - Built in Rust using egui
  - Licensed under Apache-2.0
  - Created by the team at [rerun.io](https://rerun.io)
  - [GitHub](https://github.com/rerun-io/rerun) | [Documentation](https://www.rerun.io/docs)

- **[Apple ML-SHARP](https://github.com/apple/ml-sharp)** - Sharp Monocular View Synthesis in Less Than a Second
  - Apple's model for converting 2D images/videos to 3D Gaussian Splatting scenes
  - Official GitHub repository: [apple/ml-sharp](https://github.com/apple/ml-sharp)
  - Project page: [apple.github.io/ml-sharp](https://apple.github.io/ml-sharp/)
  - Research paper: [arXiv:2512.10685](https://arxiv.org/abs/2512.10685)
  - Model weights provided by Apple
  - Model hosted at: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
  - Installation: `pip install sharp` or install from source

### Additional Dependencies

- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **OpenCV** - Computer vision library
- **Pillow** - Image processing
- **SciPy** - Scientific computing
- **tqdm** - Progress bars

### Special Thanks

- **Rerun Team** ([@rerun-io](https://github.com/rerun-io)) for creating the visualization tool
- **Apple Research** for developing ML-SHARP and making it available
- All open-source contributors

---

## 🔗 Related Links

### ML-SHARP Resources
- **ML-SHARP GitHub**: https://github.com/apple/ml-sharp
- **ML-SHARP Project Page**: https://apple.github.io/ml-sharp/
- **Research Paper**: https://arxiv.org/abs/2512.10685
- **ML-SHARP Model**: `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt`
- **Installation**: `pip install sharp` or install from GitHub
- **ML-SHARP CLI**: `sharp predict -i <input> -o <output>` (see [official docs](https://github.com/apple/ml-sharp))

### Rerun Resources
- **Rerun GitHub**: https://github.com/rerun-io/rerun
- **Rerun Documentation**: https://www.rerun.io/docs
- **Rerun Website**: https://rerun.io
- **Rerun Discord**: Join for community support

### Documentation
- **Quick Start Guide**: See [QUICKSTART.md](QUICKSTART.md)
- **Project Structure**: See [STRUCTURE.md](STRUCTURE.md) (if available)
- **Examples**: See [examples/README.md](examples/README.md)

---

## 📄 License

This project is part of the Apple ML-SHARP ecosystem and visualization tools.

**Third-party licenses:**
- **Rerun**: Apache-2.0 License
- **PyTorch**: BSD-style License
- **Other dependencies**: See their respective licenses

---

**Made with ❤️ using [Rerun](https://github.com/rerun-io/rerun) and [Apple ML-SHARP](https://github.com/apple/ml-sharp)**
