"""
Setup script for Splatline — ML-SHARP + Rerun visualization tools.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="splatline",
    version="0.1.0",
    description="Splatline: Rerun-based visualization and navigation for 3D Gaussian splats (ML-SHARP)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Splatline contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "rerun-sdk>=0.15.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "splatline-visualize=scripts.visualizers.visualize_with_rerun:main",
            "splatline-complete-viewer=scripts.visualizers.video_complete_viewer:main",
            "splatline-nav-map=scripts.navigation.build_navigation_map:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

