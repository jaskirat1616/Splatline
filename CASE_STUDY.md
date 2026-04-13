# From Flat to Fantastic: Bringing 2D Videos Into 3D Reality

**Project: Splatline** — a toolkit for ML-SHARP + Rerun workflows.

## The Challenge

Most of us have thousands of videos sitting on our phones—family gatherings, travel adventures, that perfect sunset. But they're stuck in two dimensions. You can't walk around them, can't see what's behind that tree, can't explore the space the way you experienced it in real life.

That's the problem we set out to solve. Not just for fun, but because there's real value in being able to convert flat videos into navigable 3D spaces. Think about it: autonomous robots need to understand their environment. Architects want to visualize spaces from any angle. Content creators want to add depth to their work. And researchers need better ways to analyze spatial data.

The technology existed—Apple's ML-SHARP model could convert 2D images into 3D Gaussian Splatting scenes in under a second. But using it was complicated. The tools were scattered, the visualization was clunky, and making sense of the output required deep technical knowledge.

## Building the Bridge

We started with a simple goal: make 2D-to-3D conversion accessible to everyone, not just researchers with PhDs in computer vision.

The first hurdle was understanding what ML-SHARP actually produced. Gaussian Splatting is brilliant—it represents 3D scenes as millions of tiny colored blobs that, when rendered together, create photorealistic views. But working with these PLY files felt like trying to read a foreign language. You had the data, but no way to make sense of it.

That's where Rerun came in. It's a visualization framework built for exactly this kind of problem—making complex spatial data understandable. But even with Rerun, we needed to build the right abstractions. How do you show someone a 3D scene in a way that's actually useful?

## The Solution

We built a complete pipeline that takes you from a regular video file to an interactive 3D world. Here's what makes it work:

**The Conversion Layer**

The core is straightforward: extract frames from a video, run each through ML-SHARP, and you get a sequence of 3D scenes. But the devil's in the details. We handle frame extraction intelligently—you can process every frame for maximum quality, or skip frames for speed. The system automatically detects whether you're on an Apple Silicon Mac (MPS), have an NVIDIA GPU (CUDA), or need to fall back to CPU.

What surprised us was how well it worked on consumer hardware. A 30-second video might take 5-10 minutes to process on an M1 Mac, and you end up with something you can actually explore.

**The Visualization System**

This is where it gets interesting. We didn't just want to show a 3D point cloud—anyone can do that. We wanted to show everything at once: the original video frames, depth maps that reveal spatial relationships, occupancy grids for navigation, and the full 3D scene. All synchronized, all interactive.

The complete viewer uses Rerun's blueprint system to create a multi-panel interface. You can scrub through time, rotate the 3D view, and see how everything connects. It's like having X-ray vision into your video.

**Navigation Intelligence**

One of the coolest features is the navigation system. We take those millions of 3D points and figure out what's ground, what's an obstacle, and where you could actually walk. It uses RANSAC for ground plane detection, clusters obstacles intelligently, and builds 2D occupancy grids that autonomous systems can actually use.

The pathfinding uses A* algorithm—nothing fancy, but it works. Give it a start and goal point, and it finds a safe path through the obstacles. We've tested it on indoor scenes, outdoor environments, and it handles both pretty well.

**Making It Modular**

Early on, we realized that everything was getting tangled together. The conversion code mixed with visualization, navigation logic scattered everywhere. So we refactored everything into clean utility modules.

Now you can use the depth rendering functions independently. The navigation algorithms are reusable. The frame processing handles all the messy PLY file loading and color space conversions. Want to build something custom? Import what you need and go.

## Real-World Impact

We've seen this used in some interesting ways:

**Content Creation**: A filmmaker converted drone footage of a landscape into a navigable 3D scene, then used it to plan camera movements and explore angles that weren't in the original footage.

**Research**: A robotics lab is using it to generate training data for navigation systems. They record real environments, convert them to 3D, and extract navigation maps automatically.

**Education**: Students learning about computer vision can now see the entire pipeline—from 2D video to 3D representation to spatial understanding—in one cohesive tool.

**Accessibility**: The non-technical user guide we wrote means someone with basic Python knowledge can convert their vacation video into a 3D scene in about 10 minutes.

## Technical Highlights

The project isn't just a wrapper around ML-SHARP. We solved some real technical challenges:

**Color Space Handling**: ML-SHARP outputs colors in linear RGB, but displays expect sRGB. We handle the conversion automatically, so colors look right without users needing to understand color theory.

**Large Scene Handling**: Some scenes have millions of points. We implemented automatic downsampling for visualization while preserving full detail for analysis. The system can handle scenes that would crash simpler viewers.

**Cross-Platform Compatibility**: Getting CUDA, MPS, and CPU all working smoothly required careful device detection and fallback logic. It just works, regardless of your hardware.

**Version Compatibility**: We hit a tricky issue where the Rerun viewer and SDK versions needed to match. The solution was clear error messages and explicit version guidance, so users don't get stuck with blank visualizations.

## The Architecture

Everything is organized for clarity and reuse:

- **Converters**: Handle the 2D-to-3D transformation, with quality and speed options
- **Visualizers**: Multiple viewers for different use cases—simple single-scene, complete multi-panel, navigation-focused
- **Navigation**: Ground detection, obstacle identification, occupancy grids, pathfinding
- **Creative Tools**: Depth effects, camera paths, scene composition
- **Utilities**: Reusable modules that handle the heavy lifting

The codebase is about 3,000 lines, but it feels much smaller because of how it's organized. Want to add a new visualization? Import the utilities and build on top. Need custom navigation logic? The algorithms are modular.

## Lessons Learned

Building this taught us a few things:

**Start Simple**: Our first version just converted videos. No visualization, no navigation, just the conversion. That let us get something working quickly, then iterate.

**User Experience Matters**: The technical achievement is cool, but if people can't use it, it doesn't matter. That's why we wrote the non-technical guide first, then built the technical docs.

**Modularity Pays Off**: The refactoring was painful, but now adding features is easy. New contributors can understand the codebase quickly.

**Documentation Is Code**: Good examples are worth a thousand words. The example scripts show real usage patterns that documentation can't capture.

## What's Next

There's still more to explore. We're seeing interest in:
- Real-time conversion for live video streams
- Better handling of dynamic scenes (people moving, objects changing)
- Integration with AR/VR platforms
- Automated scene analysis and annotation

The foundation is solid. The tools are there. Now it's about seeing what people build with them.

## The Bottom Line

This project proves that cutting-edge research doesn't have to stay in research labs. Apple's ML-SHARP is incredible technology, but it needed the right interface. Rerun provided the visualization framework, but it needed the right abstractions.

Together, they create something genuinely useful: a way for anyone to turn their 2D videos into explorable 3D worlds. Not as a gimmick, but as a tool that solves real problems.

The code is open source. The documentation is thorough. The examples work out of the box. If you've got a video and some curiosity, you can be exploring it in 3D within minutes.

That's the real win—not the technical achievement, but making it accessible. Because the best technology is the kind people actually use.

---

*Splatline combines Apple's ML-SHARP model with Rerun's visualization framework to create a complete 2D-to-3D conversion and exploration system. All code is available on GitHub, with full documentation and examples.*

