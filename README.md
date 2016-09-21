# GPU-path-tracing-tutorial-4
Matching Socks CUDA path tracer
by Samuel Lapere, 2016
https://raytracey.blogspot.com

Demo of high performance CUDA accelerated path tracing
based on the GPU ray tracing framework of Timo Aila, 
Samuli Laine and Tero Karras (Nvidia Research)

Source code for original framework: 
- https://code.google.com/archive/p/understanding-the-efficiency-of-ray-traversal-on-gpus/
- https://research.nvidia.com/publication/understanding-efficiency-ray-traversal-gpus-kepler-and-fermi-addendum
- https://mediatech.aalto.fi/~timo/HPG2009/

Features:

- BVH with spatial splits
- Woop ray/triangle intersection
- hand tuned traversal kernels for Fermi and Kepler GPUs
- HDRI environment mapping
- basic OBJ loader
- basic material system (diffuse, clearcoat, refractive, 
reflective, metal)
- camera depth-of-field
- real-time user interaction

Downloadable demo available at 
https://github.com/straaljager/GPU-path-tracing-tutorial-4/releases

More info about this tutorial and some screenshots rendered with this code at

https://raytracey.blogspot.co.nz/2016/09/gpu-path-tracing-tutorial-4-optimised.html 
