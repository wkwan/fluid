# Purpose
To experiment with different techniques for real-time fluid simulation and rendering in Rust/WGSL/Bevy by making a beautiful realistic lake drawing toy.

# Overview
Fluid simulation is done with a particle-based approach called Smoothed Particle Hydrodynamics (SPH). Code is heavily based off of https://github.com/SebLague/Fluid-Sim

2D mode can be enabled in the left menu but development is focused on 3D. When 2D is enabled, GPU acceleration should be disabled, otherwise the particles will overlap due to a bug (low priority bug).

3D fluid rendering can be toggled between 2 separate pipelines. Development focus is on screen space rendering to balance realism with performance, but raymarching (more realistic) is also implemented.

### Kanban board: https://github.com/users/wkwan/projects/2

### Streaming development on https://twitch.tv/willkwan

# To run demo:
```
cargo run --release
```
