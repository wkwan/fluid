// Shared constants for fluid simulation
// These values should be used consistently across 2D, 3D, CPU, and GPU implementations

/// Particle radius for collision detection and minimum separation
pub const PARTICLE_RADIUS: f32 = 2.5;

/// Boundary dampening factor for wall collisions
pub const BOUNDARY_DAMPENING: f32 = 0.3;

/// Rest density for visualization and normalization
pub const REST_DENSITY: f32 = 1500.0;

/// Gravity acceleration (2D)
pub const GRAVITY_2D: [f32; 2] = [0.0, -20.0];

/// Gravity acceleration (3D)
pub const GRAVITY_3D: [f32; 3] = [0.0, -10.0, 0.0]; 