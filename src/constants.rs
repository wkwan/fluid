// Shared constants for fluid simulation
// These values should be used consistently across 2D, 3D, CPU, and GPU implementations

/// Particle radius for collision detection and minimum separation
pub const PARTICLE_RADIUS: f32 = 2.5;

/// GPU particle radius (used in shaders and GPU calculations)
pub const GPU_PARTICLE_RADIUS: f32 = 5.0;

/// Boundary dampening factor for wall collisions
pub const BOUNDARY_DAMPENING: f32 = 0.3;

/// Rest density for visualization and normalization
pub const REST_DENSITY: f32 = 1500.0;

/// Gravity acceleration (2D)
pub const GRAVITY_2D: [f32; 2] = [0.0, -20.0];

/// Gravity acceleration (3D)
pub const GRAVITY_3D: [f32; 3] = [0.0, -40.0, 0.0];

/// 3D Boundary limits
pub const BOUNDARY_3D_MIN: [f32; 3] = [-100.0, -100.0, -100.0];
pub const BOUNDARY_3D_MAX: [f32; 3] = [100.0, 100.0, 100.0];

/// Y threshold below which 3D particles are recycled
pub const KILL_Y_THRESHOLD_3D: f32 = -250.0; 