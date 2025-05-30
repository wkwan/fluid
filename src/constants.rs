// Shared constants for fluid simulation
// These values should be used consistently across 2D, 3D, CPU, and GPU implementations

/// Particle radius for collision detection and minimum separation
pub const PARTICLE_RADIUS: f32 = 2.5;

/// GPU particle radius for rendering (can be different from collision radius)
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

// ======================== CAMERA CONSTANTS ========================

/// Camera zoom limits
pub const MIN_ZOOM: f32 = 50.0;
pub const MAX_ZOOM: f32 = 1000.0;

/// Camera reset values
pub const RESET_YAW: f32 = 0.0;
pub const RESET_PITCH: f32 = -20.0;
pub const RESET_DISTANCE: f32 = 400.0;

// ======================== SIMULATION CONSTANTS ========================

/// Maximum velocity for normalization in rendering
pub const MAX_VELOCITY: f32 = 700.0;

/// Maximum angular velocity for 3D objects (radians per second)
pub const MAX_ANGULAR_VELOCITY: f32 = 3.0;

/// Duck size relative to particle radius
pub const DUCK_SIZE: f32 = PARTICLE_RADIUS * 5.0;

// ======================== MOUSE INTERACTION CONSTANTS ========================

/// Mouse interaction strength levels
pub const MOUSE_STRENGTH_LOW: f32 = 1000.0;
pub const MOUSE_STRENGTH_MEDIUM: f32 = 2000.0;
pub const MOUSE_STRENGTH_HIGH: f32 = 3000.0;

// ======================== PHYSICS CONSTANTS ========================

/// Restitution (bounce factor) for collisions
pub const RESTITUTION: f32 = 0.3;

/// Friction coefficient for surface interactions
pub const FRICTION: f32 = 0.1;

/// Velocity damping factor for air resistance
pub const VELOCITY_DAMPING: f32 = 0.98;

/// Angular velocity damping for 3D objects
pub const ANGULAR_DAMPING: f32 = 0.95;

/// Horizontal velocity damping for ground interactions
pub const HORIZONTAL_DAMPING: f32 = 0.9; 