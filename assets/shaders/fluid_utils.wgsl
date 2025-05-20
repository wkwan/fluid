// Fluid Simulation - Common Utility Functions and Structures

// Particle structure
struct Particle {
    position: vec2<f32>,
    padding0: vec2<f32>,  // Padding for 16-byte alignment
    velocity: vec2<f32>,
    padding1: vec2<f32>,  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Fluid parameters structure
struct FluidParams {
    // Vec4 aligned group 1
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    // Vec4 aligned group 2
    viscosity_strength: f32,
    boundary_dampening: f32,
    particle_radius: f32,
    dt: f32,
    
    // Vec4 aligned group 3
    boundary_min: vec2<f32>,
    boundary_min_padding: vec2<f32>,
    
    // Vec4 aligned group 4
    boundary_max: vec2<f32>,
    boundary_max_padding: vec2<f32>,
    
    // Vec4 aligned group 5
    gravity: vec2<f32>,
    gravity_padding: vec2<f32>,
    
    // Vec4 aligned group 6
    mouse_position: vec2<f32>,
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: vec2<u32>,
}

// 2D cell offsets for neighboring cells (9 total including center)
const OFFSETS_2D: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, 1),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(-1, -1),
    vec2<i32>(0, -1),
    vec2<i32>(1, -1)
);

// Constants used for hashing
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;
const PI: f32 = 3.14159265359;

// Smoothing kernel function: Poly6
fn poly6(dist_squared: f32, h_squared: f32) -> f32 {
    if (dist_squared >= h_squared) {
        return 0.0;
    }
    
    let h_squared_minus_r_squared = h_squared - dist_squared;
    // Pre-calculated scaling factor for h^8
    let poly6_scaling_factor = 4.0 / (PI * pow(h_squared, 4.0));
    
    return poly6_scaling_factor * pow(h_squared_minus_r_squared, 3.0);
}

// Smoothing kernel function: Spiky Pow3
fn spiky_pow3(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow3_scaling_factor = 10.0 / (PI * pow(h, 5.0));
    
    return spiky_pow3_scaling_factor * pow(h_minus_r, 3.0);
}

// Smoothing kernel function: Spiky Pow2
fn spiky_pow2(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow2_scaling_factor = 6.0 / (PI * pow(h, 4.0));
    
    return spiky_pow2_scaling_factor * pow(h_minus_r, 2.0);
}

// Derivative functions for calculating forces
fn spiky_pow3_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow3_derivative_scaling_factor = 30.0 / (PI * pow(h, 5.0));
    
    return -spiky_pow3_derivative_scaling_factor * pow(h_minus_r, 2.0);
}

fn spiky_pow2_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow2_derivative_scaling_factor = 12.0 / (PI * pow(h, 4.0));
    
    return -spiky_pow2_derivative_scaling_factor * h_minus_r;
}

// Convert floating point position into an integer cell coordinate
fn get_cell_2d(position: vec2<f32>, cell_size: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / cell_size));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return ((x * HASH_K1) ^ (y * HASH_K2)) + (x * y);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    let mixed = hash ^ (hash >> 16u);
    return mixed % table_size;
}

// Calculate optimal cell size based on smoothing radius
fn calculate_optimal_cell_size(smoothing_radius: f32) -> f32 {
    return smoothing_radius;
} 