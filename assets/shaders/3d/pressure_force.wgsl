// Particle and parameter structs
struct Particle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    force: vec3<f32>,
}

struct FluidParams3D {
    // Vec4 aligned group 1
    smoothing_radius: f32,
    rest_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,

    // Vec4 aligned group 2
    viscosity: f32,
    boundary_dampening: f32,
    particle_radius: f32,
    dt: f32,

    // Vec4 aligned group 3
    bounds_min: vec3<f32>,
    _pad0: f32,

    // Vec4 aligned group 4
    bounds_max: vec3<f32>,
    _pad1: f32,

    // Vec4 aligned group 5
    gravity: vec3<f32>,
    _pad2: f32,

    // Vec4 aligned group 6
    mouse_position: vec3<f32>,
    mouse_radius: f32,

    // Vec4 aligned group 7
    mouse_strength: f32,
    mouse_active: u32,
    mouse_repel: u32,
    group6_padding: f32,

    // Vec4 aligned group 8
    padding: vec2<u32>,
    _pad3: vec2<u32>,
}

// Constants
const HASH_K1: f32 = 15823.0;
const HASH_K2: f32 = 9737333.0;
const HASH_K3: f32 = 440817757.0;
const BLOCK_SIZE: u32 = 256u;
const PI: f32 = 3.14159265359;
const MAX_NEIGHBORS: u32 = 128u;

// 3D cell offsets for neighboring cells
const OFFSETS_3D: array<vec3<i32>, 27> = array<vec3<i32>, 27>(
    vec3<i32>(-1, -1, -1), vec3<i32>(-1, -1, 0), vec3<i32>(-1, -1, 1),
    vec3<i32>(-1, 0, -1), vec3<i32>(-1, 0, 0), vec3<i32>(-1, 0, 1),
    vec3<i32>(-1, 1, -1), vec3<i32>(-1, 1, 0), vec3<i32>(-1, 1, 1),
    vec3<i32>(0, -1, -1), vec3<i32>(0, -1, 0), vec3<i32>(0, -1, 1),
    vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 0), vec3<i32>(0, 0, 1),
    vec3<i32>(0, 1, -1), vec3<i32>(0, 1, 0), vec3<i32>(0, 1, 1),
    vec3<i32>(1, -1, -1), vec3<i32>(1, -1, 0), vec3<i32>(1, -1, 1),
    vec3<i32>(1, 0, -1), vec3<i32>(1, 0, 0), vec3<i32>(1, 0, 1),
    vec3<i32>(1, 1, -1), vec3<i32>(1, 1, 0), vec3<i32>(1, 1, 1)
);

// Shared memory for particle data
struct CachedParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Bindings
@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<uniform> params: FluidParams3D;
@group(0) @binding(2) var<storage, read_write> spatial_keys_dummy: array<u32>;
@group(0) @binding(3) var<storage, read_write> spatial_offsets_dummy: array<u32>;
@group(0) @binding(4) var<storage, read_write> neighbor_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> neighbor_indices: array<u32>;

// Helper functions
fn get_cell_3d(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(floor(pos / cell_size));
}

fn hash_cell_3d(cell: vec3<i32>) -> u32 {
    let ucell = vec3<u32>(cell + vec3<i32>(i32(BLOCK_SIZE) / 2));
    return u32(HASH_K1 * f32(ucell.x) + HASH_K2 * f32(ucell.y) + HASH_K3 * f32(ucell.z));
}

fn key_from_hash(hash: u32, num_cells: u32) -> u32 {
    return hash % num_cells;
}

// Unity-style derivative kernels with safety checks
fn density_derivative(distance: f32, radius: f32) -> f32 {
    if distance >= radius || distance <= 0.001 || radius <= 0.0 {
        return 0.0;
    }
    let h = radius;
    let h2 = h * h;
    let h4 = h2 * h2;
    let h9 = h4 * h4 * h;
    if h9 <= 0.0 {
        return 0.0;
    }
    let scale = -12.0 / (PI * h9);
    let v = h - distance;
    return scale * v;
}

fn near_density_derivative(distance: f32, radius: f32) -> f32 {
    if distance >= radius || distance <= 0.001 || radius <= 0.0 {
        return 0.0;
    }
    let h = radius;
    let h2 = h * h;
    let h3 = h2 * h;
    let h9 = h3 * h3 * h3;
    if h9 <= 0.0 {
        return 0.0;
    }
    let scale = -30.0 / (PI * h9);
    let v = h - distance;
    return scale * v * v;
}

fn spiky_kernel_derivative(r: vec3<f32>, h: f32) -> vec3<f32> {
    let r_len = length(r);
    if r_len >= h || r_len <= 0.001 || h <= 0.0 {
        return vec3<f32>(0.0);
    }
    
    let h2 = h * h;
    let h6 = h2 * h2 * h2;
    if h6 <= 0.0 {
        return vec3<f32>(0.0);
    }
    
    let scale = -45.0 / (PI * h6);
    let v = h - r_len;
    let v2 = v * v;
    let direction = r / r_len; // Safe since r_len > 0.001
    return direction * scale * v2;
}

// Main compute shader for pressure force calculation - Unity style with safety checks
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= arrayLength(&particles)) {
        return;
    }
    
    var particle = particles[particle_idx];
    var pressure_force = vec3<f32>(0.0);
    
    // Safety check - ensure particle has valid density
    if particle.density <= 0.001 {
        particle.density = 0.1; // Prevent division by zero
    }
    
    // Get number of neighbors for this particle
    let num_neighbors = neighbor_counts[particle_idx];
    
    // Calculate pressure forces from neighbors using Unity approach
    for (var i = 0u; i < num_neighbors; i = i + 1u) {
        let neighbor_idx = neighbor_indices[particle_idx * MAX_NEIGHBORS + i];
        if neighbor_idx >= arrayLength(&particles) || neighbor_idx == particle_idx {
            continue; // Skip invalid or self neighbors
        }
        
        var neighbor = particles[neighbor_idx];
        
        // Safety check for neighbor density
        if neighbor.density <= 0.001 {
            neighbor.density = 0.1;
        }
        
        let offset = particle.position - neighbor.position;
        let distance = length(offset);
        
        if distance > 0.001 && distance < params.smoothing_radius {
            // Calculate shared pressure (Unity style)
            let shared_pressure = (particle.pressure + neighbor.pressure) * 0.5;
            let shared_near_pressure = (particle.near_pressure + neighbor.near_pressure) * 0.5;
            
            // Calculate direction (safe since distance > 0.001)
            let direction = offset / distance;
            
            // Calculate force components
            let pressure_grad = density_derivative(distance, params.smoothing_radius);
            let near_pressure_grad = near_density_derivative(distance, params.smoothing_radius);
            
            // Combine forces with safety scaling
            let force_magnitude = (shared_pressure * pressure_grad + shared_near_pressure * near_pressure_grad) / particle.density;
            let force = direction * force_magnitude;
            
            // Clamp force to prevent explosion
            let max_force = 100.0;
            let force_mag = length(force);
            if force_mag > max_force {
                pressure_force += normalize(force) * max_force;
            } else {
                pressure_force += force;
            }
        }
    }
    
    // Apply pressure force to velocity (Unity approach: direct velocity modification)
    particle.velocity += pressure_force * params.dt;
    
    // Safety clamp final velocity
    let max_velocity = 50.0;
    let vel_magnitude = length(particle.velocity);
    if vel_magnitude > max_velocity {
        particle.velocity = normalize(particle.velocity) * max_velocity;
    }
    
    particles[particle_idx] = particle;
} 