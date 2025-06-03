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

fn viscosity_kernel(r: vec3<f32>, h: f32) -> f32 {
    let r_len = length(r);
    if (r_len >= h || r_len < 0.0001) {
        return 0.0;
    }
    
    let h2 = h * h;
    let h3 = h2 * h;
    let h4 = h3 * h;
    let h5 = h4 * h;
    
    let coef = 45.0 / (PI * h5);
    let h_r = h - r_len;
    return coef * h_r;
}

// Main compute shader for viscosity force calculation
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= arrayLength(&particles)) {
        return;
    }
    
    var particle = particles[particle_idx];
    var viscosity_force = vec3<f32>(0.0);
    
    // Get number of neighbors for this particle
    let num_neighbors = neighbor_counts[particle_idx];
    
    // Calculate viscosity forces from neighbors
    for (var i = 0u; i < num_neighbors; i = i + 1u) {
        let neighbor_idx = neighbor_indices[particle_idx * MAX_NEIGHBORS + i];
        let neighbor = particles[neighbor_idx];
        
        let r = particle.position - neighbor.position;
        let r_len = length(r);
        
        if (r_len < params.smoothing_radius && r_len > 0.0001) {
            let viscosity = viscosity_kernel(r, params.smoothing_radius);
            let velocity_diff = neighbor.velocity - particle.velocity;
            viscosity_force = viscosity_force + velocity_diff * viscosity * params.viscosity;
        }
    }
    
    // Update particle force
    particle.force = particle.force + viscosity_force;
    particles[particle_idx] = particle;
} 