// Particle and parameter structs
struct Particle3D {
    position: vec3<f32>,
    padding0: f32,
    velocity: vec3<f32>,
    padding1: f32,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    force: vec3<f32>,
    padding2: f32,
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
    bounds_min_padding: f32,

    // Vec4 aligned group 4
    bounds_max: vec3<f32>,
    bounds_max_padding: f32,

    // Vec4 aligned group 5
    gravity: vec3<f32>,
    gravity_padding: f32,

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
    _pad2: vec2<u32>,
}

const PI: f32 = 3.14159265359;
const MAX_NEIGHBORS: u32 = 128u;

// Bindings
@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<uniform> params: FluidParams3D;
@group(0) @binding(2) var<storage, read_write> spatial_keys_dummy: array<u32>;
@group(0) @binding(3) var<storage, read_write> spatial_offsets_dummy: array<u32>;
@group(0) @binding(4) var<storage, read_write> neighbor_counts_dummy: array<u32>;
@group(0) @binding(5) var<storage, read_write> neighbor_indices_dummy: array<u32>;

// Helper struct for collision result
struct CollisionResult {
    position: vec3<f32>,
    velocity: vec3<f32>,
}

// Unity-style collision resolution
fn resolve_collisions(pos: vec3<f32>, vel: vec3<f32>) -> CollisionResult {
    var new_pos = pos;
    var new_vel = vel;
    
    // Boundary collisions with damping
    // X-axis
    if new_pos.x < params.bounds_min.x + params.particle_radius {
        new_pos.x = params.bounds_min.x + params.particle_radius;
        new_vel.x = -new_vel.x * params.boundary_dampening;
    } else if new_pos.x > params.bounds_max.x - params.particle_radius {
        new_pos.x = params.bounds_max.x - params.particle_radius;
        new_vel.x = -new_vel.x * params.boundary_dampening;
    }
    
    // Y-axis
    if new_pos.y < params.bounds_min.y + params.particle_radius {
        new_pos.y = params.bounds_min.y + params.particle_radius;
        new_vel.y = -new_vel.y * params.boundary_dampening;
    } else if new_pos.y > params.bounds_max.y - params.particle_radius {
        new_pos.y = params.bounds_max.y - params.particle_radius;
        new_vel.y = -new_vel.y * params.boundary_dampening;
    }
    
    // Z-axis
    if new_pos.z < params.bounds_min.z + params.particle_radius {
        new_pos.z = params.bounds_min.z + params.particle_radius;
        new_vel.z = -new_vel.z * params.boundary_dampening;
    } else if new_pos.z > params.bounds_max.z - params.particle_radius {
        new_pos.z = params.bounds_max.z - params.particle_radius;
        new_vel.z = -new_vel.z * params.boundary_dampening;
    }
    
    return CollisionResult(new_pos, new_vel);
}

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= arrayLength(&particles) {
        return;
    }
    
    var particle = particles[index];
    
    // Unity approach: Get original position from stored value in force field
    // (In prediction step, we stored original position there)
    let original_position = particle.force; // This was set in predict_positions.wgsl
    
    // Integrate: pos = original_pos + velocity * dt  
    // This matches Unity's: pos += vel * deltaTime;
    var new_pos = original_position + particle.velocity * params.dt;
    var new_vel = particle.velocity;
    
    // Resolve collisions
    let collision_result = resolve_collisions(new_pos, new_vel);
    new_pos = collision_result.position;
    new_vel = collision_result.velocity;
    
    // Update particle
    particle.position = new_pos;
    particle.velocity = new_vel;
    particle.force = vec3<f32>(0.0); // Reset force for next frame
    
    particles[index] = particle;
} 