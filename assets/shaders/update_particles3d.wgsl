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
    smoothing_radius: f32,
    rest_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    viscosity: f32,
    gravity: vec3<f32>,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
}

const PI: f32 = 3.14159265359;
const MAX_NEIGHBORS: u32 = 128u;

// Bindings
@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<storage, read> params: FluidParams3D;
@group(0) @binding(2) var<storage, read> neighbor_counts: array<u32>;
@group(0) @binding(3) var<storage, read> neighbor_indices: array<u32>;

// Helper function to handle boundary collisions
fn handle_boundary_collision(pos: vec3<f32>, vel: vec3<f32>, min_bounds: vec3<f32>, max_bounds: vec3<f32>) -> vec3<f32> {
    var new_vel = vel;
    
    // X-axis boundaries
    if (pos.x < min_bounds.x) {
        new_vel.x = abs(new_vel.x);
    } else if (pos.x > max_bounds.x) {
        new_vel.x = -abs(new_vel.x);
    }
    
    // Y-axis boundaries
    if (pos.y < min_bounds.y) {
        new_vel.y = abs(new_vel.y);
    } else if (pos.y > max_bounds.y) {
        new_vel.y = -abs(new_vel.y);
    }
    
    // Z-axis boundaries
    if (pos.z < min_bounds.z) {
        new_vel.z = abs(new_vel.z);
    } else if (pos.z > max_bounds.z) {
        new_vel.z = -abs(new_vel.z);
    }
    
    return new_vel;
}

// Main compute shader for updating particle positions
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= arrayLength(&particles)) {
        return;
    }
    
    var particle = particles[particle_idx];
    
    // Apply forces and update position using Verlet integration
    let dt = 0.016; // Fixed time step
    let dt2 = dt * dt;
    
    // Update velocity with forces
    particle.velocity = particle.velocity + particle.force * dt;
    
    // Add gravity
    particle.velocity = particle.velocity + params.gravity * dt;
    
    // Update position
    particle.position = particle.position + particle.velocity * dt;
    
    // Handle boundary collisions
    particle.velocity = handle_boundary_collision(
        particle.position,
        particle.velocity,
        params.bounds_min,
        params.bounds_max
    );
    
    // Clamp position to bounds
    particle.position = clamp(
        particle.position,
        params.bounds_min,
        params.bounds_max
    );
    
    // Reset force for next frame
    particle.force = vec3<f32>(0.0);
    
    // Update particle
    particles[particle_idx] = particle;
} 