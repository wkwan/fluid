// Particle and parameter structs
struct Particle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    pressure_force: vec3<f32>,
    near_pressure_force: vec3<f32>,
    viscosity_force: vec3<f32>,
}

struct FluidParams3D {
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    viscosity_strength: f32,
    gravity: vec3<f32>,
    delta_time: f32,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    damping: f32,
}

// Bindings
@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<uniform> params: FluidParams3D;

// Helper function to handle boundary collisions
fn handle_boundary_collision(pos: vec3<f32>, vel: vec3<f32>) -> vec3<f32> {
    var new_vel = vel;
    
    // X-axis boundaries
    if (pos.x < params.bounds_min.x) {
        new_vel.x = abs(new_vel.x) * params.damping;
    } else if (pos.x > params.bounds_max.x) {
        new_vel.x = -abs(new_vel.x) * params.damping;
    }
    
    // Y-axis boundaries
    if (pos.y < params.bounds_min.y) {
        new_vel.y = abs(new_vel.y) * params.damping;
    } else if (pos.y > params.bounds_max.y) {
        new_vel.y = -abs(new_vel.y) * params.damping;
    }
    
    // Z-axis boundaries
    if (pos.z < params.bounds_min.z) {
        new_vel.z = abs(new_vel.z) * params.damping;
    } else if (pos.z > params.bounds_max.z) {
        new_vel.z = -abs(new_vel.z) * params.damping;
    }
    
    return new_vel;
}

// Main compute shader for updating particle positions
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Get particle data
    var particle = particles[index];
    
    // Combine all forces
    let total_force = particle.pressure_force + 
                     particle.near_pressure_force + 
                     particle.viscosity_force + 
                     params.gravity;
    
    // Update velocity using Verlet integration
    particle.velocity += total_force * params.delta_time;
    
    // Update position
    particle.position += particle.velocity * params.delta_time;
    
    // Handle boundary collisions
    particle.velocity = handle_boundary_collision(particle.position, particle.velocity);
    
    // Clamp position to bounds
    particle.position = clamp(particle.position, params.bounds_min, params.bounds_max);
    
    // Update particle
    particles[index] = particle;
} 