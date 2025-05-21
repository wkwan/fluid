// Highly optimized Integration Compute Shader
// Matches Unity's implementation with coherent memory access and optimized boundaries

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

// Shared memory structure for integration - perfectly aligned for memory coalescing
struct ParticleData {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

// Shared memory for optimized performance - sized to match workgroup exactly
var<workgroup> shared_data: array<ParticleData, 64>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

// Optimized velocity clamping for stability
fn clamp_velocity(velocity: vec2<f32>, max_speed: f32) -> vec2<f32> {
    let speed_squared = dot(velocity, velocity);
    if (speed_squared > max_speed * max_speed) {
        return velocity * (max_speed / sqrt(speed_squared));
    }
    return velocity;
}

// Optimized boundary collision detection and response
fn handle_boundary_collision(position: vec2<f32>, velocity: vec2<f32>, boundary_min: vec2<f32>, 
                            boundary_max: vec2<f32>, particle_radius: f32, dampening: f32) -> vec4<f32> {
    let collision_margin = 0.01; // Small margin to prevent sticking to walls
    var new_position = position;
    var new_velocity = velocity;
    
    // Process X boundaries
    if (position.x < boundary_min.x + particle_radius) {
        // Left wall collision
        new_position.x = boundary_min.x + particle_radius + collision_margin;
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.x) / 200.0) * 0.5);
        new_velocity.x = -velocity.x * damping_factor;
        new_velocity.y *= 0.95; // Horizontal friction
    } else if (position.x > boundary_max.x - particle_radius) {
        // Right wall collision
        new_position.x = boundary_max.x - particle_radius - collision_margin;
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.x) / 200.0) * 0.5);
        new_velocity.x = -velocity.x * damping_factor;
        new_velocity.y *= 0.95; // Horizontal friction
    }
    
    // Process Y boundaries
    if (position.y < boundary_min.y + particle_radius) {
        // Floor collision
        new_position.y = boundary_min.y + particle_radius + collision_margin;
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.y) / 200.0) * 0.5);
        new_velocity.y = -velocity.y * damping_factor;
        new_velocity.x *= 0.9; // Horizontal friction on floor
    } else if (position.y > boundary_max.y - particle_radius) {
        // Ceiling collision
        new_position.y = boundary_max.y - particle_radius - collision_margin;
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.y) / 200.0) * 0.5);
        new_velocity.y = -velocity.y * damping_factor;
        new_velocity.x *= 0.9; // Horizontal friction on ceiling
    }
    
    return vec4<f32>(new_position.x, new_position.y, new_velocity.x, new_velocity.y);
}

// Main compute shader function
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
         @builtin(local_invocation_id) local_id: vec3<u32>,
         @builtin(workgroup_id) group_id: vec3<u32>) {
    // Get thread indices
    let global_index = global_id.x;
    let local_index = local_id.x;
    
    // Skip if beyond array bounds
    if (global_index >= arrayLength(&particles)) {
        return;
    }
    
    // Cache simulation parameters locally for better register utilization
    let particle_radius = params.particle_radius;
    let boundary_min = params.boundary_min;
    let boundary_max = params.boundary_max;
    let dampening = params.boundary_dampening;
    let dt = params.dt;
    let max_speed = 1000.0;  // Maximum allowed velocity for stability
    
    // Get a local copy of the particle data for better memory access
    let particle = particles[global_index];
    shared_data[local_index].position = particle.position;
    shared_data[local_index].velocity = particle.velocity;
    
    // Ensure all threads have loaded their data
    workgroupBarrier();
    
    // Work with local variables for better register usage
    var position = shared_data[local_index].position;
    var velocity = shared_data[local_index].velocity;
    
    // Enforce maximum velocity for stability using faster dot product
    velocity = clamp_velocity(velocity, max_speed);
    
    // Update position with velocity - no adaptive timestep to match CPU
    position += velocity * dt;
    
    // Handle boundary collisions using vectorized approach
    let collision_result = handle_boundary_collision(position, velocity, boundary_min, 
                                                   boundary_max, particle_radius, dampening);
    position = collision_result.xy;
    velocity = collision_result.zw;
    
    // Apply small damping to prevent perpetual motion
    velocity *= 0.998;
    
    // Update the particle in global memory - do this only once at the end
    var updated_particle = particles[global_index];
    updated_particle.position = position;
    updated_particle.velocity = velocity;
    particles[global_index] = updated_particle;
} 
