// Fluid Simulation Integration Compute Shader
// Updates positions and handles boundary collisions

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

// Shared memory for RTX 4090 optimization
struct CachedParticleIntegration {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

var<workgroup> shared_particles: array<CachedParticleIntegration, 128>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

// Function to clamp velocity for stability
fn clamp_velocity(velocity: vec2<f32>, max_speed: f32) -> vec2<f32> {
    let speed = length(velocity);
    if (speed > max_speed) {
        return velocity * (max_speed / speed);
    }
    return velocity;
}

// RTX 4090 optimized workgroup size
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    // Cache particle data in shared memory for better performance
    if (local_index < 128u) {
        shared_particles[local_index].position = particles[index].position;
        shared_particles[local_index].velocity = particles[index].velocity;
    }
    
    // Ensure all threads have cached their data
    workgroupBarrier();
    
    let particle_radius = params.particle_radius;
    let boundary_min = params.boundary_min;
    let boundary_max = params.boundary_max;
    let dampening = params.boundary_dampening;
    
    var position = shared_particles[local_index].position;
    var velocity = shared_particles[local_index].velocity;
    
    // Enforce maximum velocity for stability
    let max_speed = 1000.0;
    velocity = clamp_velocity(velocity, max_speed);
    
    // Adaptive timestep based on velocity magnitude
    let base_dt = params.dt;
    let speed = length(velocity);
    let adaptive_dt = select(base_dt, base_dt * min(1.0, 200.0 / speed), speed > 200.0);
    
    // Update position with velocity
    position += velocity * adaptive_dt;
    
    // Handle boundary collisions with improved damping
    let collision_margin = 0.01; // Small margin to prevent sticking to walls
    
    // X boundaries
    if (position.x < boundary_min.x + particle_radius) {
        // Apply position correction
        position.x = boundary_min.x + particle_radius + collision_margin;
        
        // Apply velocity damping with progressive factor (more damping at high speeds)
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.x) / 200.0) * 0.5);
        velocity.x = -velocity.x * damping_factor;
        
        // Additional horizontal friction when hitting vertical walls
        velocity.y *= 0.95;
    } else if (position.x > boundary_max.x - particle_radius) {
        position.x = boundary_max.x - particle_radius - collision_margin;
        
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.x) / 200.0) * 0.5);
        velocity.x = -velocity.x * damping_factor;
        
        // Additional horizontal friction when hitting vertical walls
        velocity.y *= 0.95;
    }
    
    // Y boundaries
    if (position.y < boundary_min.y + particle_radius) {
        position.y = boundary_min.y + particle_radius + collision_margin;
        
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.y) / 200.0) * 0.5);
        velocity.y = -velocity.y * damping_factor;
        
        // Additional horizontal friction when hitting floor
        velocity.x *= 0.9;
    } else if (position.y > boundary_max.y - particle_radius) {
        position.y = boundary_max.y - particle_radius - collision_margin;
        
        let damping_factor = dampening * (1.0 + min(1.0, abs(velocity.y) / 200.0) * 0.5);
        velocity.y = -velocity.y * damping_factor;
        
        // Additional horizontal friction when hitting ceiling
        velocity.x *= 0.9;
    }
    
    // Apply small damping to prevent perpetual motion
    velocity *= 0.998;
    
    // Update the particle position and velocity
    particles[index].position = position;
    particles[index].velocity = velocity;
} 