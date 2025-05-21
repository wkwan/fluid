// Minimal update positions shader with only the essential bindings

struct Particle {
    position: vec2<f32>,
    padding0: vec2<f32>,
    velocity: vec2<f32>,
    padding1: vec2<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

struct FluidParams {
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    viscosity_strength: f32,
    boundary_dampening: f32,
    particle_radius: f32,
    dt: f32,
    
    boundary_min: vec2<f32>,
    boundary_min_padding: vec2<f32>,
    
    boundary_max: vec2<f32>,
    boundary_max_padding: vec2<f32>,
    
    gravity: vec2<f32>,
    gravity_padding: vec2<f32>,
    
    mouse_position: vec2<f32>,
    mouse_radius: f32,
    mouse_strength: f32,
    
    mouse_active: u32,
    mouse_repel: u32,
    padding: vec2<u32>,
}

// Only use the essential bindings
@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

// Handle boundary collisions
fn handle_boundaries(position: vec2<f32>, velocity: vec2<f32>) -> vec2<f32> {
    var new_velocity = velocity;
    
    // Handle left/right boundary collisions
    if position.x < params.boundary_min.x {
        new_velocity.x = abs(new_velocity.x) * params.boundary_dampening;
    } else if position.x > params.boundary_max.x {
        new_velocity.x = -abs(new_velocity.x) * params.boundary_dampening;
    }
    
    // Handle top/bottom boundary collisions
    if position.y < params.boundary_min.y {
        new_velocity.y = abs(new_velocity.y) * params.boundary_dampening;
    } else if position.y > params.boundary_max.y {
        new_velocity.y = -abs(new_velocity.y) * params.boundary_dampening;
    }
    
    return new_velocity;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&particles) {
        return;
    }
    
    var particle = particles[index];
    
    // Update position based on velocity
    particle.position += particle.velocity * params.dt;
    
    // Handle boundary collisions
    particle.velocity = handle_boundaries(particle.position, particle.velocity);
    
    // Clamp positions to stay in bounds
    particle.position.x = clamp(particle.position.x, params.boundary_min.x, params.boundary_max.x);
    particle.position.y = clamp(particle.position.y, params.boundary_min.y, params.boundary_max.y);
    
    particles[index] = particle;
} 
