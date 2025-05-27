// Recompute velocities shader to match CPU behavior
// This recalculates velocity from position changes after position correction

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

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

// We need to store previous positions to calculate velocity
// For now, we'll use a simple approach and store them in a separate buffer
@group(0) @binding(4)
var<storage, read_write> previous_positions: array<vec2<f32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Skip if particle index is out of bounds
    if index >= arrayLength(&particles) {
        return;
    }
    
    var particle = particles[index];
    let dt = params.dt;
    
    // Get previous position (stored from before position correction)
    let previous_position = previous_positions[index];
    let current_position = particle.position;
    
    // Recalculate velocity from position change: velocity = (current_position - previous_position) / dt
    if (dt > 0.0001) {
        particle.velocity = (current_position - previous_position) / dt;
    }
    
    // Store the updated particle
    particles[index] = particle;
} 