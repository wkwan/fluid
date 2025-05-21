// Minimal spatial hash shader with only the essential bindings

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

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Simple placeholder implementation
    // In a real implementation, this would compute spatial hashes
    // For now, we'll just apply a small force based on position
    var particle = particles[index];
    
    // Apply a small force based on position (simulating spatial interaction)
    let force_x = sin(particle.position.x * 0.01) * 0.1;
    let force_y = cos(particle.position.y * 0.01) * 0.1;
    
    particle.velocity += vec2<f32>(force_x, force_y) * params.dt;
    
    particles[index] = particle;
} 
