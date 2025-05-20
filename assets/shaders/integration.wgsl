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

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    let particle_radius = params.particle_radius;
    let boundary_min = params.boundary_min;
    let boundary_max = params.boundary_max;
    let dampening = params.boundary_dampening;
    
    var position = particles[index].position;
    var velocity = particles[index].velocity;
    
    // Update position with velocity
    position += velocity * params.dt;
    
    // Handle boundary collisions
    // X boundaries
    if (position.x < boundary_min.x + particle_radius) {
        position.x = boundary_min.x + particle_radius;
        velocity.x = -velocity.x * dampening;
    } else if (position.x > boundary_max.x - particle_radius) {
        position.x = boundary_max.x - particle_radius;
        velocity.x = -velocity.x * dampening;
    }
    
    // Y boundaries
    if (position.y < boundary_min.y + particle_radius) {
        position.y = boundary_min.y + particle_radius;
        velocity.y = -velocity.y * dampening;
    } else if (position.y > boundary_max.y - particle_radius) {
        position.y = boundary_max.y - particle_radius;
        velocity.y = -velocity.y * dampening;
    }
    
    // Update the particle position and velocity
    particles[index].position = position;
    particles[index].velocity = velocity;
} 