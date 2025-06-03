// 3D Position Prediction Compute Shader
// Mirrors Unity's ExternalForces kernel - applies gravity and predicts positions

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

struct PredictedParticle3D {
    predicted_position: vec3<f32>,
    padding: f32,
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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<uniform> params: FluidParams3D;
@group(0) @binding(2) var<storage, read_write> spatial_keys_dummy: array<u32>;
@group(0) @binding(3) var<storage, read_write> spatial_offsets_dummy: array<u32>;
@group(0) @binding(4) var<storage, read_write> neighbor_counts_dummy: array<u32>;
@group(0) @binding(5) var<storage, read_write> neighbor_indices_dummy: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= arrayLength(&particles) {
        return;
    }
    
    var particle = particles[index];
    
    // Safety bounds for position and velocity
    let max_coord = 1000.0;
    let max_velocity = 50.0;
    
    // Clamp position to reasonable bounds
    particle.position = clamp(particle.position, vec3<f32>(-max_coord), vec3<f32>(max_coord));
    
    // Clamp velocity to reasonable bounds
    let vel_magnitude = length(particle.velocity);
    if vel_magnitude > max_velocity {
        particle.velocity = normalize(particle.velocity) * max_velocity;
    }
    
    // Apply gravity to velocity
    particle.velocity += params.gravity * params.dt;
    
    // Apply mouse interaction if active (with safety checks)
    if params.mouse_active != 0u {
        let to_mouse = params.mouse_position - particle.position;
        let distance = length(to_mouse);
        
        if distance < params.mouse_radius && distance > 0.001 {
            var direction: vec3<f32>;
            if params.mouse_repel != 0u {
                direction = -to_mouse;
            } else {
                direction = to_mouse;
            }
            let falloff = (1.0 - distance / params.mouse_radius);
            let normalized_direction = direction / distance; // Safe since distance > 0.001
            let force = normalized_direction * params.mouse_strength * falloff * params.dt;
            
            particle.velocity += force;
        }
    }
    
    // Store original position in force field (temporarily) 
    // We'll restore it later after force calculations
    particle.force = particle.position;
    
    // Predict new position for density/force calculations
    // This follows Unity's approach: predicted_pos = pos + vel * dt
    particle.position = particle.position + particle.velocity * params.dt;
    
    // Final safety clamp to prevent runaway particles
    particle.position = clamp(particle.position, vec3<f32>(-max_coord), vec3<f32>(max_coord));
    particle.velocity = clamp(particle.velocity, vec3<f32>(-max_velocity), vec3<f32>(max_velocity));
    
    particles[index] = particle;
} 