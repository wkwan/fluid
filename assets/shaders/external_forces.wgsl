// Optimized external forces shader with memory coalescing
// Implements efficient memory access patterns for better GPU performance

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

// Shared memory structure for better memory access
struct ParticleData {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

// Thread Group Shared Memory (TGSM) for particle data caching
var<workgroup> shared_particles: array<ParticleData, 64>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

// Gravity calculation function with faster vectorized operations
fn apply_gravity(velocity: vec2<f32>, gravity: vec2<f32>, dt: f32) -> vec2<f32> {
    return velocity + gravity * dt;
}

// Mouse force calculation function
fn calculate_mouse_force(position: vec2<f32>, velocity: vec2<f32>, 
                         mouse_pos: vec2<f32>, mouse_radius: f32, 
                         mouse_strength: f32, repel: bool, dt: f32) -> vec2<f32> {
    let offset_to_mouse = mouse_pos - position;
    let dist_squared = dot(offset_to_mouse, offset_to_mouse);
    
    if (dist_squared < mouse_radius * mouse_radius && dist_squared > 0.001) {
        let dist = sqrt(dist_squared);
        let dir = offset_to_mouse / dist;
        let strength = mouse_strength * (1.0 - dist / mouse_radius);
        
        // Apply attraction or repulsion based on mouse button
        let force_dir = select(dir, -dir, repel);
        return velocity + force_dir * strength * dt;
    }
    
    return velocity;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
         @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Load particle data into shared memory for this workgroup
    if index < arrayLength(&particles) {
        let particle = particles[index];
        shared_particles[local_index].position = particle.position;
        shared_particles[local_index].velocity = particle.velocity;
    } else {
        // Use invalid values for particles outside array bounds
        shared_particles[local_index].position = vec2<f32>(-99999.0, -99999.0);
        shared_particles[local_index].velocity = vec2<f32>(0.0, 0.0);
    }
    
    // Wait for all threads to finish loading shared memory
    workgroupBarrier();
    
    // Skip if particle index is out of bounds
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Cache parameter reads to reduce memory traffic
    let dt = params.dt;
    let gravity = params.gravity;
    let mouse_active = params.mouse_active != 0u;
    let mouse_position = params.mouse_position;
    let mouse_radius = params.mouse_radius;
    let mouse_strength = params.mouse_strength;
    let mouse_repel = params.mouse_repel != 0u;
    
    // Work with local copies for better register usage
    var position = shared_particles[local_index].position;
    var velocity = shared_particles[local_index].velocity;
    
    // Apply gravity force - vectorized approach for better performance
    velocity = apply_gravity(velocity, gravity, dt);
    
    // Apply mouse interaction if active - specialized function for better branching
    if (mouse_active) {
        velocity = calculate_mouse_force(position, velocity, mouse_position, 
                                        mouse_radius, mouse_strength, mouse_repel, dt);
    }
    
    // Write back to global memory only once
    var particle = particles[index];
    particle.velocity = velocity;
    particles[index] = particle;
} 
