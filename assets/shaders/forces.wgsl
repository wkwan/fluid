// Fluid Simulation Forces Compute Shader
// Computes pressure and viscosity forces for SPH fluid simulation

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    padding: array<f32, 8>,
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
    
    // Vec4 aligned group 3 and 4
    boundary_min: vec2<f32>,
    boundary_max: vec2<f32>,
    gravity: vec2<f32>,
    
    // Vec4 aligned group 5
    mouse_position: vec2<f32>,
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 6
    mouse_active: u32,
    mouse_repel: u32,
    padding1: u32,
    padding2: u32,
}

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

// SPH kernel derivatives for force calculations
fn spiky_pow3_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h || dist <= 0.0) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    let kernel_const = -45.0 / (3.14159265 * h6);
    let kernel = kernel_const * (h - dist) * (h - dist) / dist;
    return kernel;
}

fn spiky_pow2_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h || dist <= 0.0) {
        return 0.0;
    }
    let h5 = h * h * h * h * h;
    let kernel_const = -30.0 / (3.14159265 * h5);
    let kernel = kernel_const * (h - dist) / dist;
    return kernel;
}

fn viscosity_kernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    let h3 = h * h * h;
    let kernel_const = 15.0 / (2.0 * 3.14159265 * h3);
    let kernel = kernel_const * (-pow(dist / h, 3.0) + 2.0 * pow(dist / h, 2.0) - 0.5);
    return kernel;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    let h = params.smoothing_radius;
    let position = particles[index].position;
    let velocity = particles[index].velocity;
    
    // Apply gravity
    var force = params.gravity;
    
    // Apply mouse force if active
    if (params.mouse_active != 0u) {
        let to_mouse = params.mouse_position - position;
        let dist_to_mouse = length(to_mouse);
        
        if (dist_to_mouse < params.mouse_radius) {
            let force_direction = to_mouse / max(dist_to_mouse, 0.001); // Avoid division by zero
            let force_strength = params.mouse_strength * (1.0 - dist_to_mouse / params.mouse_radius);
            
            if (params.mouse_repel != 0u) {
                // Repel
                force -= force_direction * force_strength * params.dt;
            } else {
                // Attract
                force += force_direction * force_strength * params.dt;
            }
        }
    }
    
    // Calculate pressure and viscosity forces
    for (var i = 0u; i < arrayLength(&particles); i = i + 1u) {
        if (i == index) {
            continue;
        }
        
        let other_pos = particles[i].position;
        let other_vel = particles[i].velocity;
        let offset = position - other_pos;
        let dist_squared = dot(offset, offset);
        
        if (dist_squared < h * h && dist_squared > 0.0) {
            let dist = sqrt(dist_squared);
            let dir = offset / dist;
            
            // Pressure force calculation
            let shared_pressure = (particles[index].pressure + particles[i].pressure) * 0.5;
            let shared_near_pressure = (particles[index].near_pressure + particles[i].near_pressure) * 0.5;
            
            let pressure_force = dir * (
                spiky_pow3_derivative(dist, h) * shared_pressure +
                spiky_pow2_derivative(dist, h) * shared_near_pressure
            );
            
            // Viscosity force calculation
            let vel_diff = other_vel - velocity;
            let visc_strength = params.viscosity_strength * viscosity_kernel(dist, h);
            let viscosity_force = vel_diff * visc_strength;
            
            // Add forces
            force += pressure_force + viscosity_force;
        }
    }
    
    // Update velocity with forces
    let new_velocity = velocity + force * params.dt;
    
    // Update particle velocity
    particles[index].velocity = new_velocity;
} 