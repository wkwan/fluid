// Fluid Simulation Density-Pressure Compute Shader
// Computes density and pressure for SPH fluid simulation

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

// SPH kernel functions
fn poly6(dist_squared: f32, h_squared: f32) -> f32 {
    if (dist_squared >= h_squared) {
        return 0.0;
    }
    let h9 = h_squared * h_squared * h_squared * h_squared * h_squared;
    let kernel_const = 315.0 / (64.0 * 3.14159265 * sqrt(h9));
    let kernel = kernel_const * pow(h_squared - dist_squared, 3);
    return kernel;
}

fn spiky_pow2(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    let kernel_const = 15.0 / (3.14159265 * h6);
    let kernel = kernel_const * pow(h - dist, 2);
    return kernel;
}

// Brute force O(n²) approach - in a production environment, 
// you'd want to use a grid/spatial hash for neighbor finding
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    let h = params.smoothing_radius;
    let h_squared = h * h;
    
    // Self-contribution
    var density = poly6(0.0, h_squared);
    var near_density = 0.0;
    
    let position = particles[index].position;
    
    // Calculate density from all other particles
    for (var i = 0u; i < arrayLength(&particles); i = i + 1u) {
        if (i == index) {
            continue;
        }
        
        let other_pos = particles[i].position;
        let offset = position - other_pos;
        let dist_squared = dot(offset, offset);
        
        if (dist_squared < h_squared) {
            let dist = sqrt(dist_squared);
            
            // Add density contribution
            density += poly6(dist_squared, h_squared);
            
            // Near density for surface tension
            if (dist > 0.0) {
                near_density += spiky_pow2(dist, h);
            }
        }
    }
    
    // Calculate pressure from density
    let density_error = density - params.target_density;
    let pressure = density_error * params.pressure_multiplier;
    let near_pressure = near_density * params.near_pressure_multiplier;
    
    // Update the particle density and pressure
    particles[index].density = density;
    particles[index].pressure = pressure;
    particles[index].near_density = near_density;
    particles[index].near_pressure = near_pressure;
} 