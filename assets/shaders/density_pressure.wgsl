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

// 2D cell offsets for neighboring cells (9 total including center)
const OFFSETS_2D: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, 1),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(-1, -1),
    vec2<i32>(0, -1),
    vec2<i32>(1, -1)
);

// Constants used for hashing
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_indices: array<u32>;

@group(0) @binding(4)
var<storage, read_write> spatial_offsets: array<u32>;

// SPH kernel functions
fn poly6(dist_squared: f32, h_squared: f32) -> f32 {
    if (dist_squared >= h_squared) {
        return 0.0;
    }
    let h9 = h_squared * h_squared * h_squared * h_squared * h_squared;
    let kernel_const = 315.0 / (64.0 * 3.14159265 * sqrt(h9));
    let kernel = kernel_const * pow(h_squared - dist_squared, 3.0);
    return kernel;
}

fn spiky_pow2(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    let kernel_const = 15.0 / (3.14159265 * h6);
    let kernel = kernel_const * pow(h - dist, 2.0);
    return kernel;
}

// Convert floating point position into an integer cell coordinate
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return x * HASH_K1 + y * HASH_K2;
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    return hash % table_size;
}

// Using spatial hashing for efficient neighbor finding
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
    let origin_cell = get_cell_2d(position, h);
    
    // Calculate density using spatial hash for neighbor finding
    for (var i = 0u; i < 9u; i = i + 1u) {
        let neighbor_cell = origin_cell + OFFSETS_2D[i];
        let hash = hash_cell_2d(neighbor_cell);
        let key = key_from_hash(hash, u32(arrayLength(&particles)));
        
        // Get the starting index for this key
        var curr_index = spatial_offsets[key];
        
        // Iterate through all particles in this cell
        while (curr_index < arrayLength(&particles)) {
            if (curr_index == 0xFFFFFFFFu) {
                // Cell is empty (sentinel value)
                break;
            }
            
            // Check if we're still in the same cell
            if (key != spatial_keys[curr_index]) {
                break;
            }
            
            // Skip self-interaction
            if (curr_index != index) {
                let other_pos = particles[curr_index].position;
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
            
            // Move to next particle in this cell
            curr_index = curr_index + 1u;
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