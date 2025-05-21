// Update the density_pressure shader to use spatial hashing similar to Unity implementation
// This is a critical part of accurate SPH fluid simulation

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

// 2D cell offsets for neighboring cells
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

// Only use the essential bindings
@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_offsets: array<u32>;

// Get cell from position and radius
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return ((x * HASH_K1) ^ (y * HASH_K2)) + (x * y);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    let mixed = hash ^ (hash >> 16u);
    return mixed % table_size;
}

// Density kernel functions
fn poly6(dist_squared: f32, h_squared: f32) -> f32 {
    if (dist_squared >= h_squared) {
        return 0.0;
    }
    
    let h_squared_minus_r_squared = h_squared - dist_squared;
    // Pre-calculated scaling factor
    let poly6_scaling_factor = 4.0 / (3.14159 * pow(h_squared, 4.0));
    
    return poly6_scaling_factor * pow(h_squared_minus_r_squared, 3.0);
}

fn spiky_pow3(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow3_scaling_factor = 10.0 / (3.14159 * pow(h, 5.0));
    
    return spiky_pow3_scaling_factor * pow(h_minus_r, 3.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&particles) {
        return;
    }
    
    var particle = particles[index];
    let radius = params.smoothing_radius;
    let radius_squared = radius * radius;
    
    // Calculate density based on neighboring particles
    var density: f32 = 0.0;
    var near_density: f32 = 0.0;
    
    // Get the cell for this particle
    let cell = get_cell_2d(particle.position, radius);
    
    // Process each neighboring cell explicitly instead of using a dynamic loop index
    // Top row
    process_cell(cell + vec2<i32>(-1, 1), particle.position, radius_squared, index, &density, &near_density);
    process_cell(cell + vec2<i32>(0, 1), particle.position, radius_squared, index, &density, &near_density);
    process_cell(cell + vec2<i32>(1, 1), particle.position, radius_squared, index, &density, &near_density);
    
    // Middle row
    process_cell(cell + vec2<i32>(-1, 0), particle.position, radius_squared, index, &density, &near_density);
    process_cell(cell + vec2<i32>(0, 0), particle.position, radius_squared, index, &density, &near_density);
    process_cell(cell + vec2<i32>(1, 0), particle.position, radius_squared, index, &density, &near_density);
    
    // Bottom row
    process_cell(cell + vec2<i32>(-1, -1), particle.position, radius_squared, index, &density, &near_density);
    process_cell(cell + vec2<i32>(0, -1), particle.position, radius_squared, index, &density, &near_density);
    process_cell(cell + vec2<i32>(1, -1), particle.position, radius_squared, index, &density, &near_density);
    
    // Update particle densities
    particle.density = density;
    particle.near_density = near_density;
    
    // Calculate pressure from density
    particle.pressure = max(0.0, (density - params.target_density) * params.pressure_multiplier);
    particle.near_pressure = near_density * params.near_pressure_multiplier;
    
    // Write back to particle buffer
    particles[index] = particle;
}

// Helper function to process a neighboring cell
fn process_cell(neighbor_cell: vec2<i32>, pos: vec2<f32>, radius_squared: f32, self_index: u32, density_ptr: ptr<function, f32>, near_density_ptr: ptr<function, f32>) {
    let hash = hash_cell_2d(neighbor_cell);
    let key = key_from_hash(hash, arrayLength(&particles));
    
    // Get the starting index for this cell
    let start_index = spatial_offsets[key];
    var curr_index = start_index;
    
    // Iterate through particles in this cell
    while (curr_index < arrayLength(&particles)) {
        let neighbor_index = curr_index;
        curr_index = curr_index + 1u;
        
        // Skip if looking at self
        if (neighbor_index == self_index) {
            continue;
        }
        
        // Check if still in the same cell by comparing keys
        if (spatial_keys[neighbor_index] != key) {
            break;
        }
        
        let neighbor_pos = particles[neighbor_index].position;
        let offset_to_neighbor = neighbor_pos - pos;
        let sqr_dst = dot(offset_to_neighbor, offset_to_neighbor);
        
        // Skip if not within smoothing radius
        if (sqr_dst > radius_squared) {
            continue;
        }
        
        // Calculate density contribution
        *density_ptr += poly6(sqr_dst, radius_squared);
        
        // Calculate near density with spiky kernel
        let dst = sqrt(sqr_dst);
        *near_density_ptr += spiky_pow3(dst, params.smoothing_radius);
    }
} 
