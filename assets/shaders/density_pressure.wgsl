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

// Shared memory for RTX 4090 optimization
// Each workgroup will cache particle data for faster access
struct CachedParticle {
    position: vec2<f32>,
    key: u32,
}

var<workgroup> shared_particles: array<CachedParticle, 128>;

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

// Calculate optimal cell size based on particle radius
fn calculate_optimal_cell_size(particle_radius: f32, smoothing_radius: f32) -> f32 {
    return max(smoothing_radius, particle_radius * 4.0);
}

// SPH kernel functions with optimized math
fn poly6(dist_squared: f32, h_squared: f32) -> f32 {
    if (dist_squared >= h_squared) {
        return 0.0;
    }
    
    // Precompute constants for better performance
    let factor = h_squared - dist_squared;
    let factor_squared = factor * factor;
    
    // Use approximate math where possible for RTX 4090
    let h9 = h_squared * h_squared * h_squared * h_squared * h_squared;
    let kernel_const = 315.0 / (64.0 * 3.14159265 * sqrt(h9));
    
    return kernel_const * factor * factor_squared;
}

fn spiky_pow2(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let factor = h - dist;
    let h6 = h * h * h * h * h * h;
    let kernel_const = 15.0 / (3.14159265 * h6);
    
    return kernel_const * factor * factor;
}

// Convert floating point position into an integer cell coordinate
fn get_cell_2d(position: vec2<f32>, cell_size: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / cell_size));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return ((x * HASH_K1) ^ (y * HASH_K2)) + (x * y);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    let mixed = hash ^ (hash >> 16);
    return mixed % table_size;
}

// Using spatial hashing for efficient neighbor finding
// RTX 4090 optimized workgroup size
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    // Cache this particle's data in shared memory
    if (local_index < 128u) {
        shared_particles[local_index].position = particles[index].position;
        shared_particles[local_index].key = spatial_keys[index];
    }
    
    // Ensure all threads have cached their data
    workgroupBarrier();
    
    let h = params.smoothing_radius;
    let h_squared = h * h;
    let optimal_cell_size = calculate_optimal_cell_size(params.particle_radius, h);
    
    // Self-contribution
    var density = poly6(0.0, h_squared);
    var near_density = 0.0;
    
    let position = shared_particles[local_index].position;
    let origin_cell = get_cell_2d(position, optimal_cell_size);
    
    // Calculate density using spatial hash for neighbor finding
    for (var i = 0u; i < 9u; i = i + 1u) {
        let neighbor_cell = origin_cell + OFFSETS_2D[i];
        let hash = hash_cell_2d(neighbor_cell);
        let key = key_from_hash(hash, u32(arrayLength(&particles)));
        
        // Get the starting index for this key
        var curr_index = spatial_offsets[key];
        
        // Skip empty cells early
        if (curr_index == 0xFFFFFFFFu) {
            continue;
        }
        
        // Iterate through all particles in this cell
        while (curr_index < arrayLength(&particles)) {
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
    
    // Calculate pressure from density with improved stability
    let target_density = max(params.target_density, 0.1); // Prevent division by zero
    let density_error = density - target_density;
    
    // Use non-linear pressure model for better stability at high densities
    var pressure = density_error * params.pressure_multiplier;
    
    // Enforce minimum pressure for stability (prevents clustering)
    if (density > target_density * 1.5) {
        pressure *= 1.5; // Extra pressure for high density regions
    }
    
    let near_pressure = near_density * params.near_pressure_multiplier;
    
    // Update the particle density and pressure
    particles[index].density = density;
    particles[index].pressure = pressure;
    particles[index].near_density = near_density;
    particles[index].near_pressure = near_pressure;
} 