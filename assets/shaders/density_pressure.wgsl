// Highly optimized density_pressure shader with shared memory caching
// Matches Unity's implementation with improved memory access patterns

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

// 2D cell offsets for neighboring cells in array format
const CELL_OFFSETS_X: array<i32, 9> = array<i32, 9>(-1, 0, 1, -1, 0, 1, -1, 0, 1);
const CELL_OFFSETS_Y: array<i32, 9> = array<i32, 9>(1, 1, 1, 0, 0, 0, -1, -1, -1);

// Constants used for hashing - match CPU implementation
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;
// Table size constants - must be power of two
const TABLE_SIZE: u32 = 4096u;
const TABLE_SIZE_MASK: u32 = 4095u; // TABLE_SIZE - 1

// Shared memory structure for caching particle data
struct CachedParticle {
    position: vec2<f32>,
    key: u32,
}

// Thread Group Shared Memory (TGSM) for particle data caching
var<workgroup> shared_particles: array<CachedParticle, 64>;
// TGSM for cell offsets - cache up to 9 cells for faster lookups
var<workgroup> shared_cell_offsets: array<u32, 9>;
// TGSM for key cache - store spatial hash keys for neighboring cells
var<workgroup> shared_cell_keys: array<u32, 9>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_offsets: array<u32>;

// Get cell from position and radius - optimized version
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer - optimized XOR version
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return (x * HASH_K1) ^ (y * HASH_K2);
}

// Get key from hash for a table of given size - optimized with bitmask
fn key_from_hash(hash: u32) -> u32 {
    return hash & TABLE_SIZE_MASK; // Fast modulo with bit mask
}

// Density kernel functions - optimized with precomputed constants
fn poly6(dist_squared: f32, h_squared: f32) -> f32 {
    if (dist_squared >= h_squared) {
        return 0.0;
    }
    
    let h_squared_minus_r_squared = h_squared - dist_squared;
    // Pre-calculated scaling factor
    let poly6_scaling_factor = 4.0 / (3.14159 * pow(h_squared, 4.0));
    
    return poly6_scaling_factor * h_squared_minus_r_squared * h_squared_minus_r_squared * h_squared_minus_r_squared;
}

fn spiky_pow3(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow3_scaling_factor = 10.0 / (3.14159 * pow(h, 5.0));
    
    return spiky_pow3_scaling_factor * h_minus_r * h_minus_r * h_minus_r;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
         @builtin(local_invocation_id) local_id: vec3<u32>,
         @builtin(workgroup_id) group_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Load particle data into shared memory for this workgroup
    if index < arrayLength(&particles) {
        shared_particles[local_index].position = particles[index].position;
        shared_particles[local_index].key = spatial_keys[index];
    } else {
        // Use invalid values for particles outside array bounds
        shared_particles[local_index].position = vec2<f32>(-99999.0, -99999.0);
        shared_particles[local_index].key = 0xFFFFFFFFu;
    }
    
    // Wait for all threads to finish loading shared memory
    workgroupBarrier();
    
    // Skip if particle index is out of bounds
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Get particle from global memory once
    var particle = particles[index];
    let radius = params.smoothing_radius;
    let radius_squared = radius * radius;
    let pos = particle.position;
    
    // Calculate density based on neighboring particles
    var density: f32 = 0.0;
    var near_density: f32 = 0.0;
    
    // Add self contribution
    density += poly6(0.0, radius_squared);
    
    // Get the cell for this particle
    let cell = get_cell_2d(pos, radius * 2.0);
    
    // Precompute cell keys and offsets for neighboring cells - first thread in workgroup only
    if local_index == 0u {
        for (var i = 0u; i < 9u; i++) {
            // Use hardcoded offsets instead of arrays to avoid dynamic indexing
            let offset_x = select(-1, select(0, 1, i % 3u == 1u), i % 3u == 2u);
            let offset_y = select(-1, select(0, 1, (i / 3u) == 1u), (i / 3u) == 0u);
            
            let neighbor_cell = vec2<i32>(
                cell.x + offset_x,
                cell.y + offset_y
            );
            
            let hash = hash_cell_2d(neighbor_cell);
            let key = key_from_hash(hash);
            
            shared_cell_keys[i] = key;
            shared_cell_offsets[i] = spatial_offsets[key];
        }
    }
    
    // Wait for thread 0 to finish computing cell offsets
    workgroupBarrier();
    
    // Process each neighboring cell using shared memory
    for (var offset_idx = 0u; offset_idx < 9u; offset_idx++) {
        let key = shared_cell_keys[offset_idx];
        let start_index = shared_cell_offsets[offset_idx];
        if (start_index == 0xFFFFFFFFu) {
            continue; // Skip empty cells
        }
        
        var curr_index = start_index;
        
        // Iterate through particles in this cell
        while (curr_index < arrayLength(&particles)) {
            let neighbor_index = curr_index;
            
            // Check if still in the same cell by comparing keys
            if (spatial_keys[neighbor_index] != key) {
                break;
            }
            
            // Skip self
            if (neighbor_index == index) {
                curr_index = curr_index + 1u;
                continue;
            }
            
            // First check if this neighbor is in our shared memory cache
            var neighbor_pos = vec2<f32>(-99999.0, -99999.0);
            var found_in_cache = false;
            
            // Check if the neighbor is in the current workgroup's shared memory
            let workgroup_start = group_id.x * 64u;
            let workgroup_end = workgroup_start + 64u;
            
            if (neighbor_index >= workgroup_start && neighbor_index < workgroup_end) {
                let shared_idx = neighbor_index - workgroup_start;
                if (shared_idx < 64u) { // Explicit bounds check to help the compiler
                    neighbor_pos = shared_particles[shared_idx].position;
                    found_in_cache = true;
                }
            }
            
            // If not in cache, fetch from global memory
            if (!found_in_cache) {
                neighbor_pos = particles[neighbor_index].position;
            }
            
            let offset_to_neighbor = neighbor_pos - pos;
            let sqr_dst = dot(offset_to_neighbor, offset_to_neighbor);
            
            // Process if within smoothing radius
            if (sqr_dst < radius_squared && sqr_dst > 0.0001) {
                // Calculate density contribution
                density += poly6(sqr_dst, radius_squared);
                
                // Calculate near density with spiky kernel
                let dst = sqrt(sqr_dst);
                near_density += spiky_pow3(dst, radius);
            }
            
            curr_index = curr_index + 1u;
        }
    }
    
    // Update particle densities
    particle.density = density;
    particle.near_density = near_density;
    
    // Calculate pressure from density
    particle.pressure = max(0.0, (density - params.target_density) * params.pressure_multiplier);
    particle.near_pressure = near_density * params.near_pressure_multiplier;
    
    // Write back to particle buffer only once at the end
    particles[index] = particle;
} 
