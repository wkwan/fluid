// Update pressure force shader to use spatial hashing with optimized memory access patterns
// Implements Thread Group Shared Memory and memory coalescing

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

// Cached particle data for TGSM
struct CachedParticleData {
    position: vec2<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Thread Group Shared Memory (TGSM) for particle data caching
var<workgroup> shared_particles: array<CachedParticleData, 64>;
// TGSM for cell keys and offsets
var<workgroup> shared_cell_keys: array<u32, 9>;
var<workgroup> shared_cell_offsets: array<u32, 9>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_offsets: array<u32>;

// Hash table size constants - must be power of two
const TABLE_SIZE: u32 = 4096u;
const TABLE_SIZE_MASK: u32 = 4095u; // TABLE_SIZE - 1

// Get cell from position and radius
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return (x * 15823u) ^ (y * 9737333u);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32) -> u32 {
    return hash & TABLE_SIZE_MASK; // Fast modulo with bit mask
}

// Derivative functions for calculating forces
fn spiky_pow3_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow3_derivative_scaling_factor = 30.0 / (3.14159 * pow(h, 5.0));
    
    return -spiky_pow3_derivative_scaling_factor * pow(h_minus_r, 2.0);
}

fn spiky_pow2_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    
    let h_minus_r = h - dist;
    // Pre-calculated scaling factor
    let spiky_pow2_derivative_scaling_factor = 12.0 / (3.14159 * pow(h, 4.0));
    
    return -spiky_pow2_derivative_scaling_factor * h_minus_r;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
         @builtin(local_invocation_id) local_id: vec3<u32>,
         @builtin(workgroup_id) group_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Load particle data into shared memory for this workgroup
    if index < arrayLength(&particles) {
        let particle = particles[index];
        shared_particles[local_index].position = particle.position;
        shared_particles[local_index].density = particle.density;
        shared_particles[local_index].pressure = particle.pressure;
        shared_particles[local_index].near_density = particle.near_density;
        shared_particles[local_index].near_pressure = particle.near_pressure;
    } else {
        // Use invalid values for particles outside array bounds
        shared_particles[local_index].position = vec2<f32>(-99999.0, -99999.0);
        shared_particles[local_index].density = 0.0;
        shared_particles[local_index].pressure = 0.0;
        shared_particles[local_index].near_density = 0.0;
        shared_particles[local_index].near_pressure = 0.0;
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
    let pressure = particle.pressure;
    let near_pressure = particle.near_pressure;
    let density = particle.density;
    
    // Get the cell for this particle
    let cell = get_cell_2d(pos, radius * 2.0);
    
    // Precompute cell keys and offsets for neighboring cells - first thread in workgroup only
    if local_index == 0u {
        for (var i = 0u; i < 9u; i++) {
            // Use hardcoded offsets to avoid dynamic indexing
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
    
    // Initialize pressure force
    var pressure_force = vec2<f32>(0.0, 0.0);
    
    // Process all 9 neighboring cells using shared memory
    for (var i = 0u; i < 9u; i++) {
        let key = shared_cell_keys[i];
        let start_index = shared_cell_offsets[i];
        
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
            
            // First try to get neighbor data from shared memory
            var neighbor_pos = vec2<f32>(-99999.0, -99999.0);
            var neighbor_density = 0.0;
            var neighbor_pressure = 0.0;
            var neighbor_near_density = 0.0;
            var neighbor_near_pressure = 0.0;
            var found_in_cache = false;
            
            // Check if neighbor is in current workgroup's shared memory
            let workgroup_start = group_id.x * 64u;
            let workgroup_end = workgroup_start + 64u;
            
            if (neighbor_index >= workgroup_start && neighbor_index < workgroup_end) {
                let shared_idx = neighbor_index - workgroup_start;
                if (shared_idx < 64u) { // Explicit bounds check
                    neighbor_pos = shared_particles[shared_idx].position;
                    neighbor_density = shared_particles[shared_idx].density;
                    neighbor_pressure = shared_particles[shared_idx].pressure;
                    neighbor_near_density = shared_particles[shared_idx].near_density;
                    neighbor_near_pressure = shared_particles[shared_idx].near_pressure;
                    found_in_cache = true;
                }
            }
            
            // Fall back to global memory if not in shared memory
            if (!found_in_cache) {
                let neighbor = particles[neighbor_index];
                neighbor_pos = neighbor.position;
                neighbor_density = neighbor.density;
                neighbor_pressure = neighbor.pressure;
                neighbor_near_density = neighbor.near_density;
                neighbor_near_pressure = neighbor.near_pressure;
            }
            
            let offset_to_neighbor = neighbor_pos - pos;
            let sqr_dst = dot(offset_to_neighbor, offset_to_neighbor);
            
            // Skip if not within smoothing radius
            if (sqr_dst > radius_squared || sqr_dst < 0.0001) {
                curr_index = curr_index + 1u;
                continue;
            }
            
            let dst = sqrt(sqr_dst);
            
            // Avoid division by zero using a clearer approach
            var dir_to_neighbor = vec2<f32>(0.0, 1.0);
            if (dst > 0.001) {
                dir_to_neighbor = offset_to_neighbor / dst;
            }
            
            // Calculate shared pressure
            let shared_pressure = (pressure + neighbor_pressure) * 0.5;
            let shared_near_pressure = (near_pressure + neighbor_near_pressure) * 0.5;
            
            // Calculate pressure force using derivatives of the smoothing kernels
            if (neighbor_density > 0.0) {
                // Standard pressure force
                let pressure_force_magnitude = spiky_pow2_derivative(dst, radius) 
                                * shared_pressure / max(0.001, neighbor_density);
                
                // Near pressure force (provides more stable interactions at very close distances)
                let near_pressure_force_magnitude = spiky_pow3_derivative(dst, radius) 
                                * shared_near_pressure / max(0.001, neighbor_near_density);
                
                // Combine forces (removed additional_repulsion - will be handled in position correction)
                let total_force_magnitude = pressure_force_magnitude + near_pressure_force_magnitude;
                pressure_force += dir_to_neighbor * total_force_magnitude;
            }
            
            curr_index = curr_index + 1u;
        }
    }
    
    // Apply pressure force
    // Scale force by 1/density to account for mass conservation
    let acceleration = pressure_force / max(0.001, density);
    
    // Update velocity based on the pressure force
    particle.velocity += acceleration * params.dt;
    
    // Store the updated particle
    particles[index] = particle;
} 
