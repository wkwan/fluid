// Optimized viscosity shader with Thread Group Shared Memory
// Implements memory coalescing for better GPU performance

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

// Shared memory structure for caching particle data
struct CachedParticleData {
    position: vec2<f32>,
    velocity: vec2<f32>,
}

// Thread Group Shared Memory (TGSM) for particle data caching
var<workgroup> shared_particles: array<CachedParticleData, 64>;
// TGSM for cell keys and offsets
var<workgroup> shared_cell_keys: array<u32, 9>;
var<workgroup> shared_cell_offsets: array<u32, 9>;

// Hash table size constants - must be power of two
const TABLE_SIZE: u32 = 4096u;
const TABLE_SIZE_MASK: u32 = 4095u; // TABLE_SIZE - 1

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
    return (x * 15823u) ^ (y * 9737333u);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32) -> u32 {
    return hash & TABLE_SIZE_MASK; // Fast modulo with bit mask
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
    
    // Get particle from global memory once
    var particle = particles[index];
    let radius = params.smoothing_radius;
    let radius_squared = radius * radius;
    let pos = particle.position;
    let viscosity_strength = params.viscosity_strength;
    let dt = params.dt;
    
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
    
    var velocity_change = vec2<f32>(0.0, 0.0);
    
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
            var neighbor_velocity = vec2<f32>(0.0, 0.0);
            var found_in_cache = false;
            
            // Check if neighbor is in current workgroup's shared memory
            let workgroup_start = group_id.x * 64u;
            let workgroup_end = workgroup_start + 64u;
            
            if (neighbor_index >= workgroup_start && neighbor_index < workgroup_end) {
                let shared_idx = neighbor_index - workgroup_start;
                if (shared_idx < 64u) { // Explicit bounds check
                    neighbor_pos = shared_particles[shared_idx].position;
                    neighbor_velocity = shared_particles[shared_idx].velocity;
                    found_in_cache = true;
                }
            }
            
            // Fall back to global memory if not in shared memory
            if (!found_in_cache) {
                let neighbor = particles[neighbor_index];
                neighbor_pos = neighbor.position;
                neighbor_velocity = neighbor.velocity;
            }
            
            let offset_to_neighbor = neighbor_pos - pos;
            let sqr_dst = dot(offset_to_neighbor, offset_to_neighbor);
            
            // Process if within smoothing radius
            if (sqr_dst < radius_squared && sqr_dst > 0.0001) {
                let dst = sqrt(sqr_dst);
                
                // Simple viscosity effect - velocity averaging based on distance
                let influence = 1.0 - dst / radius;
                velocity_change += (neighbor_velocity - particle.velocity) * influence * viscosity_strength;
            }
            
            curr_index = curr_index + 1u;
        }
    }
    
    // Apply viscosity change to particle velocity
    particle.velocity += velocity_change * dt;
    
    // Apply damping for stability - stronger damping at high velocities
    let speed = length(particle.velocity);
    let damping_factor = select(0.98, 0.95, speed > 100.0);
    particle.velocity *= damping_factor;
    
    // Write back to global memory
    particles[index] = particle;
} 
