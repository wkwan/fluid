// Position correction shader to prevent particle overlapping
// Fixed to use correct binding layout and spatial hash logic

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
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;

// Get cell from position and radius
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return (x * HASH_K1) ^ (y * HASH_K2);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32) -> u32 {
    return hash & TABLE_SIZE_MASK; // Fast modulo with bit mask
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Skip if particle index is out of bounds
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Get particle from global memory
    var particle = particles[index];
    let radius = params.smoothing_radius;
    let pos = particle.position;
    let dt = params.dt;
    let dt_squared = dt * dt;
    
    // Clear debug info at start of frame
    particle.padding0 = vec2<f32>(0.0, 0.0); // [overlap_count, min_distance_found]
    particle.padding1 = vec2<f32>(0.0, 0.0); // [total_repulsion_force, max_displacement]
    
    // Calculate pressure from density like CPU does (paper equations 2 and 5)
    let pressure = params.pressure_multiplier * (particle.density - params.target_density);
    let near_pressure = params.near_pressure_multiplier * particle.near_density;
    
    // Get the cell for this particle (use same cell size as other shaders)
    let cell_size = radius * 2.0;
    let cell = get_cell_2d(pos, cell_size);
    
    // Initialize displacement
    var displacement = vec2<f32>(0.0, 0.0);
    var overlap_count = 0.0;
    var min_distance_found = 999999.0;
    var total_repulsion_force = 0.0;
    
    // Process all 9 neighboring cells using the CORRECT spatial hash lookup (same as density_pressure.wgsl)
    for (var cell_offset_y = -1; cell_offset_y <= 1; cell_offset_y++) {
        for (var cell_offset_x = -1; cell_offset_x <= 1; cell_offset_x++) {
            let neighbor_cell = vec2<i32>(
                cell.x + cell_offset_x,
                cell.y + cell_offset_y
            );
            
            let hash = hash_cell_2d(neighbor_cell);
            let key = key_from_hash(hash);
            let start_index = spatial_offsets[key];
            
            if (start_index == 0xFFFFFFFFu) {
                continue; // Skip empty cells
            }
            
            // FIXED: Use the correct spatial hash lookup logic (same as density_pressure.wgsl)
            var curr_index = start_index;
            
            // Iterate through particles in this cell
            while (curr_index < arrayLength(&particles)) {
                // Check if still in the same cell by comparing keys
                if (spatial_keys[curr_index] != key) {
                    break;
                }
                
                // Skip self
                if (curr_index == index) {
                    curr_index = curr_index + 1u;
                    continue;
                }
                
                let neighbor = particles[curr_index];
                let neighbor_pos = neighbor.position;
                
                let offset = pos - neighbor_pos;
                let distance = length(offset);
                
                // Track minimum distance for debugging
                if (distance > 0.0001) {
                    min_distance_found = min(min_distance_found, distance);
                }
                
                // Skip if not within smoothing radius or too close to zero
                if (distance <= 0.0001 || distance >= radius) {
                    curr_index = curr_index + 1u;
                    continue;
                }
                
                let direction = offset / distance;
                let q = distance / radius;
                
                // Paper equation 6: displacement calculation (same as CPU)
                var displacement_magnitude = dt_squared * (
                    pressure * (1.0 - q) +
                    near_pressure * (1.0 - q) * (1.0 - q)
                );
                
                // Add strong repulsion force when particles get too close to prevent overlapping
                let min_distance = params.particle_radius * 2.0; // Minimum separation distance
                if (distance < min_distance) {
                    overlap_count += 1.0;
                    let overlap_factor = (min_distance - distance) / min_distance;
                    // Increased repulsion force for stronger overlap prevention
                    let repulsion_force = overlap_factor * overlap_factor * 2000.0 * dt_squared;
                    displacement_magnitude += repulsion_force;
                    total_repulsion_force += repulsion_force;
                }
                
                let particle_displacement = direction * displacement_magnitude;
                
                // Apply half displacement (the other half will be applied to the neighbor)
                displacement -= particle_displacement * 0.5;
                
                curr_index = curr_index + 1u;
            }
        }
    }
    
    // Store debug information in padding fields
    particle.padding0.x = overlap_count; // Number of overlapping neighbors
    particle.padding0.y = min_distance_found; // Minimum distance to any neighbor
    particle.padding1.x = total_repulsion_force; // Total repulsion force applied
    particle.padding1.y = length(displacement); // Total displacement magnitude
    
    // Apply displacement to particle position
    let old_position = particle.position;
    particle.position += displacement;
    
    // Update velocity to reflect the position change (like CPU recompute_velocities)
    // This ensures gravity and other forces are preserved through position corrections
    if (dt > 0.0001) {
        let position_change = particle.position - old_position;
        particle.velocity += position_change / dt;
    }
    
    // Store the updated particle
    particles[index] = particle;
} 