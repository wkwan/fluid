// Update pressure force shader to use spatial hashing similar to Unity implementation

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

// Get cell from position and radius
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return ((x * 15823u) ^ (y * 9737333u)) + (x * y);
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    let mixed = hash ^ (hash >> 16u);
    return mixed % table_size;
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

// Process pressure force from a neighboring cell
fn process_cell(neighbor_cell: vec2<i32>, particle_index: u32, pos: vec2<f32>, pressure: f32, near_pressure: f32, density: f32, radius: f32, radius_squared: f32) -> vec2<f32> {
    var pressure_force = vec2<f32>(0.0, 0.0);
    
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
        if (neighbor_index == particle_index) {
            continue;
        }
        
        // Check if still in the same cell by comparing keys
        if (spatial_keys[neighbor_index] != key) {
            break;
        }
        
        let neighbor = particles[neighbor_index];
        let offset_to_neighbor = neighbor.position - pos;
        let sqr_dst = dot(offset_to_neighbor, offset_to_neighbor);
        
        // Skip if not within smoothing radius
        if (sqr_dst > radius_squared) {
            continue;
        }
        
        let dst = sqrt(sqr_dst);
        // Avoid division by zero using a clearer approach
        var dir_to_neighbor = vec2<f32>(0.0, 1.0);
        if (dst > 0.001) {
            dir_to_neighbor = offset_to_neighbor / dst;
        }
        
        // Calculate shared pressure
        let shared_pressure = (pressure + neighbor.pressure) * 0.5;
        let shared_near_pressure = (near_pressure + neighbor.near_pressure) * 0.5;
        
        // Calculate pressure force using derivatives of the smoothing kernels
        if (neighbor.density > 0.0) {
            // Standard pressure force
            pressure_force += dir_to_neighbor * spiky_pow2_derivative(dst, radius) 
                            * shared_pressure / max(0.001, neighbor.density);
            
            // Near pressure force (provides more stable interactions at very close distances)
            pressure_force += dir_to_neighbor * spiky_pow3_derivative(dst, radius) 
                            * shared_near_pressure / max(0.001, neighbor.near_density);
        }
    }
    
    return pressure_force;
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
    
    // Get the cell for this particle
    let cell = get_cell_2d(particle.position, radius);
    var pressure_force = vec2<f32>(0.0, 0.0);
    
    // Process all 9 neighboring cells (including current cell)
    // Top row
    pressure_force += process_cell(cell + vec2<i32>(-1, 1), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
                                  
    pressure_force += process_cell(cell + vec2<i32>(0, 1), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
                                  
    pressure_force += process_cell(cell + vec2<i32>(1, 1), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
    
    // Middle row
    pressure_force += process_cell(cell + vec2<i32>(-1, 0), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
                                  
    pressure_force += process_cell(cell + vec2<i32>(0, 0), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
                                  
    pressure_force += process_cell(cell + vec2<i32>(1, 0), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
    
    // Bottom row
    pressure_force += process_cell(cell + vec2<i32>(-1, -1), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
                                  
    pressure_force += process_cell(cell + vec2<i32>(0, -1), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
                                  
    pressure_force += process_cell(cell + vec2<i32>(1, -1), index, particle.position, 
                                  particle.pressure, particle.near_pressure, 
                                  particle.density, radius, radius_squared);
    
    // Apply pressure force
    // Scale force by 1/density to account for mass conservation
    let acceleration = pressure_force / max(0.001, particle.density);
    
    // Update velocity based on the pressure force
    particle.velocity += acceleration * params.dt;
    
    // Store the updated particle
    particles[index] = particle;
} 
