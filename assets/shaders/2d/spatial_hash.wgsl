// Optimized spatial hash shader with improved memory layout and table size
// Matches Unity's implementation with memory coalescing optimizations

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

// Constants used for hashing - match Unity's implementation exactly
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;
// Table size constants - using power of two for fast modulo
const TABLE_SIZE: u32 = 4096u;
const TABLE_SIZE_MASK: u32 = 4095u; // TABLE_SIZE - 1

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_offsets: array<atomic<u32>>;

// Get point cell from position and cell_size - optimized for SIMD processing
fn get_cell_2d(position: vec2<f32>, cell_size: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / cell_size));
}

// Improved hash function that more closely matches Unity's implementation
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    // Use unsigned values to match Unity's impl
    let x = u32(cell.x);
    let y = u32(cell.y);
    
    // Use XOR for slightly better distribution and faster calculation
    return (x * HASH_K1) ^ (y * HASH_K2);
}

// Get linear key from hash - optimized with bit masking instead of modulo
fn key_from_hash(hash: u32) -> u32 {
    return hash & TABLE_SIZE_MASK;
}

// Calculate optimal cell size - matched to forces shader
fn calculate_optimal_cell_size(particle_radius: f32, smoothing_radius: f32) -> f32 {
    return smoothing_radius * 2.0; // Match Unity's approach
}

// Main compute shader
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get particle index
    let index = global_id.x;
    
    // Skip if beyond array bounds
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    // Initialize spatial offsets table with invalid value on first workgroup
    if (index < TABLE_SIZE) {
        atomicStore(&spatial_offsets[index], 0xFFFFFFFFu);
    }
    
    // Wait for all threads to finish initializing
    storageBarrier();
    
    // Reuse particles in registers for better memory efficiency
    let position = particles[index].position;
    
    // Calculate optimized cell size
    let cell_size = calculate_optimal_cell_size(params.particle_radius, params.smoothing_radius);
    
    // Get cell from position
    let cell = get_cell_2d(position, cell_size);
    
    // Calculate hash
    let hash = hash_cell_2d(cell);
    
    // Calculate key from hash
    let key = key_from_hash(hash);
    
    // Store key for this particle
    spatial_keys[index] = key;
    
    // Wait for all threads to finish before proceeding to sorting phase
    storageBarrier();
    
    // Atomic min operation - first thread to reach a key gets to set its index
    // This creates a start index for each occupied cell
    atomicMin(&spatial_offsets[key], index);
} 
