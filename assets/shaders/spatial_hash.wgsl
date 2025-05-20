// Fluid Simulation Spatial Hashing Compute Shader

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

// Workgroup shared memory for better locality
var<workgroup> shared_positions: array<vec2<f32>, 128>;

// Spatial hash buffers
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
    // For better performance, cell size should be related to smoothing radius
    // but not too small to prevent excessive hash collisions
    return max(smoothing_radius, particle_radius * 4.0);
}

// Convert floating point position into an integer cell coordinate
fn get_cell_2d(position: vec2<f32>, cell_size: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / cell_size));
}

// Hash cell coordinate to a single unsigned integer using prime multipliers
// This improves distribution and reduces collisions
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return ((x * HASH_K1) ^ (y * HASH_K2)) + (x * y);
}

// Get key from hash for a table of given size with better distribution
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    // FNV-1a-inspired mixing for better hash distribution
    let mixed = hash ^ (hash >> 16);
    return mixed % table_size;
}

// Compute shader to calculate spatial hash for all particles
// RTX 4090 optimized workgroup size
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Check array bounds to prevent issues
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    // Cache particle position in shared memory for better performance
    if (local_index < 128u) {
        shared_positions[local_index] = particles[index].position;
    }
    
    // Ensure all threads have cached their data
    workgroupBarrier();
    
    // Calculate optimal cell size
    let cell_size = calculate_optimal_cell_size(params.particle_radius, params.smoothing_radius);
    
    // Get position from shared memory
    let position = shared_positions[local_index];
    
    // Calculate cell for particle position
    let cell = get_cell_2d(position, cell_size);
    
    // Calculate hash and key
    let hash = hash_cell_2d(cell);
    let key = key_from_hash(hash, u32(arrayLength(&particles)));
    
    // Store key and index
    spatial_keys[index] = key;
    spatial_indices[index] = index; // Initial indices (will be sorted externally)
} 