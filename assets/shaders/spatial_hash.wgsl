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

// Compute shader to calculate spatial hash for all particles
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    // Calculate cell for particle position
    let cell = get_cell_2d(particles[index].position, params.smoothing_radius);
    
    // Calculate hash and key
    let hash = hash_cell_2d(cell);
    let key = key_from_hash(hash, u32(arrayLength(&particles)));
    
    // Store key and index
    spatial_keys[index] = key;
    spatial_indices[index] = index; // Initial indices (will be sorted externally)
} 