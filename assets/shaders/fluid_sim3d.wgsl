// 3D Fluid Simulation Compute Shader
// Mirrors Unity's FluidSim.compute implementation with WGSL optimizations

struct Particle3D {
    position: vec3<f32>,
    padding0: f32,  // Padding for 16-byte alignment
    velocity: vec3<f32>,
    padding1: f32,  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

struct FluidParams3D {
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
    boundary_min: vec3<f32>,
    boundary_min_padding: f32,
    
    // Vec4 aligned group 4
    boundary_max: vec3<f32>,
    boundary_max_padding: f32,
    
    // Vec4 aligned group 5
    gravity: vec3<f32>,
    gravity_padding: f32,
    
    // Vec4 aligned group 6
    mouse_position: vec3<f32>,
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: vec2<u32>,
}

// Constants for hashing - match Unity's implementation
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;
const HASH_K3: u32 = 440817757u;
const BLOCK_SIZE: u32 = 50u;

// 3D cell offsets for neighboring cells (27 total including center)
const OFFSETS_3D: array<vec3<i32>, 27> = array<vec3<i32>, 27>(
    vec3<i32>(-1, -1, -1), vec3<i32>(0, -1, -1), vec3<i32>(1, -1, -1),
    vec3<i32>(-1, 0, -1), vec3<i32>(0, 0, -1), vec3<i32>(1, 0, -1),
    vec3<i32>(-1, 1, -1), vec3<i32>(0, 1, -1), vec3<i32>(1, 1, -1),
    vec3<i32>(-1, -1, 0), vec3<i32>(0, -1, 0), vec3<i32>(1, -1, 0),
    vec3<i32>(-1, 0, 0), vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1), vec3<i32>(0, -1, 1), vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1)
);

// Shared memory for optimized performance
struct CachedParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    pressure: f32,
    near_pressure: f32,
}

var<workgroup> shared_particles: array<CachedParticle3D, 128>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle3D>;

@group(0) @binding(1)
var<uniform> params: FluidParams3D;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_offsets: array<atomic<u32>>;

@group(0) @binding(4)
var<storage, read_write> target_particles: array<Particle3D>;

// Helper functions
fn get_cell_3d(position: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(floor(position / cell_size));
}

fn hash_cell_3d(cell: vec3<i32>) -> u32 {
    let ucell = vec3<u32>(cell + vec3<i32>(i32(BLOCK_SIZE) / 2));
    let local_cell = ucell % BLOCK_SIZE;
    let block_id = ucell / BLOCK_SIZE;
    let block_hash = block_id.x * HASH_K1 + block_id.y * HASH_K2 + block_id.z * HASH_K3;
    return local_cell.x + BLOCK_SIZE * (local_cell.y + BLOCK_SIZE * local_cell.z) + block_hash;
}

fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    return hash % table_size;
}

// Main compute shader for spatial hash update
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Initialize spatial offsets table with invalid value on first workgroup
    if index < arrayLength(&spatial_offsets) {
        atomicStore(&spatial_offsets[index], 0xFFFFFFFFu);
    }
    
    // Wait for all threads to finish initializing
    storageBarrier();
    
    // Get particle data
    let position = particles[index].position;
    
    // Calculate cell from position
    let cell = get_cell_3d(position, params.smoothing_radius * 2.0);
    
    // Calculate hash and key
    let hash = hash_cell_3d(cell);
    let key = key_from_hash(hash, arrayLength(&spatial_offsets));
    
    // Store key for this particle
    spatial_keys[index] = key;
    
    // Wait for all threads to finish before proceeding to sorting phase
    storageBarrier();
    
    // Atomic min operation - first thread to reach a key gets to set its index
    atomicMin(&spatial_offsets[key], index);
} 