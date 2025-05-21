// Optimized shader for reordering particles based on spatial hash
// Matches Unity's implementation with atomic counters and efficient sorting

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

// Define a counter structure for parallel reordering
struct Counter {
    count: atomic<u32>,
}

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_offsets: array<u32>;

@group(0) @binding(4)
var<storage, read_write> target_particles: array<Particle>;

// Shared memory counter for each cell
var<workgroup> cell_counters: array<atomic<u32>, 128>; // Support up to 128 different cells per workgroup

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>) {
    
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Initialize shared counters
    if local_index < 128u {
        atomicStore(&cell_counters[local_index], 0u);
    }
    
    // Ensure initialization is complete
    workgroupBarrier();
    
    // Skip if particle index is out of bounds
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Read the key and source particle data
    let key = spatial_keys[index];
    let particle = particles[index];
    
    // Determine base offset for this key
    let base_offset = spatial_offsets[key];
    
    // Handle case where this key has no offset (invalid or empty cell)
    if base_offset == 0xFFFFFFFFu {
        return;
    }
    
    // Get thread-safe offset using atomics
    // Since we're limited in workgroup shared memory, we'll use a hash
    // of the key to index into our counter array
    let counter_index = key & 127u; // Fast modulo by 128
    
    // Atomically increment counter and get previous value as our offset
    let relative_offset = atomicAdd(&cell_counters[counter_index], 1u);
    
    // Calculate final target index
    let target_index = base_offset + relative_offset;
    
    // Ensure target index is valid
    if target_index < arrayLength(&target_particles) {
        // Write particle to target buffer in sorted order
        target_particles[target_index] = particle;
    }
} 
