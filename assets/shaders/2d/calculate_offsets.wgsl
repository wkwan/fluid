// Optimized calculate_offsets shader - matching Unity's implementation
// This shader calculates the starting offsets for each spatial hash cell

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

// Constants for the hash table size and hashing
const TABLE_SIZE: u32 = 4096u;
const TABLE_SIZE_MASK: u32 = 4095u; // TABLE_SIZE - 1

// Workgroup-shared histogram for counting particles per cell
var<workgroup> histogram: array<atomic<u32>, 128>; // For workgroup-level histogramming

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read_write> spatial_indices: array<u32>;

@group(0) @binding(4)
var<storage, read_write> spatial_offsets: array<atomic<u32>>;

// Main compute shader function
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
         @builtin(local_invocation_id) local_id: vec3<u32>,
         @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Initialize histogram to zeros
    if local_index < 128u {
        atomicStore(&histogram[local_index], 0u);
    }
    
    // Ensure all threads have finished initializing
    workgroupBarrier();
    
    // Phase 1: Count occurrences in this workgroup
    if index < arrayLength(&particles) {
        let key = spatial_keys[index];
        let bin = key % 128u; // Modulo for workgroup-local histogram
        atomicAdd(&histogram[bin], 1u);
    }
    
    // Ensure all threads in workgroup have updated the histogram
    workgroupBarrier();
    
    // Phase 2: Update global histogram
    // Each thread in the workgroup handles a different bin
    if local_index < 128u {
        let bin_count = atomicLoad(&histogram[local_index]);
        if bin_count > 0u {
            // For each bin with particles, calculate actual keys from workgroup-local bin
            // and add to the global histogram
            let wg_offset = workgroup_id.x * 64u;
            
            // Update each key's global offset counter
            for (var i = 0u; i < bin_count; i += 1u) {
                // Find the particle in this workgroup that hashes to this bin
                for (var j = 0u; j < 64u; j += 1u) {
                    let particle_idx = wg_offset + j;
                    if particle_idx < arrayLength(&particles) {
                        if spatial_keys[particle_idx] % 128u == local_index {
                            // Add to global counter
                            let key = spatial_keys[particle_idx];
                            atomicAdd(&spatial_offsets[key], 1u);
                            break;
                        }
                    }
                }
            }
        }
    }
    
    // Final barrier before exiting
    workgroupBarrier();
    
    // Phase 3: For workgroup 0 only - convert counts to offsets
    if workgroup_id.x == 0u && local_id.x < 128u {
        let bin_id = local_id.x;
        
        // Process multiple bins per thread
        for (var key_id = bin_id; key_id < TABLE_SIZE; key_id += 128u) {
            var count = atomicLoad(&spatial_offsets[key_id]);
            if count != 0xFFFFFFFFu {
                // Replace count with index
                atomicStore(&spatial_offsets[key_id], key_id);
            }
        }
    }
} 
