// Fluid Simulation Spatial Offsets Calculator Compute Shader

// Spatial hash buffers
@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(4)
var<storage, read_write> spatial_offsets: array<u32>;

// Calculate the offset (starting index) for each spatial key
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&spatial_keys)) {
        return;
    }
    
    // Initialize all offsets to max value
    if (index == 0u) {
        for (var i = 0u; i < arrayLength(&spatial_offsets); i = i + 1u) {
            spatial_offsets[i] = 0xFFFFFFFFu; // Use max uint as sentinel value
        }
    }
    
    // Only proceed after initialization
    workgroupBarrier();
    
    // The first occurrence of each key becomes the offset
    let key = spatial_keys[index];
    
    // Ensure we're at the first occurrence of this key
    // (this check works because the keys are sorted)
    if (index == 0u || key != spatial_keys[index - 1u]) {
        spatial_offsets[key] = index;
    }
} 