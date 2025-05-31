// Bilateral Filter Compute Shader for Screen Space Fluid Rendering
// Based on the Unity implementation with depth-aware smoothing

struct FilterParams {
    screen_width: u32,
    screen_height: u32,
    filter_radius: f32,
    depth_threshold: f32,
    // Bilateral filter parameters
    sigma_spatial: f32,
    sigma_depth: f32,
    _padding1: f32,
    _padding2: f32,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: FilterParams;

// Calculate screen-space radius based on world radius and depth
fn calculate_screen_space_radius(world_radius: f32, depth: f32, screen_width: f32) -> f32 {
    // Simplified projection calculation
    // In a real implementation, this would use proper projection matrix values
    let pixels_per_meter = screen_width / (2.0 * depth * 0.5); // Approximation
    return abs(pixels_per_meter) * world_radius;
}

// Bilateral filter kernel
fn bilateral_weight(spatial_distance: f32, depth_difference: f32) -> f32 {
    let spatial_weight = exp(-(spatial_distance * spatial_distance) / (2.0 * params.sigma_spatial * params.sigma_spatial));
    let depth_weight = exp(-(depth_difference * depth_difference) / (2.0 * params.sigma_depth * params.sigma_depth));
    return spatial_weight * depth_weight;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let screen_size = vec2<i32>(i32(params.screen_width), i32(params.screen_height));
    
    // Bounds check
    if (coords.x >= screen_size.x || coords.y >= screen_size.y) {
        return;
    }
    
    // Sample center depth
    let center_depth = textureLoad(input_texture, coords, 0).r;
    
    // If no depth data (background), skip filtering
    if (center_depth <= 0.0) {
        textureStore(output_texture, coords, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    // Calculate adaptive filter radius based on depth
    let screen_radius = calculate_screen_space_radius(params.filter_radius, center_depth, f32(params.screen_width));
    let filter_radius_pixels = max(1.0, min(screen_radius, 20.0)); // Clamp radius
    
    var filtered_depth = 0.0;
    var total_weight = 0.0;
    
    // Sample in a square pattern around the center pixel
    let radius_int = i32(ceil(filter_radius_pixels));
    
    for (var dy = -radius_int; dy <= radius_int; dy++) {
        for (var dx = -radius_int; dx <= radius_int; dx++) {
            let sample_coords = coords + vec2<i32>(dx, dy);
            
            // Bounds check for sample
            if (sample_coords.x < 0 || sample_coords.x >= screen_size.x ||
                sample_coords.y < 0 || sample_coords.y >= screen_size.y) {
                continue;
            }
            
            // Calculate spatial distance
            let spatial_distance = sqrt(f32(dx * dx + dy * dy));
            
            // Skip if outside circular radius
            if (spatial_distance > filter_radius_pixels) {
                continue;
            }
            
            // Sample depth at this position
            let sample_depth = textureLoad(input_texture, sample_coords, 0).r;
            
            // Skip background pixels
            if (sample_depth <= 0.0) {
                continue;
            }
            
            // Calculate depth difference
            let depth_difference = abs(sample_depth - center_depth);
            
            // Skip if depth difference is too large (edge preservation)
            if (depth_difference > params.depth_threshold) {
                continue;
            }
            
            // Calculate bilateral weight
            let weight = bilateral_weight(spatial_distance, depth_difference);
            
            filtered_depth += sample_depth * weight;
            total_weight += weight;
        }
    }
    
    // Normalize and output
    if (total_weight > 0.0) {
        filtered_depth /= total_weight;
    } else {
        filtered_depth = center_depth; // Fallback to original depth
    }
    
    textureStore(output_texture, coords, vec4<f32>(filtered_depth, 0.0, 0.0, 1.0));
}