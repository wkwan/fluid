#import bevy_pbr::forward_io::VertexOutput

@group(2) @binding(0) var<uniform> material: RayMarchMaterial;
@group(2) @binding(1) var density_texture: texture_3d<f32>;
@group(2) @binding(2) var density_sampler: sampler;

struct RayMarchMaterial {
    camera_pos: vec3<f32>,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    step_size: f32,
    density_multiplier: f32,
    density_threshold: f32,
    max_density: f32,
    absorption: f32,
    scattering: f32,
    light_intensity: f32,
    refraction_enabled: u32,
    reflection_enabled: u32,
    environment_sampling: u32,
    max_bounces: u32,
    ior_water: f32,
    ior_air: f32,
    extinction_coefficient: vec3<f32>,
    surface_smoothness: f32,
}

struct HitInfo {
    did_hit: bool,
    is_inside: bool,
    dst: f32,
    hit_point: vec3<f32>,
    normal: vec3<f32>,
}

struct SurfaceInfo {
    pos: vec3<f32>,
    normal: vec3<f32>,
    density_along_ray: f32,
    found_surface: bool,
}

// Constants
const IOR_AIR: f32 = 1.0;
const IOR_WATER: f32 = 1.33;
const TINY_NUDGE: f32 = 0.01;

// Ray-box intersection
fn ray_box_intersection(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;
    let t_min = (box_min - ray_origin) * inv_dir;
    let t_max = (box_max - ray_origin) * inv_dir;
    
    let t1 = min(t_min, t_max);
    let t2 = max(t_min, t_max);
    
    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);
    
    // Return negative values if no intersection
    if (t_near > t_far || t_far < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    
    return vec2<f32>(max(t_near, 0.0), t_far);
}

// Sample density from 3D texture with trilinear interpolation
fn sample_density(world_pos: vec3<f32>) -> f32 {
    // Convert world position to texture coordinates (0-1)
    let bounds_size = material.bounds_max - material.bounds_min;
    let normalized_pos = (world_pos - material.bounds_min) / bounds_size;
    
    // Clamp to valid texture coordinates
    let tex_coords = clamp(normalized_pos, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Check if we're at the edge of the volume
    let epsilon = 0.0001;
    let is_edge = any(tex_coords >= vec3<f32>(1.0 - epsilon)) || any(tex_coords <= vec3<f32>(epsilon));
    if (is_edge) {
        return -material.density_threshold; // Return negative value at edges
    }
    
    // Sample the density texture with trilinear interpolation
    let density_sample = textureSampleLevel(density_texture, density_sampler, tex_coords, 0.0);
    
    // Extract density from red channel
    let raw_density = density_sample.r;
    
    // Only check for completely invalid textures (all channels zero)
    if (density_sample.r == 0.0 && density_sample.g == 0.0 && density_sample.b == 0.0 && density_sample.a == 0.0) {
        return -material.density_threshold; // Return negative for invalid texture
    }
    
    // Apply density multiplier and subtract threshold (like Unity's volumeValueOffset)
    let final_density = raw_density * material.density_multiplier - material.density_threshold;
    
    return final_density;
}

// Calculate normal using improved gradient estimation
fn calculate_normal(pos: vec3<f32>) -> vec3<f32> {
    let offset = 0.25; // Very small offset for fine detail
    
    // Use central differences for more accurate gradient
    let dx = sample_density(pos + vec3<f32>(offset, 0.0, 0.0)) - sample_density(pos - vec3<f32>(offset, 0.0, 0.0));
    let dy = sample_density(pos + vec3<f32>(0.0, offset, 0.0)) - sample_density(pos - vec3<f32>(0.0, offset, 0.0));
    let dz = sample_density(pos + vec3<f32>(0.0, 0.0, offset)) - sample_density(pos - vec3<f32>(0.0, 0.0, offset));
    
    let gradient = vec3<f32>(dx, dy, dz);
    let gradient_length = length(gradient);
    
    // Calculate a base normal from the gradient
    let volume_normal = select(
        vec3<f32>(0.0, 1.0, 0.0), // Default up normal
        normalize(gradient),
        gradient_length > 0.00001 // Very low threshold for responsiveness
    );
    
    // Sample additional points for smoother normal estimation
    let offset2 = offset * 0.5;
    let dx2 = sample_density(pos + vec3<f32>(offset2, 0.0, 0.0)) - sample_density(pos - vec3<f32>(offset2, 0.0, 0.0));
    let dy2 = sample_density(pos + vec3<f32>(0.0, offset2, 0.0)) - sample_density(pos - vec3<f32>(0.0, offset2, 0.0));
    let dz2 = sample_density(pos + vec3<f32>(0.0, 0.0, offset2)) - sample_density(pos - vec3<f32>(0.0, 0.0, offset2));
    
    let gradient2 = vec3<f32>(dx2, dy2, dz2);
    let gradient2_length = length(gradient2);
    
    let volume_normal2 = select(
        vec3<f32>(0.0, 1.0, 0.0),
        normalize(gradient2),
        gradient2_length > 0.00001
    );
    
    // Blend the two normal estimates for smoother results
    let blended_normal = normalize(volume_normal + volume_normal2);
    
    // Apply surface smoothness
    let smoothing_factor = material.surface_smoothness;
    let final_normal = normalize(mix(blended_normal, vec3<f32>(0.0, 1.0, 0.0), smoothing_factor * 0.3));
    
    return final_normal;
}

// Check if position is inside fluid
fn is_inside_fluid(pos: vec3<f32>) -> bool {
    let bounds_intersection = ray_box_intersection(pos, vec3<f32>(0.0, 0.0, 1.0), material.bounds_min, material.bounds_max);
    let inside_bounds = bounds_intersection.x <= 0.0 && bounds_intersection.y > 0.0;
    return inside_bounds && sample_density(pos) > 0.0;
}

// Calculate density along a ray (for lighting)
fn calculate_density_along_ray(ray_pos: vec3<f32>, ray_dir: vec3<f32>, step_size: f32) -> f32 {
    let intersection = ray_box_intersection(ray_pos, ray_dir, material.bounds_min, material.bounds_max);
    
    if (intersection.x < 0.0) {
        return 0.0;
    }
    
    let t_start = intersection.x;
    let t_end = intersection.y;
    let ray_length = t_end - t_start;
    
    var total_density = 0.0;
    let num_steps = max(1u, u32(ray_length / step_size));
    let actual_step_size = ray_length / f32(num_steps);
    
    for (var i = 0u; i < num_steps; i++) {
        let t = t_start + f32(i) * actual_step_size;
        let sample_pos = ray_pos + ray_dir * t;
        let density = sample_density(sample_pos);
        total_density += density * actual_step_size;
    }
    
    return total_density;
}

// Find next surface along ray using gradient-based detection
fn find_next_surface(origin: vec3<f32>, ray_dir: vec3<f32>, find_entry_point: bool) -> SurfaceInfo {
    var info: SurfaceInfo;
    info.found_surface = false;
    
    // Check for invalid ray direction
    if (dot(ray_dir, ray_dir) < 0.5) {
        return info;
    }
    
    let intersection = ray_box_intersection(origin, ray_dir, material.bounds_min, material.bounds_max);
    if (intersection.x < 0.0) {
        return info;
    }
    
    let t_start = intersection.x + TINY_NUDGE;
    let t_end = intersection.y - TINY_NUDGE;
    let ray_length = t_end - t_start;
    
    if (ray_length <= 0.0) {
        return info;
    }
    
    let step_size = material.step_size * 0.25; // Even smaller steps for smoother detection
    let num_steps = max(1u, u32(ray_length / step_size));
    let actual_step_size = ray_length / f32(num_steps);
    
    // Use a density threshold for surface detection
    let surface_threshold = material.density_threshold * 3.0;
    
    var last_density = sample_density(origin);
    var last_inside = last_density > surface_threshold;
    
    for (var i = 1u; i < num_steps; i++) {
        let t = t_start + f32(i) * actual_step_size;
        let sample_pos = origin + ray_dir * t;
        let density = sample_density(sample_pos);
        let inside_fluid = density > surface_threshold;
        
        // Accumulate density for lighting calculations
        if (inside_fluid) {
            info.density_along_ray += max(0.0, density) * actual_step_size;
        }
        
        // Check for surface crossing
        let surface_crossing = select(
            last_inside && !inside_fluid, // Exit point (inside to outside)
            !last_inside && inside_fluid, // Entry point (outside to inside)
            find_entry_point
        );
        
        if (surface_crossing) {
            // Use binary search for precise surface location
            let prev_t = t_start + f32(i - 1u) * actual_step_size;
            let prev_pos = origin + ray_dir * prev_t;
            
            // Binary search between prev_pos and sample_pos
            var search_start = prev_pos;
            var search_end = sample_pos;
            var search_density_start = last_density;
            var search_density_end = density;
            
            // Perform 8 iterations of binary search for precision
            for (var search_iter = 0u; search_iter < 8u; search_iter++) {
                let mid_pos = (search_start + search_end) * 0.5;
                let mid_density = sample_density(mid_pos);
                let mid_inside = mid_density > surface_threshold;
                
                if (select(!mid_inside, mid_inside, find_entry_point)) {
                    search_end = mid_pos;
                    search_density_end = mid_density;
                } else {
                    search_start = mid_pos;
                    search_density_start = mid_density;
                }
            }
            
            // Final interpolation for sub-voxel precision
            let final_pos = (search_start + search_end) * 0.5;
            let density_diff = abs(search_density_end - search_density_start);
            
            if (density_diff > 0.001) {
                let interpolation_factor = abs(surface_threshold - search_density_start) / density_diff;
                info.pos = mix(search_start, search_end, clamp(interpolation_factor, 0.0, 1.0));
            } else {
                info.pos = final_pos;
            }
            
            info.normal = calculate_normal(info.pos);
            info.found_surface = true;
            break;
        }
        
        last_density = density;
        last_inside = inside_fluid;
    }
    
    return info;
}

// Calculate Fresnel reflectance
fn calculate_reflectance(in_dir: vec3<f32>, normal: vec3<f32>, ior_a: f32, ior_b: f32) -> f32 {
    let refract_ratio = ior_a / ior_b;
    let cos_angle_in = -dot(in_dir, normal);
    let sin_sqr_angle_refraction = refract_ratio * refract_ratio * (1.0 - cos_angle_in * cos_angle_in);
    
    if (sin_sqr_angle_refraction >= 1.0) {
        return 1.0; // Total internal reflection
    }
    
    let cos_angle_refraction = sqrt(1.0 - sin_sqr_angle_refraction);
    
    // Fresnel equations
    let r_perp = (ior_a * cos_angle_in - ior_b * cos_angle_refraction) / (ior_a * cos_angle_in + ior_b * cos_angle_refraction);
    let r_parallel = (ior_b * cos_angle_in - ior_a * cos_angle_refraction) / (ior_b * cos_angle_in + ior_a * cos_angle_refraction);
    
    return (r_perp * r_perp + r_parallel * r_parallel) * 0.5;
}

// Calculate refraction direction
fn refract_ray(in_dir: vec3<f32>, normal: vec3<f32>, ior_a: f32, ior_b: f32) -> vec3<f32> {
    let refract_ratio = ior_a / ior_b;
    let cos_angle_in = -dot(in_dir, normal);
    let sin_sqr_angle_refraction = refract_ratio * refract_ratio * (1.0 - cos_angle_in * cos_angle_in);
    
    if (sin_sqr_angle_refraction > 1.0) {
        return vec3<f32>(0.0); // Total internal reflection
    }
    
    let refract_dir = refract_ratio * in_dir + (refract_ratio * cos_angle_in - sqrt(1.0 - sin_sqr_angle_refraction)) * normal;
    return refract_dir;
}

// Calculate reflection direction
fn reflect_ray(in_dir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return in_dir - 2.0 * dot(in_dir, normal) * normal;
}

// Sample environment (simplified sky)
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    // More realistic sky colors (less saturated, more neutral)
    let sky_horizon = vec3<f32>(0.9, 0.95, 1.0);  // Lighter, less blue
    let sky_zenith = vec3<f32>(0.5, 0.7, 0.9);    // Less saturated blue
    let ground_color = vec3<f32>(0.4, 0.35, 0.3); // Warmer ground
    
    let sky_gradient_t = pow(smoothstep(0.0, 0.4, dir.y), 0.35);
    let ground_to_sky_t = smoothstep(-0.01, 0.0, dir.y);
    let sky_gradient = mix(sky_horizon, sky_zenith, sky_gradient_t);
    
    // Brighter, more prominent sun
    let sun_dir = normalize(vec3<f32>(0.3, 0.8, 0.3));
    let sun = pow(max(0.0, dot(dir, sun_dir)), 200.0) * 2.0; // Larger, brighter sun
    
    return mix(ground_color, sky_gradient, ground_to_sky_t) + sun * ground_to_sky_t;
}

// Calculate transmittance through medium
fn transmittance(thickness: f32) -> vec3<f32> {
    // More realistic water absorption - water absorbs red more than blue/green
    let extinction_coeff = select(
        material.extinction_coefficient,
        vec3<f32>(0.45, 0.15, 0.1), // Red absorbed more, blue/green less
        all(material.extinction_coefficient == vec3<f32>(0.0))
    );
    
    // Apply water color tinting - subtle blue-green tint
    let water_tint = vec3<f32>(0.9, 0.95, 1.0); // Slight blue tint
    return exp(-thickness * extinction_coeff) * water_tint;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_origin = material.camera_pos;
    let ray_dir = normalize(in.world_position.xyz - ray_origin);
    
    // Check if advanced features are enabled
    let use_refraction = material.refraction_enabled != 0u;
    let use_reflection = material.reflection_enabled != 0u;
    let use_environment = material.environment_sampling != 0u;
    
    // If no advanced features, fall back to simple volumetric rendering
    if (!use_refraction && !use_reflection && !use_environment) {
        return simple_volumetric_render(ray_origin, ray_dir);
    }
    
    var current_ray_pos = ray_origin;
    var current_ray_dir = ray_dir;
    var travelling_through_fluid = is_inside_fluid(ray_origin);
    
    var accumulated_light = vec3<f32>(0.0);
    var transmittance_factor = vec3<f32>(1.0);
    
    // Multiple refraction bounces
    let max_bounces = min(material.max_bounces, 4u); // Cap at 4 for performance
    for (var bounce = 0u; bounce < max_bounces; bounce++) {
        let search_for_entry = !travelling_through_fluid;
        let surface_info = find_next_surface(current_ray_pos, current_ray_dir, search_for_entry);
        
        if (!surface_info.found_surface) {
            // Ray escaped - only sample environment if enabled and we have significant transmittance
            if (use_environment && any(transmittance_factor > vec3<f32>(0.01))) {
                accumulated_light += sample_environment(current_ray_dir) * transmittance_factor;
            }
            break;
        }
        
        // Apply transmittance through the medium
        transmittance_factor *= transmittance(surface_info.density_along_ray);
        
        // Early exit if transmittance is too low
        if (all(transmittance_factor < vec3<f32>(0.01))) {
            break;
        }
        
        // Calculate surface normal (flip if needed)
        var surface_normal = surface_info.normal;
        if (dot(surface_normal, current_ray_dir) > 0.0) {
            surface_normal = -surface_normal;
        }
        
        // Determine indices of refraction
        let ior_a = select(material.ior_water, material.ior_air, travelling_through_fluid);
        let ior_b = select(material.ior_air, material.ior_water, travelling_through_fluid);
        
        // Calculate reflection and refraction
        let reflectance = calculate_reflectance(current_ray_dir, surface_normal, ior_a, ior_b);
        let reflect_dir = reflect_ray(current_ray_dir, surface_normal);
        let refract_dir = refract_ray(current_ray_dir, surface_normal, ior_a, ior_b);
        
        // Choose the more significant path based on enabled features
        let can_refract = use_refraction && length(refract_dir) > 0.5;
        let can_reflect = use_reflection;
        
        // More sophisticated path selection like Unity
        if (can_refract && can_reflect) {
            // Calculate density along both paths to choose the more interesting one
            let density_along_refract = calculate_density_along_ray(surface_info.pos, refract_dir, material.step_size);
            let density_along_reflect = calculate_density_along_ray(surface_info.pos, reflect_dir, material.step_size);
            
            // Weight by both density and fresnel terms
            let refract_weight = density_along_refract * (1.0 - reflectance);
            let reflect_weight = density_along_reflect * reflectance;
            
            let use_refraction_path = refract_weight > reflect_weight;
            
            if (use_refraction_path) {
                // Approximate the reflection path
                accumulated_light += sample_environment(reflect_dir) * transmittance_factor * transmittance(density_along_reflect) * reflectance;
                
                // Follow refraction path
                current_ray_pos = surface_info.pos + refract_dir * TINY_NUDGE;
                current_ray_dir = refract_dir;
                travelling_through_fluid = !travelling_through_fluid;
                transmittance_factor *= (1.0 - reflectance);
            } else {
                // Approximate the refraction path
                accumulated_light += sample_environment(refract_dir) * transmittance_factor * transmittance(density_along_refract) * (1.0 - reflectance);
                
                // Follow reflection path
                current_ray_pos = surface_info.pos + reflect_dir * TINY_NUDGE;
                current_ray_dir = reflect_dir;
                transmittance_factor *= reflectance;
            }
        } else if (can_refract) {
            // Follow refraction path
            current_ray_pos = surface_info.pos + refract_dir * TINY_NUDGE;
            current_ray_dir = refract_dir;
            travelling_through_fluid = !travelling_through_fluid;
            transmittance_factor *= (1.0 - reflectance);
        } else if (can_reflect) {
            // Follow reflection path
            current_ray_pos = surface_info.pos + reflect_dir * TINY_NUDGE;
            current_ray_dir = reflect_dir;
            transmittance_factor *= reflectance;
        } else {
            // No valid path - sample environment and exit
            if (use_environment && any(transmittance_factor > vec3<f32>(0.01))) {
                accumulated_light += sample_environment(current_ray_dir) * transmittance_factor;
            }
            break;
        }
    }
    
    // Final environment sample for remaining path (only if we haven't already sampled)
    if (use_environment && any(transmittance_factor > vec3<f32>(0.01))) {
        accumulated_light += sample_environment(current_ray_dir) * transmittance_factor;
    }
    
    return vec4<f32>(accumulated_light, 1.0);
}

// Simple volumetric rendering fallback
fn simple_volumetric_render(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    let intersection = ray_box_intersection(ray_origin, ray_dir, material.bounds_min, material.bounds_max);
    
    if (intersection.x < 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let t_start = intersection.x;
    let t_end = intersection.y;
    let ray_length = t_end - t_start;
    
    let step_count = 32u;
    let actual_step_size = ray_length / f32(step_count);
    
    var accumulated_color = vec3<f32>(0.0);
    var accumulated_alpha = 0.0;
    var t = t_start + actual_step_size * 0.5;
    
    for (var i = 0u; i < step_count && accumulated_alpha < 0.95; i++) {
        let sample_pos = ray_origin + ray_dir * t;
        let density = sample_density(sample_pos);
        
        if (density > 0.0) { // Use 0.0 threshold like Unity
            let sample_color = vec3<f32>(0.3, 0.7, 1.0) * material.light_intensity;
            let sample_alpha = min(density * material.absorption * 0.1, 0.3);
            
            let alpha_factor = sample_alpha * (1.0 - accumulated_alpha);
            accumulated_color += sample_color * alpha_factor;
            accumulated_alpha += alpha_factor;
        }
        
        t += actual_step_size;
    }
    
    return vec4<f32>(accumulated_color, accumulated_alpha);
} 