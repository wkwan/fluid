// Screen Space Fluid Rendering Shader - Phase 1 (Simple Billboard Circles)

struct ScreenSpaceFluidMaterial {
    base_color: vec4<f32>,
    particle_scale: f32,
    depth_mode: f32,
    _padding1: f32,
    _padding2: f32,
};

@group(2) @binding(0)
var<uniform> material: ScreenSpaceFluidMaterial;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_pos: vec3<f32>,
    @location(2) particle_center_view: vec3<f32>,
};

// Simplified approach - use Bevy's built-in bindings
#import bevy_pbr::{
    mesh_functions::get_world_from_local,
    view_transformations::{position_world_to_clip, position_local_to_world},
}
#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    
    // Use Bevy's built-in mesh functions
    let world_position = position_local_to_world(get_world_from_local(0u), vec4<f32>(vertex.position, 1.0));
    let world_center = position_local_to_world(get_world_from_local(0u), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    
    // Transform to view space
    let view_position = view.view * world_position;
    let view_center = view.view * world_center;
    
    out.clip_position = position_world_to_clip(world_position.xyz);
    out.uv = vertex.uv;
    out.view_pos = view_position.xyz;
    out.particle_center_view = view_center.xyz;
    
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    // Calculate distance from center in UV space
    let center_offset = (in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let sqr_dist = dot(center_offset, center_offset);
    
    // Discard pixels outside the circle
    if (sqr_dist > 1.0) {
        discard;
    }
    
    // Calculate smooth edge for anti-aliasing
    let edge_width = fwidth(sqr_dist);
    let alpha = 1.0 - smoothstep(1.0 - edge_width * 2.0, 1.0, sqr_dist);
    
    // Check if we're in depth mode
    if (material.depth_mode > 0.5) {
        // Depth mode: calculate spherical depth correction
        let z_offset = sqrt(1.0 - sqr_dist);
        
        // Calculate the corrected view position
        let particle_radius = material.particle_scale;
        let corrected_view_pos = in.particle_center_view + vec3<f32>(0.0, 0.0, -z_offset * particle_radius);
        
        // Transform to clip space for depth calculation
        let corrected_clip_pos = view.projection * vec4<f32>(corrected_view_pos, 1.0);
        
        // Calculate linear depth (positive values, increasing with distance)
        let linear_depth = -corrected_view_pos.z;
        
        // Output linear depth as color for visualization and set hardware depth
        out.color = vec4<f32>(linear_depth / 100.0, linear_depth / 100.0, linear_depth / 100.0, alpha);
        out.depth = corrected_clip_pos.z / corrected_clip_pos.w;
    } else {
        // Billboard mode: regular particle rendering
        out.color = vec4<f32>(material.base_color.rgb, material.base_color.a * alpha);
        out.depth = in.clip_position.z / in.clip_position.w;
    }
    
    return out;
} 