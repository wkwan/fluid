// Screen Space Fluid Rendering Shader - Phase 1 (Simple Billboard Circles)

struct ScreenSpaceFluidMaterial {
    base_color: vec4<f32>,
    particle_scale: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
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
};

// Bevy automatically provides these bindings
struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    viewport: vec4<f32>,
};

struct Mesh {
    model: mat4x4<f32>,
    previous_model: mat4x4<f32>,
    inverse_transpose_model: mat4x4<f32>,
    flags: u32,
};

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> mesh: Mesh;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform vertex position from model space to world space, then to clip space
    let world_position = mesh.model * vec4<f32>(vertex.position, 1.0);
    out.clip_position = view.view_proj * world_position;
    out.uv = vertex.uv;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
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
    
    // Apply material color with alpha
    return vec4<f32>(material.base_color.rgb, material.base_color.a * alpha);
} 