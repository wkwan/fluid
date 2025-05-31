// Fluid Normals Shader - Phase 3 of Screen Space Fluid Rendering
// Simplified version that should compile properly

struct FluidNormalsMaterial {
    base_color: vec4<f32>,
    particle_scale: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
};

@group(2) @binding(0) var<uniform> material: FluidNormalsMaterial;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Use Bevy's standard mesh bindings
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
    
    // Transform to world space
    let world_pos = mesh.model * vec4<f32>(vertex.position, 1.0);
    
    // Transform to clip space
    out.clip_position = view.view_proj * world_pos;
    out.uv = vertex.uv;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Make it very obvious this shader is working - use bright red color
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);  // Bright red
    
    // TODO: Add normal calculation once we confirm the shader is loading
}