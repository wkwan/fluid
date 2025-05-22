#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_types

@group(1) @binding(0) var<uniform> velocity_color: vec4<f32>;
@group(1) @binding(1) var velocity_gradient: texture_2d<f32>;
@group(1) @binding(2) var gradient_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // This is a simple passthrough shader that uses the velocity color 
    // set by the CPU-side code (we don't actually sample from the gradient texture yet)
    return velocity_color;
} 