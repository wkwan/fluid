// Viscosity (placeholder)
struct Particle { position: vec4<f32>; velocity: vec4<f32>; data1: vec4<f32>; data2: vec4<f32>; };
struct Params { dummy: vec4<f32>; };
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> spatialKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> spatialOffsets: array<u32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {} 