struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) velocity: vec3<f32>,
    @location(1) size: f32,
}

struct Particle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    force: vec3<f32>,
}

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle3D>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

const PI: f32 = 3.14159265359;
const PARTICLE_SIZE: f32 = 0.1;

@vertex
fn vertex(@builtin(instance_index) instance_index: u32) -> VertexOutput {
    let particle = particles[instance_index];
    let position = vec4<f32>(particle.position, 1.0);
    
    var output: VertexOutput;
    output.position = camera.view_proj * position;
    output.velocity = particle.velocity;
    output.size = PARTICLE_SIZE;
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize velocity to [0, 1] range
    let max_speed = 10.0;
    let normalized_vel = clamp(length(input.velocity) / max_speed, 0.0, 1.0);
    
    // Convert velocity to HSV color
    let hue = (atan2(input.velocity.z, input.velocity.x) + PI) / (2.0 * PI);
    let saturation = 1.0;
    let value = normalized_vel;
    
    // Convert HSV to RGB
    let c = value * saturation;
    let h = hue * 6.0;
    let x = c * (1.0 - abs(fract(vec3<f32>(h, h - 4.0, h - 2.0)) * 6.0 - 3.0));
    let m = value - c;
    
    let rgb = select(
        vec3<f32>(c, x.x, 0.0),
        select(
            vec3<f32>(x.z, c, 0.0),
            vec3<f32>(0.0, c, x.y),
            hue < 0.333,
        ),
        hue < 0.167,
    ) + m;
    
    return vec4<f32>(rgb, 1.0);
} 