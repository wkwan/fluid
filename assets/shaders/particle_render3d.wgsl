struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) velocity: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) velocity: vec3<f32>,
}

struct Material {
    max_velocity: f32,
    time: f32,
}

@group(1) @binding(0) var<uniform> material: Material;

const POINT_SIZE: f32 = 10.0;
const MAX_SPEED: f32 = 700.0;  // Adjusted to match our observed velocity range
const PI: f32 = 3.14159265359;

@vertex
fn vertex(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Pass through position and velocity
    out.world_position = vertex.position;
    out.world_normal = vertex.normal;
    out.velocity = vertex.velocity;
    
    // Set clip position (with adjusted point size for visibility)
    var pos = vec4<f32>(vertex.position, 1.0);
    out.clip_position = pos;
    
    // Set gl_PointSize equivalent in WebGPU
    out.clip_position.w = POINT_SIZE / 2.0;
    
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize velocity to [0, 1] range
    let velocity_magnitude = length(in.velocity);
    let normalized_vel = clamp(velocity_magnitude / material.max_velocity, 0.0, 1.0);
    
    // Use velocity direction for hue
    let velocity_dir = normalize(in.velocity);
    // Use atan2 of xz components for hue to create a "top-down" color wheel
    let hue = (atan2(velocity_dir.z, velocity_dir.x) + PI) / (2.0 * PI);
    let saturation = 0.8;
    // Use velocity magnitude for value (brightness)
    let value = 0.5 + normalized_vel * 0.5;
    
    // Convert HSV to RGB
    let c = value * saturation;
    let h = hue * 6.0;
    let x = c * (1.0 - abs(fract(h) * 2.0 - 1.0));
    let m = value - c;
    
    var rgb: vec3<f32>;
    if (h < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    rgb = rgb + m;
    
    // Add simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let normal = normalize(in.world_normal);
    let diffuse = max(dot(normal, light_dir), 0.0);
    let ambient = 0.3;
    let lighting = ambient + diffuse * 0.7;
    
    // Make the point a circle by discarding pixels outside radius
    // (not needed for PointList topology)
    
    // Add subtle animation based on time
    let pulse = sin(material.time * 2.0) * 0.1 + 0.9;
    
    return vec4<f32>(rgb * lighting * pulse, 1.0);
} 