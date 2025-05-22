// 3D Density and Pressure Calculation Compute Shader
// Mirrors Unity's CalculateDensities kernel with WGSL optimizations

const PI: f32 = 3.14159265359;

struct Particle3D {
    position: vec3<f32>,
    padding0: f32,  // Padding for 16-byte alignment
    velocity: vec3<f32>,
    padding1: f32,  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

struct FluidParams3D {
    // Vec4 aligned group 1
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    // Vec4 aligned group 2
    viscosity_strength: f32,
    boundary_dampening: f32,
    particle_radius: f32,
    dt: f32,
    
    // Vec4 aligned group 3
    boundary_min: vec3<f32>,
    boundary_min_padding: f32,
    
    // Vec4 aligned group 4
    boundary_max: vec3<f32>,
    boundary_max_padding: f32,
    
    // Vec4 aligned group 5
    gravity: vec3<f32>,
    gravity_padding: f32,
    
    // Vec4 aligned group 6
    mouse_position: vec3<f32>,
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: vec2<u32>,
}

// Constants for hashing - match Unity's implementation
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;
const HASH_K3: u32 = 440817757u;
const BLOCK_SIZE: u32 = 50u;

// 3D cell offsets for neighboring cells (27 total including center)
const OFFSETS_3D: array<vec3<i32>, 27> = array<vec3<i32>, 27>(
    vec3<i32>(-1, -1, -1), vec3<i32>(0, -1, -1), vec3<i32>(1, -1, -1),
    vec3<i32>(-1, 0, -1), vec3<i32>(0, 0, -1), vec3<i32>(1, 0, -1),
    vec3<i32>(-1, 1, -1), vec3<i32>(0, 1, -1), vec3<i32>(1, 1, -1),
    vec3<i32>(-1, -1, 0), vec3<i32>(0, -1, 0), vec3<i32>(1, -1, 0),
    vec3<i32>(-1, 0, 0), vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0),
    vec3<i32>(-1, 1, 0), vec3<i32>(0, 1, 0), vec3<i32>(1, 1, 0),
    vec3<i32>(-1, -1, 1), vec3<i32>(0, -1, 1), vec3<i32>(1, -1, 1),
    vec3<i32>(-1, 0, 1), vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1),
    vec3<i32>(-1, 1, 1), vec3<i32>(0, 1, 1), vec3<i32>(1, 1, 1)
);

// Shared memory for optimized performance
struct CachedParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    pressure: f32,
    near_pressure: f32,
}

var<workgroup> shared_particles: array<CachedParticle3D, 128>;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle3D>;

@group(0) @binding(1)
var<uniform> params: FluidParams3D;

@group(0) @binding(2)
var<storage, read> spatial_keys: array<u32>;

@group(0) @binding(3)
var<storage, read> spatial_offsets: array<atomic<u32>>;

// Helper functions
fn get_cell_3d(position: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(floor(position / cell_size));
}

fn hash_cell_3d(cell: vec3<i32>) -> u32 {
    let ucell = vec3<u32>(cell + vec3<i32>(i32(BLOCK_SIZE) / 2));
    let local_cell = ucell % BLOCK_SIZE;
    let block_id = ucell / BLOCK_SIZE;
    let block_hash = block_id.x * HASH_K1 + block_id.y * HASH_K2 + block_id.z * HASH_K3;
    return local_cell.x + BLOCK_SIZE * (local_cell.y + BLOCK_SIZE * local_cell.z) + block_hash;
}

fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    return hash % table_size;
}

// Smoothing kernel functions
fn spiky_kernel_pow2(distance: f32, radius: f32) -> f32 {
    let h = radius;
    let h2 = h * h;
    let h4 = h2 * h2;
    let h9 = h4 * h4 * h;
    let scale = 6.0 / (PI * h9);
    let v = h - distance;
    return scale * v * v;
}

fn spiky_kernel_pow3(distance: f32, radius: f32) -> f32 {
    let h = radius;
    let h2 = h * h;
    let h3 = h2 * h;
    let h9 = h3 * h3 * h3;
    let scale = 10.0 / (PI * h9);
    let v = h - distance;
    return scale * v * v * v;
}

// Main compute shader for density and pressure calculation
@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if index >= arrayLength(&particles) {
        return;
    }
    
    // Get particle data
    let position = particles[index].position;
    let cell = get_cell_3d(position, params.smoothing_radius * 2.0);
    let sqr_radius = params.smoothing_radius * params.smoothing_radius;
    
    var density: f32 = 0.0;
    var near_density: f32 = 0.0;
    
    // Search neighboring cells
    for (var i = 0u; i < 27u; i = i + 1u) {
        let offset_cell = cell + OFFSETS_3D[i];
        let hash = hash_cell_3d(offset_cell);
        let key = key_from_hash(hash, arrayLength(&spatial_offsets));
        var curr_index = atomicLoad(&spatial_offsets[key]);
        
        while (curr_index < arrayLength(&particles)) {
            let neighbor_index = curr_index;
            curr_index = curr_index + 1u;
            
            let neighbor_key = spatial_keys[neighbor_index];
            if (neighbor_key != key) {
                break;
            }
            
            // Skip self
            if (neighbor_index == index) {
                continue;
            }
            
            let neighbor_pos = particles[neighbor_index].position;
            let offset = neighbor_pos - position;
            let sqr_distance = dot(offset, offset);
            
            // Skip if not within radius
            if (sqr_distance > sqr_radius) {
                continue;
            }
            
            let distance = sqrt(sqr_distance);
            density += spiky_kernel_pow2(distance, params.smoothing_radius);
            near_density += spiky_kernel_pow3(distance, params.smoothing_radius);
        }
    }
    
    // Calculate pressure from density
    let density_error = density - params.target_density;
    let pressure = density_error * params.pressure_multiplier;
    let near_pressure = near_density * params.near_pressure_multiplier;
    
    // Update particle
    var particle = particles[index];
    particle.density = density;
    particle.pressure = pressure;
    particle.near_density = near_density;
    particle.near_pressure = near_pressure;
    particles[index] = particle;
} 