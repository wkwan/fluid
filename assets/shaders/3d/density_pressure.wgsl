// 3D Density and Pressure Calculation Compute Shader
// Mirrors Unity's CalculateDensities kernel with WGSL optimizations

struct Particle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    force: vec3<f32>,
}

struct FluidParams3D {
    smoothing_radius: f32,
    rest_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    viscosity: f32,
    gravity: vec3<f32>,
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
}

const PI: f32 = 3.14159265359;
const MAX_NEIGHBORS: u32 = 128u;

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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<storage, read> params: FluidParams3D;
@group(0) @binding(2) var<storage, read> spatial_keys_dummy: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_offsets_dummy: array<u32>;
@group(0) @binding(4) var<storage, read> neighbor_counts: array<u32>;
@group(0) @binding(5) var<storage, read> neighbor_indices: array<u32>;

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

fn poly6_kernel(r: vec3<f32>, h: f32) -> f32 {
    let r_len = length(r);
    if (r_len >= h || r_len < 0.0001) {
        return 0.0;
    }
    
    let h2 = h * h;
    let h3 = h2 * h;
    let h4 = h3 * h;
    let h5 = h4 * h;
    let h6 = h5 * h;
    let h9 = h6 * h3;
    
    let coef = 315.0 / (64.0 * PI * h9);
    let h_r = h - r_len;
    return coef * h_r * h_r * h_r;
}

// Main compute shader for density and pressure calculation
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= arrayLength(&particles)) {
        return;
    }
    
    var particle = particles[particle_idx];
    var density = 0.0;
    var near_density = 0.0;
    
    // Get number of neighbors for this particle
    let num_neighbors = neighbor_counts[particle_idx];
    
    // Calculate densities from neighbors
    for (var i = 0u; i < num_neighbors; i = i + 1u) {
        let neighbor_idx = neighbor_indices[particle_idx * MAX_NEIGHBORS + i];
        let neighbor = particles[neighbor_idx];
        
        let r = particle.position - neighbor.position;
        let r_len = length(r);
        
        if (r_len < params.smoothing_radius && r_len > 0.0001) {
            let kernel = poly6_kernel(r, params.smoothing_radius);
            density = density + kernel;
            near_density = near_density + kernel * kernel;
        }
    }
    
    // Add self-contribution
    density = density + poly6_kernel(vec3<f32>(0.0), params.smoothing_radius);
    near_density = near_density + poly6_kernel(vec3<f32>(0.0), params.smoothing_radius) * 
        poly6_kernel(vec3<f32>(0.0), params.smoothing_radius);
    
    // Calculate pressure
    let pressure = max(0.0, params.pressure_multiplier * (density - params.rest_density));
    let near_pressure = max(0.0, params.near_pressure_multiplier * near_density);
    
    // Update particle
    particle.density = density;
    particle.near_density = near_density;
    particle.pressure = pressure;
    particle.near_pressure = near_pressure;
    particles[particle_idx] = particle;
} 