// Particle and parameter structs
struct Particle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
    pressure_force: vec3<f32>,
    near_pressure_force: vec3<f32>,
    viscosity_force: vec3<f32>,
}

struct FluidParams3D {
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    viscosity_strength: f32,
    gravity: vec3<f32>,
    delta_time: f32,
}

// Constants
const HASH_K1: f32 = 15823.0;
const HASH_K2: f32 = 9737333.0;
const HASH_K3: f32 = 440817757.0;
const BLOCK_SIZE: u32 = 256u;
const PI: f32 = 3.14159265359;

// Shared memory for particle data
struct CachedParticle3D {
    position: vec3<f32>,
    velocity: vec3<f32>,
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Bindings
@group(0) @binding(0) var<storage, read_write> particles: array<Particle3D>;
@group(0) @binding(1) var<uniform> params: FluidParams3D;
@group(0) @binding(2) var<storage, read> spatial_keys: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_offsets: array<atomic<u32>>;

// Helper functions
fn get_cell_3d(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(floor(pos / cell_size));
}

fn hash_cell_3d(cell: vec3<i32>) -> u32 {
    let ucell = vec3<u32>(cell + vec3<i32>(i32(BLOCK_SIZE) / 2));
    return u32(HASH_K1 * f32(ucell.x) + HASH_K2 * f32(ucell.y) + HASH_K3 * f32(ucell.z));
}

fn key_from_hash(hash: u32, num_cells: u32) -> u32 {
    return hash % num_cells;
}

fn viscosity_kernel(distance: f32, radius: f32) -> f32 {
    if (distance >= radius) {
        return 0.0;
    }
    let x = 1.0 - distance / radius;
    return 45.0 / (PI * radius * radius * radius) * x * x * x;
}

// Main compute shader for viscosity force calculation
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
    
    var viscosity_force = vec3<f32>(0.0);
    
    // Search neighboring cells
    for (var i = 0u; i < 27u; i = i + 1u) {
        var offset: vec3<i32>;
        switch(i) {
            case 0u: { offset = vec3<i32>(-1, -1, -1); }
            case 1u: { offset = vec3<i32>(-1, -1, 0); }
            case 2u: { offset = vec3<i32>(-1, -1, 1); }
            case 3u: { offset = vec3<i32>(-1, 0, -1); }
            case 4u: { offset = vec3<i32>(-1, 0, 0); }
            case 5u: { offset = vec3<i32>(-1, 0, 1); }
            case 6u: { offset = vec3<i32>(-1, 1, -1); }
            case 7u: { offset = vec3<i32>(-1, 1, 0); }
            case 8u: { offset = vec3<i32>(-1, 1, 1); }
            case 9u: { offset = vec3<i32>(0, -1, -1); }
            case 10u: { offset = vec3<i32>(0, -1, 0); }
            case 11u: { offset = vec3<i32>(0, -1, 1); }
            case 12u: { offset = vec3<i32>(0, 0, -1); }
            case 13u: { offset = vec3<i32>(0, 0, 0); }
            case 14u: { offset = vec3<i32>(0, 0, 1); }
            case 15u: { offset = vec3<i32>(0, 1, -1); }
            case 16u: { offset = vec3<i32>(0, 1, 0); }
            case 17u: { offset = vec3<i32>(0, 1, 1); }
            case 18u: { offset = vec3<i32>(1, -1, -1); }
            case 19u: { offset = vec3<i32>(1, -1, 0); }
            case 20u: { offset = vec3<i32>(1, -1, 1); }
            case 21u: { offset = vec3<i32>(1, 0, -1); }
            case 22u: { offset = vec3<i32>(1, 0, 0); }
            case 23u: { offset = vec3<i32>(1, 0, 1); }
            case 24u: { offset = vec3<i32>(1, 1, -1); }
            case 25u: { offset = vec3<i32>(1, 1, 0); }
            case 26u: { offset = vec3<i32>(1, 1, 1); }
            default: { continue; }
        }
        
        let offset_cell = cell + offset;
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
            let velocity_diff = particles[neighbor_index].velocity - particles[index].velocity;
            
            // Calculate viscosity force
            let viscosity_force_magnitude = viscosity_kernel(distance, params.smoothing_radius);
            viscosity_force += velocity_diff * viscosity_force_magnitude;
        }
    }
    
    // Apply viscosity strength
    viscosity_force *= params.viscosity_strength;
    
    // Update particle
    var particle = particles[index];
    particle.viscosity_force = viscosity_force;
    particles[index] = particle;
} 