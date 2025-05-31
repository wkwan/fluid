// Parallel reduction for neighbor finding in 3D fluid simulation
// Uses workgroup shared memory for efficient neighbor lookup within cells

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
const BLOCK_SIZE: u32 = 128u;

// Shared memory for particle data
struct SharedParticle {
    position: vec3<f32>,
    index: u32,
}

var<workgroup> shared_particles: array<SharedParticle, BLOCK_SIZE>;

@group(0) @binding(0) var<storage, read> particles: array<Particle3D>;
@group(0) @binding(1) var<storage, read> params: FluidParams3D;
@group(0) @binding(2) var<storage, read> spatial_keys_dummy: array<u32>;
@group(0) @binding(3) var<storage, read> spatial_offsets_dummy: array<u32>;
@group(0) @binding(4) var<storage, read_write> neighbor_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> neighbor_indices: array<u32>;

fn get_cell_3d(pos: vec3<f32>, cell_size: f32) -> vec3<i32> {
    return vec3<i32>(
        i32(floor(pos.x / cell_size)),
        i32(floor(pos.y / cell_size)),
        i32(floor(pos.z / cell_size))
    );
}

fn hash_cell_3d(cell: vec3<i32>) -> u32 {
    let k1: u32 = 0x9e3779b1u;
    let k2: u32 = 0x4f1bbcddu;
    let k3: u32 = 0xef649f1fu;
    
    let x = u32(cell.x);
    let y = u32(cell.y);
    let z = u32(cell.z);
    
    var h = x * k1 + y * k2 + z * k3;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

fn key_from_hash(hash: u32, num_particles: u32) -> u32 {
    return hash % num_particles;
}

@compute @workgroup_size(BLOCK_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= arrayLength(&particles)) {
        return;
    }
    
    // Load particle data into shared memory
    let particle = particles[particle_idx];
    shared_particles[global_id.x] = SharedParticle(
        particle.position,
        particle_idx
    );
    
    workgroupBarrier();
    
    // Calculate neighbors within smoothing radius
    var neighbor_count: u32 = 0u;
    var neighbors: array<u32, MAX_NEIGHBORS>;
    
    for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
        let other_particle = shared_particles[i];
        
        // Skip self
        if (other_particle.index == particle_idx) {
            continue;
        }
        
        let r = particle.position - other_particle.position;
        let r_len = length(r);
        
        if (r_len < params.smoothing_radius && r_len > 0.0001) {
            if (neighbor_count < MAX_NEIGHBORS) {
                neighbors[neighbor_count] = other_particle.index;
                neighbor_count = neighbor_count + 1u;
            }
        }
    }
    
    // Store neighbor count and indices
    neighbor_counts[particle_idx] = neighbor_count;
    
    for (var i = 0u; i < neighbor_count; i = i + 1u) {
        neighbor_indices[particle_idx * MAX_NEIGHBORS + i] = neighbors[i];
    }
} 