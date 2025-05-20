// Fluid Simulation Forces Compute Shader
// Computes pressure and viscosity forces for SPH fluid simulation

struct Particle {
    position: vec2<f32>,
    padding0: vec2<f32>,  // Padding for 16-byte alignment
    velocity: vec2<f32>,
    padding1: vec2<f32>,  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

struct FluidParams {
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
    boundary_min: vec2<f32>,
    boundary_min_padding: vec2<f32>,
    
    // Vec4 aligned group 4
    boundary_max: vec2<f32>,
    boundary_max_padding: vec2<f32>,
    
    // Vec4 aligned group 5
    gravity: vec2<f32>,
    gravity_padding: vec2<f32>,
    
    // Vec4 aligned group 6
    mouse_position: vec2<f32>,
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: vec2<u32>,
}

// 2D cell offsets for neighboring cells (9 total including center)
const OFFSETS_2D: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(-1, 1),
    vec2<i32>(0, 1),
    vec2<i32>(1, 1),
    vec2<i32>(-1, 0),
    vec2<i32>(0, 0),
    vec2<i32>(1, 0),
    vec2<i32>(-1, -1),
    vec2<i32>(0, -1),
    vec2<i32>(1, -1)
);

// Constants used for hashing
const HASH_K1: u32 = 15823u;
const HASH_K2: u32 = 9737333u;

@group(0) @binding(0)
var<storage, read_write> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> params: FluidParams;

@group(0) @binding(2)
var<storage, read_write> spatial_keys: array<u32>;

@group(0) @binding(4)
var<storage, read_write> spatial_offsets: array<u32>;

// SPH kernel derivatives for force calculations
fn spiky_pow3_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h || dist <= 0.0) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    let kernel_const = -45.0 / (3.14159265 * h6);
    let kernel = kernel_const * (h - dist) * (h - dist) / dist;
    return kernel;
}

fn spiky_pow2_derivative(dist: f32, h: f32) -> f32 {
    if (dist >= h || dist <= 0.0) {
        return 0.0;
    }
    let h5 = h * h * h * h * h;
    let kernel_const = -30.0 / (3.14159265 * h5);
    let kernel = kernel_const * (h - dist) / dist;
    return kernel;
}

fn viscosity_kernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) {
        return 0.0;
    }
    let h3 = h * h * h;
    let kernel_const = 15.0 / (2.0 * 3.14159265 * h3);
    let kernel = kernel_const * (-pow(dist / h, 3.0) + 2.0 * pow(dist / h, 2.0) - 0.5);
    return kernel;
}

// Convert floating point position into an integer cell coordinate
fn get_cell_2d(position: vec2<f32>, radius: f32) -> vec2<i32> {
    return vec2<i32>(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
fn hash_cell_2d(cell: vec2<i32>) -> u32 {
    let x = u32(cell.x);
    let y = u32(cell.y);
    return x * HASH_K1 + y * HASH_K2;
}

// Get key from hash for a table of given size
fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    return hash % table_size;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles)) {
        return;
    }
    
    let h = params.smoothing_radius;
    let position = particles[index].position;
    let velocity = particles[index].velocity;
    
    // Apply gravity
    var force = params.gravity;
    
    // Apply mouse force if active
    if (params.mouse_active != 0u) {
        let to_mouse = params.mouse_position - position;
        let dist_to_mouse = length(to_mouse);
        
        if (dist_to_mouse < params.mouse_radius) {
            let force_direction = to_mouse / max(dist_to_mouse, 0.001); // Avoid division by zero
            let force_strength = params.mouse_strength * (1.0 - dist_to_mouse / params.mouse_radius);
            
            if (params.mouse_repel != 0u) {
                // Repel
                force -= force_direction * force_strength * params.dt;
            } else {
                // Attract
                force += force_direction * force_strength * params.dt;
            }
        }
    }
    
    // Calculate pressure and viscosity forces using spatial hash
    let origin_cell = get_cell_2d(position, h);
    
    for (var i = 0u; i < 9u; i = i + 1u) {
        let neighbor_cell = origin_cell + OFFSETS_2D[i];
        let hash = hash_cell_2d(neighbor_cell);
        let key = key_from_hash(hash, u32(arrayLength(&particles)));
        
        // Get the starting index for this key
        var curr_index = spatial_offsets[key];
        
        // Iterate through all particles in this cell
        while (curr_index < arrayLength(&particles)) {
            if (curr_index == 0xFFFFFFFFu) {
                // Cell is empty (sentinel value)
                break;
            }
            
            // Check if we're still in the same cell
            if (key != spatial_keys[curr_index]) {
                break;
            }
            
            // Skip self-interaction
            if (curr_index != index) {
                let other_pos = particles[curr_index].position;
                let other_vel = particles[curr_index].velocity;
                let offset = position - other_pos;
                let dist_squared = dot(offset, offset);
                
                if (dist_squared < h * h && dist_squared > 0.0) {
                    let dist = sqrt(dist_squared);
                    let dir = offset / dist;
                    
                    // Pressure force calculation
                    let shared_pressure = (particles[index].pressure + particles[curr_index].pressure) * 0.5;
                    let shared_near_pressure = (particles[index].near_pressure + particles[curr_index].near_pressure) * 0.5;
                    
                    let pressure_force = dir * (
                        spiky_pow3_derivative(dist, h) * shared_pressure +
                        spiky_pow2_derivative(dist, h) * shared_near_pressure
                    );
                    
                    // Viscosity force calculation
                    let vel_diff = other_vel - velocity;
                    let visc_strength = params.viscosity_strength * viscosity_kernel(dist, h);
                    let viscosity_force = vel_diff * visc_strength;
                    
                    // Add forces
                    force += pressure_force + viscosity_force;
                }
            }
            
            // Move to next particle in this cell
            curr_index = curr_index + 1u;
        }
    }
    
    // Update velocity with forces
    let new_velocity = velocity + force * params.dt;
    
    // Update particle velocity
    particles[index].velocity = new_velocity;
} 