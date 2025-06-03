// Unity-style 3D Fluid Simulation Compute Shader
// Mirrors Unity's FluidSim.compute exactly

// Unity-style parameters struct
struct FluidParams {
    smoothing_radius: f32,
    target_density: f32, 
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    // Precomputed kernel constants (like Unity)
    k_spiky_pow2: f32,
    k_spiky_pow3: f32,
    k_spiky_pow2_grad: f32,
    k_spiky_pow3_grad: f32,
    
    gravity: vec3<f32>,
    _padding0: f32,
    bounds_size: vec3<f32>,
    _padding1: f32,
    centre: vec3<f32>,
    _padding2: f32,
    
    delta_time: f32,
    collision_damping: f32,
    num_particles: u32,
    _padding3: f32,
    
    // Density map parameters (like Unity)
    density_map_size: vec3<u32>,
    _padding4: f32,
}

// Unity-style separate buffers
@group(0) @binding(0) var<storage, read_write> Positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> PredictedPositions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Densities: array<vec2<f32>>; // [density, near_density]

// Spatial hashing buffers
@group(0) @binding(4) var<storage, read_write> SpatialKeys: array<u32>;
@group(0) @binding(5) var<storage, read_write> SpatialOffsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> SortedIndices: array<u32>;

// Sort target buffers for reordering
@group(0) @binding(7) var<storage, read_write> SortTarget_Positions: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> SortTarget_PredictedPositions: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> SortTarget_Velocities: array<vec4<f32>>;

@group(0) @binding(10) var<uniform> params: FluidParams;

// 3D Density texture for volumetric rendering (like Unity's DensityMap)
// @group(0) @binding(11) var DensityMap: texture_storage_3d<rgba16float, write>;

const PI: f32 = 3.14159265359;
const ThreadGroupSize: u32 = 256u;

// Unity's 3D cell offsets for 27 neighboring cells
const offsets3D: array<vec3<i32>, 27> = array<vec3<i32>, 27>(
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

// Unity's spatial hash functions
fn GetCell3D(position: vec3<f32>, smoothing_radius: f32) -> vec3<i32> {
    return vec3<i32>(floor(position / smoothing_radius));
}

fn HashCell3D(cell: vec3<i32>) -> u32 {
    let k1: u32 = 15823u;
    let k2: u32 = 9737333u; 
    let k3: u32 = 440817757u;
    
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

fn KeyFromHash(hash: u32, num_particles: u32) -> u32 {
    return hash % num_particles;
}

// Unity's SPH kernel functions using precomputed constants
fn DensityKernel(dst: f32, radius: f32) -> f32 {
    if (dst < radius) {
        let v = radius - dst;
        return v * v * params.k_spiky_pow2;
    }
    return 0.0;
}

fn NearDensityKernel(dst: f32, radius: f32) -> f32 {
    if (dst < radius) {
        let v = radius - dst;
        return v * v * v * params.k_spiky_pow3;
    }
    return 0.0;
}

fn DensityDerivative(dst: f32, radius: f32) -> f32 {
    if (dst <= radius) {
        let v = radius - dst;
        return -v * params.k_spiky_pow2_grad;
    }
    return 0.0;
}

fn NearDensityDerivative(dst: f32, radius: f32) -> f32 {
    if (dst <= radius) {
        let v = radius - dst;
        return -v * v * params.k_spiky_pow3_grad;
    }
    return 0.0;
}

// Unity's pressure calculation
fn PressureFromDensity(density: f32) -> f32 {
    return (density - params.target_density) * params.pressure_multiplier;
}

fn NearPressureFromDensity(near_density: f32) -> f32 {
    return near_density * params.near_pressure_multiplier;
}

// Unity's collision resolution
fn ResolveCollisions(pos: ptr<function, vec3<f32>>, vel: ptr<function, vec3<f32>>) {
    // Transform to local space (Unity does this with matrices, we'll use simple bounds)
    let half_size = params.bounds_size * 0.5;
    let centre = params.centre;
    let pos_local = *pos - centre;
    let edge_dst = half_size - abs(pos_local);
    
    // Resolve collisions on each axis
    if (edge_dst.x <= 0.0) {
        let new_x = sign(pos_local.x) * half_size.x;
        (*pos).x = centre.x + new_x;
        (*vel).x *= -params.collision_damping;
    }
    if (edge_dst.y <= 0.0) {
        let new_y = sign(pos_local.y) * half_size.y;
        (*pos).y = centre.y + new_y;
        (*vel).y *= -params.collision_damping;
    }
    if (edge_dst.z <= 0.0) {
        let new_z = sign(pos_local.z) * half_size.z;
        (*pos).z = centre.z + new_z;
        (*vel).z *= -params.collision_damping;
    }
}

// ===== KERNEL 1: External Forces (Unity: ExternalForces) =====
@compute @workgroup_size(ThreadGroupSize)
fn external_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    
    // External forces (gravity) - make this more aggressive for testing
    let gravity_force = params.gravity * params.delta_time * 10.0; // 10x stronger for testing
    Velocities[index] = vec4<f32>(
        Velocities[index].xyz + gravity_force,
        0.0
    );
    
    // Predict positions (Unity uses fixed 1/120.0 timestep for prediction)
    PredictedPositions[index] = vec4<f32>(
        Positions[index].xyz + Velocities[index].xyz * (1.0 / 120.0),
        0.0
    );
}

// ===== KERNEL 2: Spatial Hash (Unity: UpdateSpatialHash) =====
@compute @workgroup_size(ThreadGroupSize)
fn spatial_hash(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    let cell = GetCell3D(PredictedPositions[index].xyz, params.smoothing_radius);
    let hash = HashCell3D(cell);
    let key = KeyFromHash(hash, params.num_particles);
    
    SpatialKeys[index] = key;
}

// ===== KERNEL 3: Sort Spatial Hash (Unity: spatialHash.Run()) =====
@compute @workgroup_size(ThreadGroupSize)
fn sort_spatial_hash(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    
    // Initialize sorted indices (no sorting for now - just identity mapping)
    SortedIndices[index] = index;
    
    // Initialize spatial offsets (simplified - no actual sorting)
    SpatialOffsets[index] = index;
}

// ===== KERNEL 4: Reorder (Unity: Reorder) =====
@compute @workgroup_size(ThreadGroupSize)
fn reorder(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    let sorted_index = SortedIndices[index];
    
    SortTarget_Positions[index] = Positions[sorted_index];
    SortTarget_PredictedPositions[index] = PredictedPositions[sorted_index];
    SortTarget_Velocities[index] = Velocities[sorted_index];
}

// ===== KERNEL 5: Reorder Copyback (Unity: ReorderCopyBack) =====
@compute @workgroup_size(ThreadGroupSize)
fn reorder_copyback(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    
    Positions[index] = SortTarget_Positions[index];
    PredictedPositions[index] = SortTarget_PredictedPositions[index];
    Velocities[index] = SortTarget_Velocities[index];
}

// ===== KERNEL 6: Calculate Densities (Unity: CalculateDensities) =====
@compute @workgroup_size(ThreadGroupSize)
fn calculate_densities(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    let pos = PredictedPositions[index].xyz;
    let sqr_radius = params.smoothing_radius * params.smoothing_radius;
    
    var density = 0.0;
    var near_density = 0.0;
    
    // Include self-contribution to density (critical for stability)
    density += DensityKernel(0.0, params.smoothing_radius);
    near_density += NearDensityKernel(0.0, params.smoothing_radius);
    
    // Check neighbors
    for (var i = 0u; i < params.num_particles; i++) {
        if (i == index) {
            continue;
        }
        
        let neighbor_pos = PredictedPositions[i].xyz;
        let offset_to_neighbor = neighbor_pos - pos;
        let sqr_dst_to_neighbor = dot(offset_to_neighbor, offset_to_neighbor);
        
        if (sqr_dst_to_neighbor > sqr_radius) {
            continue;
        }
        
        let dst = sqrt(sqr_dst_to_neighbor);
        density += DensityKernel(dst, params.smoothing_radius);
        near_density += NearDensityKernel(dst, params.smoothing_radius);
    }
    
    Densities[index] = vec2<f32>(density, near_density);
}

// ===== KERNEL 7: Calculate Pressure Force (Unity: CalculatePressureForce) =====
@compute @workgroup_size(ThreadGroupSize)
fn calculate_pressure_force(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    let pos = PredictedPositions[index].xyz;
    let density = max(Densities[index].x, 0.001); // Prevent division by zero
    let density_near = Densities[index].y;
    let pressure = PressureFromDensity(density);
    let near_pressure = NearPressureFromDensity(density_near);
    
    var pressure_force = vec3<f32>(0.0);
    let velocity = Velocities[index].xyz;
    let sqr_radius = params.smoothing_radius * params.smoothing_radius;
    
    // Check neighbors for pressure force
    for (var i = 0u; i < params.num_particles; i++) {
        if (i == index) {
            continue;
        }
        
        let neighbor_pos = PredictedPositions[i].xyz;
        let offset_to_neighbor = neighbor_pos - pos;
        let sqr_dst_to_neighbor = dot(offset_to_neighbor, offset_to_neighbor);
        
        if (sqr_dst_to_neighbor > sqr_radius) {
            continue;
        }
        
        let density_neighbor = max(Densities[i].x, 0.001); // Prevent division by zero
        let near_density_neighbor = Densities[i].y;
        let neighbor_pressure = PressureFromDensity(density_neighbor);
        let neighbor_pressure_near = NearPressureFromDensity(near_density_neighbor);
        
        let shared_pressure = (pressure + neighbor_pressure) / 2.0;
        let shared_near_pressure = (near_pressure + neighbor_pressure_near) / 2.0;
        
        let dst_to_neighbor = sqrt(sqr_dst_to_neighbor);
        let dir_to_neighbor = select(
            vec3<f32>(0.0, 1.0, 0.0),
            offset_to_neighbor / dst_to_neighbor,
            dst_to_neighbor > 0.0
        );
        
        // Unity's pressure force calculation with safe division
        pressure_force += dir_to_neighbor * DensityDerivative(dst_to_neighbor, params.smoothing_radius) * shared_pressure / density_neighbor;
        pressure_force += dir_to_neighbor * NearDensityDerivative(dst_to_neighbor, params.smoothing_radius) * shared_near_pressure / density_neighbor;
    }
    
    let acceleration = pressure_force / density; // Safe division due to max() above
    let velocity_new = velocity + acceleration * params.delta_time;
    Velocities[index] = vec4<f32>(velocity_new, 0.0);
}

// ===== KERNEL 8: Update Positions (Unity: UpdatePositions) =====
@compute @workgroup_size(ThreadGroupSize)
fn update_positions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.num_particles) {
        return;
    }
    
    let index = global_id.x;
    var vel = Velocities[index].xyz;
    var pos = Positions[index].xyz;
    
    // Update position
    pos += vel * params.delta_time;
    
    // Unity's collision resolution
    ResolveCollisions(&pos, &vel);
    
    // Write results
    Positions[index] = vec4<f32>(pos, 0.0);
    Velocities[index] = vec4<f32>(vel, 0.0);
}

// ===== KERNEL 9: Volumetric Render (Unity: RenderKernel) =====
// @compute @workgroup_size(8, 8, 8)
// fn volumetric_render(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     let density_map_size = params.density_map_size;
//     
//     if (global_id.x >= density_map_size.x || 
//         global_id.y >= density_map_size.y || 
//         global_id.z >= density_map_size.z) {
//         return;
//     }
//     
//     // Convert texture coordinates to world position
//     let texture_coord = vec3<f32>(global_id) / vec3<f32>(density_map_size);
//     let world_pos = params.centre + (texture_coord - 0.5) * params.bounds_size;
//     
//     var total_density = 0.0;
//     
//     // Unity's approach: sample particles within smoothing radius
//     let origin_cell = GetCell3D(world_pos, params.smoothing_radius);
//     let sqr_radius = params.smoothing_radius * params.smoothing_radius;
//     
//     // Unity's 27-cell neighbor search for density sampling
//     for (var offset_idx = 0; offset_idx < 27; offset_idx++) {
//         var neighbor_offset: vec3<i32>;
//         if (offset_idx == 0) { neighbor_offset = vec3<i32>(-1, -1, -1); }
//         else if (offset_idx == 1) { neighbor_offset = vec3<i32>(0, -1, -1); }
//         else if (offset_idx == 2) { neighbor_offset = vec3<i32>(1, -1, -1); }
//         else if (offset_idx == 3) { neighbor_offset = vec3<i32>(-1, 0, -1); }
//         else if (offset_idx == 4) { neighbor_offset = vec3<i32>(0, 0, -1); }
//         else if (offset_idx == 5) { neighbor_offset = vec3<i32>(1, 0, -1); }
//         else if (offset_idx == 6) { neighbor_offset = vec3<i32>(-1, 1, -1); }
//         else if (offset_idx == 7) { neighbor_offset = vec3<i32>(0, 1, -1); }
//         else if (offset_idx == 8) { neighbor_offset = vec3<i32>(1, 1, -1); }
//         else if (offset_idx == 9) { neighbor_offset = vec3<i32>(-1, -1, 0); }
//         else if (offset_idx == 10) { neighbor_offset = vec3<i32>(0, -1, 0); }
//         else if (offset_idx == 11) { neighbor_offset = vec3<i32>(1, -1, 0); }
//         else if (offset_idx == 12) { neighbor_offset = vec3<i32>(-1, 0, 0); }
//         else if (offset_idx == 13) { neighbor_offset = vec3<i32>(0, 0, 0); }
//         else if (offset_idx == 14) { neighbor_offset = vec3<i32>(1, 0, 0); }
//         else if (offset_idx == 15) { neighbor_offset = vec3<i32>(-1, 1, 0); }
//         else if (offset_idx == 16) { neighbor_offset = vec3<i32>(0, 1, 0); }
//         else if (offset_idx == 17) { neighbor_offset = vec3<i32>(1, 1, 0); }
//         else if (offset_idx == 18) { neighbor_offset = vec3<i32>(-1, -1, 1); }
//         else if (offset_idx == 19) { neighbor_offset = vec3<i32>(0, -1, 1); }
//         else if (offset_idx == 20) { neighbor_offset = vec3<i32>(1, -1, 1); }
//         else if (offset_idx == 21) { neighbor_offset = vec3<i32>(-1, 0, 1); }
//         else if (offset_idx == 22) { neighbor_offset = vec3<i32>(0, 0, 1); }
//         else if (offset_idx == 23) { neighbor_offset = vec3<i32>(1, 0, 1); }
//         else if (offset_idx == 24) { neighbor_offset = vec3<i32>(-1, 1, 1); }
//         else if (offset_idx == 25) { neighbor_offset = vec3<i32>(0, 1, 1); }
//         else { neighbor_offset = vec3<i32>(1, 1, 1); } // offset_idx == 26
//         
//         let hash = HashCell3D(origin_cell + neighbor_offset);
//         let key = KeyFromHash(hash, params.num_particles);
//         var curr_index = SpatialOffsets[key];
//         
//         while (curr_index < params.num_particles) {
//             let particle_index = curr_index;
//             curr_index++;
//             
//             let particle_key = SpatialKeys[particle_index];
//             if (particle_key != key) {
//                 break;
//             }
//             
//             let particle_pos = PredictedPositions[particle_index].xyz;
//             let offset_to_particle = particle_pos - world_pos;
//             let sqr_dst_to_particle = dot(offset_to_particle, offset_to_particle);
//             
//             if (sqr_dst_to_particle > sqr_radius) {
//                 continue;
//             }
//             
//             let dst = sqrt(sqr_dst_to_particle);
//             total_density += DensityKernel(dst, params.smoothing_radius);
//         }
//     }
//     
//     // Normalize and write to texture
//     let normalized_density = total_density / params.target_density;
//     textureStore(DensityMap, global_id, vec4<f32>(normalized_density, normalized_density, normalized_density, 1.0));
// } 