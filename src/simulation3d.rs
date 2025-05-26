use bevy::prelude::*;
use bevy::math::primitives::{Sphere, Plane3d};
use crate::math::FluidMath3D;
use crate::simulation::SimulationDimension;
use crate::spatial_hash3d::SpatialHashResource3D;
use rand;
use serde::{Serialize, Deserialize};
use bevy::pbr::MeshMaterial3d;

// 3D particle component
#[derive(Component)]
pub struct Particle3D {
    pub velocity: Vec3,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
}

// Marker for 3D entities to allow cleanup
#[derive(Component)]
pub struct Marker3D;

// Marker for the ground plane
#[derive(Component)]
pub struct GroundPlane;

// Constants for 3D sim (match 2D values where possible)
const GRAVITY_3D: Vec3 = Vec3::new(0.0, -9.81, 0.0);
pub const BOUNDARY_MIN: Vec3 = Vec3::new(-300.0, -300.0, -300.0);
pub const BOUNDARY_MAX: Vec3 = Vec3::new(300.0, 300.0, 300.0);
const PARTICLE_RADIUS: f32 = 5.0;
const BOUNDARY_DAMPENING: f32 = 0.3;
const KILL_Y_THRESHOLD: f32 = -400.0; // Below this Y value, particles are recycled

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct Fluid3DParams {
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub viscosity_strength: f32,
}

impl Default for Fluid3DParams {
    fn default() -> Self {
        Self {
            smoothing_radius: 35.0,
            target_density: 1000.0,
            pressure_multiplier: 200.0,
            viscosity_strength: 0.1,
        }
    }
}

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct SpawnRegion3D {
    pub min: Vec3,
    pub max: Vec3,
    pub spacing: f32,
    pub active: bool,
}

impl Default for SpawnRegion3D {
    fn default() -> Self {
        Self {
            min: Vec3::new(-25.0, 100.0, -25.0),
            max: Vec3::new(25.0, 200.0, 25.0),
            spacing: PARTICLE_RADIUS * 1.15,
            active: true,
        }
    }
}

// ======================== SETUP ============================
pub fn setup_3d_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    _asset_server: Res<AssetServer>,
    query_cam: Query<(), With<Camera3d>>, // only spawn if none
    query_ground: Query<(), With<GroundPlane>>, // check if ground already exists
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }

    if !query_cam.is_empty() {
        return;
    }

    // Add a basic directional light so we can see the spheres
    commands.spawn((
        DirectionalLight {
            shadows_enabled: false,
            illuminance: 20000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 300.0, 300.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
        Marker3D,
    ));

    // Add ground plane if it doesn't exist
    if query_ground.is_empty() {
        let ground_size = (BOUNDARY_MAX.x - BOUNDARY_MIN.x) * 1.2; // Make it slightly larger than boundaries
        let ground_mesh = meshes.add(
            Plane3d::default()
                .mesh()
                .size(ground_size, ground_size)
        );
        
        let ground_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.4, 0.2), // Brown color
            perceptual_roughness: 0.9,
            metallic: 0.0,
            ..default()
        });

        commands.spawn((
            Mesh3d(ground_mesh),
            MeshMaterial3d(ground_material),
            Transform::from_xyz(0.0, BOUNDARY_MIN.y, 0.0), // Position at bottom boundary
            GroundPlane,
            Marker3D,
        ));
    }
}

// ======================== SPAWNER ==========================
pub fn spawn_particles_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim_dim: Res<State<SimulationDimension>>,
    spawn_region: Res<SpawnRegion3D>,
    existing: Query<(), With<Particle3D>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 || !spawn_region.active {
        return;
    }

    // Only spawn once (when no particles exist)
    if !existing.is_empty() {
        return;
    }

    // Create shared mesh - but NOT shared material
    let sphere_mesh = meshes.add(
        Sphere::new(PARTICLE_RADIUS)
            .mesh()
            .ico(2)
            .unwrap(),
    );

    // Calculate grid dimensions
    let size = spawn_region.max - spawn_region.min;
    let grid_x = (size.x / spawn_region.spacing).floor() as i32;
    let grid_y = (size.y / spawn_region.spacing).floor() as i32;
    let grid_z = (size.z / spawn_region.spacing).floor() as i32;

    for xi in 0..grid_x {
        for yi in 0..grid_y {
            for zi in 0..grid_z {
                let pos = Vec3::new(
                    spawn_region.min.x + xi as f32 * spawn_region.spacing,
                    spawn_region.min.y + yi as f32 * spawn_region.spacing,
                    spawn_region.min.z + zi as f32 * spawn_region.spacing,
                );
                
                // Create a unique material for each particle
                let material = materials.add(StandardMaterial {
                    base_color: Color::srgb(0.1, 0.4, 1.0),
                    perceptual_roughness: 0.8,
                    ..default()
                });

                commands.spawn((
                    Mesh3d(sphere_mesh.clone()),
                    MeshMaterial3d(material),
                    Transform::from_translation(pos),
                    Particle3D {
                        velocity: Vec3::ZERO,
                        density: 0.0,
                        pressure: 0.0,
                        near_density: 0.0,
                        near_pressure: 0.0,
                    },
                    Marker3D,
                ));
            }
        }
    }
}

// =================== PHYSICS SYSTEMS =======================

pub fn apply_external_forces_3d(
    time: Res<Time>,
    mut particles: Query<&mut Particle3D>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }

    let dt = time.delta_secs();
    for mut p in particles.iter_mut() {
        p.velocity += GRAVITY_3D * dt;
    }
}

pub fn update_spatial_hash_3d(
    mut spatial_hash: ResMut<SpatialHashResource3D>,
    particle_query: Query<(Entity, &Transform), With<Particle3D>>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }

    spatial_hash.spatial_hash.clear();
    
    for (entity, transform) in particle_query.iter() {
        spatial_hash.spatial_hash.insert(transform.translation, entity);
    }
}

pub fn calculate_density_pressure_3d(
    mut particles_q: Query<(Entity, &Transform, &mut Particle3D)>,
    spatial_hash: Res<SpatialHashResource3D>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }

    let smoothing_radius: f32 = 35.0; // TODO param
    let smoothing_radius_squared = smoothing_radius * smoothing_radius;
    let math = FluidMath3D::new(smoothing_radius);
    let target_density = 1000.0;
    let pressure_mult = 200.0;

    // Cache positions and store entities order
    let mut positions: Vec<Vec3> = Vec::with_capacity(particles_q.iter().count());
    let mut entities: Vec<Entity> = Vec::with_capacity(positions.capacity());

    for (e, t, _) in particles_q.iter() {
        entities.push(e);
        positions.push(t.translation);
    }

    let count = positions.len();
    let mut densities = vec![0.0f32; count];

    for i in 0..count {
        let pos_i = positions[i];
        let mut density = 0.0;
        
        // Get neighbors using spatial hash
        let neighbors = spatial_hash.spatial_hash.get_neighbors(pos_i, smoothing_radius);
        
        // Add self-contribution
        density += math.poly6(0.0, smoothing_radius_squared);
        
        // Add neighbor contributions
        for &neighbor_entity in &neighbors {
            if let Ok((_, transform, _)) = particles_q.get(neighbor_entity) {
                let r2 = (pos_i - transform.translation).length_squared();
                if r2 < smoothing_radius_squared {
                    density += math.poly6(r2, smoothing_radius_squared);
                }
            }
        }
        
        densities[i] = density;
    }

    // Write back density/pressure
    for (idx, entity) in entities.iter().enumerate() {
        if let Ok((_, _, mut part)) = particles_q.get_mut(*entity) {
            let density = densities[idx];
            part.density = density;
            part.pressure = (density - target_density) * pressure_mult;
        }
    }
}

pub fn apply_pressure_viscosity_3d(
    mut particles_q: Query<(Entity, &Transform, &mut Particle3D)>,
    spatial_hash: Res<SpatialHashResource3D>,
    params: Res<Fluid3DParams>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }

    let smoothing_radius = params.smoothing_radius;
    let smoothing_radius_squared = smoothing_radius * smoothing_radius;
    let math = FluidMath3D::new(smoothing_radius);
    let viscosity = params.viscosity_strength;

    // Cache positions and store entities order
    let mut positions: Vec<Vec3> = Vec::with_capacity(particles_q.iter().count());
    let mut velocities: Vec<Vec3> = Vec::with_capacity(positions.capacity());
    let mut entities: Vec<Entity> = Vec::with_capacity(positions.capacity());

    for (e, t, p) in particles_q.iter() {
        entities.push(e);
        positions.push(t.translation);
        velocities.push(p.velocity);
    }

    let count = positions.len();
    let mut delta_vs = vec![Vec3::ZERO; count];

    for i in 0..count {
        let pos_i = positions[i];
        let vel_i = velocities[i];
        let entity_i = entities[i];
        let mut pressure_force = Vec3::ZERO;
        let mut viscosity_force = Vec3::ZERO;

        // Get neighbors using spatial hash
        let neighbors = spatial_hash.spatial_hash.get_neighbors(pos_i, smoothing_radius);
        for &neighbor_entity in &neighbors {
            if neighbor_entity == entity_i { continue; }
            if let Ok((_, t_j, p_j)) = particles_q.get(neighbor_entity) {
                let pos_j = t_j.translation;
                let vel_j = p_j.velocity;
                let r = pos_i - pos_j;
                let dist = r.length();
                if dist > 0.0 && dist < smoothing_radius {
                    // Pressure force (spiky gradient)
                    let dir = r / dist;
                    let pressure_term = (p_j.pressure + p_j.pressure) * 0.5; // symmetric
                    let grad = math.spiky_pow3_derivative(dist, smoothing_radius);
                    pressure_force -= dir * pressure_term * grad;

                    // Viscosity force (Laplacian)
                    let lap = math.spiky_pow2(dist, smoothing_radius);
                    viscosity_force += (vel_j - vel_i) * lap;
                }
            }
        }
        // Apply viscosity strength
        viscosity_force *= viscosity;
        // Accumulate
        delta_vs[i] = pressure_force + viscosity_force;
    }

    // Write back velocity changes
    for (i, entity) in entities.iter().enumerate() {
        if let Ok((_, _, mut p)) = particles_q.get_mut(*entity) {
            p.velocity += delta_vs[i];
        }
    }
}

pub fn integrate_positions_3d(
    time: Res<Time>,
    mut particles: Query<(&mut Transform, &mut Particle3D)>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }
    let dt = time.delta_secs();
    for (mut transform, mut particle) in particles.iter_mut() {
        transform.translation += particle.velocity * dt;

        // Boundary collisions with damping and friction
        let mut pos = transform.translation;
        let mut vel = particle.velocity;

        // X-axis
        if pos.x < BOUNDARY_MIN.x + PARTICLE_RADIUS {
            pos.x = BOUNDARY_MIN.x + PARTICLE_RADIUS;
            vel.x = -vel.x * BOUNDARY_DAMPENING;
        } else if pos.x > BOUNDARY_MAX.x - PARTICLE_RADIUS {
            pos.x = BOUNDARY_MAX.x - PARTICLE_RADIUS;
            vel.x = -vel.x * BOUNDARY_DAMPENING;
        }

        // Y-axis
        if pos.y < BOUNDARY_MIN.y + PARTICLE_RADIUS {
            pos.y = BOUNDARY_MIN.y + PARTICLE_RADIUS;
            vel.y = -vel.y * BOUNDARY_DAMPENING;
        } else if pos.y > BOUNDARY_MAX.y - PARTICLE_RADIUS {
            pos.y = BOUNDARY_MAX.y - PARTICLE_RADIUS;
            vel.y = -vel.y * BOUNDARY_DAMPENING;
        }

        // Z-axis
        if pos.z < BOUNDARY_MIN.z + PARTICLE_RADIUS {
            pos.z = BOUNDARY_MIN.z + PARTICLE_RADIUS;
            vel.z = -vel.z * BOUNDARY_DAMPENING;
        } else if pos.z > BOUNDARY_MAX.z - PARTICLE_RADIUS {
            pos.z = BOUNDARY_MAX.z - PARTICLE_RADIUS;
            vel.z = -vel.z * BOUNDARY_DAMPENING;
        }

        transform.translation = pos;
        particle.velocity = vel;
    }
}

pub fn recycle_particles_3d(
    mut commands: Commands,
    mut particles: Query<(Entity, &Transform, &mut Particle3D)>,
    spawn_region: Res<SpawnRegion3D>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 || !spawn_region.active {
        return;
    }

    for (entity, transform, mut particle) in particles.iter_mut() {
        if transform.translation.y < KILL_Y_THRESHOLD {
            // Reset particle to a random position in spawn region
            let size = spawn_region.max - spawn_region.min;
            let pos = Vec3::new(
                spawn_region.min.x + rand::random::<f32>() * size.x,
                spawn_region.min.y + rand::random::<f32>() * size.y,
                spawn_region.min.z + rand::random::<f32>() * size.z,
            );

            // Update transform and reset particle state
            commands.entity(entity).insert(Transform::from_translation(pos));
            particle.velocity = Vec3::ZERO;
            particle.density = 0.0;
            particle.pressure = 0.0;
            particle.near_density = 0.0;
            particle.near_pressure = 0.0;
        }
    }
}

pub fn update_particle_colors_3d(
    mut materials: ResMut<Assets<StandardMaterial>>,
    particles: Query<(&Particle3D, &MeshMaterial3d<StandardMaterial>)>,
    time: Res<Time>,
) {
    // Adjusted maximum velocity for normalization to match actual simulation velocity values
    const MAX_VELOCITY: f32 = 700.0;
    
    // Debug info
    let mut total_magnitude = 0.0;
    let mut count = 0;
    let mut max_seen: f32 = 0.0;
    
    for (particle, mat_handle) in particles.iter() {
        // Calculate velocity magnitude and normalize
        let velocity_magnitude = particle.velocity.length();
        total_magnitude += velocity_magnitude;
        count += 1;
        max_seen = max_seen.max(velocity_magnitude);
        
        let normalized_velocity = (velocity_magnitude / MAX_VELOCITY).clamp(0.0, 1.0);
        
        // Create a blue -> green -> red gradient based on velocity
        let color = if normalized_velocity < 0.5 {
            // Blue to green
            let local_t = normalized_velocity * 2.0;
            Color::srgb(0.0, local_t, 1.0 - local_t)
        } else {
            // Green to red
            let local_t = (normalized_velocity - 0.5) * 2.0;
            Color::srgb(local_t, 1.0 - local_t, 0.0)
        };
        
        // Apply the color to the material
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.base_color = color;
        }
    }
    
    // Only print debug info occasionally to avoid flooding the console
    if count > 0 && ((time.elapsed_secs_f64() % 2.0) < 0.1) {
        println!("3D Particles - Avg velocity: {:.2}, Max velocity: {:.2}, Using MAX_VELOCITY={:.2} for normalization", 
                 total_magnitude / count as f32, 
                 max_seen,
                 MAX_VELOCITY);
    }
} 