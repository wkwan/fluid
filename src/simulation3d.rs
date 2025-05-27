use bevy::prelude::*;
use bevy::math::primitives::{Sphere, Plane3d};
use bevy::pbr::MeshMaterial3d;
use bevy::time::{Timer, TimerMode};
use crate::math::FluidMath3D;
use crate::simulation::SimulationDimension;
use crate::spatial_hash3d::SpatialHashResource3D;
use rand::{self, Rng};
use serde::{Serialize, Deserialize};
use crate::constants::{PARTICLE_RADIUS, BOUNDARY_DAMPENING, GRAVITY_3D};

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

// Component to store deformable ground mesh data
#[derive(Component)]
pub struct DeformableGround {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub width_segments: u32,
    pub height_segments: u32,
    pub size: f32,
}

// Component for solid rubber ducks that interact with particles
#[derive(Component)]
pub struct RubberDuck {
    pub velocity: Vec3,
    pub angular_velocity: Vec3, // Rotation velocity in radians per second
    pub size: f32,
}

#[derive(Component)]
pub struct MouseIndicator;

#[derive(Component)]
pub struct BoundaryWireframe;

// 3D Mouse interaction resource
#[derive(Resource, Clone)]
pub struct MouseInteraction3D {
    pub position: Vec3,
    pub active: bool,
    pub repel: bool,
    pub strength: f32,
    pub radius: f32,
}

impl Default for MouseInteraction3D {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            active: false,
            repel: false,
            strength: 10000.0,  // Increased from 1000.0
            radius: 150.0,     // Increased from 50.0
        }
    }
}

// Constants for 3D sim (match 2D values where possible)
const GRAVITY_VEC3: Vec3 = Vec3::new(GRAVITY_3D[0], GRAVITY_3D[1], GRAVITY_3D[2]);
const BOUNDARY_MIN: Vec3 = Vec3::new(-300.0, -300.0, -300.0);
const BOUNDARY_MAX: Vec3 = Vec3::new(300.0, 300.0, 300.0);
const DUCK_SIZE: f32 = PARTICLE_RADIUS * 5.0; // 5x bigger than particles
const KILL_Y_THRESHOLD: f32 = -400.0; // Below this Y value, particles are recycled
const MAX_ANGULAR_VELOCITY: f32 = 3.0; // Maximum angular velocity in radians per second

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct Fluid3DParams {
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub viscosity_strength: f32,
    pub collision_damping: f32,
}

impl Default for Fluid3DParams {
    fn default() -> Self {
        Self {
            smoothing_radius: 35.0,
            target_density: 1200.0,  // Reduced from 1500 to allow settling
            pressure_multiplier: 100.0,  // Reduced from 150 to let gravity dominate
            near_pressure_multiplier: 12.0,  // Reduced from 25 to allow settling
            viscosity_strength: 0.12,  // Slightly reduced for better flow
            collision_damping: 0.95,
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
            min: Vec3::new(-15.0, 100.0, -15.0),    // Reduced from -25.0 to -15.0 (smaller area)
            max: Vec3::new(15.0, 150.0, 15.0),      // Reduced from 25.0 to 15.0 and 200.0 to 150.0 (smaller volume)
            spacing: PARTICLE_RADIUS * 1.5,         // Increased from 0.95 to 1.5 (looser packing)
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
        let segments = 50; // Higher resolution for better deformation
        
        // Create custom deformable ground mesh
        let (vertices, indices) = create_deformable_plane_mesh(ground_size, segments, segments);
        let ground_mesh = create_mesh_from_vertices(&vertices, &indices);
        let mesh_handle = meshes.add(ground_mesh);
        
        let ground_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.4, 0.2), // Brown color
            perceptual_roughness: 0.9,
            metallic: 0.0,
            ..default()
        });

        commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(ground_material),
            Transform::from_xyz(0.0, BOUNDARY_MIN.y, 0.0), // Position at bottom boundary
            GroundPlane,
            DeformableGround {
                vertices: vertices.clone(),
                indices,
                width_segments: segments,
                height_segments: segments,
                size: ground_size,
            },
            Marker3D,
        ));
    }

    // Add boundary wireframe cube
    create_boundary_wireframe(&mut commands, &mut meshes, &mut materials);
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

pub fn handle_mouse_input_3d(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<crate::orbit_camera::OrbitCamera>>,
    mut mouse_interaction_3d: ResMut<MouseInteraction3D>,
    sim_dim: Res<State<SimulationDimension>>,
    particles: Query<&Transform, With<Particle3D>>,
    draw_lake_mode: Res<crate::simulation::DrawLakeMode>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    // Handle mouse interaction
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                // Convert screen position to world ray
                if let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_position) {
                    // Find the closest particle to the ray to determine interaction depth
                    let mut closest_distance = f32::INFINITY;
                    let mut best_position = Vec3::ZERO;
                    
                    // Check all particles to find the one closest to the ray
                    for particle_transform in particles.iter() {
                        let particle_pos = particle_transform.translation;
                        
                        // Calculate distance from particle to ray
                        let to_particle = particle_pos - ray.origin;
                        let projection_length = to_particle.dot(*ray.direction);
                        let closest_point_on_ray = ray.origin + *ray.direction * projection_length;
                        let distance_to_ray = (particle_pos - closest_point_on_ray).length();
                        
                        // If this particle is closer to the ray and within reasonable distance
                        if distance_to_ray < closest_distance && distance_to_ray < 200.0 {
                            closest_distance = distance_to_ray;
                            best_position = closest_point_on_ray;
                        }
                    }
                    
                    // If we found a good particle, use that position
                    // Otherwise, fall back to projecting to the middle of the spawn region
                    if closest_distance < 200.0 {
                        mouse_interaction_3d.position = best_position;
                    } else {
                        // Fallback: project to Y=150 (middle of spawn region)
                        let interaction_plane_y = 150.0;
                        let t = if ray.direction.y.abs() > 0.001 {
                            (interaction_plane_y - ray.origin.y) / ray.direction.y
                        } else {
                            100.0
                        };
                        mouse_interaction_3d.position = ray.origin + *ray.direction * t;
                    }
                }
            }
        }
    }

    // Update mouse interaction state (disabled when Draw Lake mode is active)
    if !draw_lake_mode.enabled {
    mouse_interaction_3d.active = mouse_buttons.pressed(MouseButton::Left) || 
                                  mouse_buttons.pressed(MouseButton::Right);
    mouse_interaction_3d.repel = mouse_buttons.pressed(MouseButton::Right);
    } else {
        mouse_interaction_3d.active = false;
        mouse_interaction_3d.repel = false;
    }
}

pub fn apply_external_forces_3d(
    time: Res<Time>,
    mouse_interaction_3d: Res<MouseInteraction3D>,
    mut particles: Query<(&Transform, &mut Particle3D)>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    let dt = time.delta_secs();
    
    for (transform, mut particle) in particles.iter_mut() {
        // Apply gravity
        particle.velocity += GRAVITY_VEC3 * dt;
        
        // Apply mouse force if active
        if mouse_interaction_3d.active {
            let direction = mouse_interaction_3d.position - transform.translation;
            let distance = direction.length();
            
            if distance < mouse_interaction_3d.radius {
                let force_direction = if mouse_interaction_3d.repel { -direction } else { direction };
                
                // Use a smoother falloff function for more natural interaction
                let distance_ratio = distance / mouse_interaction_3d.radius;
                let falloff = (1.0 - distance_ratio).powi(2); // Quadratic falloff
                let force_strength = mouse_interaction_3d.strength * falloff;
                
                if distance > 0.001 { // Avoid division by zero
                    let force = force_direction.normalize() * force_strength * dt;
                    particle.velocity += force;
                    
                    // Add some damping to prevent excessive velocities
                    particle.velocity *= 0.98;
                }
            }
        }
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
    params: Res<Fluid3DParams>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }

    let smoothing_radius = params.smoothing_radius;
    let smoothing_radius_squared = smoothing_radius * smoothing_radius;
    let math = FluidMath3D::new(smoothing_radius);
    let target_density = params.target_density;
    let pressure_mult = params.pressure_multiplier;
    let near_pressure_mult = params.near_pressure_multiplier;

    // Cache positions and store entities order
    let mut positions: Vec<Vec3> = Vec::with_capacity(particles_q.iter().count());
    let mut entities: Vec<Entity> = Vec::with_capacity(positions.capacity());

    for (e, t, _) in particles_q.iter() {
        entities.push(e);
        positions.push(t.translation);
    }

    let count = positions.len();
    let mut densities = vec![0.0f32; count];
    let mut near_densities = vec![0.0f32; count];

    for i in 0..count {
        let pos_i = positions[i];
        let mut density = 0.0;
        let mut near_density = 0.0;
        
        // Get neighbors using spatial hash
        let neighbors = spatial_hash.spatial_hash.get_neighbors(pos_i, smoothing_radius);
        
        // Add self-contribution
        density += math.poly6(0.0, smoothing_radius_squared);
        near_density += math.spiky_pow2(0.0, smoothing_radius);
        
        // Add neighbor contributions
        for &neighbor_entity in &neighbors {
            if let Ok((_, transform, _)) = particles_q.get(neighbor_entity) {
                let r_vec = pos_i - transform.translation;
                let r2 = r_vec.length_squared();
                if r2 < smoothing_radius_squared && r2 > 1e-10 {
                    let r = r2.sqrt();
                    density += math.poly6(r2, smoothing_radius_squared);
                    near_density += math.spiky_pow2(r, smoothing_radius);
                }
            }
        }
        
        densities[i] = density;
        near_densities[i] = near_density;
    }

    // Write back density/pressure
    for (idx, entity) in entities.iter().enumerate() {
        if let Ok((_, _, mut part)) = particles_q.get_mut(*entity) {
            let density = densities[idx];
            let near_density = near_densities[idx];
            part.density = density;
            part.near_density = near_density;
            part.pressure = (density - target_density) * pressure_mult;
            part.near_pressure = near_density * near_pressure_mult;
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
                    let near_pressure_term = (p_j.near_pressure + p_j.near_pressure) * 0.5; // symmetric
                    let grad = math.spiky_pow3_derivative(dist, smoothing_radius);
                    let near_grad = math.spiky_pow2_derivative(dist, smoothing_radius);
                    pressure_force -= dir * (pressure_term * grad + near_pressure_term * near_grad);

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
    params: Res<Fluid3DParams>,
    spatial_hash: Res<SpatialHashResource3D>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if sim_dim.get() != &SimulationDimension::Dim3 {
        return;
    }
    let dt = time.delta_secs();
    let collision_damping = params.collision_damping;
    let particle_diameter = PARTICLE_RADIUS * 2.0;
    
    // Cache positions for particle-particle collision detection
    let mut positions: Vec<Vec3> = Vec::new();
    for (transform, _) in particles.iter() {
        positions.push(transform.translation);
    }
    
    for (mut transform, mut particle) in particles.iter_mut() {
        transform.translation += particle.velocity * dt;

        // Boundary collisions with damping and friction
        let mut pos = transform.translation;
        let mut vel = particle.velocity;

        // X-axis
        if pos.x < BOUNDARY_MIN.x + PARTICLE_RADIUS {
            pos.x = BOUNDARY_MIN.x + PARTICLE_RADIUS;
            vel.x = -vel.x * collision_damping;
        } else if pos.x > BOUNDARY_MAX.x - PARTICLE_RADIUS {
            pos.x = BOUNDARY_MAX.x - PARTICLE_RADIUS;
            vel.x = -vel.x * collision_damping;
        }

        // Y-axis
        if pos.y < BOUNDARY_MIN.y + PARTICLE_RADIUS {
            pos.y = BOUNDARY_MIN.y + PARTICLE_RADIUS;
            vel.y = -vel.y * collision_damping;
        } else if pos.y > BOUNDARY_MAX.y - PARTICLE_RADIUS {
            pos.y = BOUNDARY_MAX.y - PARTICLE_RADIUS;
            vel.y = -vel.y * collision_damping;
        }

        // Z-axis
        if pos.z < BOUNDARY_MIN.z + PARTICLE_RADIUS {
            pos.z = BOUNDARY_MIN.z + PARTICLE_RADIUS;
            vel.z = -vel.z * collision_damping;
        } else if pos.z > BOUNDARY_MAX.z - PARTICLE_RADIUS {
            pos.z = BOUNDARY_MAX.z - PARTICLE_RADIUS;
            vel.z = -vel.z * collision_damping;
        }

        // Handle particle-to-particle collisions
        let neighbors = spatial_hash.spatial_hash.get_neighbors(pos, particle_diameter);
        
        for &neighbor_pos in &positions {
            if (neighbor_pos - pos).length() < 0.001 {
                continue; // Skip self (same position)
            }
            
            let offset = pos - neighbor_pos;
            let distance = offset.length();
            
            // Check if particles are overlapping
            if distance > 0.0 && distance < particle_diameter {
                let overlap = particle_diameter - distance;
                let separation_direction = offset.normalize();
                
                // Move this particle away by half the overlap
                let separation = separation_direction * (overlap * 0.5);
                pos += separation;
                
                // Apply collision response to velocity
                let velocity_along_normal = vel.dot(separation_direction);
                
                // Only resolve if particles are moving towards each other
                if velocity_along_normal < 0.0 {
                    let restitution = 0.3; // Bounce factor
                    let impulse = -(1.0 + restitution) * velocity_along_normal;
                    vel += separation_direction * impulse;
                }
            }
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

pub fn update_mouse_indicator_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mouse_interaction_3d: Res<MouseInteraction3D>,
    mut indicator_query: Query<(Entity, &mut Transform), With<MouseIndicator>>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        // Remove indicator if not in 3D mode
        for (entity, _) in indicator_query.iter() {
            commands.entity(entity).despawn();
        }
        return;
    }

    if mouse_interaction_3d.active {
        // Update or create indicator
        if let Ok((_, mut transform)) = indicator_query.single_mut() {
            // Update existing indicator position
            transform.translation = mouse_interaction_3d.position;
        } else {
            // Create new indicator
            let indicator_mesh = meshes.add(
                Sphere::new(mouse_interaction_3d.radius * 0.1) // Small sphere to show interaction point
                    .mesh()
                    .ico(2)
                    .unwrap(),
            );
            
            let indicator_material = materials.add(StandardMaterial {
                base_color: if mouse_interaction_3d.repel { 
                    Color::srgb(1.0, 0.2, 0.2) // Red for repel
                } else { 
                    Color::srgb(0.2, 1.0, 0.2) // Green for attract
                },
                emissive: LinearRgba::rgb(0.5, 0.5, 0.5),
                perceptual_roughness: 0.1,
                metallic: 0.8,
                ..default()
            });

            commands.spawn((
                Mesh3d(indicator_mesh),
                MeshMaterial3d(indicator_material),
                Transform::from_translation(mouse_interaction_3d.position),
                MouseIndicator,
                Marker3D,
            ));
        }
    } else {
        // Remove indicator when not active
        for (entity, _) in indicator_query.iter() {
            commands.entity(entity).despawn();
        }
    }
}

pub fn spawn_duck_at_cursor(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut spawn_duck_ev: EventReader<crate::simulation::SpawnDuck>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<crate::orbit_camera::OrbitCamera>>,
    sim_dim: Res<State<SimulationDimension>>,
    asset_server: Res<AssetServer>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        spawn_duck_ev.clear();
        return;
    }

    for _ in spawn_duck_ev.read() {
        // Get cursor position and convert to world space
        if let Some(window) = windows.iter().next() {
            if let Some(cursor_position) = window.cursor_position() {
                if let Ok((camera, camera_transform)) = camera_q.single() {
                    // Convert screen position to world ray
                    if let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_position) {
                        // Project ray to a reasonable distance (middle of spawn region)
                        let spawn_distance = 150.0; // Distance from camera to spawn duck
                        let spawn_position = ray.origin + *ray.direction * spawn_distance;
                        
                        // Calculate initial velocity based on camera direction
                        let camera_forward = camera_transform.forward();
                        let initial_velocity = camera_forward.as_vec3() * 200.0; // Launch speed
                        
                        // Load rubber duck model
                        spawn_rubber_duck_model(&mut commands, &mut meshes, &mut materials, &asset_server, spawn_position, initial_velocity);
                        
                        info!("Spawned rubber duck at position: {:?} with velocity: {:?}", spawn_position, initial_velocity);
                    }
                }
            }
        }
    }
}

fn spawn_rubber_duck_model(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    asset_server: &Res<AssetServer>,
    position: Vec3,
    velocity: Vec3,
) {
    // Try to load the rubber duck model from assets
    // For now, we'll create a simple duck-like shape using primitives
    // This can be replaced with actual model loading when a GLTF file is available
    
    // Create a yellow material for the duck
    let duck_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.9, 0.0), // Bright yellow
        perceptual_roughness: 0.4,
        metallic: 0.0,
        ..default()
    });
    
    // Create a simple duck shape using a sphere (body) and smaller sphere (head)
    let body_mesh = meshes.add(
        Sphere::new(DUCK_SIZE * 0.4)
            .mesh()
            .ico(3)
            .unwrap()
    );
    
    // Spawn the main duck entity with initial angular velocity (much slower)
    let initial_angular_velocity = Vec3::new(
        (rand::random::<f32>() - 0.5) * 1.0, // Random rotation around X axis
        (rand::random::<f32>() - 0.5) * 1.0, // Random rotation around Y axis
        (rand::random::<f32>() - 0.5) * 1.0, // Random rotation around Z axis
    );
    
    let duck_entity = commands.spawn((
        Transform::from_translation(position),
        RubberDuck {
            velocity,
            angular_velocity: initial_angular_velocity,
            size: DUCK_SIZE,
        },
        Marker3D,
    )).id();
    
    // Add the body as a child
    let body_entity = commands.spawn((
        Mesh3d(body_mesh),
        MeshMaterial3d(duck_material.clone()),
        Transform::from_translation(Vec3::ZERO)
            .with_scale(Vec3::new(1.2, 0.8, 1.0)), // Flatten slightly for duck body
    )).id();
    
    // Add a head
    let head_mesh = meshes.add(
        Sphere::new(DUCK_SIZE * 0.25)
            .mesh()
            .ico(3)
            .unwrap()
    );
    
    let head_entity = commands.spawn((
        Mesh3d(head_mesh),
        MeshMaterial3d(duck_material),
        Transform::from_translation(Vec3::new(0.0, DUCK_SIZE * 0.3, DUCK_SIZE * 0.3)),
    )).id();
    
    // Attach body and head to the main duck entity
    commands.entity(duck_entity).add_children(&[body_entity, head_entity]);
}

pub fn update_duck_physics(
    time: Res<Time>,
    mut ducks: Query<(&mut Transform, &mut RubberDuck)>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    let dt = time.delta_secs();
    
    for (mut transform, mut duck) in ducks.iter_mut() {
        // Apply gravity
        duck.velocity += GRAVITY_VEC3 * dt;
        
        // Update position
        transform.translation += duck.velocity * dt;
        
        // Update rotation based on angular velocity
        let rotation_delta = Quat::from_scaled_axis(duck.angular_velocity * dt);
        transform.rotation = rotation_delta * transform.rotation;
        
        // Add angular damping (air resistance) - stronger damping for smoother motion
        duck.angular_velocity *= 0.95;
        
        // Handle boundary collisions
        let half_size = duck.size * 0.5;
        let mut pos = transform.translation;
        let mut vel = duck.velocity;
        let mut angular_vel = duck.angular_velocity;

        // X-axis boundaries
        if pos.x - half_size < BOUNDARY_MIN.x {
            pos.x = BOUNDARY_MIN.x + half_size;
            vel.x = -vel.x * BOUNDARY_DAMPENING;
            // Add gentle spin when hitting walls
            angular_vel.y += vel.x * 0.02;
            angular_vel.z += vel.x * 0.01;
        } else if pos.x + half_size > BOUNDARY_MAX.x {
            pos.x = BOUNDARY_MAX.x - half_size;
            vel.x = -vel.x * BOUNDARY_DAMPENING;
            // Add gentle spin when hitting walls
            angular_vel.y += vel.x * 0.02;
            angular_vel.z += vel.x * 0.01;
        }

        // Y-axis boundaries
        if pos.y - half_size < BOUNDARY_MIN.y {
            pos.y = BOUNDARY_MIN.y + half_size;
            vel.y = -vel.y * BOUNDARY_DAMPENING;
            // Add gentle tumbling when hitting ground/ceiling
            angular_vel.x += vel.y * 0.02;
            angular_vel.z += vel.y * 0.02;
        } else if pos.y + half_size > BOUNDARY_MAX.y {
            pos.y = BOUNDARY_MAX.y - half_size;
            vel.y = -vel.y * BOUNDARY_DAMPENING;
            // Add gentle tumbling when hitting ground/ceiling
            angular_vel.x += vel.y * 0.02;
            angular_vel.z += vel.y * 0.02;
        }

        // Z-axis boundaries
        if pos.z - half_size < BOUNDARY_MIN.z {
            pos.z = BOUNDARY_MIN.z + half_size;
            vel.z = -vel.z * BOUNDARY_DAMPENING;
            // Add gentle spin when hitting walls
            angular_vel.x += vel.z * 0.02;
            angular_vel.y += vel.z * 0.01;
        } else if pos.z + half_size > BOUNDARY_MAX.z {
            pos.z = BOUNDARY_MAX.z - half_size;
            vel.z = -vel.z * BOUNDARY_DAMPENING;
            // Add gentle spin when hitting walls
            angular_vel.x += vel.z * 0.02;
            angular_vel.y += vel.z * 0.01;
        }

        // Apply velocity-based rotation (gentle tumbling through air)
        let velocity_magnitude = vel.length();
        if velocity_magnitude > 50.0 {
            // Add subtle rotation based on movement direction for realistic tumbling
            let velocity_normalized = vel.normalize();
            angular_vel += velocity_normalized.cross(Vec3::Y) * 0.005 * (velocity_magnitude / 100.0).min(1.0);
        }

        // Clamp angular velocity to prevent excessive spinning
        angular_vel = angular_vel.clamp_length_max(MAX_ANGULAR_VELOCITY);
        
        transform.translation = pos;
        duck.velocity = vel;
        duck.angular_velocity = angular_vel;
    }
}

pub fn handle_particle_duck_collisions(
    mut particles: Query<(&mut Transform, &mut Particle3D), Without<RubberDuck>>,
    mut ducks: Query<(&Transform, &mut RubberDuck), Without<Particle3D>>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    for (mut particle_transform, mut particle) in particles.iter_mut() {
        for (duck_transform, mut duck) in ducks.iter_mut() {
            let particle_pos = particle_transform.translation;
            let duck_pos = duck_transform.translation;
            let half_duck_size = duck.size * 0.5;
            
            // Check if particle is inside or near the duck (using AABB collision)
            let diff = particle_pos - duck_pos;
            let abs_diff = diff.abs();
            
            // Check if particle is within duck bounds + particle radius
            if abs_diff.x < half_duck_size + PARTICLE_RADIUS &&
               abs_diff.y < half_duck_size + PARTICLE_RADIUS &&
               abs_diff.z < half_duck_size + PARTICLE_RADIUS {
                
                // Find the closest face of the duck
                let penetration_x = (half_duck_size + PARTICLE_RADIUS) - abs_diff.x;
                let penetration_y = (half_duck_size + PARTICLE_RADIUS) - abs_diff.y;
                let penetration_z = (half_duck_size + PARTICLE_RADIUS) - abs_diff.z;
                
                // Find the axis with minimum penetration (closest face)
                let min_penetration = penetration_x.min(penetration_y).min(penetration_z);
                
                let mut normal = Vec3::ZERO;
                let mut penetration = 0.0;
                
                if min_penetration == penetration_x {
                    normal.x = if diff.x > 0.0 { 1.0 } else { -1.0 };
                    penetration = penetration_x;
                } else if min_penetration == penetration_y {
                    normal.y = if diff.y > 0.0 { 1.0 } else { -1.0 };
                    penetration = penetration_y;
                } else {
                    normal.z = if diff.z > 0.0 { 1.0 } else { -1.0 };
                    penetration = penetration_z;
                }
                
                // Push particle out of duck
                particle_transform.translation += normal * penetration;
                
                // Apply collision response
                let relative_velocity = particle.velocity - duck.velocity;
                let velocity_along_normal = relative_velocity.dot(normal);
                
                // Only resolve if objects are moving towards each other
                if velocity_along_normal < 0.0 {
                    let restitution = 0.3; // Bounce factor
                    let impulse = -(1.0 + restitution) * velocity_along_normal;
                    
                    // Apply impulse to particle (duck is much heavier, so it doesn't move much)
                    particle.velocity += normal * impulse;
                    
                    // Add some friction
                    let friction = 0.1;
                    let tangent_velocity = relative_velocity - normal * velocity_along_normal;
                    particle.velocity -= tangent_velocity * friction;
                    
                    // Add gentle angular velocity to duck based on collision
                    let collision_point = particle_pos - duck_pos;
                    let impulse_vector = normal * impulse * 0.02; // Much smaller effect
                    let torque = collision_point.cross(impulse_vector);
                    duck.angular_velocity += torque * 0.1; // Apply gentle torque to angular velocity
                }
            }
        }
    }
}

// Helper function to create a deformable plane mesh
fn create_deformable_plane_mesh(size: f32, width_segments: u32, height_segments: u32) -> (Vec<Vec3>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    let half_size = size * 0.5;
    
    // Generate vertices
    for y in 0..=height_segments {
        for x in 0..=width_segments {
            let u = x as f32 / width_segments as f32;
            let v = y as f32 / height_segments as f32;
            
            let pos_x = (u - 0.5) * size;
            let pos_z = (v - 0.5) * size;
            let pos_y = 0.0; // Start flat
            
            vertices.push(Vec3::new(pos_x, pos_y, pos_z));
        }
    }
    
    // Generate indices for triangles
    for y in 0..height_segments {
        for x in 0..width_segments {
            let i0 = y * (width_segments + 1) + x;
            let i1 = i0 + 1;
            let i2 = i0 + (width_segments + 1);
            let i3 = i2 + 1;
            
            // First triangle
            indices.push(i0);
            indices.push(i2);
            indices.push(i1);
            
            // Second triangle
            indices.push(i1);
            indices.push(i2);
            indices.push(i3);
        }
    }
    
    (vertices, indices)
}

// Helper function to create a Bevy mesh from vertices and indices
fn create_mesh_from_vertices(vertices: &[Vec3], indices: &[u32]) -> Mesh {
    use bevy::render::mesh::{Indices, PrimitiveTopology};
    use bevy::render::render_asset::RenderAssetUsages;
    
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    
    // Set positions
    let positions: Vec<[f32; 3]> = vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    
    // Calculate normals (all pointing up initially)
    let normals: Vec<[f32; 3]> = vertices.iter().map(|_| [0.0, 1.0, 0.0]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    
    // Set UVs
    let uvs: Vec<[f32; 2]> = vertices.iter().enumerate().map(|(i, _)| {
        let segments = (vertices.len() as f32).sqrt() as u32;
        let x = (i as u32) % (segments);
        let y = (i as u32) / (segments);
        [x as f32 / (segments - 1) as f32, y as f32 / (segments - 1) as f32]
    }).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    
    // Set indices
    mesh.insert_indices(Indices::U32(indices.to_vec()));
    
    mesh
}

// Resource to track ground deformation timing
#[derive(Resource)]
pub struct GroundDeformationTimer {
    pub timer: Timer,
}

impl Default for GroundDeformationTimer {
    fn default() -> Self {
        Self {
            // Allow deformation every 50ms when holding mouse button (20 times per second)
            timer: Timer::from_seconds(0.05, TimerMode::Repeating),
        }
    }
}

// System to handle ground deformation when Draw Lake mode is active
pub fn handle_ground_deformation(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<crate::orbit_camera::OrbitCamera>>,
    mut ground_query: Query<(&mut DeformableGround, &Mesh3d, &Transform), With<GroundPlane>>,
    mut meshes: ResMut<Assets<Mesh>>,
    draw_lake_mode: Res<crate::simulation::DrawLakeMode>,
    sim_dim: Res<State<SimulationDimension>>,
    mut deformation_timer: ResMut<GroundDeformationTimer>,
    time: Res<Time>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 || !draw_lake_mode.enabled {
        return;
    }
    
    // Update the deformation timer
    deformation_timer.timer.tick(time.delta());
    
    // Deform on left mouse click or when left mouse button is held down
    if !mouse_buttons.pressed(MouseButton::Left) {
        // Reset timer when mouse is released so next click is immediate
        deformation_timer.timer.reset();
        return;
    }
    
    // For initial click, deform immediately. For held clicks, use timer to throttle
    let should_deform = mouse_buttons.just_pressed(MouseButton::Left) || 
                       deformation_timer.timer.finished();
    
    if !should_deform {
        return;
    }
    
    // Reset timer for next deformation when holding
    if mouse_buttons.pressed(MouseButton::Left) && !mouse_buttons.just_pressed(MouseButton::Left) {
        deformation_timer.timer.reset();
    }
    
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(ray) = camera.viewport_to_world(camera_transform, cursor_position) {
                    // Check if ray intersects with ground plane
                    let ground_y = BOUNDARY_MIN.y;
                    
                    if ray.direction.y.abs() > 0.001 {
                        let t = (ground_y - ray.origin.y) / ray.direction.y;
                        if t > 0.0 {
                            let intersection_point = ray.origin + ray.direction * t;
                            
                            // Deform the ground at this point
                            if let Ok((mut deformable_ground, mesh_handle, ground_transform)) = ground_query.single_mut() {
                                deform_ground_at_point(
                                    &mut deformable_ground,
                                    &mut meshes,
                                    mesh_handle,
                                    intersection_point,
                                    ground_transform,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// Function to deform the ground mesh at a specific point
fn deform_ground_at_point(
    deformable_ground: &mut DeformableGround,
    meshes: &mut ResMut<Assets<Mesh>>,
    mesh_handle: &Mesh3d,
    world_point: Vec3,
    ground_transform: &Transform,
) {
    let deformation_radius = 50.0; // Radius of deformation
    let deformation_depth = 20.0; // How deep to make the cavity
    
    // Convert world point to local ground coordinates
    let local_point = ground_transform.compute_matrix().inverse().transform_point3(world_point);
    
    // Deform vertices within radius
    for vertex in &mut deformable_ground.vertices {
        let distance = (Vec3::new(vertex.x, 0.0, vertex.z) - Vec3::new(local_point.x, 0.0, local_point.z)).length();
        
        if distance < deformation_radius {
            // Use a smooth falloff function
            let falloff = 1.0 - (distance / deformation_radius);
            let falloff_smooth = falloff * falloff * (3.0 - 2.0 * falloff); // Smoothstep
            
            // Deform downward
            vertex.y -= deformation_depth * falloff_smooth;
        }
    }
    
    // Update the mesh with new vertices
    if let Some(mesh) = meshes.get_mut(&mesh_handle.0) {
        let positions: Vec<[f32; 3]> = deformable_ground.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        
        // Recalculate normals for proper lighting
        recalculate_normals(mesh, &deformable_ground.vertices, &deformable_ground.indices);
    }
}

// Helper function to recalculate normals after mesh deformation
fn recalculate_normals(mesh: &mut Mesh, vertices: &[Vec3], indices: &[u32]) {
    let mut normals = vec![Vec3::ZERO; vertices.len()];
    
    // Calculate face normals and accumulate vertex normals
    for triangle in indices.chunks(3) {
        let i0 = triangle[0] as usize;
        let i1 = triangle[1] as usize;
        let i2 = triangle[2] as usize;
        
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(edge2).normalize();
        
        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }
    
    // Normalize vertex normals
    for normal in &mut normals {
        *normal = normal.normalize();
    }
    
    let normal_array: Vec<[f32; 3]> = normals.iter().map(|n| [n.x, n.y, n.z]).collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normal_array);
}

fn create_boundary_wireframe(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    // Create wireframe material
    let wireframe_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.3), // Semi-transparent white
        alpha_mode: AlphaMode::Blend,
        unlit: true, // Make it unlit so it's always visible
        cull_mode: None, // Render both sides
        ..default()
    });

    // Define the 8 corners of the cube
    let corners = [
        Vec3::new(BOUNDARY_MIN.x, BOUNDARY_MIN.y, BOUNDARY_MIN.z), // 0: bottom-back-left
        Vec3::new(BOUNDARY_MAX.x, BOUNDARY_MIN.y, BOUNDARY_MIN.z), // 1: bottom-back-right
        Vec3::new(BOUNDARY_MAX.x, BOUNDARY_MIN.y, BOUNDARY_MAX.z), // 2: bottom-front-right
        Vec3::new(BOUNDARY_MIN.x, BOUNDARY_MIN.y, BOUNDARY_MAX.z), // 3: bottom-front-left
        Vec3::new(BOUNDARY_MIN.x, BOUNDARY_MAX.y, BOUNDARY_MIN.z), // 4: top-back-left
        Vec3::new(BOUNDARY_MAX.x, BOUNDARY_MAX.y, BOUNDARY_MIN.z), // 5: top-back-right
        Vec3::new(BOUNDARY_MAX.x, BOUNDARY_MAX.y, BOUNDARY_MAX.z), // 6: top-front-right
        Vec3::new(BOUNDARY_MIN.x, BOUNDARY_MAX.y, BOUNDARY_MAX.z), // 7: top-front-left
    ];

    // Define the 12 edges of the cube (each edge connects two corners)
    let edges = [
        // Bottom face edges
        (0, 1), (1, 2), (2, 3), (3, 0),
        // Top face edges
        (4, 5), (5, 6), (6, 7), (7, 4),
        // Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    // Create a line mesh for each edge
    for (start_idx, end_idx) in edges.iter() {
        let start = corners[*start_idx];
        let end = corners[*end_idx];
        
        // Create a thin cylinder to represent the line
        let direction = end - start;
        let length = direction.length();
        let center = (start + end) * 0.5;
        
        // Create a thin cylinder mesh
        let cylinder_mesh = meshes.add(
            bevy::math::primitives::Cylinder::new(0.5, length) // Very thin radius
                .mesh()
        );
        
        // Calculate rotation to align cylinder with the edge direction
        let up = Vec3::Y;
        let rotation = if direction.normalize().dot(up).abs() > 0.99 {
            // Handle case where direction is parallel to Y axis
            if direction.y > 0.0 {
                Quat::IDENTITY
            } else {
                Quat::from_rotation_z(std::f32::consts::PI)
            }
        } else {
            Quat::from_rotation_arc(up, direction.normalize())
        };
        
        commands.spawn((
            Mesh3d(cylinder_mesh),
            MeshMaterial3d(wireframe_material.clone()),
            Transform::from_translation(center).with_rotation(rotation),
            BoundaryWireframe,
            Marker3D,
        ));
    }
} 