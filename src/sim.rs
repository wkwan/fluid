use crate::camera::{control_orbit_camera, spawn_orbit_camera};
use crate::constants::{
    ANGULAR_DAMPING, BOUNDARY_3D_MAX, BOUNDARY_3D_MIN, BOUNDARY_DAMPENING, DUCK_SIZE, FRICTION,
    GRAVITY_3D, HORIZONTAL_DAMPING, KILL_Y_THRESHOLD_3D, MAX_ANGULAR_VELOCITY, MAX_VELOCITY,
    MOUSE_STRENGTH_HIGH, MOUSE_STRENGTH_LOW, MOUSE_STRENGTH_MEDIUM, PARTICLE_RADIUS, RESTITUTION,
    VELOCITY_DAMPING,
};
use crate::spatial_hash::SpatialHashResource3D;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::Camera3d;
use bevy::prelude::*;
use bevy::math::primitives::Sphere;
use bevy::pbr::MeshMaterial3d;
use bevy::time::{Timer, TimerMode};
use rand;
use serde::{Deserialize, Serialize};

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DrawLakeMode>()
            .init_resource::<Fluid3DParams>()
            .init_resource::<SpawnRegion3D>()
            .init_resource::<SpatialHashResource3D>()
            .init_resource::<MouseInteraction3D>()
            .init_resource::<PresetManager3D>()
            .init_resource::<GroundDeformationTimer>()
            .init_resource::<GpuState>()
            .init_resource::<crate::raymarch::RayMarchingSettings>()
            .add_plugins(crate::raymarch::RayMarchPlugin)
            .add_plugins(crate::screenspace::ScreenSpaceFluidPlugin)
            .add_systems(Startup, setup_simulation)
            .add_systems(Startup, spawn_orbit_camera)
            .add_event::<ResetSim>()
            .add_event::<SpawnDuck>()
            .add_systems(Update, handle_input)
            .add_systems(Update, handle_draw_lake_toggle)
            // ===== 3D Setup =====
            .add_systems(Update, setup_3d_environment)
            .add_systems(Update, spawn_particles_3d)
            .add_systems(Update, update_spatial_hash_on_radius_change_3d)
            // ===== 3D Physics =====
            .add_systems(
                Update,
                (
                    handle_mouse_input_3d,
                    apply_external_forces_3d,
                    predict_positions_3d,
                    update_spatial_hash_3d,
                    calculate_density_3d,
                    double_density_relaxation_3d,
                    recompute_velocities_3d,
                    integrate_positions_3d,
                    recycle_particles_3d,
                )
                    .chain(),
            )
            .add_systems(Update, spawn_duck_at_cursor)
            .add_systems(Update, update_duck_physics)
            .add_systems(Update, handle_particle_duck_collisions)
            .add_systems(Update, update_particle_colors_3d)
            .add_systems(Update, update_mouse_indicator_3d)
            .add_systems(Update, handle_ground_deformation)
            .add_systems(Update, update_fps_display)
            .add_systems(Update, handle_reset_sim)
            // Orbit camera (3D only)
            .add_systems(Update, control_orbit_camera)
            // Preset hotkey
            .add_systems(Update, preset_hotkey_3d)
            // Handle duck spawning with spacebar in 3D mode
            .add_systems(Update, handle_duck_spawning);
    }
}

// Track GPU state and errors
#[derive(Resource, Clone)]
pub struct GpuState {
    pub enabled: bool,
}

impl Default for GpuState {
    fn default() -> Self {
        Self { enabled: true }
    }
}

// Components

// Mark the FPS text for updating
#[derive(Component)]
struct FpsText;

// Systems
fn setup_simulation(mut commands: Commands) {
    // Set up UI
    commands.spawn((
        Text::new("Fluid Simulation (Bevy 0.16)"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            right: Val::Px(20.0),
            ..default()
        },
    ));

    // Performance monitor
    commands.spawn((
        Text::new("FPS: --"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(30.0),
            right: Val::Px(20.0),
            ..default()
        },
        FpsText,
    ));
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_interaction: ResMut<MouseInteraction3D>,
    mut fluid3d_params: ResMut<Fluid3DParams>,
    mut raymarching_settings: ResMut<crate::raymarch::RayMarchingSettings>,
) {
    // Toggle raymarching with Q key (only in 3D mode)
    if keys.just_pressed(KeyCode::KeyQ) {
        raymarching_settings.enabled = !raymarching_settings.enabled;
        info!(
            "Ray marching: {}",
            if raymarching_settings.enabled {
                "enabled"
            } else {
                "disabled"
            }
        );
    }

    // Toggle force strength with number keys
    if keys.just_pressed(KeyCode::Digit1) {
        mouse_interaction.strength = MOUSE_STRENGTH_LOW;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = MOUSE_STRENGTH_MEDIUM;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = MOUSE_STRENGTH_HIGH;
    }

    // Parameter adjustment keys - always available now (no debug UI visibility check)
    // Smoothing radius (Up/Down arrows)
    if keys.pressed(KeyCode::ArrowUp) {
        fluid3d_params.smoothing_radius = (fluid3d_params.smoothing_radius + 0.5).min(100.0);
    }
    if keys.pressed(KeyCode::ArrowDown) {
        fluid3d_params.smoothing_radius = (fluid3d_params.smoothing_radius - 0.5).max(1.0);
    }

    // Pressure multiplier (Left/Right arrows)
    if keys.pressed(KeyCode::ArrowRight) {
        fluid3d_params.pressure_multiplier = (fluid3d_params.pressure_multiplier + 5.0).min(500.0);
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        fluid3d_params.pressure_multiplier = (fluid3d_params.pressure_multiplier - 5.0).max(50.0);
    }

    // Surface tension for 3D mode (T/R keys)
    if keys.pressed(KeyCode::KeyT) {
        fluid3d_params.near_pressure_multiplier =
            (fluid3d_params.near_pressure_multiplier + 0.1).min(10.0);
    }
    if keys.pressed(KeyCode::KeyR) {
        fluid3d_params.near_pressure_multiplier =
            (fluid3d_params.near_pressure_multiplier - 0.1).max(0.0);
    }

    // Collision damping (B/V keys)
    if keys.pressed(KeyCode::KeyB) {
        fluid3d_params.collision_damping = (fluid3d_params.collision_damping + 0.01).min(1.0);
    }
    if keys.pressed(KeyCode::KeyV) {
        fluid3d_params.collision_damping = (fluid3d_params.collision_damping - 0.01).max(0.0);
    }

    // Viscosity (Y/H keys)
    if keys.pressed(KeyCode::KeyY) {
        fluid3d_params.viscosity_strength = (fluid3d_params.viscosity_strength + 0.01).min(0.5);
    }
    if keys.pressed(KeyCode::KeyH) {
        fluid3d_params.viscosity_strength = (fluid3d_params.viscosity_strength - 0.01).max(0.0);
    }

    // Reset to defaults
    if keys.just_pressed(KeyCode::KeyX) {
        *fluid3d_params = Fluid3DParams::default();
        *mouse_interaction = MouseInteraction3D::default();
    }
}

// System to update the FPS display
fn update_fps_display(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
    time: Res<Time>,
) {
    if time.elapsed_secs().fract() < 0.1 {
        // Update at most 10 times per second
        if let Ok(mut text) = query.single_mut() {
            // Get FPS from diagnostics if available
            if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(avg) = fps.average() {
                    *text = Text::new(format!("FPS: {:.1}", avg));
                    return;
                }
            }

            // Fallback - calculate FPS from delta time
            let fps = 1.0 / time.delta_secs();
            *text = Text::new(format!("FPS: {:.1}", fps));
        }
    }
}

#[derive(Event)]
pub struct ResetSim;

fn handle_reset_sim(
    mut ev: EventReader<ResetSim>,
    mut commands: Commands,
    q_particles3d: Query<Entity, With<Particle3D>>,
    q_marker3d: Query<Entity, With<Marker3D>>,
    q_ducks: Query<Entity, With<RubberDuck>>,
    q_orbit: Query<Entity, With<crate::camera::OrbitCamera>>,
    q_cam3d: Query<Entity, With<Camera3d>>,
    world: &World,
) {
    if ev.is_empty() {
        return;
    }
    ev.clear();

    // Safe despawn helper that ensures we don't try to despawn entities that don't exist
    let safe_despawn = |entity: Entity, commands: &mut Commands| {
        // Only attempt to despawn if the entity exists in the world
        if world.get_entity(entity).is_ok() {
            commands.entity(entity).despawn();
        }
    };

    // Log counts for debugging
    let p3d_count = q_particles3d.iter().count();
    let marker_count = q_marker3d.iter().count();
    let duck_count = q_ducks.iter().count();
    info!(
        "Cleaning up: {}x 3D particles, {}x 3D markers, {}x ducks",
        p3d_count, marker_count, duck_count
    );

    // Always clean up all particle types and associated entities to ensure a fresh state.

    for e in q_particles3d.iter() {
        safe_despawn(e, &mut commands);
    }

    for e in q_marker3d.iter() {
        safe_despawn(e, &mut commands);
    }

    for e in q_ducks.iter() {
        safe_despawn(e, &mut commands);
    }

    for e in q_orbit.iter() {
        safe_despawn(e, &mut commands);
    }

    for e in q_cam3d.iter() {
        safe_despawn(e, &mut commands);
    }

    info!("Dimension transition cleanup complete");
}

// 3D particle component
#[derive(Component)]
pub struct Particle3D {
    pub velocity: Vec3,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
    pub previous_position: Vec3, // Add this for prediction-relaxation scheme
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
            strength: 3000.0, // Balanced mouse interaction force
            radius: 150.0,    // Increased from 50.0
        }
    }
}

// Constants for 3D sim (match 2D values where possible)
const GRAVITY_VEC3: Vec3 = Vec3::new(GRAVITY_3D[0], GRAVITY_3D[1], GRAVITY_3D[2]);
const BOUNDARY_MIN: Vec3 = Vec3::new(BOUNDARY_3D_MIN[0], BOUNDARY_3D_MIN[1], BOUNDARY_3D_MIN[2]);
const BOUNDARY_MAX: Vec3 = Vec3::new(BOUNDARY_3D_MAX[0], BOUNDARY_3D_MAX[1], BOUNDARY_3D_MAX[2]);
const KILL_Y_THRESHOLD: f32 = KILL_Y_THRESHOLD_3D;

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
            smoothing_radius: 1.0,          // Default to 1.0 for tighter interactions
            target_density: 40.0,           // Increased from 30.0 for stronger density
            pressure_multiplier: 120.0,     // Increased from 100.0 for better cohesion
            near_pressure_multiplier: 60.0, // Increased from 50.0 for better surface tension
            viscosity_strength: 0.0,        // Same as 2D working value (no viscosity)
            collision_damping: 0.85,        // Keep reasonable collision damping
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
            min: Vec3::new(-20.0, 120.0, -20.0), // Slightly larger area
            max: Vec3::new(20.0, 160.0, 20.0),   // More compact height
            spacing: PARTICLE_RADIUS * 1.4,      // Reduced from 1.55 for denser packing
            active: true,
        }
    }
}

#[derive(Event)]
pub struct SpawnDuck;

// Resource to track Draw Lake mode state
#[derive(Resource, Default)]
pub struct DrawLakeMode {
    pub enabled: bool,
}

#[derive(Clone)]
pub struct Preset3D {
    pub name: String,
    pub params: Fluid3DParams,
}

#[derive(Resource, Default)]
pub struct PresetManager3D {
    pub presets: Vec<Preset3D>,
    pub current: usize,
}

impl PresetManager3D {
    pub fn current_preset(&self) -> Option<&Preset3D> {
        self.presets.get(self.current)
    }

    pub fn next(&mut self) {
        if !self.presets.is_empty() {
            self.current = (self.current + 1) % self.presets.len();
        }
    }
}

/// Helper function to get cursor world ray from orbit camera
fn get_cursor_world_ray(
    windows: &Query<&Window>,
    camera_q: &Query<(&Camera, &GlobalTransform), With<crate::camera::OrbitCamera>>,
) -> Option<(Vec2, Ray3d)> {
    let window = windows.iter().next()?;
    let cursor_position = window.cursor_position()?;
    let (camera, camera_transform) = camera_q.single().ok()?;
    let ray = camera
        .viewport_to_world(camera_transform, cursor_position)
        .ok()?;
    Some((cursor_position, ray))
}

// ======================== SETUP ============================
fn setup_3d_environment(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    _asset_server: Res<AssetServer>,
    query_ground: Query<(), With<GroundPlane>>, // check if ground already exists
) {
    // Add a basic directional light so we can see the spheres
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 2000.0,
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
fn spawn_particles_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    spawn_region: Res<SpawnRegion3D>,
    existing: Query<(), With<Particle3D>>,
) {
    if !spawn_region.active {
        return;
    }

    // Only spawn once (when no particles exist)
    if !existing.is_empty() {
        return;
    }

    // Create shared mesh - but NOT shared material
    let sphere_mesh = meshes.add(Sphere::new(PARTICLE_RADIUS).mesh().ico(2).unwrap());

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
                        previous_position: pos,
                    },
                    Marker3D,
                ));
            }
        }
    }
}

// =================== PHYSICS SYSTEMS =======================

fn handle_mouse_input_3d(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<crate::camera::OrbitCamera>>,
    mut mouse_interaction_3d: ResMut<MouseInteraction3D>,
    particles: Query<&Transform, With<Particle3D>>,
    draw_lake_mode: Res<DrawLakeMode>,
) {
    // Handle mouse interaction
    if let Some((_cursor_position, ray)) = get_cursor_world_ray(&windows, &camera_q) {
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

    // Update mouse interaction state (disabled when Draw Lake mode is active)
    if !draw_lake_mode.enabled {
        mouse_interaction_3d.active =
            mouse_buttons.pressed(MouseButton::Left) || mouse_buttons.pressed(MouseButton::Right);
        mouse_interaction_3d.repel = mouse_buttons.pressed(MouseButton::Right);
    } else {
        mouse_interaction_3d.active = false;
        mouse_interaction_3d.repel = false;
    }
}

fn apply_external_forces_3d(
    time: Res<Time>,
    mouse_interaction_3d: Res<MouseInteraction3D>,
    mut particles: Query<(&Transform, &mut Particle3D)>,
) {
    let dt = time.delta_secs();

    for (transform, mut particle) in particles.iter_mut() {
        // Store previous position for prediction-relaxation
        particle.previous_position = transform.translation;

        // Apply gravity (external forces modify velocity)
        particle.velocity += GRAVITY_VEC3 * dt;

        // Apply mouse force if active
        if mouse_interaction_3d.active {
            let direction = mouse_interaction_3d.position - transform.translation;
            let distance = direction.length();

            if distance < mouse_interaction_3d.radius {
                let force_direction = if mouse_interaction_3d.repel {
                    -direction
                } else {
                    direction
                };

                // Use a smoother falloff function for more natural interaction
                let distance_ratio = distance / mouse_interaction_3d.radius;
                let falloff = (1.0 - distance_ratio).powi(2); // Quadratic falloff
                let force_strength = mouse_interaction_3d.strength * falloff;

                if distance > 0.001 {
                    // Avoid division by zero
                    let force = force_direction.normalize() * force_strength * dt;
                    particle.velocity += force;

                    // Add some damping to prevent excessive velocities
                    particle.velocity *= VELOCITY_DAMPING;
                }
            }
        }
    }
}

fn predict_positions_3d(time: Res<Time>, mut query: Query<(&mut Transform, &Particle3D)>) {
    let dt = time.delta_secs();

    for (mut transform, particle) in query.iter_mut() {
        // Predict position based on current velocity
        let predicted_pos = particle.previous_position + particle.velocity * dt;
        transform.translation = predicted_pos;
    }
}

fn update_spatial_hash_3d(
    mut spatial_hash: ResMut<SpatialHashResource3D>,
    particle_query: Query<(Entity, &Transform), With<Particle3D>>,
) {
    spatial_hash.spatial_hash.clear();

    for (entity, transform) in particle_query.iter() {
        spatial_hash
            .spatial_hash
            .insert(transform.translation, entity);
    }
}

fn calculate_density_3d(
    mut particles_q: Query<(Entity, &Transform, &mut Particle3D)>,
    spatial_hash: Res<SpatialHashResource3D>,
    params: Res<Fluid3DParams>,
) {
    let smoothing_radius = params.smoothing_radius;

    // Cache positions
    let positions: Vec<(Entity, Vec3)> = particles_q
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation))
        .collect();

    let mut position_map = std::collections::HashMap::with_capacity(positions.len());
    for (entity, position) in &positions {
        position_map.insert(*entity, *position);
    }

    // Calculate densities using paper's kernels
    for (_entity_i, transform_i, mut particle) in particles_q.iter_mut() {
        let position_i = transform_i.translation;
        let neighbors = spatial_hash
            .spatial_hash
            .get_neighbors(position_i, smoothing_radius);

        let mut density = 0.0;
        let mut near_density = 0.0;

        for neighbor_entity in neighbors {
            if let Some(&position_j) = position_map.get(&neighbor_entity) {
                let offset = position_i - position_j;
                let distance = offset.length();

                if distance < smoothing_radius {
                    let q = distance / smoothing_radius;

                    // Paper equation 1: density kernel (1-r/h)²
                    density += (1.0 - q) * (1.0 - q);

                    // Paper equation 4: near-density kernel (1-r/h)³
                    near_density += (1.0 - q) * (1.0 - q) * (1.0 - q);
                }
            }
        }

        particle.density = density;
        particle.near_density = near_density;
    }
}

fn double_density_relaxation_3d(
    fluid_params: Res<Fluid3DParams>,
    spatial_hash: Res<SpatialHashResource3D>,
    time: Res<Time>,
    mut particle_query: Query<(Entity, &mut Transform, &mut Particle3D)>,
) {
    let dt = time.delta_secs();
    let dt_squared = dt * dt;
    let smoothing_radius = fluid_params.smoothing_radius;
    let target_density = fluid_params.target_density;
    let k = fluid_params.pressure_multiplier;
    let k_near = fluid_params.near_pressure_multiplier;

    // Cache particle data
    let particle_data: Vec<(Entity, Vec3, f32, f32)> = particle_query
        .iter()
        .map(|(entity, transform, particle)| {
            (
                entity,
                transform.translation,
                particle.density,
                particle.near_density,
            )
        })
        .collect();

    let mut position_map = std::collections::HashMap::with_capacity(particle_data.len());
    let mut density_map = std::collections::HashMap::with_capacity(particle_data.len());

    for &(entity, position, density, near_density) in &particle_data {
        position_map.insert(entity, position);
        density_map.insert(entity, (density, near_density));
    }

    // Calculate position displacements for each particle
    let mut displacements = std::collections::HashMap::with_capacity(particle_data.len());

    for &(entity_i, position_i, density_i, near_density_i) in &particle_data {
        let neighbors = spatial_hash
            .spatial_hash
            .get_neighbors(position_i, smoothing_radius);

        // Calculate pressure and near-pressure (paper equations 2 and 5)
        let pressure_i = k * (density_i - target_density);
        let near_pressure_i = k_near * near_density_i;

        let mut displacement_i = Vec3::ZERO;

        for neighbor_entity in neighbors {
            if neighbor_entity == entity_i {
                continue;
            }

            if let Some(&position_j) = position_map.get(&neighbor_entity) {
                let offset = position_i - position_j;
                let distance = offset.length();

                if distance > 0.0 && distance < smoothing_radius {
                    let direction = offset / distance;
                    let q = distance / smoothing_radius;

                    // Paper equation 6: displacement calculation
                    let mut displacement_magnitude = dt_squared
                        * (pressure_i * (1.0 - q) + near_pressure_i * (1.0 - q) * (1.0 - q));

                    // Add strong repulsion force when particles get too close to prevent overlapping
                    let min_distance = PARTICLE_RADIUS * 2.0; // Minimum separation distance
                    if distance < min_distance {
                        let overlap_factor = (min_distance - distance) / min_distance;
                        let repulsion_force = overlap_factor * overlap_factor * 500.0 * dt_squared;
                        displacement_magnitude += repulsion_force;
                    }

                    let displacement = direction * displacement_magnitude;

                    // Apply half displacement to each particle (action-reaction)
                    displacement_i -= displacement * 0.5;

                    // Store displacement for neighbor
                    *displacements.entry(neighbor_entity).or_insert(Vec3::ZERO) +=
                        displacement * 0.5;
                }
            }
        }

        displacements.insert(entity_i, displacement_i);
    }

    // Apply displacements to particle positions
    for (entity, mut transform, _) in particle_query.iter_mut() {
        if let Some(&displacement) = displacements.get(&entity) {
            let new_pos = transform.translation + displacement;
            transform.translation = new_pos;
        }
    }
}

fn recompute_velocities_3d(time: Res<Time>, mut query: Query<(&Transform, &mut Particle3D)>) {
    let dt = time.delta_secs();

    for (transform, mut particle) in query.iter_mut() {
        let current_position = transform.translation;
        // Velocity = (current_position - previous_position) / dt
        particle.velocity = (current_position - particle.previous_position) / dt;
    }
}

fn integrate_positions_3d(
    time: Res<Time>,
    mut particles: Query<(&mut Transform, &mut Particle3D)>,
    params: Res<Fluid3DParams>,
    spatial_hash: Res<SpatialHashResource3D>,
    ground_query: Query<(&DeformableGround, &Transform), (With<GroundPlane>, Without<Particle3D>)>,
) {
    let dt = time.delta_secs();
    let collision_damping = params.collision_damping;
    let particle_diameter = PARTICLE_RADIUS * 2.0;

    // Cache positions for particle-particle collision detection
    let mut positions: Vec<Vec3> = Vec::new();
    for (transform, _) in particles.iter() {
        positions.push(transform.translation);
    }

    // Get ground data for collision detection
    let ground_data = ground_query.single().ok();

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

        // Y-axis (top boundary only, ground collision handled separately)
        if pos.y > BOUNDARY_MAX.y - PARTICLE_RADIUS {
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

        // Ground collision detection with deformed terrain
        if let Some((deformable_ground, ground_transform)) = ground_data {
            let ground_height =
                sample_ground_height_at_position(pos.x, pos.z, deformable_ground, ground_transform);

            if pos.y < ground_height + PARTICLE_RADIUS {
                pos.y = ground_height + PARTICLE_RADIUS;
                vel.y = -vel.y * collision_damping;
                // Add some horizontal friction when hitting the ground
                vel.x *= HORIZONTAL_DAMPING;
                vel.z *= HORIZONTAL_DAMPING;
            }
        } else {
            // Fallback to flat ground collision if no deformable ground exists
            if pos.y < BOUNDARY_MIN.y + PARTICLE_RADIUS {
                pos.y = BOUNDARY_MIN.y + PARTICLE_RADIUS;
                vel.y = -vel.y * collision_damping;
            }
        }

        // Handle particle-to-particle collisions
        let _neighbors = spatial_hash
            .spatial_hash
            .get_neighbors(pos, particle_diameter);

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
                    let impulse = -(1.0 + RESTITUTION) * velocity_along_normal;
                    vel += separation_direction * impulse;
                }
            }
        }

        transform.translation = pos;
        particle.velocity = vel;
    }
}

fn recycle_particles_3d(
    mut commands: Commands,
    mut particles: Query<(Entity, &Transform, &mut Particle3D)>,
    spawn_region: Res<SpawnRegion3D>,
) {
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
            commands
                .entity(entity)
                .insert(Transform::from_translation(pos));
            particle.velocity = Vec3::ZERO;
            particle.density = 0.0;
            particle.pressure = 0.0;
            particle.near_density = 0.0;
            particle.near_pressure = 0.0;
            particle.previous_position = pos;
        }
    }
}

fn update_particle_colors_3d(
    mut materials: ResMut<Assets<StandardMaterial>>,
    particles: Query<(&Particle3D, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (particle, mat_handle) in particles.iter() {
        // Calculate velocity magnitude and normalize
        let velocity_magnitude = particle.velocity.length();
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
}

fn update_mouse_indicator_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mouse_interaction_3d: Res<MouseInteraction3D>,
    mut indicator_query: Query<(Entity, &mut Transform), With<MouseIndicator>>,
) {
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

fn spawn_duck_at_cursor(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut spawn_duck_ev: EventReader<SpawnDuck>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<crate::camera::OrbitCamera>>,
) {
    for _ in spawn_duck_ev.read() {
        // Get cursor position and convert to world space
        if let Some((_cursor_position, ray)) = get_cursor_world_ray(&windows, &camera_q) {
            // Project ray to a reasonable distance (middle of spawn region)
            let spawn_distance = 150.0; // Distance from camera to spawn duck
            let spawn_position = ray.origin + *ray.direction * spawn_distance;

            // Calculate initial velocity based on camera direction
            let camera_forward = ray.direction.normalize();
            let initial_velocity = camera_forward * 200.0; // Launch speed

            // Create a yellow material for the duck
            let duck_material = materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.9, 0.0), // Bright yellow
                perceptual_roughness: 0.4,
                metallic: 0.0,
                ..default()
            });

            // Create a simple duck shape using a sphere (body) and smaller sphere (head)
            let body_mesh = meshes.add(Sphere::new(DUCK_SIZE * 0.4).mesh().ico(3).unwrap());

            // Spawn the main duck entity with initial angular velocity (much slower)
            let initial_angular_velocity = Vec3::new(
                (rand::random::<f32>() - 0.5) * 1.0, // Random rotation around X axis
                (rand::random::<f32>() - 0.5) * 1.0, // Random rotation around Y axis
                (rand::random::<f32>() - 0.5) * 1.0, // Random rotation around Z axis
            );

            let duck_entity = commands
                .spawn((
                    Transform::from_translation(spawn_position),
                    RubberDuck {
                        velocity: initial_velocity,
                        angular_velocity: initial_angular_velocity,
                        size: DUCK_SIZE,
                    },
                    Marker3D,
                ))
                .id();

            // Add the body as a child
            let body_entity = commands
                .spawn((
                    Mesh3d(body_mesh),
                    MeshMaterial3d(duck_material.clone()),
                    Transform::from_translation(Vec3::ZERO).with_scale(Vec3::new(1.2, 0.8, 1.0)), // Flatten slightly for duck body
                ))
                .id();

            // Add a head
            let head_mesh = meshes.add(Sphere::new(DUCK_SIZE * 0.25).mesh().ico(3).unwrap());

            let head_entity = commands
                .spawn((
                    Mesh3d(head_mesh),
                    MeshMaterial3d(duck_material),
                    Transform::from_translation(Vec3::new(0.0, DUCK_SIZE * 0.3, DUCK_SIZE * 0.3)),
                ))
                .id();

            // Attach body and head to the main duck entity
            commands
                .entity(duck_entity)
                .add_children(&[body_entity, head_entity]);

            info!(
                "Spawned rubber duck at position: {:?} with velocity: {:?}",
                spawn_position, initial_velocity
            );
        }
    }
}

fn update_duck_physics(time: Res<Time>, mut ducks: Query<(&mut Transform, &mut RubberDuck)>) {
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
        duck.angular_velocity *= ANGULAR_DAMPING;

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
            angular_vel +=
                velocity_normalized.cross(Vec3::Y) * 0.005 * (velocity_magnitude / 100.0).min(1.0);
        }

        // Clamp angular velocity to prevent excessive spinning
        angular_vel = angular_vel.clamp_length_max(MAX_ANGULAR_VELOCITY);

        transform.translation = pos;
        duck.velocity = vel;
        duck.angular_velocity = angular_vel;
    }
}

fn handle_particle_duck_collisions(
    mut particles: Query<(&mut Transform, &mut Particle3D), Without<RubberDuck>>,
    mut ducks: Query<(&Transform, &mut RubberDuck), Without<Particle3D>>,
) {
    for (mut particle_transform, mut particle) in particles.iter_mut() {
        for (duck_transform, mut duck) in ducks.iter_mut() {
            let particle_pos = particle_transform.translation;
            let duck_pos = duck_transform.translation;
            let half_duck_size = duck.size * 0.5;

            // Check if particle is inside or near the duck (using AABB collision)
            let diff = particle_pos - duck_pos;
            let abs_diff = diff.abs();

            // Check if particle is within duck bounds + particle radius
            if abs_diff.x < half_duck_size + PARTICLE_RADIUS
                && abs_diff.y < half_duck_size + PARTICLE_RADIUS
                && abs_diff.z < half_duck_size + PARTICLE_RADIUS
            {
                // Find the closest face of the duck
                let penetration_x = (half_duck_size + PARTICLE_RADIUS) - abs_diff.x;
                let penetration_y = (half_duck_size + PARTICLE_RADIUS) - abs_diff.y;
                let penetration_z = (half_duck_size + PARTICLE_RADIUS) - abs_diff.z;

                // Find the axis with minimum penetration (closest face)
                let min_penetration = penetration_x.min(penetration_y).min(penetration_z);

                let mut normal = Vec3::ZERO;
                let penetration;

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
                    let impulse = -(1.0 + RESTITUTION) * velocity_along_normal;

                    // Apply impulse to particle (duck is much heavier, so it doesn't move much)
                    particle.velocity += normal * impulse;

                    // Add some friction
                    let tangent_velocity = relative_velocity - normal * velocity_along_normal;
                    particle.velocity -= tangent_velocity * FRICTION;

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
fn create_deformable_plane_mesh(
    size: f32,
    width_segments: u32,
    height_segments: u32,
) -> (Vec<Vec3>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

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
    let uvs: Vec<[f32; 2]> = vertices
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let segments = (vertices.len() as f32).sqrt() as u32;
            let x = (i as u32) % (segments);
            let y = (i as u32) / (segments);
            [
                x as f32 / (segments - 1) as f32,
                y as f32 / (segments - 1) as f32,
            ]
        })
        .collect();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    // Set indices
    mesh.insert_indices(Indices::U32(indices.to_vec()));

    mesh
}

// Resource to track ground deformation timing and last position
#[derive(Resource)]
pub struct GroundDeformationTimer {
    pub timer: Timer,
    pub last_position: Option<Vec3>,
    pub last_deform_position: Option<Vec3>,
}

impl Default for GroundDeformationTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.05, TimerMode::Repeating),
            last_position: None,
            last_deform_position: None,
        }
    }
}

// System to handle ground deformation when terrain doodling mode is active
fn handle_ground_deformation(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<crate::camera::OrbitCamera>>,
    mut ground_query: Query<(&mut DeformableGround, &Mesh3d, &Transform), With<GroundPlane>>,
    mut meshes: ResMut<Assets<Mesh>>,
    draw_lake_mode: Res<DrawLakeMode>,
    mut deformation_timer: ResMut<GroundDeformationTimer>,
    time: Res<Time>,
) {
    if !draw_lake_mode.enabled {
        return;
    }

    // Check for either left or right mouse button
    let left_pressed = mouse_buttons.pressed(MouseButton::Left);
    let right_pressed = mouse_buttons.pressed(MouseButton::Right);

    if !left_pressed && !right_pressed {
        // Reset tracking when mouse is released
        deformation_timer.last_position = None;
        deformation_timer.last_deform_position = None;
        return;
    }

    // Determine deformation direction: left = emboss (down), right = extrude (up)
    let deform_up = right_pressed;

    // Get current mouse intersection point
    let current_intersection =
        if let Some((_cursor_position, ray)) = get_cursor_world_ray(&windows, &camera_q) {
            let ground_y = BOUNDARY_MIN.y;
            if ray.direction.y.abs() > 0.001 {
                let t = (ground_y - ray.origin.y) / ray.direction.y;
                if t > 0.0 {
                    Some(ray.origin + ray.direction * t)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

    // Update last position and check if we should deform
    if let Some(current_pos) = current_intersection {
        // Only tick timer when mouse is held down
        if left_pressed || right_pressed {
            deformation_timer.timer.tick(time.delta());
        }

        let should_deform = mouse_buttons.just_pressed(MouseButton::Left)
            || mouse_buttons.just_pressed(MouseButton::Right)
            || deformation_timer.timer.just_finished();

        if should_deform {
            // Get the last deformation position
            let last_pos = deformation_timer
                .last_deform_position
                .unwrap_or(current_pos);

            // Calculate intermediate points for smooth deformation
            let distance = (current_pos - last_pos).length();
            let num_steps = (distance / 5.0).ceil() as i32; // One deformation every 5 units

            if num_steps > 1 {
                // Interpolate between last and current position
                for i in 0..=num_steps {
                    let t = i as f32 / num_steps as f32;
                    let interpolated_pos = last_pos.lerp(current_pos, t);

                    // Deform the ground at interpolated point
                    if let Ok((mut deformable_ground, mesh_handle, ground_transform)) =
                        ground_query.single_mut()
                    {
                        deform_ground_at_point(
                            &mut deformable_ground,
                            &mut meshes,
                            mesh_handle,
                            interpolated_pos,
                            ground_transform,
                            deform_up,
                        );
                    }
                }
            } else {
                // Single deformation at current position
                if let Ok((mut deformable_ground, mesh_handle, ground_transform)) =
                    ground_query.single_mut()
                {
                    deform_ground_at_point(
                        &mut deformable_ground,
                        &mut meshes,
                        mesh_handle,
                        current_pos,
                        ground_transform,
                        deform_up,
                    );
                }
            }

            // Update last deformation position
            deformation_timer.last_deform_position = Some(current_pos);
        }

        // Always update last position
        deformation_timer.last_position = Some(current_pos);
    }
}

// Function to deform the ground mesh at a specific point
fn deform_ground_at_point(
    deformable_ground: &mut DeformableGround,
    meshes: &mut ResMut<Assets<Mesh>>,
    mesh_handle: &Mesh3d,
    world_point: Vec3,
    ground_transform: &Transform,
    deform_up: bool,
) {
    let deformation_radius = 12.5; // Reduced from 50.0 to quarter size
    let deformation_depth = 5.0; // Reduced from 20.0 to quarter size

    // Convert world point to local ground coordinates
    let local_point = ground_transform
        .compute_matrix()
        .inverse()
        .transform_point3(world_point);

    // Deform vertices within radius
    for vertex in &mut deformable_ground.vertices {
        let distance = (Vec3::new(vertex.x, 0.0, vertex.z)
            - Vec3::new(local_point.x, 0.0, local_point.z))
        .length();

        if distance < deformation_radius {
            // Use a smooth falloff function
            let falloff = 1.0 - (distance / deformation_radius);
            let falloff_smooth = falloff * falloff * (3.0 - 2.0 * falloff); // Smoothstep

            // Deform up or down based on mouse button
            if deform_up {
                vertex.y += deformation_depth * falloff_smooth; // Right-click: extrude upward
            } else {
                vertex.y -= deformation_depth * falloff_smooth; // Left-click: emboss downward
            }
        }
    }

    // Update the mesh with new vertices
    if let Some(mesh) = meshes.get_mut(&mesh_handle.0) {
        let positions: Vec<[f32; 3]> = deformable_ground
            .vertices
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

        // Recalculate normals for proper lighting
        recalculate_normals(
            mesh,
            &deformable_ground.vertices,
            &deformable_ground.indices,
        );
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

// Function to sample the ground height at a given world XZ position
fn sample_ground_height_at_position(
    world_x: f32,
    world_z: f32,
    deformable_ground: &DeformableGround,
    ground_transform: &Transform,
) -> f32 {
    // Convert world position to local ground coordinates
    let world_pos = Vec3::new(world_x, 0.0, world_z);
    let local_pos = ground_transform
        .compute_matrix()
        .inverse()
        .transform_point3(world_pos);

    let half_size = deformable_ground.size * 0.5;

    // Check if position is outside the ground mesh bounds
    if local_pos.x < -half_size
        || local_pos.x > half_size
        || local_pos.z < -half_size
        || local_pos.z > half_size
    {
        // Return the base ground level if outside bounds
        return ground_transform.translation.y;
    }

    // Convert local position to grid coordinates
    let u = (local_pos.x + half_size) / deformable_ground.size;
    let v = (local_pos.z + half_size) / deformable_ground.size;

    // Clamp to valid range
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    // Convert to vertex indices
    let grid_x = u * deformable_ground.width_segments as f32;
    let grid_z = v * deformable_ground.height_segments as f32;

    // Get the four surrounding vertices for bilinear interpolation
    let x0 = grid_x.floor() as u32;
    let x1 = (x0 + 1).min(deformable_ground.width_segments);
    let z0 = grid_z.floor() as u32;
    let z1 = (z0 + 1).min(deformable_ground.height_segments);

    // Calculate interpolation weights
    let fx = grid_x - x0 as f32;
    let fz = grid_z - z0 as f32;

    // Get vertex indices in the vertex array
    let idx00 = (z0 * (deformable_ground.width_segments + 1) + x0) as usize;
    let idx10 = (z0 * (deformable_ground.width_segments + 1) + x1) as usize;
    let idx01 = (z1 * (deformable_ground.width_segments + 1) + x0) as usize;
    let idx11 = (z1 * (deformable_ground.width_segments + 1) + x1) as usize;

    // Ensure indices are within bounds
    if idx00 >= deformable_ground.vertices.len()
        || idx10 >= deformable_ground.vertices.len()
        || idx01 >= deformable_ground.vertices.len()
        || idx11 >= deformable_ground.vertices.len()
    {
        return ground_transform.translation.y;
    }

    // Get the heights at the four corners
    let h00 = deformable_ground.vertices[idx00].y;
    let h10 = deformable_ground.vertices[idx10].y;
    let h01 = deformable_ground.vertices[idx01].y;
    let h11 = deformable_ground.vertices[idx11].y;

    // Bilinear interpolation
    let h0 = h00 * (1.0 - fx) + h10 * fx;
    let h1 = h01 * (1.0 - fx) + h11 * fx;
    let interpolated_height = h0 * (1.0 - fz) + h1 * fz;

    // Transform back to world space
    ground_transform.translation.y + interpolated_height
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
        unlit: true,     // Make it unlit so it's always visible
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
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        // Top face edges
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        // Vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
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
                .mesh(),
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

fn handle_draw_lake_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut draw_lake_mode: ResMut<DrawLakeMode>,
) {
    if keys.just_pressed(KeyCode::KeyL) {
        draw_lake_mode.enabled = !draw_lake_mode.enabled;
    }
}

fn handle_duck_spawning(
    keys: Res<ButtonInput<KeyCode>>,
    mut spawn_duck_ev: EventWriter<SpawnDuck>,
) {
    if keys.just_pressed(KeyCode::Space) {
        spawn_duck_ev.write(SpawnDuck);
    }
}

// Hotkey to cycle 3D presets (P key)
fn preset_hotkey_3d(
    keys: Res<ButtonInput<KeyCode>>,
    mut preset_mgr: ResMut<PresetManager3D>,
    mut fluid3d_params: ResMut<Fluid3DParams>,
) {
    if keys.just_pressed(KeyCode::KeyP) {
        preset_mgr.next();
        if let Some(p) = preset_mgr.current_preset() {
            *fluid3d_params = p.params.clone();
            info!("Loaded preset: {}", p.name);
        }
    }
}

// System to update spatial hash when smoothing radius changes in 3D
fn update_spatial_hash_on_radius_change_3d(
    fluid3d_params: Res<Fluid3DParams>,
    mut spatial_hash_3d: ResMut<SpatialHashResource3D>,
) {
    // Check if the smoothing radius has changed from the spatial hash's current cell size
    if (fluid3d_params.smoothing_radius - spatial_hash_3d.spatial_hash.cell_size).abs() > 0.1 {
        // Update the spatial hash with the new smoothing radius
        spatial_hash_3d.spatial_hash =
            crate::spatial_hash::SpatialHash3D::new(fluid3d_params.smoothing_radius);
        info!(
            "Updated 3D spatial hash cell size to: {}",
            fluid3d_params.smoothing_radius
        );
    }
}
