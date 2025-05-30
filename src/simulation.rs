use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, DiagnosticsStore};
use crate::spatial_hash::SpatialHash;
use crate::gpu_fluid::{GpuState, GpuPerformanceStats};
use crate::orbit_camera::{spawn_orbit_camera, control_orbit_camera, spawn_2d_camera, despawn_2d_camera};
use bevy::prelude::Camera3d;
use crate::constants::{GRAVITY_2D, BOUNDARY_DAMPENING, PARTICLE_RADIUS, REST_DENSITY, MOUSE_STRENGTH_LOW, MOUSE_STRENGTH_MEDIUM, MOUSE_STRENGTH_HIGH};
use crate::simulation3d::{
    apply_external_forces_3d, predict_positions_3d, calculate_density_3d, double_density_relaxation_3d, 
    recompute_velocities_3d, integrate_positions_3d, update_spatial_hash_3d,
    Fluid3DParams, SpawnRegion3D, recycle_particles_3d, MouseInteraction3D,
    handle_mouse_input_3d, update_mouse_indicator_3d, handle_ground_deformation, GroundDeformationTimer,
};
use crate::spatial_hash3d::SpatialHashResource3D;
use bevy::prelude::{States, Reflect};
use bevy::time::Timer;
use bevy::time::TimerMode;
use crate::presets::{PresetManager3D, load_presets_system};
// 3D simulation systems are referenced via full paths to avoid module ordering issues.

// Define ColorMapParams locally since we removed the utility module
#[derive(Resource, Clone, Copy)]
pub struct ColorMapParams {
    pub min_speed: f32,
    pub max_speed: f32,
    pub use_velocity_color: bool,
}

// Resource to track Draw Lake mode state
#[derive(Resource, Default)]
pub struct DrawLakeMode {
    pub enabled: bool,
}

impl Default for ColorMapParams {
    fn default() -> Self {
        Self {
            min_speed: 0.0,
            max_speed: 300.0,
            use_velocity_color: true,
        }
    }
}

// Function to map velocity to color
fn velocity_to_color(velocity: Vec2, min_speed: f32, max_speed: f32) -> Color {
    // Normalize velocity magnitude to 0-1 range
    let speed = velocity.length();
    let normalized_speed = ((speed - min_speed) / (max_speed - min_speed)).clamp(0.0, 1.0);
    
    // Create color mapping:
    // Blue (0, 0, 1) -> Cyan (0, 1, 1) -> Green (0, 1, 0) -> Yellow (1, 1, 0) -> Red (1, 0, 0)
    let color = if normalized_speed < 0.25 {
        // Blue to Cyan
        let t = normalized_speed * 4.0;
        Color::srgb(0.0, t, 1.0)
    } else if normalized_speed < 0.5 {
        // Cyan to Green
        let t = (normalized_speed - 0.25) * 4.0;
        Color::srgb(0.0, 1.0, 1.0 - t)
    } else if normalized_speed < 0.75 {
        // Green to Yellow
        let t = (normalized_speed - 0.5) * 4.0;
        Color::srgb(t, 1.0, 0.0)
    } else {
        // Yellow to Red
        let t = (normalized_speed - 0.75) * 4.0;
        Color::srgb(1.0, 1.0 - t, 0.0)
    };
    
    color
}

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<SimulationDimension>()
            .init_resource::<FluidParams>()
           .init_resource::<MouseInteraction>()
           .init_resource::<SpatialHashResource>()
           .init_resource::<ColorMapParams>()
           .init_resource::<DrawLakeMode>()
            .init_resource::<Fluid3DParams>()
            .init_resource::<SpawnRegion3D>()
            .init_resource::<SpatialHashResource3D>()
            .init_resource::<MouseInteraction3D>()
            .init_resource::<ToggleCooldown>()
            .init_resource::<PresetManager3D>()
            .init_resource::<GroundDeformationTimer>()
            .init_resource::<crate::marching::RayMarchingSettings>()
            .add_plugins(crate::marching::RayMarchPlugin)
            .add_systems(Startup, load_presets_system)
           .add_systems(Startup, setup_simulation)
            .add_event::<ResetSim>()
            .add_event::<SpawnDuck>()
           .add_systems(Update, handle_input)
           .add_systems(Update, handle_draw_lake_toggle)
           .add_systems(Update, handle_mouse_input_2d)
            // ===== 2D Systems =====
           .add_systems(Update, (
               apply_external_forces_paper,
               predict_positions,
               update_spatial_hash,
               calculate_density_paper,
               double_density_relaxation,
               apply_viscosity_paper,
               handle_collisions,
               recompute_velocities,
           ).chain().run_if(gpu_disabled).run_if(in_state(SimulationDimension::Dim2)))

            // ===== 3D Setup =====
            .add_systems(Update, crate::simulation3d::setup_3d_environment.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, crate::simulation3d::spawn_particles_3d.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, update_spatial_hash_on_radius_change_3d.run_if(in_state(SimulationDimension::Dim3)))

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
                    .chain()
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            // ===== 3D Duck Systems =====
            .add_systems(Update, crate::simulation3d::spawn_duck_at_cursor.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, crate::simulation3d::update_duck_physics.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, crate::simulation3d::handle_particle_duck_collisions.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, crate::simulation3d::update_particle_colors_3d.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, update_mouse_indicator_3d.run_if(in_state(SimulationDimension::Dim3)))
            .add_systems(Update, handle_ground_deformation.run_if(in_state(SimulationDimension::Dim3)))
           .add_systems(Update, update_particle_colors)
           .add_systems(Update, update_fps_display)
            .add_systems(Update, track_max_velocity)
            .add_systems(Update, handle_reset_sim)
            .add_systems(Update, render_free_surface_simple
                .run_if(in_state(SimulationDimension::Dim3))
            )
            // Orbit camera (3D only)
            .add_systems(Update, (
                spawn_orbit_camera,
                control_orbit_camera,
            ).run_if(in_state(SimulationDimension::Dim3)))
            // 2D camera (2D only)
            .add_systems(Update, (
                spawn_2d_camera,
            ).run_if(in_state(SimulationDimension::Dim2)))
            // Camera cleanup when switching dimensions - run on state transitions
            .add_systems(OnEnter(SimulationDimension::Dim2), crate::orbit_camera::despawn_orbit_camera)
            .add_systems(OnEnter(SimulationDimension::Dim3), despawn_2d_camera)
            // Preset hotkey
            .add_systems(Update, preset_hotkey_3d)
            // Handle duck spawning with spacebar in 3D mode
            .add_systems(Update, handle_duck_spawning);

        app.register_type::<SimulationDimension>();
    }
}

// Run condition to skip CPU physics when GPU is enabled
fn gpu_disabled(gpu_state: Res<GpuState>) -> bool {
    !gpu_state.enabled
}

// Components
#[derive(Component)]
pub struct Particle {
    pub velocity: Vec2,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
    pub previous_position: Vec2,  // Add this for prediction-relaxation scheme
}

// Resources
#[derive(Resource, Clone)]
pub struct FluidParams {
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub viscosity_strength: f32,
    pub boundary_min: Vec2,
    pub boundary_max: Vec2,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            smoothing_radius: 10.0,    // Set to 10.0 as this works well with GPU accel off
            target_density: 30.0,      // Reduced from 50.0 to balance with stronger gravity
            pressure_multiplier: 100.0, // Increased from 0.004 for stronger pressure forces
            near_pressure_multiplier: 50.0, // Increased from 0.01 for stronger surface tension
            viscosity_strength: 0.0,   // Start with no viscosity
            boundary_min: Vec2::new(-300.0, -300.0),
            boundary_max: Vec2::new(300.0, 300.0),
        }
    }
}

#[derive(Resource, Clone)]
pub struct MouseInteraction {
    pub position: Vec2,
    pub active: bool,
    pub repel: bool,
    pub strength: f32,
    pub radius: f32,
}

impl Default for MouseInteraction {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            active: false,
            repel: false,
            strength: 1000.0,
            radius: 50.0,
        }
    }
}

#[derive(Resource)]
struct SpatialHashResource {
    spatial_hash: SpatialHash,
}

impl Default for SpatialHashResource {
    fn default() -> Self {
        // Initialize with the default FluidParams smoothing radius
        let default_params = FluidParams::default();
        Self {
            spatial_hash: SpatialHash::new(default_params.smoothing_radius),
        }
    }
}

// Cooldown resource to debounce Z dimension toggle
#[derive(Resource, Default)]
struct ToggleCooldown {
    timer: Timer,
}

// Mark the FPS text for updating
#[derive(Component)]
struct FpsText;

// Constants - now using shared constants from constants module
const GRAVITY: Vec2 = Vec2::new(GRAVITY_2D[0], GRAVITY_2D[1]);

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

// Separate system for handling Draw Lake mode toggle
fn handle_draw_lake_toggle(
    keys: Res<ButtonInput<KeyCode>>,
    mut draw_lake_mode: ResMut<DrawLakeMode>,
) {
    if keys.just_pressed(KeyCode::KeyL) {
        draw_lake_mode.enabled = !draw_lake_mode.enabled;
    }
}

// Separate system for mouse input handling
fn handle_mouse_input_2d(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    draw_lake_mode: Res<DrawLakeMode>,
) {
    // Handle mouse interaction (disabled when Draw Lake mode is active)
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
                    mouse_interaction.position = world_position;
                    // Disable mouse forces when Draw Lake mode is active
                    if !draw_lake_mode.enabled {
                        mouse_interaction.active = mouse_buttons.pressed(MouseButton::Left) || 
                                                  mouse_buttons.pressed(MouseButton::Right);
                        mouse_interaction.repel = mouse_buttons.pressed(MouseButton::Right);
                    } else {
                        mouse_interaction.active = false;
                        mouse_interaction.repel = false;
                    }
                }
            }
        }
    }
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    mut fluid_params: ResMut<FluidParams>,
    mut gpu_state: ResMut<GpuState>,
    mut perf_stats: ResMut<GpuPerformanceStats>,
    mut color_params: ResMut<ColorMapParams>,
    sim_dim: Res<State<SimulationDimension>>,
    mut next_sim_dim: ResMut<NextState<SimulationDimension>>,
    mut reset_ev: EventWriter<ResetSim>,
    mut fluid3d_params: ResMut<Fluid3DParams>,
    mut toggle_cooldown: ResMut<ToggleCooldown>,
    mut raymarching_settings: ResMut<crate::marching::RayMarchingSettings>,
    time: Res<Time>,
) {

    // Toggle raymarching with Q key (only in 3D mode)
    if *sim_dim.get() == SimulationDimension::Dim3 && keys.just_pressed(KeyCode::KeyQ) {
        raymarching_settings.enabled = !raymarching_settings.enabled;
        info!("Ray marching: {}", if raymarching_settings.enabled { "enabled" } else { "disabled" });
    }

    // Toggle force strength with number keys
    if keys.just_pressed(KeyCode::Digit1) {
        mouse_interaction.strength = MOUSE_STRENGTH_LOW;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = MOUSE_STRENGTH_MEDIUM;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = MOUSE_STRENGTH_HIGH;
    }
    
    // Toggle color mode with C key
    if keys.just_pressed(KeyCode::KeyC) {
        color_params.use_velocity_color = !color_params.use_velocity_color;
    }
    
    // Adjust color map min/max speed
    if keys.pressed(KeyCode::KeyM) {
        color_params.min_speed += 5.0;
    }
    if keys.pressed(KeyCode::KeyN) {
        color_params.min_speed = (color_params.min_speed - 5.0).max(0.0);
    }
    if keys.pressed(KeyCode::KeyK) {
        color_params.max_speed += 5.0;
    }
    if keys.pressed(KeyCode::KeyJ) {
        color_params.max_speed = (color_params.max_speed - 5.0).max(color_params.min_speed + 1.0);
    }
    
    // Toggle adaptive iterations with I key
    if keys.just_pressed(KeyCode::KeyI) {
        perf_stats.adaptive_iterations = !perf_stats.adaptive_iterations;
    }
    
    // Adjust base iterations with U/O keys
    if keys.just_pressed(KeyCode::KeyU) {
        perf_stats.base_iterations = (perf_stats.base_iterations - 1).max(1);
    }
    if keys.just_pressed(KeyCode::KeyO) {
        perf_stats.base_iterations = (perf_stats.base_iterations + 1).min(8);
    }
    
    // Parameter adjustment keys - always available now (no debug UI visibility check)
        // Smoothing radius (Up/Down arrows)
        if keys.pressed(KeyCode::ArrowUp) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
                fluid_params.smoothing_radius = (fluid_params.smoothing_radius + 0.5).min(100.0);
            } else {
                fluid3d_params.smoothing_radius = (fluid3d_params.smoothing_radius + 0.5).min(100.0);
            }
        }
        if keys.pressed(KeyCode::ArrowDown) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
                fluid_params.smoothing_radius = (fluid_params.smoothing_radius - 0.5).max(5.0);
            } else {
                fluid3d_params.smoothing_radius = (fluid3d_params.smoothing_radius - 0.5).max(1.0);
            }
        }
        
        // Pressure multiplier (Left/Right arrows)
        if keys.pressed(KeyCode::ArrowRight) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
            fluid_params.pressure_multiplier = (fluid_params.pressure_multiplier + 5.0).min(500.0);
            } else {
                fluid3d_params.pressure_multiplier = (fluid3d_params.pressure_multiplier + 5.0).min(500.0);
            }
        }
        if keys.pressed(KeyCode::ArrowLeft) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
            fluid_params.pressure_multiplier = (fluid_params.pressure_multiplier - 5.0).max(50.0);
            } else {
                fluid3d_params.pressure_multiplier = (fluid3d_params.pressure_multiplier - 5.0).max(50.0);
            }
        }
        
        // Surface tension (T/R keys) - only available in 2D mode
        if *sim_dim.get() == SimulationDimension::Dim2 {
            if keys.pressed(KeyCode::KeyT) {
                fluid_params.near_pressure_multiplier = (fluid_params.near_pressure_multiplier + 1.0).min(100.0);
            }
            if keys.pressed(KeyCode::KeyR) {
                fluid_params.near_pressure_multiplier = (fluid_params.near_pressure_multiplier - 1.0).max(5.0);
            }
        } else {
            // Surface tension for 3D mode (T/R keys)
            if keys.pressed(KeyCode::KeyT) {
                fluid3d_params.near_pressure_multiplier = (fluid3d_params.near_pressure_multiplier + 0.1).min(10.0);
            }
            if keys.pressed(KeyCode::KeyR) {
                fluid3d_params.near_pressure_multiplier = (fluid3d_params.near_pressure_multiplier - 0.1).max(0.0);
            }
        }
        
        // Collision damping (B/V keys) - only available in 3D mode
        if *sim_dim.get() == SimulationDimension::Dim3 {
            if keys.pressed(KeyCode::KeyB) {
                fluid3d_params.collision_damping = (fluid3d_params.collision_damping + 0.01).min(1.0);
            }
            if keys.pressed(KeyCode::KeyV) {
                fluid3d_params.collision_damping = (fluid3d_params.collision_damping - 0.01).max(0.0);
            }
        }
        
        // Viscosity (Y/H keys)
        if keys.pressed(KeyCode::KeyY) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
            fluid_params.viscosity_strength = (fluid_params.viscosity_strength + 0.01).min(0.5);
            } else {
                fluid3d_params.viscosity_strength = (fluid3d_params.viscosity_strength + 0.01).min(0.5);
            }
        }
        if keys.pressed(KeyCode::KeyH) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
            fluid_params.viscosity_strength = (fluid_params.viscosity_strength - 0.01).max(0.0);
            } else {
                fluid3d_params.viscosity_strength = (fluid3d_params.viscosity_strength - 0.01).max(0.0);
            }
        }
        
        // Reset to defaults
        if keys.just_pressed(KeyCode::KeyX) {
            if *sim_dim.get() == SimulationDimension::Dim2 {
            *fluid_params = FluidParams::default();
            } else {
                *fluid3d_params = Fluid3DParams::default();
            }
            *mouse_interaction = MouseInteraction::default();
    }
    
    // tick cooldown
    toggle_cooldown.timer.tick(time.delta());

    if keys.just_pressed(KeyCode::KeyZ) && toggle_cooldown.timer.finished() {
        let new_dim = match *sim_dim.get() {
            SimulationDimension::Dim2 => SimulationDimension::Dim3,
            SimulationDimension::Dim3 => SimulationDimension::Dim2,
        };

        // Set GPU accel based on new dimension
        match new_dim {
            SimulationDimension::Dim2 => gpu_state.enabled = false,
            SimulationDimension::Dim3 => gpu_state.enabled = true,
        }

        info!("Starting transition from {:?} to {:?} mode", *sim_dim.get(), new_dim);
        
        // Trigger cleanup first and wait for it to complete
        reset_ev.write(ResetSim);

        // Schedule the state transition for the next frame
        next_sim_dim.set(new_dim);

        // Use a longer cooldown (1.0s) to ensure transitions complete safely
        toggle_cooldown.timer = Timer::from_seconds(1.0, TimerMode::Once);
    }
}

fn apply_external_forces_paper(
    time: Res<Time>,
    mouse_interaction: Res<MouseInteraction>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    
    for (transform, mut particle) in particle_query.iter_mut() {
        // Store previous position for prediction-relaxation
        particle.previous_position = transform.translation.truncate();
        
        // Apply gravity (external forces modify velocity)
        particle.velocity += GRAVITY * dt;
        
        // Apply mouse force if active
        if mouse_interaction.active {
            let direction = mouse_interaction.position - transform.translation.truncate();
            let distance = direction.length();
            
            if distance < mouse_interaction.radius {
                let force_direction = if mouse_interaction.repel { -direction } else { direction };
                let force_strength = mouse_interaction.strength * (1.0 - distance / mouse_interaction.radius);
                particle.velocity += force_direction.normalize() * force_strength * dt;
            }
        }
    }
}

fn predict_positions(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Particle)>,
) {
    let dt = time.delta_secs();
    
    for (mut transform, particle) in query.iter_mut() {
        // Predict position based on current velocity
        let predicted_pos = particle.previous_position + particle.velocity * dt;
        transform.translation = Vec3::new(predicted_pos.x, predicted_pos.y, 0.0);
    }
}

fn double_density_relaxation(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(Entity, &mut Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    let dt_squared = dt * dt;
    let smoothing_radius = fluid_params.smoothing_radius;
    let target_density = fluid_params.target_density;
    let k = fluid_params.pressure_multiplier;
    let k_near = fluid_params.near_pressure_multiplier;
    
    // Cache particle data
    let particle_data: Vec<(Entity, Vec2, f32, f32)> = particle_query
        .iter()
        .map(|(entity, transform, particle)| 
            (entity, transform.translation.truncate(), particle.density, particle.near_density))
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
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_i, smoothing_radius);
        
        // Calculate pressure and near-pressure (paper equations 2 and 5)
        let pressure_i = k * (density_i - target_density);
        let near_pressure_i = k_near * near_density_i;
        
        let mut displacement_i = Vec2::ZERO;
        
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
                    let mut displacement_magnitude = dt_squared * (
                        pressure_i * (1.0 - q) +
                        near_pressure_i * (1.0 - q) * (1.0 - q)
                    );
                    
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
                    *displacements.entry(neighbor_entity).or_insert(Vec2::ZERO) += displacement * 0.5;
                }
            }
        }
        
        displacements.insert(entity_i, displacement_i);
    }
    
    // Apply displacements to particle positions
    for (entity, mut transform, _) in particle_query.iter_mut() {
        if let Some(&displacement) = displacements.get(&entity) {
            let new_pos = transform.translation.truncate() + displacement;
            transform.translation = Vec3::new(new_pos.x, new_pos.y, 0.0);
        }
    }
}

fn calculate_density_paper(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    mut particle_query: Query<(Entity, &Transform, &mut Particle)>,
) {
    let smoothing_radius = fluid_params.smoothing_radius;
    
    // Cache positions
    let positions: Vec<(Entity, Vec2)> = particle_query
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation.truncate()))
        .collect();
    
    let mut position_map = std::collections::HashMap::with_capacity(positions.len());
    for (entity, position) in &positions {
        position_map.insert(*entity, *position);
    }
    
    // Calculate densities using paper's kernels
    for (_entity_i, transform_i, mut particle) in particle_query.iter_mut() {
        let position_i = transform_i.translation.truncate();
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_i, smoothing_radius);
        
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

fn recompute_velocities(
    time: Res<Time>,
    mut query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    
    for (transform, mut particle) in query.iter_mut() {
        let current_position = transform.translation.truncate();
        // Velocity = (current_position - previous_position) / dt
        particle.velocity = (current_position - particle.previous_position) / dt;
    }
}

fn apply_viscosity_paper(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(Entity, &Transform, &mut Particle)>,
) {
    if fluid_params.viscosity_strength <= 0.0 {
        return;
    }
    
    let dt = time.delta_secs();
    let smoothing_radius = fluid_params.smoothing_radius;
    let sigma = fluid_params.viscosity_strength;
    let beta = fluid_params.viscosity_strength * 0.1; // Quadratic term
    
    // Cache data
    let particle_data: Vec<(Entity, Vec2, Vec2)> = particle_query
        .iter()
        .map(|(entity, transform, particle)| 
            (entity, transform.translation.truncate(), particle.velocity))
        .collect();
    
    let mut position_map = std::collections::HashMap::with_capacity(particle_data.len());
    let mut velocity_map = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity, position, velocity) in &particle_data {
        position_map.insert(entity, position);
        velocity_map.insert(entity, velocity);
    }
    
    let mut velocity_changes = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity_i, position_i, velocity_i) in &particle_data {
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_i, smoothing_radius);
        
        for neighbor_entity in neighbors {
            if neighbor_entity == entity_i {
                continue;
            }
            
            if let (Some(&position_j), Some(&velocity_j)) = 
                (position_map.get(&neighbor_entity), velocity_map.get(&neighbor_entity)) {
                
                let offset = position_i - position_j;
                let distance = offset.length();
                
                if distance > 0.0 && distance < smoothing_radius {
                    let direction = offset / distance;
                    let q = distance / smoothing_radius;
                    
                    // Inward radial velocity
                    let u = (velocity_i - velocity_j).dot(direction);
                    
                    if u > 0.0 {
                        // Paper's viscosity impulse: I = Δt(1-q)(σu + βu²)
                        let impulse_magnitude = dt * (1.0 - q) * (sigma * u + beta * u * u);
                        let impulse = direction * impulse_magnitude;
                        
                        // Apply impulse to both particles
                        *velocity_changes.entry(entity_i).or_insert(Vec2::ZERO) -= impulse * 0.5;
                        *velocity_changes.entry(neighbor_entity).or_insert(Vec2::ZERO) += impulse * 0.5;
                    }
                }
            }
        }
    }
    
    // Apply velocity changes
    for (entity, _, mut particle) in particle_query.iter_mut() {
        if let Some(&velocity_change) = velocity_changes.get(&entity) {
            particle.velocity += velocity_change;
        }
    }
}

fn update_spatial_hash(
    fluid_params: Res<FluidParams>,
    mut spatial_hash: ResMut<SpatialHashResource>,
    particle_query: Query<(Entity, &Transform), With<Particle>>,
) {
    // Check if the smoothing radius has changed and recreate spatial hash if needed
    let current_cell_size = spatial_hash.spatial_hash.cell_size;
    let new_radius = fluid_params.smoothing_radius;
    
    // Use a more reasonable threshold (0.1) and always recreate if there's a significant difference
    if (new_radius - current_cell_size).abs() > 0.1 {
        spatial_hash.spatial_hash = SpatialHash::new(new_radius);
    } else {
        // Just clear the existing hash without recreating
        spatial_hash.spatial_hash.clear();
    }
    
    // Repopulate the spatial hash with current particle positions
    for (entity, transform) in particle_query.iter() {
        spatial_hash.spatial_hash.insert(transform.translation.truncate(), entity);
    }
}

fn handle_collisions(
    fluid_params: Res<FluidParams>,
    mut query: Query<(&mut Transform, &mut Particle)>,
) {
    let min_bounds = fluid_params.boundary_min;
    let max_bounds = fluid_params.boundary_max;
    
    // Only handle boundary collisions, like Unity does
    for (mut transform, mut particle) in query.iter_mut() {
        let pos = &mut transform.translation;
        
        // Handle boundary collisions with damping
        if pos.x < min_bounds.x + PARTICLE_RADIUS {
            pos.x = min_bounds.x + PARTICLE_RADIUS;
            particle.velocity.x = -particle.velocity.x * BOUNDARY_DAMPENING;
        } else if pos.x > max_bounds.x - PARTICLE_RADIUS {
            pos.x = max_bounds.x - PARTICLE_RADIUS;
            particle.velocity.x = -particle.velocity.x * BOUNDARY_DAMPENING;
        }
        
        if pos.y < min_bounds.y + PARTICLE_RADIUS {
            pos.y = min_bounds.y + PARTICLE_RADIUS;
            particle.velocity.y = -particle.velocity.y * BOUNDARY_DAMPENING;
        } else if pos.y > max_bounds.y - PARTICLE_RADIUS {
            pos.y = max_bounds.y - PARTICLE_RADIUS;
            particle.velocity.y = -particle.velocity.y * BOUNDARY_DAMPENING;
        }
    }
}

fn update_particle_colors(
    particles_query: Query<(&Particle, &MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    color_params: Res<ColorMapParams>,
) {
    for (particle, mesh_material) in particles_query.iter() {
        let color = if color_params.use_velocity_color {
            // Use velocity-based color mapping
            velocity_to_color(particle.velocity, color_params.min_speed, color_params.max_speed)
        } else {
            // Fallback to density-based coloring (existing behavior)
            let normalized_density = (particle.density / REST_DENSITY).clamp(0.0, 3.0) / 3.0;
            Color::srgb(
                normalized_density,
                0.5 + normalized_density * 0.5,
                1.0
            )
        };
        
        // Update material color
        if let Some(material) = materials.get_mut(&mesh_material.0) {
            material.color = color;
        }
    }
}

// System to update the FPS display
fn update_fps_display(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<FpsText>>,
    time: Res<Time>,
) {
    if time.elapsed_secs().fract() < 0.1 { // Update at most 10 times per second
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

// Track maximum velocity for performance optimization
fn track_max_velocity(
    mut perf_stats: ResMut<GpuPerformanceStats>,
    particles: Query<&Particle>,
) {
    let mut max_velocity = 0.0;
    
    for particle in particles.iter() {
        let velocity_magnitude = particle.velocity.length();
        if velocity_magnitude > max_velocity {
            max_velocity = velocity_magnitude;
        }
    }
    
    // Update maximum velocity
    perf_stats.max_velocity = max_velocity;
    
    // Adjust iterations based on velocity if adaptive iterations is enabled
    if perf_stats.adaptive_iterations {
        // Scale iterations based on maximum velocity
        // Higher velocities need more iterations for stability
        let base_iterations = perf_stats.base_iterations;
        
        let velocity_scale = if max_velocity > 0.0 {
            let normalized_velocity = (max_velocity / 500.0).clamp(1.0, 3.0);
            perf_stats.velocity_iteration_scale * normalized_velocity
        } else {
            1.0
        };
        
        // Cap min/max iterations based on performance
        if perf_stats.avg_frame_time > 20.0 { // < 50 FPS
            perf_stats.iterations_per_frame = 2; // Force lower iterations for performance
        } else {
            perf_stats.iterations_per_frame = (base_iterations as f32 * velocity_scale).max(1.0) as u32;
        }
    }
} 

#[derive(States, Reflect, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
#[reflect(State)]
pub enum SimulationDimension {
    Dim2,
    #[default]
    Dim3,
}

#[derive(Event)]
pub struct ResetSim;

#[derive(Event)]
pub struct SpawnDuck;

fn handle_reset_sim(
    mut ev: EventReader<ResetSim>,
    mut commands: Commands,
    q_particles2d: Query<Entity, With<Particle>>,
    q_particles3d: Query<Entity, With<crate::simulation3d::Particle3D>>,
    q_marker3d: Query<Entity, With<crate::simulation3d::Marker3D>>,
    q_ducks: Query<Entity, With<crate::simulation3d::RubberDuck>>,
    q_orbit: Query<Entity, With<crate::orbit_camera::OrbitCamera>>,
    q_cam3d: Query<Entity, With<Camera3d>>,
    q_cam2d: Query<Entity, With<crate::orbit_camera::Camera2DMarker>>,
    _sim_dim: Res<State<SimulationDimension>>,
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
    let p2d_count = q_particles2d.iter().count();
    let p3d_count = q_particles3d.iter().count();
    let marker_count = q_marker3d.iter().count();
    let duck_count = q_ducks.iter().count();
    info!("Cleaning up: {}x 2D particles, {}x 3D particles, {}x 3D markers, {}x ducks", 
          p2d_count, p3d_count, marker_count, duck_count);

    // Always clean up all particle types and associated entities to ensure a fresh state.
    for e in q_particles2d.iter() {
        safe_despawn(e, &mut commands);
    }
    
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
    
    for e in q_cam2d.iter() {
        safe_despawn(e, &mut commands);
    }
    
    info!("Dimension transition cleanup complete");
}

// Hotkey to cycle 3D presets (P key)
fn preset_hotkey_3d(
    keys: Res<ButtonInput<KeyCode>>,
    mut preset_mgr: ResMut<PresetManager3D>,
    mut fluid3d_params: ResMut<Fluid3DParams>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    if keys.just_pressed(KeyCode::KeyP) {
        preset_mgr.next();
        if let Some(p) = preset_mgr.current_preset() {
            *fluid3d_params = p.params.clone();
            info!("Loaded preset: {}", p.name);
        }
    }
}

// Handle cube spawning with spacebar in 3D mode
fn handle_duck_spawning(
    keys: Res<ButtonInput<KeyCode>>,
    mut spawn_duck_ev: EventWriter<SpawnDuck>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    if keys.just_pressed(KeyCode::Space) {
        spawn_duck_ev.write(SpawnDuck);
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
        spatial_hash_3d.spatial_hash = crate::spatial_hash3d::SpatialHash3D::new(fluid3d_params.smoothing_radius);
        info!("Updated 3D spatial hash cell size to: {}", fluid3d_params.smoothing_radius);
    }
}

// Simplified system that checks render mode and calls appropriate renderer
fn render_free_surface_simple(
    render_settings: Res<crate::marching::FluidRenderSettings>,
) {
    // The actual rendering is handled by mode-specific systems
    // This just tracks which mode is active for debugging
    if render_settings.show_free_surface {
        match render_settings.render_mode {
            crate::marching::FluidRenderMode::ScreenSpace => {
                // Screen space rendering is handled by render_screen_space_fluid_system
            }
            crate::marching::FluidRenderMode::RayMarching => {
                // Ray marching is handled by existing systems
            }
        }
    }
}