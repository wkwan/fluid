use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, DiagnosticsStore};
use crate::math::FluidMath;
use crate::spatial_hash::SpatialHash;
use crate::gpu_fluid::{GpuState, GpuPerformanceStats};

// Define ColorMapParams locally since we removed the utility module
#[derive(Resource, Clone, Copy)]
pub struct ColorMapParams {
    pub min_speed: f32,
    pub max_speed: f32,
    pub use_velocity_color: bool,
}

impl Default for ColorMapParams {
    fn default() -> Self {
        Self {
            min_speed: 0.0,
            max_speed: 500.0,
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
        app.init_resource::<FluidParams>()
           .init_resource::<MouseInteraction>()
           .init_resource::<SpatialHashResource>()
           .init_resource::<DebugUiState>()
           .init_resource::<ColorMapParams>()
           .add_systems(Startup, setup_simulation)
           .add_systems(Update, handle_input)
           .add_systems(Update, apply_external_forces)
           .add_systems(Update, update_spatial_hash)
           .add_systems(Update, calculate_density)
           .add_systems(Update, calculate_pressure_force)
           .add_systems(Update, calculate_viscosity)
           .add_systems(Update, update_positions)
           .add_systems(Update, handle_collisions)
           .add_systems(Update, update_particle_colors)
           .add_systems(Update, update_fps_display)
           .add_systems(Update, handle_debug_ui_toggle)
           .add_systems(Update, track_max_velocity);
    }
}

// Mark the FPS text for updating
#[derive(Component)]
struct FpsText;

// Simple Debug UI State
#[derive(Resource, Default)]
struct DebugUiState {
    visible: bool,
    settings_text: String,
}

// Constants
const GRAVITY: Vec2 = Vec2::new(0.0, -9.81);
const BOUNDARY_DAMPENING: f32 = 0.3;
const PARTICLE_RADIUS: f32 = 5.0;
const REST_DENSITY: f32 = 1000.0;

// Components
#[derive(Component)]
pub struct Particle {
    pub velocity: Vec2,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
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
            smoothing_radius: 35.0,
            target_density: REST_DENSITY,
            pressure_multiplier: 200.0,
            near_pressure_multiplier: 30.0,
            viscosity_strength: 0.1,
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
        Self {
            spatial_hash: SpatialHash::new(35.0),
        }
    }
}

// Marker for settings text
#[derive(Component)]
struct SettingsText;

// Systems
fn setup_simulation(mut commands: Commands) {
    // Set up UI
    commands.spawn((
        Text::new("Fluid Simulation (Bevy 0.16)"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));

    // Performance monitor
    commands.spawn((
        Text::new("FPS: --"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(30.0),
            left: Val::Px(10.0),
            ..default()
        },
        FpsText,
    ));
    
    // Debug settings text (initially hidden)
    commands.spawn((
        Text::new(""),
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(10.0),
            top: Val::Px(10.0),
            padding: UiRect::all(Val::Px(10.0)),
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.7)),
        SettingsText,
    ));
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut fluid_params: ResMut<FluidParams>,
    mut debug_ui_state: ResMut<DebugUiState>,
    gpu_state: Res<GpuState>,
    mut perf_stats: ResMut<GpuPerformanceStats>,
    mut color_params: ResMut<ColorMapParams>,
) {
    // Handle mouse interaction
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
                    mouse_interaction.position = world_position;
                    mouse_interaction.active = mouse_buttons.pressed(MouseButton::Left) || 
                                              mouse_buttons.pressed(MouseButton::Right);
                    mouse_interaction.repel = mouse_buttons.pressed(MouseButton::Right);
                }
            }
        }
    }

    // Toggle force strength with number keys
    if keys.just_pressed(KeyCode::Digit1) {
        mouse_interaction.strength = 1000.0;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = 2000.0;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = 3000.0;
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
    
    // Parameter adjustment keys - only when debug UI is visible
    if debug_ui_state.visible {
        // Smoothing radius
        if keys.pressed(KeyCode::KeyQ) {
            fluid_params.smoothing_radius = (fluid_params.smoothing_radius + 0.5).min(100.0);
        }
        if keys.pressed(KeyCode::KeyA) {
            fluid_params.smoothing_radius = (fluid_params.smoothing_radius - 0.5).max(5.0);
        }
        
        // Pressure multiplier
        if keys.pressed(KeyCode::KeyW) {
            fluid_params.pressure_multiplier = (fluid_params.pressure_multiplier + 5.0).min(500.0);
        }
        if keys.pressed(KeyCode::KeyS) {
            fluid_params.pressure_multiplier = (fluid_params.pressure_multiplier - 5.0).max(50.0);
        }
        
        // Surface tension
        if keys.pressed(KeyCode::KeyE) {
            fluid_params.near_pressure_multiplier = (fluid_params.near_pressure_multiplier + 1.0).min(100.0);
        }
        if keys.pressed(KeyCode::KeyD) {
            fluid_params.near_pressure_multiplier = (fluid_params.near_pressure_multiplier - 1.0).max(5.0);
        }
        
        // Viscosity
        if keys.pressed(KeyCode::KeyR) {
            fluid_params.viscosity_strength = (fluid_params.viscosity_strength + 0.01).min(0.5);
        }
        if keys.pressed(KeyCode::KeyF) {
            fluid_params.viscosity_strength = (fluid_params.viscosity_strength - 0.01).max(0.0);
        }
        
        // Reset to defaults
        if keys.just_pressed(KeyCode::KeyX) {
            *fluid_params = FluidParams::default();
            *mouse_interaction = MouseInteraction::default();
        }
    }
    
    // Update settings text content
    update_settings_text(&mut debug_ui_state, &fluid_params, &mouse_interaction, &gpu_state, &perf_stats, &color_params);
}

// Helper function to update settings text
fn update_settings_text(
    debug_ui_state: &mut DebugUiState, 
    fluid_params: &FluidParams,
    mouse_interaction: &MouseInteraction,
    gpu_state: &GpuState,
    perf_stats: &GpuPerformanceStats,
    color_params: &ColorMapParams,
) {
    debug_ui_state.settings_text = format!(
        "Simulation Parameters (F1 to hide)\n\n\
        [Q/A] Smoothing Radius: {:.1}\n\
        [W/S] Pressure Multiplier: {:.1}\n\
        [E/D] Surface Tension: {:.1}\n\
        [R/F] Viscosity: {:.3}\n\n\
        Target Density: {:.1}\n\
        Mouse Force: {:.1}\n\
        Mouse Radius: {:.1}\n\n\
        [G] GPU Acceleration: {}\n\
        Avg Frame Time: {:.2} ms\n\
        [I] Adaptive Iterations: {}\n\
        [U/O] Iterations: {}/{}\n\
        Max Velocity: {:.1}\n\
        \n\
        [C] Color by Velocity: {}\n\
        [M/N] Min Speed: {:.1}\n\
        [K/J] Max Speed: {:.1}\n\
        {}\n\n\
        [X] Reset to Defaults",
        fluid_params.smoothing_radius,
        fluid_params.pressure_multiplier,
        fluid_params.near_pressure_multiplier,
        fluid_params.viscosity_strength,
        fluid_params.target_density,
        mouse_interaction.strength,
        mouse_interaction.radius,
        if gpu_state.enabled { "Enabled" } else { "Disabled (CPU)" },
        perf_stats.avg_frame_time,
        if perf_stats.adaptive_iterations { "On" } else { "Off" },
        perf_stats.iterations_per_frame, 
        perf_stats.base_iterations,
        perf_stats.max_velocity,
        if color_params.use_velocity_color { "Yes" } else { "No" },
        color_params.min_speed,
        color_params.max_speed,
        if let Some(err) = &gpu_state.last_error {
            format!("GPU Error: {}", err)
        } else {
            String::new()
        }
    );
}

// Toggle debug UI visibility
fn handle_debug_ui_toggle(
    mut debug_ui_state: ResMut<DebugUiState>,
    mut query: Query<(&mut Node, &mut Text), With<SettingsText>>,
    buttons: Res<ButtonInput<KeyCode>>,
) {
    if buttons.just_pressed(KeyCode::F1) {
        debug_ui_state.visible = !debug_ui_state.visible;
        
        if let Ok((mut node, mut text)) = query.single_mut() {
            node.display = if debug_ui_state.visible {
                Display::Flex
            } else {
                Display::None
            };
            
            if debug_ui_state.visible {
                // Update text content when showing
                *text = Text::new(debug_ui_state.settings_text.clone());
            }
        }
    } else if debug_ui_state.visible {
        // Update text content even when not toggling visibility
        if let Ok((_, mut text)) = query.single_mut() {
            *text = Text::new(debug_ui_state.settings_text.clone());
        }
    }
}

fn apply_external_forces(
    time: Res<Time>,
    mouse_interaction: Res<MouseInteraction>,
    _params: Res<FluidParams>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    
    for (transform, mut particle) in particle_query.iter_mut() {
        // Apply gravity
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

fn update_spatial_hash(
    mut spatial_hash: ResMut<SpatialHashResource>,
    particle_query: Query<(Entity, &Transform), With<Particle>>,
) {
    spatial_hash.spatial_hash.clear();
    
    for (entity, transform) in particle_query.iter() {
        spatial_hash.spatial_hash.insert(transform.translation.truncate(), entity);
    }
}

fn calculate_density(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    mut particle_query: Query<(Entity, &Transform, &mut Particle)>,
) {
    let smoothing_radius = fluid_params.smoothing_radius;
    let smoothing_radius_squared = smoothing_radius * smoothing_radius;
    let math = FluidMath::new(smoothing_radius);
    let target_density = fluid_params.target_density;
    let pressure_multiplier = fluid_params.pressure_multiplier;
    let near_pressure_multiplier = fluid_params.near_pressure_multiplier;

    // Pre-compute self-contribution once
    let self_density_contribution = math.poly6(0.0, smoothing_radius_squared);
    
    // Cache entity positions to avoid repeated Transform reads
    let positions: Vec<(Entity, Vec2)> = particle_query
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation.truncate()))
        .collect();
    
    // Create a hashmap for fast position lookups
    let mut position_map = std::collections::HashMap::with_capacity(positions.len());
    for (entity, position) in &positions {
        position_map.insert(*entity, *position);
    }
    
    // Use iter_mut() for bevy's built-in optimizations
    for (entity_a, transform_a, mut particle) in particle_query.iter_mut() {
        let position_a = transform_a.translation.truncate();
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_a, smoothing_radius);
        
        // Initial values with self-contribution
        let mut density = self_density_contribution;
        let mut near_density = 0.0;
        
        // Calculate contributions from neighbors
        for neighbor_entity in neighbors {
            // Skip self comparison
            if neighbor_entity == entity_a {
                continue;
            }
            
            // Get position from our cached map - faster than querying again
            if let Some(&position_b) = position_map.get(&neighbor_entity) {
                let offset = position_b - position_a;
                let distance_squared = offset.length_squared();
                
                if distance_squared < smoothing_radius_squared {
                    // Fast path for density calculation
                    density += math.poly6(distance_squared, smoothing_radius_squared);
                    
                    // Near density for surface tension - only needed if distance > 0
                    if distance_squared > 1e-10 {
                        let distance = distance_squared.sqrt();
                        near_density += math.spiky_pow2(distance, smoothing_radius);
                    }
                }
            }
        }
        
        // Update the particle with calculated densities
        particle.density = density;
        particle.near_density = near_density;
        
        // Calculate pressure from density
        let density_error = density - target_density;
        particle.pressure = density_error * pressure_multiplier;
        particle.near_pressure = near_density * near_pressure_multiplier;
    }
}

fn calculate_pressure_force(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    let smoothing_radius = fluid_params.smoothing_radius;
    let math = FluidMath::new(smoothing_radius);
    
    // Cache positions and pressure values to avoid repeated Transform/Component access
    let particle_data: Vec<(Entity, Vec2, f32, f32)> = particle_query
        .iter()
        .enumerate()
        .map(|(i, (transform, particle))| 
            (Entity::from_raw(i as u32), transform.translation.truncate(), particle.pressure, particle.near_pressure))
        .collect();
    
    // Create lookup tables
    let mut position_map = std::collections::HashMap::with_capacity(particle_data.len());
    let mut pressure_map = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity, position, pressure, near_pressure) in &particle_data {
        position_map.insert(entity, position);
        pressure_map.insert(entity, (pressure, near_pressure));
    }
    
    let mut pressure_forces: Vec<Vec2> = vec![Vec2::ZERO; particle_query.iter().count()];
    
    // Calculate pressure forces between all particles
    for (i, (transform_a, _)) in particle_query.iter().enumerate() {
        let position_a = transform_a.translation.truncate();
        let entity_a = Entity::from_raw(i as u32);
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_a, smoothing_radius);
        
        // Get pressure values from our lookup table
        let (pressure_a, near_pressure_a) = *pressure_map.get(&entity_a).unwrap_or(&(0.0, 0.0));
        
        for neighbor_entity in neighbors {
            // Skip self
            if neighbor_entity == entity_a {
                continue;
            }
            
            // Get neighbor data from our maps - faster than querying
            if let (Some(&position_b), Some(&(pressure_b, near_pressure_b))) = 
                (position_map.get(&neighbor_entity), pressure_map.get(&neighbor_entity)) {
                
                let offset = position_a - position_b;
                let distance_squared = offset.length_squared();
                
                if distance_squared > 0.0 && distance_squared < smoothing_radius * smoothing_radius {
                    let distance = distance_squared.sqrt();
                    let direction = offset / distance;
                    
                    // Pressure force calculation based on both pressure values
                    let shared_pressure = (pressure_a + pressure_b) * 0.5;
                    let shared_near_pressure = (near_pressure_a + near_pressure_b) * 0.5;
                    
                    let pressure_force = direction * 
                        (math.spiky_pow3_derivative(distance, smoothing_radius) * shared_pressure +
                         math.spiky_pow2_derivative(distance, smoothing_radius) * shared_near_pressure);
                    
                    pressure_forces[i] += pressure_force;
                }
            }
        }
    }
    
    // Apply the pressure forces
    for (i, (_, mut particle)) in particle_query.iter_mut().enumerate() {
        particle.velocity += pressure_forces[i] * dt;
    }
}

fn calculate_viscosity(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    let smoothing_radius = fluid_params.smoothing_radius;
    let viscosity_strength = fluid_params.viscosity_strength;
    let math = FluidMath::new(smoothing_radius);
    
    // Cache positions and velocities to avoid repeated Transform/Component access
    let particle_data: Vec<(Entity, Vec2, Vec2)> = particle_query
        .iter()
        .enumerate()
        .map(|(i, (transform, particle))| 
            (Entity::from_raw(i as u32), transform.translation.truncate(), particle.velocity))
        .collect();
    
    // Create lookup tables
    let mut position_map = std::collections::HashMap::with_capacity(particle_data.len());
    let mut velocity_map = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity, position, velocity) in &particle_data {
        position_map.insert(entity, position);
        velocity_map.insert(entity, velocity);
    }
    
    // Process viscosity (velocity damping)
    let mut velocity_changes: Vec<Vec2> = vec![Vec2::ZERO; particle_query.iter().count()];
    
    for (i, (transform_a, _)) in particle_query.iter().enumerate() {
        let position_a = transform_a.translation.truncate();
        let entity_a = Entity::from_raw(i as u32);
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_a, smoothing_radius);
        
        // Get velocity from our lookup table
        let velocity_a = *velocity_map.get(&entity_a).unwrap_or(&Vec2::ZERO);
        
        for neighbor_entity in neighbors {
            // Skip self
            if neighbor_entity == entity_a {
                continue;
            }
            
            // Get neighbor data from our maps - faster than querying
            if let (Some(&position_b), Some(&velocity_b)) = 
                (position_map.get(&neighbor_entity), velocity_map.get(&neighbor_entity)) {
                
                let offset = position_a - position_b;
                let distance_squared = offset.length_squared();
                
                if distance_squared > 0.0 && distance_squared < smoothing_radius * smoothing_radius {
                    let distance = distance_squared.sqrt();
                    
                    // Viscosity is based on the velocity difference
                    let velocity_diff = velocity_b - velocity_a;
                    let influence = math.spiky_pow3(distance, smoothing_radius) * viscosity_strength;
                    velocity_changes[i] += velocity_diff * influence;
                }
            }
        }
    }
    
    // Apply the viscosity forces
    for (i, (_, mut particle)) in particle_query.iter_mut().enumerate() {
        particle.velocity += velocity_changes[i] * dt;
    }
}

fn update_positions(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Particle)>,
) {
    let dt = time.delta_secs();
    
    for (mut transform, particle) in query.iter_mut() {
        transform.translation += Vec3::new(particle.velocity.x, particle.velocity.y, 0.0) * dt;
    }
}

fn handle_collisions(
    fluid_params: Res<FluidParams>,
    mut query: Query<(&mut Transform, &mut Particle)>,
) {
    let min_bounds = fluid_params.boundary_min;
    let max_bounds = fluid_params.boundary_max;
    
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
) {
    if let Ok(mut text) = query.single_mut() {
        if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                *text = Text::new(format!("FPS: {:.1}", value));
            }
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