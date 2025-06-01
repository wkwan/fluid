use crate::constants::{
    MOUSE_STRENGTH_HIGH, MOUSE_STRENGTH_LOW, MOUSE_STRENGTH_MEDIUM,
};
use crate::three_d::camera::{control_orbit_camera, spawn_orbit_camera};
use crate::three_d::simulation::{
    apply_external_forces_3d, calculate_density_3d, double_density_relaxation_3d,
    handle_draw_lake_toggle, handle_duck_spawning, handle_ground_deformation,
    handle_mouse_input_3d, integrate_positions_3d, predict_positions_3d, preset_hotkey_3d,
    recompute_velocities_3d, recycle_particles_3d, update_mouse_indicator_3d,
    update_spatial_hash_3d, update_spatial_hash_on_radius_change_3d, DrawLakeMode, Fluid3DParams,
    GroundDeformationTimer, MouseInteraction3D, PresetManager3D, SpawnDuck, SpawnRegion3D,
};
use crate::three_d::spatial_hash::SpatialHashResource3D;
use crate::two_d::camera::spawn_2d_camera;
use crate::two_d::gpu_fluid::{GpuPerformanceStats, GpuState};
use crate::two_d::simulation::{
    apply_external_forces_paper, apply_viscosity_paper, calculate_density_paper,
    double_density_relaxation, handle_collisions, handle_mouse_input_2d, predict_positions,
    recompute_velocities, track_max_velocity, update_particle_colors, update_spatial_hash,
    ColorMapParams, FluidParams, MouseInteraction, Particle, SpatialHashResource,
};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::Camera3d;
use bevy::prelude::*;
use bevy::prelude::{Reflect, States};
use bevy::time::Timer;
use bevy::time::TimerMode;

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
            .init_resource::<crate::three_d::raymarch::RayMarchingSettings>()
            .add_plugins(crate::three_d::raymarch::RayMarchPlugin)
            .add_plugins(crate::two_d::spawner::SpawnerPlugin)
            .add_plugins(crate::two_d::gpu_fluid::GpuFluidPlugin)
            .add_plugins(crate::three_d::screenspace::ScreenSpaceFluidPlugin)
            .add_systems(Startup, setup_simulation)
            .add_event::<ResetSim>()
            .add_event::<SpawnDuck>()
            .add_systems(Update, handle_input)
            .add_systems(Update, handle_draw_lake_toggle)
            .add_systems(Update, handle_mouse_input_2d)
            // ===== 2D Systems =====
            .add_systems(
                Update,
                (
                    apply_external_forces_paper,
                    predict_positions,
                    update_spatial_hash,
                    calculate_density_paper,
                    double_density_relaxation,
                    apply_viscosity_paper,
                    handle_collisions,
                    recompute_velocities,
                )
                    .chain()
                    .run_if(gpu_disabled)
                    .run_if(in_state(SimulationDimension::Dim2)),
            )
            // ===== 3D Setup =====
            .add_systems(
                Update,
                crate::three_d::simulation::setup_3d_environment
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                crate::three_d::simulation::spawn_particles_3d
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                update_spatial_hash_on_radius_change_3d.run_if(in_state(SimulationDimension::Dim3)),
            )
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
            .add_systems(
                Update,
                crate::three_d::simulation::spawn_duck_at_cursor
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                crate::three_d::simulation::update_duck_physics
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                crate::three_d::simulation::handle_particle_duck_collisions
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                crate::three_d::simulation::update_particle_colors_3d
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                update_mouse_indicator_3d.run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(
                Update,
                handle_ground_deformation.run_if(in_state(SimulationDimension::Dim3)),
            )
            .add_systems(Update, update_particle_colors)
            .add_systems(Update, update_fps_display)
            .add_systems(Update, track_max_velocity)
            .add_systems(Update, handle_reset_sim)
            // Orbit camera (3D only)
            .add_systems(
                Update,
                (spawn_orbit_camera, control_orbit_camera)
                    .run_if(in_state(SimulationDimension::Dim3)),
            )
            // 2D camera (2D only)
            .add_systems(
                Update,
                (spawn_2d_camera,).run_if(in_state(SimulationDimension::Dim2)),
            )
            // Camera cleanup when switching dimensions - run on state transitions
            .add_systems(
                OnEnter(SimulationDimension::Dim2),
                crate::three_d::camera::despawn_orbit_camera,
            )
            .add_systems(
                OnEnter(SimulationDimension::Dim3),
                crate::two_d::camera::despawn_2d_camera,
            )
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

// Cooldown resource to debounce Z dimension toggle
#[derive(Resource, Default)]
struct ToggleCooldown {
    timer: Timer,
}

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
    mut raymarching_settings: ResMut<crate::three_d::raymarch::RayMarchingSettings>,
    time: Res<Time>,
) {
    // Toggle raymarching with Q key (only in 3D mode)
    if *sim_dim.get() == SimulationDimension::Dim3 && keys.just_pressed(KeyCode::KeyQ) {
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
            fluid3d_params.pressure_multiplier =
                (fluid3d_params.pressure_multiplier + 5.0).min(500.0);
        }
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        if *sim_dim.get() == SimulationDimension::Dim2 {
            fluid_params.pressure_multiplier = (fluid_params.pressure_multiplier - 5.0).max(50.0);
        } else {
            fluid3d_params.pressure_multiplier =
                (fluid3d_params.pressure_multiplier - 5.0).max(50.0);
        }
    }

    // Surface tension (T/R keys) - only available in 2D mode
    if *sim_dim.get() == SimulationDimension::Dim2 {
        if keys.pressed(KeyCode::KeyT) {
            fluid_params.near_pressure_multiplier =
                (fluid_params.near_pressure_multiplier + 1.0).min(100.0);
        }
        if keys.pressed(KeyCode::KeyR) {
            fluid_params.near_pressure_multiplier =
                (fluid_params.near_pressure_multiplier - 1.0).max(5.0);
        }
    } else {
        // Surface tension for 3D mode (T/R keys)
        if keys.pressed(KeyCode::KeyT) {
            fluid3d_params.near_pressure_multiplier =
                (fluid3d_params.near_pressure_multiplier + 0.1).min(10.0);
        }
        if keys.pressed(KeyCode::KeyR) {
            fluid3d_params.near_pressure_multiplier =
                (fluid3d_params.near_pressure_multiplier - 0.1).max(0.0);
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

        info!(
            "Starting transition from {:?} to {:?} mode",
            *sim_dim.get(),
            new_dim
        );

        // Trigger cleanup first and wait for it to complete
        reset_ev.write(ResetSim);

        // Schedule the state transition for the next frame
        next_sim_dim.set(new_dim);

        // Use a longer cooldown (1.0s) to ensure transitions complete safely
        toggle_cooldown.timer = Timer::from_seconds(1.0, TimerMode::Once);
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

#[derive(States, Reflect, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
#[reflect(State)]
pub enum SimulationDimension {
    Dim2,
    #[default]
    Dim3,
}

#[derive(Event)]
pub struct ResetSim;

fn handle_reset_sim(
    mut ev: EventReader<ResetSim>,
    mut commands: Commands,
    q_particles2d: Query<Entity, With<Particle>>,
    q_particles3d: Query<Entity, With<crate::three_d::simulation::Particle3D>>,
    q_marker3d: Query<Entity, With<crate::three_d::simulation::Marker3D>>,
    q_ducks: Query<Entity, With<crate::three_d::simulation::RubberDuck>>,
    q_orbit: Query<Entity, With<crate::three_d::camera::OrbitCamera>>,
    q_cam3d: Query<Entity, With<Camera3d>>,
    q_cam2d: Query<Entity, With<crate::two_d::camera::Camera2DMarker>>,
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
    info!(
        "Cleaning up: {}x 2D particles, {}x 3D particles, {}x 3D markers, {}x ducks",
        p2d_count, p3d_count, marker_count, duck_count
    );

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
