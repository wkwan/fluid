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
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::Camera3d;
use bevy::prelude::*;

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
            .init_resource::<crate::three_d::raymarch::RayMarchingSettings>()
            .add_plugins(crate::three_d::raymarch::RayMarchPlugin)
            .add_plugins(crate::three_d::screenspace::ScreenSpaceFluidPlugin)
            .add_systems(Startup, setup_simulation)
            .add_systems(Startup, spawn_orbit_camera)
            .add_event::<ResetSim>()
            .add_event::<SpawnDuck>()
            .add_systems(Update, handle_input)
            .add_systems(Update, handle_draw_lake_toggle)
            // ===== 3D Setup =====
            .add_systems(
                Update,
                crate::three_d::simulation::setup_3d_environment
            )
            .add_systems(
                Update,
                crate::three_d::simulation::spawn_particles_3d
            )
            .add_systems(
                Update,
                update_spatial_hash_on_radius_change_3d
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
            )
            // ===== 3D Duck Systems =====
            .add_systems(
                Update,
                crate::three_d::simulation::spawn_duck_at_cursor
            )
            .add_systems(
                Update,
                crate::three_d::simulation::update_duck_physics
            )
            .add_systems(
                Update,
                crate::three_d::simulation::handle_particle_duck_collisions
            )
            .add_systems(
                Update,
                crate::three_d::simulation::update_particle_colors_3d
            )
            .add_systems(
                Update,
                update_mouse_indicator_3d,
            )
            .add_systems(
                Update,
                handle_ground_deformation,
            )
            .add_systems(Update, update_fps_display)
            .add_systems(Update, handle_reset_sim)
            // Orbit camera (3D only)
            .add_systems(
                Update,control_orbit_camera
            )
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
        Self {
            enabled: true,
        }
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
    mut raymarching_settings: ResMut<crate::three_d::raymarch::RayMarchingSettings>
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
        fluid3d_params.pressure_multiplier =
            (fluid3d_params.pressure_multiplier + 5.0).min(500.0);
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        fluid3d_params.pressure_multiplier =
            (fluid3d_params.pressure_multiplier - 5.0).max(50.0);
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
    q_particles3d: Query<Entity, With<crate::three_d::simulation::Particle3D>>,
    q_marker3d: Query<Entity, With<crate::three_d::simulation::Marker3D>>,
    q_ducks: Query<Entity, With<crate::three_d::simulation::RubberDuck>>,
    q_orbit: Query<Entity, With<crate::three_d::camera::OrbitCamera>>,
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
