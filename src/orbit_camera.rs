use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use crate::simulation::SimulationDimension;
use crate::constants::{MIN_ZOOM, MAX_ZOOM, RESET_YAW, RESET_PITCH, RESET_DISTANCE};

#[derive(Component, Default)]
pub struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    center: Vec3, // The point the camera orbits around
}

// Marker component for 2D camera
#[derive(Component)]
pub struct Camera2DMarker;

/// Spawn a 3-D orbit camera when entering 3-D mode (if none exists).
pub fn spawn_orbit_camera(
    mut commands: Commands,
    sim_dim: Res<State<SimulationDimension>>,
    existing: Query<(), With<OrbitCamera>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 || !existing.is_empty() {
        return;
    }

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 200.0, 400.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
        OrbitCamera {
            yaw: RESET_YAW,
            pitch: RESET_PITCH,
            distance: RESET_DISTANCE,
            center: Vec3::ZERO,
        },
    ));
}

/// Despawn orbit camera when returning to 2-D
pub fn despawn_orbit_camera(
    mut commands: Commands,
    sim_dim: Res<State<SimulationDimension>>,
    query: Query<Entity, With<OrbitCamera>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim2 {
        return;
    }
    
    let count = query.iter().count();
    if count > 0 {
        info!("Cleaning up {} orbit cameras in 2D mode", count);
        for entity in query.iter() {
            commands.entity(entity).despawn();
        }
    }
}

/// Mouse-driven orbit / scroll-wheel zoom with reset key and WASD movement.
pub fn control_orbit_camera(
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
    mut mouse_evr: EventReader<MouseMotion>,
    mut scroll_evr: EventReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    sim_dim: Res<State<SimulationDimension>>,
    draw_lake_mode: Res<crate::simulation::DrawLakeMode>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    let Ok((mut cam, mut transform)) = query.single_mut() else { return; };

    // Mouse movement rotates camera
    // When terrain doodling mode is active, use middle mouse button for rotation
    // When terrain doodling mode is off, rotate freely without holding any button
    let should_rotate_camera = if draw_lake_mode.enabled {
        buttons.pressed(MouseButton::Middle)
    } else {
        true // Free rotation when terrain doodling is off
    };
    
    if should_rotate_camera {
        for ev in mouse_evr.read() {
            cam.yaw -= ev.delta.x * 0.25;
            cam.pitch -= ev.delta.y * 0.25;
            cam.pitch = cam.pitch.clamp(-89.0, 89.0);
        }
    } else {
        // Clear mouse events when not rotating to prevent them from accumulating
        mouse_evr.clear();
    }

    // Scroll zoom
    for ev in scroll_evr.read() {
        cam.distance -= ev.y * 25.0;
        cam.distance = cam.distance.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    // WASD movement controls
    let movement_speed = 200.0 * time.delta_secs(); // Units per second
    let mut movement = Vec3::ZERO;

    // Calculate camera's full rotation (yaw + pitch) to get true forward direction
    let camera_rotation = Quat::from_euler(EulerRot::YXZ,
                                          cam.yaw.to_radians(),
                                          cam.pitch.to_radians(),
                                          0.0);
    let forward = camera_rotation * Vec3::NEG_Z; // Camera looks down -Z, so forward is -Z
    let right = camera_rotation * Vec3::X;

    if keys.pressed(KeyCode::KeyW) {
        movement += forward * movement_speed; // Forward relative to camera
    }
    if keys.pressed(KeyCode::KeyS) {
        movement -= forward * movement_speed; // Backward relative to camera
    }
    if keys.pressed(KeyCode::KeyA) {
        movement -= right * movement_speed; // Left relative to camera
    }
    if keys.pressed(KeyCode::KeyD) {
        movement += right * movement_speed; // Right relative to camera
    }

    // Apply movement to camera center
    if movement != Vec3::ZERO {
        cam.center += movement;
    }

    // Reset orientation and position key (Home)
    if keys.just_pressed(KeyCode::Home) {
        cam.yaw = RESET_YAW;
        cam.pitch = RESET_PITCH;
        cam.distance = RESET_DISTANCE.clamp(MIN_ZOOM, MAX_ZOOM);
        cam.center = Vec3::ZERO;
    }

    // Calculate camera position based on center point
    let rot = Quat::from_euler(EulerRot::YXZ,
                               cam.yaw.to_radians(),
                               cam.pitch.to_radians(),
                               0.0);
    transform.translation = cam.center + (rot * Vec3::Z * cam.distance);
    transform.look_at(cam.center, Vec3::Y);
}

/// Spawn a 2D camera when entering 2D mode (if none exists).
pub fn spawn_2d_camera(
    mut commands: Commands,
    sim_dim: Res<State<SimulationDimension>>,
    existing: Query<(), With<Camera2DMarker>>,
) {
    if *sim_dim.get() != SimulationDimension::Dim2 || !existing.is_empty() {
        return;
    }

    commands.spawn((
        Camera2d::default(),
        Camera2DMarker,
    ));
    info!("Spawned 2D camera");
}

/// Despawn 2D camera when leaving 2D mode.
pub fn despawn_2d_camera(
    mut commands: Commands,
    sim_dim: Res<State<SimulationDimension>>,
    existing: Query<Entity, With<Camera2DMarker>>,
) {
    if *sim_dim.get() == SimulationDimension::Dim2 {
        return;
    }

    let count = existing.iter().count();
    if count > 0 {
        info!("Cleaning up {} 2D cameras in 3D mode", count);
        for entity in existing.iter() {
            commands.entity(entity).despawn();
        }
    }
} 