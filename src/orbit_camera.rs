use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use crate::simulation::SimulationDimension;

#[derive(Component, Default)]
pub struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

/// Constants for zoom limits computed from simulation bounds
const MIN_ZOOM: f32 = 50.0;
const MAX_ZOOM: f32 = 1000.0; // Upper zoom limit
const RESET_YAW: f32 = 0.0;
const RESET_PITCH: f32 = -20.0;
const RESET_DISTANCE: f32 = 400.0;

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
            yaw: 0.0,
            pitch: -20.0,
            distance: 400.0,
        },
    ));
}

/// Despawn orbit camera when returning to 2-D
pub fn despawn_orbit_camera(
    mut commands: Commands,
    sim_dim: Res<State<SimulationDimension>>,
    query: Query<Entity, With<OrbitCamera>>,
    world: &World,
) {
    if *sim_dim.get() == SimulationDimension::Dim2 {
        let count = query.iter().count();
        info!("Cleaning up {} orbit cameras in 2D mode", count);
        
        for e in query.iter() {
            // Only attempt to despawn if the entity exists in the world
            if world.get_entity(e).is_ok() {
                commands.entity(e).despawn();
            }
        }
    }
}

/// Mouse-driven orbit / scroll-wheel zoom with reset key.
pub fn control_orbit_camera(
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
    mut mouse_evr: EventReader<MouseMotion>,
    mut scroll_evr: EventReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    sim_dim: Res<State<SimulationDimension>>,)
{
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    let Ok((mut cam, mut transform)) = query.single_mut() else { return; };

    // Right-mouse drag rotates
    if buttons.pressed(MouseButton::Right) {
        for ev in mouse_evr.read() {
            cam.yaw -= ev.delta.x * 0.25;
            cam.pitch -= ev.delta.y * 0.25;
            cam.pitch = cam.pitch.clamp(-89.0, 89.0);
        }
    }

    // Scroll zoom
    for ev in scroll_evr.read() {
        cam.distance -= ev.y * 25.0;
        cam.distance = cam.distance.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    // Reset orientation key (Home)
    if keys.just_pressed(KeyCode::Home) {
        cam.yaw = RESET_YAW;
        cam.pitch = RESET_PITCH;
        cam.distance = RESET_DISTANCE.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    let rot = Quat::from_euler(EulerRot::YXZ,
                               cam.yaw.to_radians(),
                               cam.pitch.to_radians(),
                               0.0);
    transform.translation = rot * Vec3::Z * cam.distance;
    transform.look_at(Vec3::ZERO, Vec3::Y);
} 