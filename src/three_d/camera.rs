use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use crate::sim::SimulationDimension;
use crate::constants::{MIN_ZOOM, MAX_ZOOM, RESET_YAW, RESET_PITCH, RESET_DISTANCE};
use crate::utils::despawn_entities;

#[derive(Component, Default)]
pub struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    center: Vec3,
}

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
        despawn_entities(&mut commands, &query);
    }
}

pub fn control_orbit_camera(
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
    mut mouse_evr: EventReader<MouseMotion>,
    mut scroll_evr: EventReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    sim_dim: Res<State<SimulationDimension>>,
    draw_lake_mode: Res<crate::three_d::simulation::DrawLakeMode>,
) {
    if *sim_dim.get() != SimulationDimension::Dim3 {
        return;
    }

    let Ok((mut cam, mut transform)) = query.single_mut() else { return; };

    let should_rotate_camera = if draw_lake_mode.enabled {
        buttons.pressed(MouseButton::Middle)
    } else {
        true
    };
    
    if should_rotate_camera {
        for ev in mouse_evr.read() {
            cam.yaw -= ev.delta.x * 0.25;
            cam.pitch -= ev.delta.y * 0.25;
            cam.pitch = cam.pitch.clamp(-89.0, 89.0);
        }
    } else {
        mouse_evr.clear();
    }

    for ev in scroll_evr.read() {
        cam.distance -= ev.y * 25.0;
        cam.distance = cam.distance.clamp(MIN_ZOOM, MAX_ZOOM);
    }

    let movement_speed = 200.0 * time.delta_secs();
    let mut movement = Vec3::ZERO;

    let camera_rotation = Quat::from_euler(EulerRot::YXZ,
                                          cam.yaw.to_radians(),
                                          cam.pitch.to_radians(),
                                          0.0);
    let forward = camera_rotation * Vec3::NEG_Z;
    let right = camera_rotation * Vec3::X;

    if keys.pressed(KeyCode::KeyW) {
        movement += forward * movement_speed;
    }
    if keys.pressed(KeyCode::KeyS) {
        movement -= forward * movement_speed;
    }
    if keys.pressed(KeyCode::KeyA) {
        movement -= right * movement_speed;
    }
    if keys.pressed(KeyCode::KeyD) {
        movement += right * movement_speed;
    }

    if movement != Vec3::ZERO {
        cam.center += movement;
    }

    if keys.just_pressed(KeyCode::Home) {
        cam.yaw = RESET_YAW;
        cam.pitch = RESET_PITCH;
        cam.distance = RESET_DISTANCE.clamp(MIN_ZOOM, MAX_ZOOM);
        cam.center = Vec3::ZERO;
    }

    let rot = Quat::from_euler(EulerRot::YXZ,
                               cam.yaw.to_radians(),
                               cam.pitch.to_radians(),
                               0.0);
    transform.translation = cam.center + (rot * Vec3::Z * cam.distance);
    transform.look_at(cam.center, Vec3::Y);
} 