use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use crate::simulation::SimulationDimension;

#[derive(Component, Default)]
pub struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

/// Spawn a 3-D orbit camera when entering 3-D mode (if none exists).
pub fn spawn_orbit_camera(
    mut commands: Commands,
    sim_dim: Res<SimulationDimension>,
    existing: Query<(), With<OrbitCamera>>,
) {
    if *sim_dim != SimulationDimension::Dim3 || !existing.is_empty() {
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

/// Mouse-driven orbit / scroll-wheel zoom. Only active in 3-D.
pub fn control_orbit_camera(
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
    mut mouse_evr: EventReader<MouseMotion>,
    mut scroll_evr: EventReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    sim_dim: Res<SimulationDimension>,
) {
    if *sim_dim != SimulationDimension::Dim3 {
        return;
    }

    let Ok((mut cam, mut transform)) = query.get_single_mut() else { return; };

    if buttons.pressed(MouseButton::Right) {
        for ev in mouse_evr.read() {
            cam.yaw -= ev.delta.x * 0.25;
            cam.pitch -= ev.delta.y * 0.25;
            cam.pitch = cam.pitch.clamp(-89.0, 89.0);
        }
    }

    for ev in scroll_evr.read() {
        cam.distance -= ev.y * 25.0;
        cam.distance = cam.distance.clamp(50.0, 800.0);
    }

    let rot = Quat::from_euler(EulerRot::YXZ,
                               cam.yaw.to_radians(),
                               cam.pitch.to_radians(),
                               0.0);
    transform.translation = rot * Vec3::Z * cam.distance;
    transform.look_at(Vec3::ZERO, Vec3::Y);
} 