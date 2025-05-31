use bevy::prelude::*;
use crate::sim::SimulationDimension;
use crate::utils::despawn_entities;

#[derive(Component)]
pub struct Camera2DMarker;

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
        despawn_entities(&mut commands, &existing);
    }
} 