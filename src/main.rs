use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin};

mod simulation;
mod spawner;
mod spatial_hash;
mod math;
mod gpu_fluid;
mod simulation3d;
mod orbit_camera;
use fluid::spatial_hash3d;
mod presets;

use simulation::SimulationPlugin;
use spawner::SpawnerPlugin;
use gpu_fluid::GpuFluidPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((
            SimulationPlugin,
            SpawnerPlugin,
            GpuFluidPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
} 