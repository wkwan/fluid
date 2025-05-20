use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};

mod simulation;
mod spawner;
mod spatial_hash;
mod math;
mod gpu_fluid;

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
            LogDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
} 