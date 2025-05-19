use bevy::prelude::*;

mod simulation;
mod spawner;
mod spatial_hash;
mod math;

use simulation::SimulationPlugin;
use spawner::SpawnerPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((
            SimulationPlugin,
            SpawnerPlugin,
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d::default());
} 