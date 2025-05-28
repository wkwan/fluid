use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin};
use bevy_egui::EguiPlugin;

mod simulation;
mod spawner;
mod spatial_hash;
mod math;
mod marching;
mod gpu_fluid;
mod simulation3d;
mod orbit_camera;
mod ui;
mod constants;
use fluid::spatial_hash3d;
mod presets;

use simulation::SimulationPlugin;
use spawner::SpawnerPlugin;
use gpu_fluid::GpuFluidPlugin;
use ui::UiPlugin;
use simulation::SurfaceDebugSettings;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((
            EguiPlugin { enable_multipass_for_primary_context: false },
            UiPlugin,
            SimulationPlugin,
            SpawnerPlugin,
            GpuFluidPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(_commands: Commands) {
    // Camera spawning is now handled by dimension-specific systems
} 