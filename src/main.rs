use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin};
use bevy_egui::EguiPlugin;

mod simulation;
mod spawner;
mod spatial_hash;
mod marching;
mod gpu_fluid;
mod simulation3d;
mod orbit_camera;
mod ui;
mod constants;
mod spatial_hash3d;
mod presets;
mod screen_space_fluid;
mod reordering;
mod gpu_fluid3d;
mod gpu_render3d;
mod fluid3d;

use simulation::SimulationPlugin;
use spawner::SpawnerPlugin;
use gpu_fluid::GpuFluidPlugin;
use ui::UiPlugin;
use screen_space_fluid::ScreenSpaceFluidPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((
            EguiPlugin { enable_multipass_for_primary_context: false },
            UiPlugin,
            SimulationPlugin,
            SpawnerPlugin,
            GpuFluidPlugin,
            ScreenSpaceFluidPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .run();
}