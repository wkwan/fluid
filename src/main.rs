use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin};
use bevy_egui::EguiPlugin;

mod two_d;
mod three_d;

mod ui;
mod constants;
mod sim;
mod utils;

use sim::SimulationPlugin;
use two_d::spawner::SpawnerPlugin;
use two_d::gpu_fluid::GpuFluidPlugin;
use ui::UiPlugin;
use three_d::screenspace::ScreenSpaceFluidPlugin;

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