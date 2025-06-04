use bevy::prelude::*;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin};
use bevy_egui::EguiPlugin;

mod ui;
mod constants;
mod sim;
mod utils;
mod camera;
mod gpu_fluid;
mod raymarch;
mod screenspace;
mod gpu_fluid_simple;

use sim::SimulationPlugin;
use ui::UiPlugin;
// use gpu_fluid::GpuSim3DPlugin;
use gpu_fluid_simple::GpuSim3DSimplePlugin;

fn main() {
    // Ensure the process working directory is the project root so the default
    // "assets" folder is found regardless of where the executable is launched
    std::env::set_current_dir(env!("CARGO_MANIFEST_DIR")).ok();

    App::new()
        .add_plugins(DefaultPlugins.set(bevy::log::LogPlugin {
            level: bevy::log::Level::INFO,
            filter: "wgpu=warn,naga=warn,info".into(),
            ..default()
        }))
        .add_plugins((
            EguiPlugin { enable_multipass_for_primary_context: false },
            UiPlugin,
            // GpuSim3DPlugin,
            GpuSim3DSimplePlugin,
            SimulationPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .run();
}