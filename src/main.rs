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
mod spatial_hash;

use sim::SimulationPlugin;
use ui::UiPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins((
            EguiPlugin { enable_multipass_for_primary_context: false },
            UiPlugin,
            SimulationPlugin,
            FrameTimeDiagnosticsPlugin::default(),
        ))
        .run();
}