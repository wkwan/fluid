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