mod math;
mod simulation;
mod spatial_hash;
mod gpu_fluid;
mod gpu_sim;
mod spawner;
mod render;

use bevy::prelude::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use gpu_sim::GpuSimPlugin;
use gpu_sim::GpuParticles;

pub fn app() -> App {
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins)
       .add_plugins(FrameTimeDiagnosticsPlugin::default())
       .add_plugins(simulation::SimulationPlugin)
       .add_plugins(gpu_fluid::GpuFluidPlugin)
       .add_plugins(spawner::SpawnerPlugin)
       .add_plugins(GpuSimPlugin)
       .init_resource::<GpuParticles>();
    
    app
} 