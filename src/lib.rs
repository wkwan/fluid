mod math;
mod simulation;
mod spatial_hash;
mod gpu_fluid;
// mod gpu_sim; // Commented out to avoid duplication
mod spawner;
mod render;

use bevy::prelude::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
// use gpu_sim::GpuSimPlugin; // Commented out to avoid duplication
// use gpu_sim::GpuParticles; // Commented out to avoid duplication
use gpu_fluid::GpuParticles; // Use GpuParticles from gpu_fluid instead

pub fn app() -> App {
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins)
       .add_plugins(FrameTimeDiagnosticsPlugin::default())
       .add_plugins(simulation::SimulationPlugin)
       .add_plugins(gpu_fluid::GpuFluidPlugin)
       .add_plugins(spawner::SpawnerPlugin)
       // .add_plugins(GpuSimPlugin) // Commented out to avoid duplication
       .init_resource::<GpuParticles>();
    
    app
} 