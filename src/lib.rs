mod math;
mod simulation;
mod spatial_hash;
mod gpu_fluid;
// mod gpu_sim; // Commented out to avoid duplication
mod spawner;
mod render;
mod reordering;

use bevy::prelude::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
// use gpu_sim::GpuSimPlugin; // Commented out to avoid duplication
// use gpu_sim::GpuParticles; // Commented out to avoid duplication
use gpu_fluid::GpuParticles; // Use GpuParticles from gpu_fluid instead

// Re-exports
pub use gpu_fluid::{GpuFluidPlugin, GpuState, GpuPerformanceStats};
pub use simulation::FluidParams;

pub fn app() -> App {
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(simulation::SimulationPlugin)
        .add_plugins(render::RenderPlugin)
        .add_plugins(gpu_fluid::GpuFluidPlugin)
        .add_plugins(spawner::SpawnerPlugin)
        .add_plugins(reordering::ParticleReorderingPlugin)
        .init_resource::<GpuParticles>();
    
    app
}

// Plugin to add all fluid simulation functionality to a Bevy app
#[derive(Default)]
pub struct FluidPlugin;

impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            simulation::SimulationPlugin,
            gpu_fluid::GpuFluidPlugin,
            reordering::ParticleReorderingPlugin,
        ));
    }
} 