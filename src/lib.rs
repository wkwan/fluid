pub mod spatial_hash;
pub mod spatial_hash3d;
pub mod marching;
pub mod simulation;
pub mod simulation3d;
pub mod gpu_fluid;
pub mod spawner;
pub mod reordering;
pub mod orbit_camera;
pub mod fluid3d;
pub mod gpu_fluid3d;
pub mod gpu_render3d;
pub mod presets;
pub mod constants;
pub mod screen_space_fluid;

use bevy::prelude::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use gpu_fluid::GpuParticles; // Use GpuParticles from gpu_fluid instead

// Re-exports
pub use gpu_fluid::{GpuFluidPlugin, GpuState, GpuPerformanceStats};
pub use simulation::FluidParams;
pub use marching::RayMarchPlugin;
pub use screen_space_fluid::ScreenSpaceFluidPlugin;

pub fn app() -> App {
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(simulation::SimulationPlugin)
        .add_plugins(gpu_fluid::GpuFluidPlugin)
        .add_plugins(spawner::SpawnerPlugin)
        .add_plugins(reordering::ParticleReorderingPlugin)
        .add_plugins(gpu_render3d::GpuRender3DPlugin)
        .add_plugins(gpu_fluid3d::GpuSim3DPlugin)
        .add_plugins(RayMarchPlugin)
        .add_plugins(ScreenSpaceFluidPlugin)
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
            gpu_render3d::GpuRender3DPlugin,
            gpu_fluid3d::GpuSim3DPlugin,
            RayMarchPlugin,
            ScreenSpaceFluidPlugin,
        ));
    }
} 