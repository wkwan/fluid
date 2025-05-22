use bevy::prelude::*;
use crate::gpu_fluid3d::{GpuSim3DPlugin, GpuParticles3D};

/// Plugin for 3D Fluid Simulation
pub struct Fluid3DPlugin;

impl Plugin for Fluid3DPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<GpuParticles3D>()
            .add_plugins(GpuSim3DPlugin);
    }
} 