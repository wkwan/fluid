use bevy::prelude::*;
use crate::gpu_fluid3d::GpuSim3DPlugin;
use crate::gpu_render3d::GpuRender3DPlugin;

/// Plugin for 3D Fluid Simulation
pub struct Fluid3DPlugin;

impl Plugin for Fluid3DPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins((
                GpuSim3DPlugin,
                GpuRender3DPlugin,
            ));
    }
} 