//! Stub GPU render plugin (3-D)
use bevy::prelude::*;

#[derive(Default)]
pub struct GpuRender3DPlugin;

impl Plugin for GpuRender3DPlugin {
    fn build(&self, _app: &mut App) {}
} 