use serde::{Deserialize, Serialize};
use bevy::prelude::*;
use crate::simulation3d::{Fluid3DParams, SpawnRegion3D};

#[derive(Serialize, Deserialize, Clone)]
pub struct Preset3D {
    pub name: String,
    pub params: Fluid3DParams,
    pub spawn_region: SpawnRegion3D,
    pub seed: u64,
}

#[derive(Resource, Default)]
pub struct PresetManager3D {
    pub presets: Vec<Preset3D>,
    pub current: usize,
}

impl PresetManager3D {
    pub fn current_preset(&self) -> Option<&Preset3D> {
        self.presets.get(self.current)
    }

    pub fn next(&mut self) {
        if !self.presets.is_empty() {
            self.current = (self.current + 1) % self.presets.len();
        }
    }
}

// Load presets file
pub fn load_presets_system(mut manager: ResMut<PresetManager3D>) {
    if !manager.presets.is_empty() {
        return;
    }
    let path = std::path::Path::new("presets3d.json");
    if let Ok(data) = std::fs::read_to_string(path) {
        if let Ok(presets) = serde_json::from_str::<Vec<Preset3D>>(&data) {
            manager.presets = presets;
            return;
        }
    }
    // Fallback default preset
    manager.presets = vec![Preset3D {
        name: "Default".into(),
        params: Fluid3DParams::default(),
        spawn_region: SpawnRegion3D::default(),
        seed: 0,
    }];
} 