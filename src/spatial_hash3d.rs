use bevy::prelude::*;
use std::collections::HashMap;

pub struct SpatialHash3D {
    pub cell_size: f32,
    hash_map: HashMap<(i32, i32, i32), Vec<Entity>>,
}

impl SpatialHash3D {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            hash_map: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.hash_map.clear();
    }

    pub fn insert(&mut self, position: Vec3, entity: Entity) {
        let cell = self.get_cell(position);
        self.hash_map.entry(cell).or_insert_with(Vec::new).push(entity);
    }

    pub fn get_neighbors(&self, position: Vec3, radius: f32) -> Vec<Entity> {
        let mut neighbors = Vec::new();
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        let center_cell = self.get_cell(position);

        for x in -cell_radius..=cell_radius {
            for y in -cell_radius..=cell_radius {
                for z in -cell_radius..=cell_radius {
                    let cell = (
                        center_cell.0 + x,
                        center_cell.1 + y,
                        center_cell.2 + z,
                    );
                    if let Some(entities) = self.hash_map.get(&cell) {
                        neighbors.extend(entities.iter().copied());
                    }
                }
            }
        }

        neighbors
    }

    fn get_cell(&self, position: Vec3) -> (i32, i32, i32) {
        (
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
            (position.z / self.cell_size).floor() as i32,
        )
    }
}

#[derive(Resource)]
pub struct SpatialHashResource3D {
    pub spatial_hash: SpatialHash3D,
}

impl Default for SpatialHashResource3D {
    fn default() -> Self {
        Self {
            spatial_hash: SpatialHash3D::new(35.0),
        }
    }
} 