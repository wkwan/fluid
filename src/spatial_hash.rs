use bevy::prelude::*;
use std::collections::HashMap;

pub struct SpatialHash {
    cell_size: f32,
    hash_map: HashMap<(i32, i32), Vec<Entity>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            hash_map: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.hash_map.clear();
    }

    pub fn insert(&mut self, position: Vec2, entity: Entity) {
        let cell = self.position_to_cell(position);
        self.hash_map.entry(cell).or_insert_with(Vec::new).push(entity);
    }

    pub fn get_neighbors(&self, position: Vec2, radius: f32) -> Vec<Entity> {
        let center_cell = self.position_to_cell(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        
        let mut neighbors = Vec::new();
        
        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                let cell = (center_cell.0 + dx, center_cell.1 + dy);
                if let Some(entities) = self.hash_map.get(&cell) {
                    neighbors.extend(entities);
                }
            }
        }
        
        neighbors
    }

    fn position_to_cell(&self, position: Vec2) -> (i32, i32) {
        (
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
        )
    }
}

// Constants for hashing if needed for more complex implementations
pub const HASH_K1: u32 = 15823;
pub const HASH_K2: u32 = 9737333;

// Hash cell coordinate to a single unsigned integer (for GPU implementation)
pub fn hash_cell_2d(cell: (i32, i32)) -> u32 {
    let a = cell.0 as u32 * HASH_K1;
    let b = cell.1 as u32 * HASH_K2;
    a + b
}

// Get key from hash for a table of given size
pub fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    hash % table_size
} 