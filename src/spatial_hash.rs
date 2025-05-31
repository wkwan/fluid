use bevy::prelude::*;
use std::collections::HashMap;

pub struct SpatialHash {
    pub cell_size: f32,
    hash_map: HashMap<(i32, i32), Vec<Entity>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        // Pre-allocate a reasonably sized hash map
        let initial_capacity = 4096;
        let hash_map = HashMap::with_capacity(initial_capacity);
        
        Self {
            cell_size,
            hash_map,
        }
    }

    pub fn clear(&mut self) {
        // Clear but retain capacity for all hash map entries
        for entities in self.hash_map.values_mut() {
            entities.clear();
        }
    }

    pub fn insert(&mut self, position: Vec2, entity: Entity) {
        let cell = self.position_to_cell(position);
        self.hash_map.entry(cell).or_insert_with(|| Vec::with_capacity(8)).push(entity);
    }

    pub fn get_neighbors(&self, position: Vec2, radius: f32) -> Vec<Entity> {
        let center_cell = self.position_to_cell(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        
        let mut neighbors = Vec::new();
        // Optimize common case - preallocate for expected number of neighbors
        // Convert i32 to usize safely with as conversion and improved sizing estimate
        let expected_cells = (2 * cell_radius + 1) * (2 * cell_radius + 1);
        let average_particles_per_cell = 8; // Better estimate based on typical density
        let reserve_size = average_particles_per_cell * expected_cells as usize;
        neighbors.reserve(reserve_size);
        
        // Cache cell bounds calculation outside loop
        let min_x = center_cell.0 - cell_radius;
        let max_x = center_cell.0 + cell_radius;
        let min_y = center_cell.1 - cell_radius;
        let max_y = center_cell.1 + cell_radius;
        
        // Loop with direct indices rather than range - faster for large cell counts
        let mut cell_x = min_x;
        while cell_x <= max_x {
            let mut cell_y = min_y;
            while cell_y <= max_y {
                if let Some(entities) = self.hash_map.get(&(cell_x, cell_y)) {
                    // Use extend which is more optimized than individual pushes
                    neighbors.extend_from_slice(entities);
                }
                cell_y += 1;
            }
            cell_x += 1;
        }
        
        neighbors
    }

    #[inline]
    fn position_to_cell(&self, position: Vec2) -> (i32, i32) {
        let inv_cell_size = 1.0 / self.cell_size;
        (
            (position.x * inv_cell_size).floor() as i32,
            (position.y * inv_cell_size).floor() as i32,
        )
    }
} 