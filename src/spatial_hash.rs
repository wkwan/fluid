use bevy::prelude::*;
use std::collections::HashMap;

pub struct SpatialHash {
    cell_size: f32,
    hash_map: HashMap<(i32, i32), Vec<Entity>>,
    // Cache for neighbor queries to reduce allocations
    neighbor_cache: Vec<Entity>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        // Pre-allocate a reasonably sized hash map and neighbor cache
        let initial_capacity = 512;
        let hash_map = HashMap::with_capacity(initial_capacity);
        
        Self {
            cell_size,
            hash_map,
            neighbor_cache: Vec::with_capacity(initial_capacity),
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
        // We can't use the cached vec directly since it's not &mut self,
        // so we return a new vector instead
        
        let center_cell = self.position_to_cell(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        
        let mut neighbors = Vec::new();
        // Optimize common case - preallocate for expected number of neighbors
        // Convert i32 to usize safely with as conversion
        let expected_cells = (2 * cell_radius + 1) * (2 * cell_radius + 1);
        let reserve_size = 16 * expected_cells as usize;
        neighbors.reserve(reserve_size);
        
        // Cache cell bounds calculation outside loop
        let min_x = center_cell.0 - cell_radius;
        let max_x = center_cell.0 + cell_radius;
        let min_y = center_cell.1 - cell_radius;
        let max_y = center_cell.1 + cell_radius;
        
        // Loop with direct indices rather than range
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

    // Optimized version that allows reusing a results vector
    // This would be the ideal API in a real implementation
    #[inline]
    pub fn get_neighbors_into(&self, position: Vec2, radius: f32, results: &mut Vec<Entity>) {
        results.clear();
        
        let center_cell = self.position_to_cell(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        
        // Pre-reserve capacity - convert i32 to usize properly
        let expected_cells = (2 * cell_radius + 1) * (2 * cell_radius + 1);
        let expected_count = 16 * expected_cells as usize;
        
        if results.capacity() < expected_count {
            results.reserve(expected_count - results.capacity());
        }
        
        // Cache cell bounds calculation outside loop
        let min_x = center_cell.0 - cell_radius;
        let max_x = center_cell.0 + cell_radius;
        let min_y = center_cell.1 - cell_radius;
        let max_y = center_cell.1 + cell_radius;
        
        for cell_x in min_x..=max_x {
            for cell_y in min_y..=max_y {
                if let Some(entities) = self.hash_map.get(&(cell_x, cell_y)) {
                    results.extend_from_slice(entities);
                }
            }
        }
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

// Constants for hashing if needed for more complex implementations
pub const HASH_K1: u32 = 15823;
pub const HASH_K2: u32 = 9737333;

// Hash cell coordinate to a single unsigned integer (for GPU implementation)
#[inline]
pub fn hash_cell_2d(cell: (i32, i32)) -> u32 {
    let a = cell.0 as u32 * HASH_K1;
    let b = cell.1 as u32 * HASH_K2;
    a + b
}

// Get key from hash for a table of given size
#[inline]
pub fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    hash % table_size
} 