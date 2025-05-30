use bevy::prelude::*;
use std::collections::HashMap;
use crate::constants::{HASH_K1, HASH_K2};

pub struct SpatialHash {
    pub cell_size: f32,
    hash_map: HashMap<(i32, i32), Vec<Entity>>,
    // Cache for neighbor queries to reduce allocations
    neighbor_cache: Vec<Entity>,
    // Power-of-two size optimization
    table_size: u32,
    table_size_mask: u32,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        // Pre-allocate a reasonably sized hash map and neighbor cache
        // Use power-of-two size for fast modulo with bitmask
        let table_size: u32 = 4096; // Power of two for optimal hashing
        let table_size_mask = table_size - 1; // Used for fast modulo with bitwise AND
        let initial_capacity = table_size as usize;
        let hash_map = HashMap::with_capacity(initial_capacity);
        
        Self {
            cell_size,
            hash_map,
            neighbor_cache: Vec::with_capacity(initial_capacity),
            table_size,
            table_size_mask,
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

    // Optimized version that allows reusing a results vector
    #[inline]
    pub fn get_neighbors_into(&self, position: Vec2, radius: f32, results: &mut Vec<Entity>) {
        results.clear();
        
        let center_cell = self.position_to_cell(position);
        let cell_radius = (radius / self.cell_size).ceil() as i32;
        
        // Pre-reserve capacity - convert i32 to usize properly with improved estimate
        let expected_cells = (2 * cell_radius + 1) * (2 * cell_radius + 1);
        let average_particles_per_cell = 8; // Better estimate based on typical density
        let expected_count = average_particles_per_cell * expected_cells as usize;
        
        if results.capacity() < expected_count {
            results.reserve(expected_count - results.capacity());
        }
        
        // Cache cell bounds calculation outside loop
        let min_x = center_cell.0 - cell_radius;
        let max_x = center_cell.0 + cell_radius;
        let min_y = center_cell.1 - cell_radius;
        let max_y = center_cell.1 + cell_radius;
        
        // Use optimized pointer-based copying for better performance
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
    
    // Optimized hash function for GPU implementation
    #[inline]
    pub fn hash_cell(&self, cell: (i32, i32)) -> u32 {
        let a = cell.0 as u32 * HASH_K1;
        let b = cell.1 as u32 * HASH_K2;
        let hash = a ^ b; // XOR is faster than addition
        self.key_from_hash(hash)
    }
    
    // Fast power-of-two modulo using bitmask
    #[inline]
    fn key_from_hash(&self, hash: u32) -> u32 {
        hash & self.table_size_mask // Faster than hash % table_size
    }
}

// Hash cell coordinate to a single unsigned integer (for GPU implementation)
#[inline]
pub fn hash_cell_2d(cell: (i32, i32)) -> u32 {
    let a = cell.0 as u32 * HASH_K1;
    let b = cell.1 as u32 * HASH_K2;
    a ^ b // XOR is faster than addition
}

// Get key from hash for a table of given size
#[inline]
pub fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    if (table_size & (table_size - 1)) == 0 {
        // If table_size is a power of two, use fast bitwise AND
        hash & (table_size - 1)
    } else {
        // Otherwise use modulo
        hash % table_size
    }
} 