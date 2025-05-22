use bevy::prelude::*;
use crate::simulation3d::Particle3D;

/// CPU fallback for neighbor finding when GPU compute is not available
pub struct NeighborReduction3D {
    pub neighbor_counts: Vec<u32>,
    pub neighbor_indices: Vec<u32>,
    pub max_neighbors: usize,
}

impl NeighborReduction3D {
    pub fn new(max_particles: usize, max_neighbors: usize) -> Self {
        Self {
            neighbor_counts: vec![0; max_particles],
            neighbor_indices: vec![0; max_particles * max_neighbors],
            max_neighbors,
        }
    }

    pub fn find_neighbors(
        &mut self,
        particles: &[(Entity, &Transform, &Particle3D)],
        smoothing_radius: f32,
    ) {
        let sqr_radius = smoothing_radius * smoothing_radius;
        
        // Reset neighbor counts
        self.neighbor_counts.fill(0);
        
        // For each particle
        for (i, (_, transform_i, _)) in particles.iter().enumerate() {
            let pos_i = transform_i.translation;
            let mut neighbor_count = 0;
            
            // Check all other particles
            for (j, (_, transform_j, _)) in particles.iter().enumerate() {
                if i == j {
                    continue;
                }
                
                let pos_j = transform_j.translation;
                let offset = pos_j - pos_i;
                let sqr_distance = offset.length_squared();
                
                // If within radius, add to neighbor list
                if sqr_distance <= sqr_radius {
                    let base_offset = i * self.max_neighbors;
                    if neighbor_count < self.max_neighbors {
                        self.neighbor_indices[base_offset + neighbor_count] = j as u32;
                        neighbor_count += 1;
                    }
                }
            }
            
            self.neighbor_counts[i] = neighbor_count as u32;
        }
    }
    
    pub fn get_neighbors(&self, particle_index: usize) -> &[u32] {
        let base_offset = particle_index * self.max_neighbors;
        let count = self.neighbor_counts[particle_index] as usize;
        &self.neighbor_indices[base_offset..base_offset + count]
    }
} 