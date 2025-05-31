use bevy::prelude::*;
use crate::sim::{Particle, FluidParams};

/// Resource to track when particles were last reordered
#[derive(Resource, Default)]
pub struct ParticleReorderingState {
    frames_since_last_reorder: u32,
}

/// A frame counter for tracking when to reorder particles
#[derive(Resource, Default)]
pub struct FluidFrameCounter {
    count: u32,
}

/// System to reorder particles based on spatial locality for better cache coherence
/// This mimics what Unity does on the GPU but implemented on the CPU for simplicity
pub fn reorder_particles_system(
    mut particles_query: Query<(Entity, &mut Transform, &mut Particle)>,
    mut reordering_state: ResMut<ParticleReorderingState>,
    mut frame_counter: ResMut<FluidFrameCounter>,
    fluid_params: Res<FluidParams>,
    _time: Res<Time>,
) {
    // Increment frame counter
    frame_counter.count = frame_counter.count.wrapping_add(1);
    reordering_state.frames_since_last_reorder += 1;
    
    // Only reorder periodically to avoid constant overhead
    if reordering_state.frames_since_last_reorder < 100 {
        return;
    }
    
    // Reset counter
    reordering_state.frames_since_last_reorder = 0;
    
    // Skip if no particles
    if particles_query.is_empty() {
        return;
    }
    
    // Start a timer to measure performance impact
    let start_time = std::time::Instant::now();
    
    // Create a spatial hash with a cell size based on smoothing radius
    let smoothing_radius = fluid_params.smoothing_radius;
    let cell_size = smoothing_radius * 2.0;  // Match GPU shader cell size
    
    // Create a mapping from spatial cells to particles
    let mut spatial_map: Vec<(u32, Entity)> = Vec::with_capacity(particles_query.iter().count());
    
    // Build the spatial hash for all particles
    for (entity, transform, _) in particles_query.iter() {
        let position = transform.translation.truncate();
        let cell_x = (position.x / cell_size).floor() as i32;
        let cell_y = (position.y / cell_size).floor() as i32;
        
        // Use the same hash function as the GPU shader for consistency
        let hash = (cell_x as u32 * 15823) ^ (cell_y as u32 * 9737333);
        
        // Store hash and entity
        spatial_map.push((hash, entity));
    }
    
    // Sort particles by their spatial hash for better cache locality
    spatial_map.sort_by_key(|&(hash, _)| hash);
    
    // Store current transforms and velocities
    let mut transforms = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut pressures = Vec::new();
    let mut near_densities = Vec::new();
    let mut near_pressures = Vec::new();
    
    // Collect current data
    for (_, entity) in &spatial_map {
        if let Ok((_, transform, particle)) = particles_query.get(*entity) {
            transforms.push(transform.clone());
            velocities.push(particle.velocity);
            densities.push(particle.density);
            pressures.push(particle.pressure);
            near_densities.push(particle.near_density);
            near_pressures.push(particle.near_pressure);
        }
    }
    
    // Apply reordering
    for (i, (_, entity)) in spatial_map.iter().enumerate() {
        if i < transforms.len() {
            if let Ok((_, mut transform, mut particle)) = particles_query.get_mut(*entity) {
                *transform = transforms[i].clone();
                particle.velocity = velocities[i];
                particle.density = densities[i];
                particle.pressure = pressures[i];
                particle.near_density = near_densities[i];
                particle.near_pressure = near_pressures[i];
            }
        }
    }
    
    // Log reordering time for performance monitoring
    let elapsed = start_time.elapsed();
    info!("Reordered {} particles for cache locality in {:?}", spatial_map.len(), elapsed);
}

/// Plugin to add CPU-side particle reordering systems
#[derive(Default)]
pub struct ParticleReorderingPlugin;

impl Plugin for ParticleReorderingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ParticleReorderingState>()
           .init_resource::<FluidFrameCounter>()
           .add_systems(Update, reorder_particles_system);
    }
} 