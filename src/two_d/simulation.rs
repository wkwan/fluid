use bevy::prelude::*;
use crate::two_d::spatial_hash::SpatialHash;
use crate::two_d::gpu_fluid::GpuPerformanceStats;
use crate::constants::{GRAVITY_2D, BOUNDARY_DAMPENING, PARTICLE_RADIUS, REST_DENSITY};

#[derive(Component)]
pub struct Particle {
    pub velocity: Vec2,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
    pub previous_position: Vec2,
}

#[derive(Resource, Clone)]
pub struct FluidParams {
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub viscosity_strength: f32,
    pub boundary_min: Vec2,
    pub boundary_max: Vec2,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            smoothing_radius: 10.0,
            target_density: 30.0,
            pressure_multiplier: 100.0,
            near_pressure_multiplier: 50.0,
            viscosity_strength: 0.0,
            boundary_min: Vec2::new(-300.0, -300.0),
            boundary_max: Vec2::new(300.0, 300.0),
        }
    }
}

#[derive(Resource, Clone)]
pub struct MouseInteraction {
    pub position: Vec2,
    pub active: bool,
    pub repel: bool,
    pub strength: f32,
    pub radius: f32,
}

impl Default for MouseInteraction {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            active: false,
            repel: false,
            strength: 1000.0,
            radius: 50.0,
        }
    }
}

#[derive(Resource)]
pub struct SpatialHashResource {
    pub spatial_hash: SpatialHash,
}

impl Default for SpatialHashResource {
    fn default() -> Self {
        let default_params = FluidParams::default();
        Self {
            spatial_hash: SpatialHash::new(default_params.smoothing_radius),
        }
    }
}

const GRAVITY: Vec2 = Vec2::new(GRAVITY_2D[0], GRAVITY_2D[1]);

// Define ColorMapParams locally since we removed the utility module
#[derive(Resource, Clone, Copy)]
pub struct ColorMapParams {
    pub min_speed: f32,
    pub max_speed: f32,
    pub use_velocity_color: bool,
}

impl Default for ColorMapParams {
    fn default() -> Self {
        Self {
            min_speed: 0.0,
            max_speed: 300.0,
            use_velocity_color: true,
        }
    }
}

pub fn velocity_to_color(velocity: Vec2, min_speed: f32, max_speed: f32) -> Color {
    let speed = velocity.length();
    let normalized_speed = ((speed - min_speed) / (max_speed - min_speed)).clamp(0.0, 1.0);
    if normalized_speed < 0.25 {
        let t = normalized_speed * 4.0;
        Color::srgb(0.0, t, 1.0)
    } else if normalized_speed < 0.5 {
        let t = (normalized_speed - 0.25) * 4.0;
        Color::srgb(0.0, 1.0, 1.0 - t)
    } else if normalized_speed < 0.75 {
        let t = (normalized_speed - 0.5) * 4.0;
        Color::srgb(t, 1.0, 0.0)
    } else {
        let t = (normalized_speed - 0.75) * 4.0;
        Color::srgb(1.0, 1.0 - t, 0.0)
    }
}

pub fn apply_external_forces_paper(
    time: Res<Time>,
    mouse_interaction: Res<MouseInteraction>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    for (transform, mut particle) in particle_query.iter_mut() {
        particle.previous_position = transform.translation.truncate();
        particle.velocity += GRAVITY * dt;
        if mouse_interaction.active {
            let direction = mouse_interaction.position - transform.translation.truncate();
            let distance = direction.length();
            if distance < mouse_interaction.radius {
                let force_direction = if mouse_interaction.repel { -direction } else { direction };
                let force_strength = mouse_interaction.strength * (1.0 - distance / mouse_interaction.radius);
                particle.velocity += force_direction.normalize() * force_strength * dt;
            }
        }
    }
}

pub fn predict_positions(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Particle)>,
) {
    let dt = time.delta_secs();
    for (mut transform, particle) in query.iter_mut() {
        let predicted_pos = particle.previous_position + particle.velocity * dt;
        transform.translation = Vec3::new(predicted_pos.x, predicted_pos.y, 0.0);
    }
}

pub fn double_density_relaxation(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(Entity, &mut Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    let dt_squared = dt * dt;
    let smoothing_radius = fluid_params.smoothing_radius;
    let target_density = fluid_params.target_density;
    let k = fluid_params.pressure_multiplier;
    let k_near = fluid_params.near_pressure_multiplier;
    
    // Cache particle data
    let particle_data: Vec<(Entity, Vec2, f32, f32)> = particle_query
        .iter()
        .map(|(entity, transform, particle)| 
            (entity, transform.translation.truncate(), particle.density, particle.near_density))
        .collect();
    
    let mut position_map = std::collections::HashMap::with_capacity(particle_data.len());
    let mut density_map = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity, position, density, near_density) in &particle_data {
        position_map.insert(entity, position);
        density_map.insert(entity, (density, near_density));
    }
    
    // Calculate position displacements for each particle
    let mut displacements = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity_i, position_i, density_i, near_density_i) in &particle_data {
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_i, smoothing_radius);
        
        // Calculate pressure and near-pressure (paper equations 2 and 5)
        let pressure_i = k * (density_i - target_density);
        let near_pressure_i = k_near * near_density_i;
        
        let mut displacement_i = Vec2::ZERO;
        
        for neighbor_entity in neighbors {
            if neighbor_entity == entity_i {
                continue;
            }
            
            if let Some(&position_j) = position_map.get(&neighbor_entity) {
                let offset = position_i - position_j;
                let distance = offset.length();
                
                if distance > 0.0 && distance < smoothing_radius {
                    let direction = offset / distance;
                    let q = distance / smoothing_radius;
                    
                    // Paper equation 6: displacement calculation
                    let mut displacement_magnitude = dt_squared * (
                        pressure_i * (1.0 - q) +
                        near_pressure_i * (1.0 - q) * (1.0 - q)
                    );
                    
                    // Add strong repulsion force when particles get too close to prevent overlapping
                    let min_distance = PARTICLE_RADIUS * 2.0; // Minimum separation distance
                    if distance < min_distance {
                        let overlap_factor = (min_distance - distance) / min_distance;
                        let repulsion_force = overlap_factor * overlap_factor * 500.0 * dt_squared;
                        displacement_magnitude += repulsion_force;
                    }
                    
                    let displacement = direction * displacement_magnitude;
                    
                    // Apply half displacement to each particle (action-reaction)
                    displacement_i -= displacement * 0.5;
                    
                    // Store displacement for neighbor
                    *displacements.entry(neighbor_entity).or_insert(Vec2::ZERO) += displacement * 0.5;
                }
            }
        }
        
        displacements.insert(entity_i, displacement_i);
    }
    
    // Apply displacements to particle positions
    for (entity, mut transform, _) in particle_query.iter_mut() {
        if let Some(&displacement) = displacements.get(&entity) {
            let new_pos = transform.translation.truncate() + displacement;
            transform.translation = Vec3::new(new_pos.x, new_pos.y, 0.0);
        }
    }
}

pub fn calculate_density_paper(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    mut particle_query: Query<(Entity, &Transform, &mut Particle)>,
) {
    let smoothing_radius = fluid_params.smoothing_radius;
    
    // Cache positions
    let positions: Vec<(Entity, Vec2)> = particle_query
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation.truncate()))
        .collect();
    
    let mut position_map = std::collections::HashMap::with_capacity(positions.len());
    for (entity, position) in &positions {
        position_map.insert(*entity, *position);
    }
    
    // Calculate densities using paper's kernels
    for (_entity_i, transform_i, mut particle) in particle_query.iter_mut() {
        let position_i = transform_i.translation.truncate();
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_i, smoothing_radius);
        
        let mut density = 0.0;
        let mut near_density = 0.0;
        
        for neighbor_entity in neighbors {
            if let Some(&position_j) = position_map.get(&neighbor_entity) {
                let offset = position_i - position_j;
                let distance = offset.length();
                
                if distance < smoothing_radius {
                    let q = distance / smoothing_radius;
                    
                    // Paper equation 1: density kernel (1-r/h)²
                    density += (1.0 - q) * (1.0 - q);
                    
                    // Paper equation 4: near-density kernel (1-r/h)³
                    near_density += (1.0 - q) * (1.0 - q) * (1.0 - q);
                }
            }
        }
        
        particle.density = density;
        particle.near_density = near_density;
    }
}

pub fn recompute_velocities(
    time: Res<Time>,
    mut query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    
    for (transform, mut particle) in query.iter_mut() {
        let current_position = transform.translation.truncate();
        // Velocity = (current_position - previous_position) / dt
        particle.velocity = (current_position - particle.previous_position) / dt;
    }
}

pub fn apply_viscosity_paper(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(Entity, &Transform, &mut Particle)>,
) {
    if fluid_params.viscosity_strength <= 0.0 {
        return;
    }
    
    let dt = time.delta_secs();
    let smoothing_radius = fluid_params.smoothing_radius;
    let sigma = fluid_params.viscosity_strength;
    let beta = fluid_params.viscosity_strength * 0.1; // Quadratic term
    
    // Cache data
    let particle_data: Vec<(Entity, Vec2, Vec2)> = particle_query
        .iter()
        .map(|(entity, transform, particle)| 
            (entity, transform.translation.truncate(), particle.velocity))
        .collect();
    
    let mut position_map = std::collections::HashMap::with_capacity(particle_data.len());
    let mut velocity_map = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity, position, velocity) in &particle_data {
        position_map.insert(entity, position);
        velocity_map.insert(entity, velocity);
    }
    
    let mut velocity_changes = std::collections::HashMap::with_capacity(particle_data.len());
    
    for &(entity_i, position_i, velocity_i) in &particle_data {
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_i, smoothing_radius);
        
        for neighbor_entity in neighbors {
            if neighbor_entity == entity_i {
                continue;
            }
            
            if let (Some(&position_j), Some(&velocity_j)) = 
                (position_map.get(&neighbor_entity), velocity_map.get(&neighbor_entity)) {
                
                let offset = position_i - position_j;
                let distance = offset.length();
                
                if distance > 0.0 && distance < smoothing_radius {
                    let direction = offset / distance;
                    let q = distance / smoothing_radius;
                    
                    // Inward radial velocity
                    let u = (velocity_i - velocity_j).dot(direction);
                    
                    if u > 0.0 {
                        // Paper's viscosity impulse: I = Δt(1-q)(σu + βu²)
                        let impulse_magnitude = dt * (1.0 - q) * (sigma * u + beta * u * u);
                        let impulse = direction * impulse_magnitude;
                        
                        // Apply impulse to both particles
                        *velocity_changes.entry(entity_i).or_insert(Vec2::ZERO) -= impulse * 0.5;
                        *velocity_changes.entry(neighbor_entity).or_insert(Vec2::ZERO) += impulse * 0.5;
                    }
                }
            }
        }
    }
    
    // Apply velocity changes
    for (entity, _, mut particle) in particle_query.iter_mut() {
        if let Some(&velocity_change) = velocity_changes.get(&entity) {
            particle.velocity += velocity_change;
        }
    }
}

pub fn update_spatial_hash(
    fluid_params: Res<FluidParams>,
    mut spatial_hash: ResMut<SpatialHashResource>,
    particle_query: Query<(Entity, &Transform), With<Particle>>,
) {
    // Check if the smoothing radius has changed and recreate spatial hash if needed
    let current_cell_size = spatial_hash.spatial_hash.cell_size;
    let new_radius = fluid_params.smoothing_radius;
    
    // Use a more reasonable threshold (0.1) and always recreate if there's a significant difference
    if (new_radius - current_cell_size).abs() > 0.1 {
        spatial_hash.spatial_hash = SpatialHash::new(new_radius);
    } else {
        // Just clear the existing hash without recreating
        spatial_hash.spatial_hash.clear();
    }
    
    // Repopulate the spatial hash with current particle positions
    for (entity, transform) in particle_query.iter() {
        spatial_hash.spatial_hash.insert(transform.translation.truncate(), entity);
    }
}

pub fn handle_collisions(
    fluid_params: Res<FluidParams>,
    mut query: Query<(&mut Transform, &mut Particle)>,
) {
    let min_bounds = fluid_params.boundary_min;
    let max_bounds = fluid_params.boundary_max;
    
    // Only handle boundary collisions, like Unity does
    for (mut transform, mut particle) in query.iter_mut() {
        let pos = &mut transform.translation;
        
        // Handle boundary collisions with damping
        if pos.x < min_bounds.x + PARTICLE_RADIUS {
            pos.x = min_bounds.x + PARTICLE_RADIUS;
            particle.velocity.x = -particle.velocity.x * BOUNDARY_DAMPENING;
        } else if pos.x > max_bounds.x - PARTICLE_RADIUS {
            pos.x = max_bounds.x - PARTICLE_RADIUS;
            particle.velocity.x = -particle.velocity.x * BOUNDARY_DAMPENING;
        }
        
        if pos.y < min_bounds.y + PARTICLE_RADIUS {
            pos.y = min_bounds.y + PARTICLE_RADIUS;
            particle.velocity.y = -particle.velocity.y * BOUNDARY_DAMPENING;
        } else if pos.y > max_bounds.y - PARTICLE_RADIUS {
            pos.y = max_bounds.y - PARTICLE_RADIUS;
            particle.velocity.y = -particle.velocity.y * BOUNDARY_DAMPENING;
        }
    }
}

pub fn update_particle_colors(
    particles_query: Query<(&Particle, &MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    color_params: Res<ColorMapParams>,
) {
    for (particle, mesh_material) in particles_query.iter() {
        let color = if color_params.use_velocity_color {
            // Use velocity-based color mapping
            velocity_to_color(particle.velocity, color_params.min_speed, color_params.max_speed)
        } else {
            // Fallback to density-based coloring (existing behavior)
            let normalized_density = (particle.density / REST_DENSITY).clamp(0.0, 3.0) / 3.0;
            Color::srgb(
                normalized_density,
                0.5 + normalized_density * 0.5,
                1.0
            )
        };
        
        // Update material color
        if let Some(material) = materials.get_mut(&mesh_material.0) {
            material.color = color;
        }
    }
}

pub fn track_max_velocity(
    mut perf_stats: ResMut<GpuPerformanceStats>,
    particles: Query<&Particle>,
) {
    let mut max_velocity = 0.0;
    
    for particle in particles.iter() {
        let velocity_magnitude = particle.velocity.length();
        if velocity_magnitude > max_velocity {
            max_velocity = velocity_magnitude;
        }
    }
    
    // Update maximum velocity
    perf_stats.max_velocity = max_velocity;
    
    // Adjust iterations based on velocity if adaptive iterations is enabled
    if perf_stats.adaptive_iterations {
        // Scale iterations based on maximum velocity
        // Higher velocities need more iterations for stability
        let base_iterations = perf_stats.base_iterations;
        
        let velocity_scale = if max_velocity > 0.0 {
            let normalized_velocity = (max_velocity / 500.0).clamp(1.0, 3.0);
            perf_stats.velocity_iteration_scale * normalized_velocity
        } else {
            1.0
        };
        
        // Cap min/max iterations based on performance
        if perf_stats.avg_frame_time > 20.0 { // < 50 FPS
            perf_stats.iterations_per_frame = 2; // Force lower iterations for performance
        } else {
            perf_stats.iterations_per_frame = (base_iterations as f32 * velocity_scale).max(1.0) as u32;
        }
    }
}

pub fn handle_mouse_input_2d(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
) {
    // Handle mouse interaction (disabled when Draw Lake mode is active)
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
                    mouse_interaction.position = world_position;
                    mouse_interaction.active = mouse_buttons.pressed(MouseButton::Left) || 
                                                mouse_buttons.pressed(MouseButton::Right);
                    mouse_interaction.repel = mouse_buttons.pressed(MouseButton::Right);
                }
            }
        }
    }
}