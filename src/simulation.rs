use bevy::prelude::*;
use crate::math::FluidMath;
use crate::spatial_hash::SpatialHash;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FluidParams>()
           .init_resource::<MouseInteraction>()
           .init_resource::<SpatialHashResource>()
           .add_systems(Startup, setup_simulation)
           .add_systems(Update, handle_input)
           .add_systems(Update, apply_external_forces)
           .add_systems(Update, update_spatial_hash)
           .add_systems(Update, calculate_density)
           .add_systems(Update, calculate_pressure_force)
           .add_systems(Update, calculate_viscosity)
           .add_systems(Update, update_positions)
           .add_systems(Update, handle_collisions)
           .add_systems(Update, update_sprite_colors);
    }
}

// Constants
const GRAVITY: Vec2 = Vec2::new(0.0, -9.81);
const BOUNDARY_DAMPENING: f32 = 0.3;
const PARTICLE_RADIUS: f32 = 5.0;
const REST_DENSITY: f32 = 1000.0;

// Components
#[derive(Component)]
pub struct Particle {
    pub velocity: Vec2,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
}

// Resources
#[derive(Resource)]
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
            smoothing_radius: 35.0,
            target_density: REST_DENSITY,
            pressure_multiplier: 200.0,
            near_pressure_multiplier: 30.0,
            viscosity_strength: 0.1,
            boundary_min: Vec2::new(-300.0, -300.0),
            boundary_max: Vec2::new(300.0, 300.0),
        }
    }
}

#[derive(Resource)]
struct MouseInteraction {
    position: Vec2,
    active: bool,
    repel: bool,
    strength: f32,
    radius: f32,
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
struct SpatialHashResource {
    spatial_hash: SpatialHash,
}

impl Default for SpatialHashResource {
    fn default() -> Self {
        Self {
            spatial_hash: SpatialHash::new(35.0),
        }
    }
}

// Systems
fn setup_simulation(mut commands: Commands) {
    // Set up UI
    commands.spawn((
        Text::new("Fluid Simulation (Bevy 0.16)"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));

    // Performance monitor
    commands.spawn((
        Text::new("FPS: --"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(30.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
) {
    // Handle mouse interaction
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

    // Toggle force strength with number keys
    if keys.just_pressed(KeyCode::Digit1) {
        mouse_interaction.strength = 1000.0;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = 2000.0;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = 3000.0;
    }
}

fn apply_external_forces(
    time: Res<Time>,
    mouse_interaction: Res<MouseInteraction>,
    _params: Res<FluidParams>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    
    for (transform, mut particle) in particle_query.iter_mut() {
        // Apply gravity
        particle.velocity += GRAVITY * dt;
        
        // Apply mouse force if active
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

fn update_spatial_hash(
    mut spatial_hash: ResMut<SpatialHashResource>,
    particle_query: Query<(Entity, &Transform), With<Particle>>,
) {
    spatial_hash.spatial_hash.clear();
    
    for (entity, transform) in particle_query.iter() {
        spatial_hash.spatial_hash.insert(transform.translation.truncate(), entity);
    }
}

fn calculate_density(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    mut particle_query: Query<(Entity, &Transform, &mut Particle)>,
) {
    let smoothing_radius = fluid_params.smoothing_radius;
    let smoothing_radius_squared = smoothing_radius * smoothing_radius;
    let math = FluidMath::new(smoothing_radius);

    // First pass: collect positions to avoid borrow checker issues
    let positions: Vec<(Entity, Vec2)> = particle_query
        .iter()
        .map(|(entity, transform, _)| (entity, transform.translation.truncate()))
        .collect();
        
    // Second pass: perform density calculations
    for (entity_a, position_a) in &positions {
        let neighbors = spatial_hash.spatial_hash.get_neighbors(*position_a, smoothing_radius);
        
        // Initial values
        let mut density = math.poly6(0.0, smoothing_radius_squared); // Self-contribution
        let mut near_density = 0.0;
        
        // Calculate contributions from neighbors
        for neighbor_entity in &neighbors {
            // Find the neighbor's position from our positions list
            if let Some((_, position_b)) = positions.iter().find(|(e, _)| e == neighbor_entity) {
                let offset = *position_b - *position_a;
                let distance_squared = offset.length_squared();
                
                if distance_squared < smoothing_radius_squared {
                    let distance = distance_squared.sqrt();
                    
                    // Add density contribution
                    density += math.poly6(distance_squared, smoothing_radius_squared);
                    
                    // Near density for surface tension
                    if distance > 0.0 {
                        near_density += math.spiky_pow2(distance, smoothing_radius);
                    }
                }
            }
        }
        
        // Update the particle with calculated densities
        if let Ok((_, _, mut particle)) = particle_query.get_mut(*entity_a) {
            particle.density = density;
            particle.near_density = near_density;
            
            // Calculate pressure from density
            let density_error = density - fluid_params.target_density;
            particle.pressure = density_error * fluid_params.pressure_multiplier;
            particle.near_pressure = near_density * fluid_params.near_pressure_multiplier;
        }
    }
}

fn calculate_pressure_force(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    let smoothing_radius = fluid_params.smoothing_radius;
    let math = FluidMath::new(smoothing_radius);
    
    let mut pressure_forces: Vec<Vec2> = vec![Vec2::ZERO; particle_query.iter().len()];
    
    // Calculate pressure forces between all particles
    for (i, (transform_a, particle_a)) in particle_query.iter().enumerate() {
        let position_a = transform_a.translation.truncate();
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_a, smoothing_radius);
        
        for neighbor_entity in neighbors {
            if let Ok((transform_b, particle_b)) = particle_query.get(neighbor_entity) {
                let position_b = transform_b.translation.truncate();
                let offset = position_a - position_b;
                let distance_squared = offset.length_squared();
                
                if distance_squared > 0.0 && distance_squared < smoothing_radius * smoothing_radius {
                    let distance = distance_squared.sqrt();
                    let direction = offset / distance;
                    
                    // Pressure force calculation based on both pressure values
                    let shared_pressure = (particle_a.pressure + particle_b.pressure) * 0.5;
                    let shared_near_pressure = (particle_a.near_pressure + particle_b.near_pressure) * 0.5;
                    
                    let pressure_force = direction * 
                        (math.spiky_pow3_derivative(distance, smoothing_radius) * shared_pressure +
                         math.spiky_pow2_derivative(distance, smoothing_radius) * shared_near_pressure);
                    
                    pressure_forces[i] += pressure_force;
                }
            }
        }
    }
    
    // Apply the pressure forces
    for (i, (_, mut particle)) in particle_query.iter_mut().enumerate() {
        particle.velocity += pressure_forces[i] * dt;
    }
}

fn calculate_viscosity(
    fluid_params: Res<FluidParams>,
    spatial_hash: Res<SpatialHashResource>,
    time: Res<Time>,
    mut particle_query: Query<(&Transform, &mut Particle)>,
) {
    let dt = time.delta_secs();
    let smoothing_radius = fluid_params.smoothing_radius;
    let viscosity_strength = fluid_params.viscosity_strength;
    let math = FluidMath::new(smoothing_radius);
    
    // Process viscosity (velocity damping)
    let mut velocity_changes: Vec<Vec2> = vec![Vec2::ZERO; particle_query.iter().len()];
    
    for (i, (transform_a, particle_a)) in particle_query.iter().enumerate() {
        let position_a = transform_a.translation.truncate();
        let neighbors = spatial_hash.spatial_hash.get_neighbors(position_a, smoothing_radius);
        
        for neighbor_entity in neighbors {
            if let Ok((transform_b, particle_b)) = particle_query.get(neighbor_entity) {
                let position_b = transform_b.translation.truncate();
                let offset = position_a - position_b;
                let distance_squared = offset.length_squared();
                
                if distance_squared > 0.0 && distance_squared < smoothing_radius * smoothing_radius {
                    let distance = distance_squared.sqrt();
                    
                    // Viscosity is based on the velocity difference
                    let velocity_diff = particle_b.velocity - particle_a.velocity;
                    let influence = math.spiky_pow3(distance, smoothing_radius) * viscosity_strength;
                    velocity_changes[i] += velocity_diff * influence;
                }
            }
        }
    }
    
    // Apply the viscosity forces
    for (i, (_, mut particle)) in particle_query.iter_mut().enumerate() {
        particle.velocity += velocity_changes[i] * dt;
    }
}

fn update_positions(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Particle)>,
) {
    let dt = time.delta_secs();
    
    for (mut transform, particle) in query.iter_mut() {
        transform.translation += Vec3::new(particle.velocity.x, particle.velocity.y, 0.0) * dt;
    }
}

fn handle_collisions(
    fluid_params: Res<FluidParams>,
    mut query: Query<(&mut Transform, &mut Particle)>,
) {
    let min_bounds = fluid_params.boundary_min;
    let max_bounds = fluid_params.boundary_max;
    
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

fn update_sprite_colors(
    mut query: Query<(&Particle, &mut Sprite)>,
) {
    for (particle, mut sprite) in query.iter_mut() {
        // Normalize the density
        let normalized_density = (particle.density / REST_DENSITY).clamp(0.0, 3.0) / 3.0;
        
        // Create a color gradient from blue to cyan to white based on density
        let color = Color::srgb(
            normalized_density,
            0.5 + normalized_density * 0.5,
            1.0
        );
        
        // Update sprite color
        sprite.color = color;
    }
} 