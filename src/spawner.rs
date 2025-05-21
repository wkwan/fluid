use bevy::prelude::*;
use rand::Rng;

use crate::simulation::{Particle, FluidParams};

pub struct SpawnerPlugin;

impl Plugin for SpawnerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SpawnRegions>()
           .add_systems(Startup, spawn_particles)
           .add_systems(Update, handle_spawn_input);
    }
}

#[derive(Clone, Resource)]
pub struct SpawnRegions {
    pub regions: Vec<SpawnRegion>,
    pub spawn_density: f32,
    pub initial_velocity: Vec2,
    pub jitter_strength: f32,
}

impl Default for SpawnRegions {
    fn default() -> Self {
        Self {
            regions: vec![SpawnRegion {
                position: Vec2::new(0.0, 100.0),
                size: Vec2::new(300.0, 200.0),
                debug_color: Color::srgb(0.0, 0.0, 1.0),
            }],
            spawn_density: 6.0,
            initial_velocity: Vec2::new(0.0, -50.0),
            jitter_strength: 10.0,
        }
    }
}

#[derive(Clone)]
pub struct SpawnRegion {
    pub position: Vec2,
    pub size: Vec2,
    pub debug_color: Color,
}

impl SpawnRegion {
    pub fn random_position_within(&self, rng: &mut impl Rng) -> Vec2 {
        let half_size = self.size * 0.5;
        let offset = Vec2::new(
            rng.gen_range(-half_size.x..half_size.x),
            rng.gen_range(-half_size.y..half_size.y),
        );
        self.position + offset
    }
}

fn handle_spawn_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    spawn_regions: Res<SpawnRegions>,
    _fluid_params: Res<FluidParams>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if keys.just_pressed(KeyCode::Enter) {
        let mut rng = rand::thread_rng();
        let spawned_particles = spawn_in_regions(&spawn_regions, &mut rng);
        
        // Create a shared circle mesh
        let circle_mesh = meshes.add(Circle::new(5.0));
        
        for (position, velocity) in spawned_particles {
            // Create a unique material for each particle
            let particle_material = materials.add(Color::srgb(0.0, 0.5, 0.9));
            
            // Spawn with proper mesh and material handles
            commands.spawn((
                Mesh2d(circle_mesh.clone()),
                MeshMaterial2d(particle_material),
                Transform::from_translation(position.extend(0.0)),
                GlobalTransform::default(),
                Visibility::default(),
                ViewVisibility::default(),
                InheritedVisibility::default(),
                Particle { 
                    velocity, 
                    density: 0.0,
                    pressure: 0.0,
                    near_density: 0.0,
                    near_pressure: 0.0,
                },
            ));
        }
    }
}

fn spawn_particles(
    mut commands: Commands,
    spawn_regions: Res<SpawnRegions>,
    fluid_params: Res<FluidParams>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let mut rng = rand::thread_rng();
    
    // Visualize spawn regions
    for region in &spawn_regions.regions {
        commands.spawn((
            Sprite {
                color: region.debug_color.with_alpha(0.2),
                custom_size: Some(region.size),
                ..default()
            },
            Transform::from_translation(region.position.extend(-0.1)),
            GlobalTransform::default(),
            Visibility::default(),
            ViewVisibility::default(),
            InheritedVisibility::default(),
        ));
    }
    
    // Visualize simulation boundaries
    let boundary_min = fluid_params.boundary_min;
    let boundary_max = fluid_params.boundary_max;
    let boundary_width = boundary_max.x - boundary_min.x;
    let boundary_height = boundary_max.y - boundary_min.y;
    let boundary_center = (boundary_min + boundary_max) / 2.0;
    
    // Create boundary rectangle
    commands.spawn((
        Sprite {
            color: Color::WHITE,
            custom_size: Some(Vec2::new(boundary_width, boundary_height)),
            ..default()
        },
        Transform::from_translation(boundary_center.extend(-0.2)),
        GlobalTransform::default(),
        Visibility::default(),
        ViewVisibility::default(),
        InheritedVisibility::default(),
    ));
    
    // Create shared circle mesh
    let circle_mesh = meshes.add(Circle::new(5.0));
    
    // Spawn initial particles
    let spawned_particles = spawn_in_regions(&spawn_regions, &mut rng);
    
    for (position, velocity) in spawned_particles {
        // Create a unique material for each particle
        let particle_material = materials.add(Color::srgb(0.0, 0.3, 1.0));
        
        // Spawn with proper mesh and material handles
        commands.spawn((
            Mesh2d(circle_mesh.clone()),
            MeshMaterial2d(particle_material),
            Transform::from_translation(position.extend(0.0)),
            GlobalTransform::default(),
            Visibility::default(),
            ViewVisibility::default(),
            InheritedVisibility::default(),
            Particle {
                velocity,
                density: 0.0,
                pressure: 0.0,
                near_density: 0.0,
                near_pressure: 0.0,
            },
        ));
    }
}

fn spawn_in_regions(
    spawn_regions: &SpawnRegions,
    rng: &mut impl Rng,
) -> Vec<(Vec2, Vec2)> {
    let mut spawned_particles = Vec::new();
    
    for region in &spawn_regions.regions {
        let area = region.size.x * region.size.y;
        let particle_size = 10.0; // Particle diameter
        let max_particles = (area / (particle_size * particle_size) * spawn_regions.spawn_density) as usize;
        
        for _ in 0..max_particles {
            let position = region.random_position_within(rng);
            
            // Add some random jitter to the initial velocity
            let jitter = Vec2::new(
                rng.gen_range(-spawn_regions.jitter_strength..spawn_regions.jitter_strength),
                rng.gen_range(-spawn_regions.jitter_strength..spawn_regions.jitter_strength),
            );
            let velocity = spawn_regions.initial_velocity + jitter;
            
            spawned_particles.push((position, velocity));
        }
    }
    
    spawned_particles
} 