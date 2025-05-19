use bevy::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use bevy::sprite::{Wireframe2dConfig, Wireframe2dPlugin};
use bevy::pbr::wireframe::Wireframe;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::PrimitiveTopology;
use std::f32::consts::PI;
use rand::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        #[cfg(not(target_arch = "wasm32"))]
        Wireframe2dPlugin::default(),
    ))
    .add_systems(Startup, setup)
    .add_systems(Update, (update_particles, toggle_wireframe));
    app.run();
}

const GRAVITY: f32 = 300.0;
const BOX_SIZE: Vec2 = Vec2::new(800.0, 600.0);
const COLLISION_DAMPING: f32 = 0.8;
const PARTICLE_SIZE: f32 = 10.0;
const PARTICLE_MASS: f32 = 1.0;

#[derive(Component)]
struct Particle {
    velocity: Vec2
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        Camera2d,
        Camera::default()
    ));

    // Add border box
    let half_size = BOX_SIZE / 2.0;
    let vertices = vec![
        Vec3::new(-half_size.x, -half_size.y, 0.0),
        Vec3::new(half_size.x, -half_size.y, 0.0),
        Vec3::new(half_size.x, half_size.y, 0.0),
        Vec3::new(-half_size.x, half_size.y, 0.0),
    ];
    
    let mut mesh = Mesh::new(PrimitiveTopology::LineStrip, bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(Indices::U32(vec![0, 1, 2, 3, 0]));

    commands.spawn((
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(materials.add(Color::WHITE)),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Spawn 1000 random particles
    let mut rng = rand::rng();
    let particle_count = 1000;
    
    for _ in 0..particle_count {
        let pos = Vec2::new(
            rng.gen_range(-half_size.x + PARTICLE_SIZE..half_size.x - PARTICLE_SIZE),
            rng.gen_range(-half_size.y + PARTICLE_SIZE..half_size.y - PARTICLE_SIZE)
        );
        
        commands.spawn((
            Mesh2d(meshes.add(Circle::new(PARTICLE_SIZE))),
            MeshMaterial2d(materials.add(Color::hsl(223., 0.95, 0.7))),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Particle {
                velocity: Vec2::new(0.0, 0.0),
            }
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    commands.spawn((
        Text::new("Press space to toggle wireframes"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn smoothing_kernel(radius: f32, dist: f32) -> f32 {
    let volume = PI * radius.powi(8) / 4.0;
    let value = f32::max(0.0, radius * radius - dist * dist);
    value * value * value / volume
}

fn calculate_density(sample_point: Vec2, mut particles: Query<(&mut Particle, &mut Transform)>) -> f32 {
    let mut density = 0.0;

    for (particle, transform) in &particles {
        let dist = (sample_point - transform.translation.xy()).length();
        let influence = smoothing_kernel(PARTICLE_SIZE, dist);
        density += PARTICLE_MASS * influence;
    }
    density
}

fn update_particles(time: Res<Time>, mut particles: Query<(&mut Particle, &mut Transform)>) {
    for (mut particle, mut transform) in &mut particles {
        let half_size = BOX_SIZE / 2.0;
        
        // Update position
        particle.velocity.y -= GRAVITY * time.delta_secs();
        transform.translation.y += particle.velocity.y * time.delta_secs();
        transform.translation.x += particle.velocity.x * time.delta_secs();
        
        // Handle collisions
        if transform.translation.x.abs() > half_size.x {
            transform.translation.x = transform.translation.x.signum() * half_size.x;
            particle.velocity.x *= -COLLISION_DAMPING;
        }
        if transform.translation.y.abs() > half_size.y {
            transform.translation.y = transform.translation.y.signum() * half_size.y;
            particle.velocity.y *= -COLLISION_DAMPING;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn toggle_wireframe(
    mut wireframe_config: ResMut<Wireframe2dConfig>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        wireframe_config.global = !wireframe_config.global;
    }
}