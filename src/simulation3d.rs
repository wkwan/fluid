use bevy::prelude::*;
use bevy::math::primitives::Sphere;
use crate::math::FluidMath3D;
use crate::simulation::SimulationDimension;

// 3D particle component
#[derive(Component)]
pub struct Particle3D {
    pub velocity: Vec3,
    pub density: f32,
    pub pressure: f32,
    pub near_density: f32,
    pub near_pressure: f32,
}

// Marker for 3D entities to allow cleanup
#[derive(Component)]
pub struct Marker3D;

// Constants for 3D sim (match 2D values where possible)
const GRAVITY_3D: Vec3 = Vec3::new(0.0, -9.81, 0.0);
const BOUNDARY_MIN: Vec3 = Vec3::new(-300.0, -300.0, -300.0);
const BOUNDARY_MAX: Vec3 = Vec3::new(300.0, 300.0, 300.0);
const PARTICLE_RADIUS: f32 = 5.0;
const BOUNDARY_DAMPENING: f32 = 0.3;

// ======================== SETUP ============================
pub fn setup_3d_environment(
    mut commands: Commands,
    _asset_server: Res<AssetServer>,
    query_cam: Query<(), With<Camera3d>>, // only spawn if none
    sim_dim: Res<SimulationDimension>,
) {
    if *sim_dim != SimulationDimension::Dim3 {
        return;
    }

    if !query_cam.is_empty() {
        return;
    }

    // Add a basic directional light so we can see the spheres
    commands.spawn((
        DirectionalLight {
            shadows_enabled: false,
            illuminance: 20000.0,
            ..default()
        },
        Transform::from_xyz(0.0, 300.0, 300.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
        Marker3D,
    ));
}

// ======================== SPAWNER ==========================
pub fn spawn_particles_3d(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sim_dim: Res<SimulationDimension>,
    existing: Query<(), With<Particle3D>>,
) {
    if *sim_dim != SimulationDimension::Dim3 {
        return;
    }

    // Only spawn once (when no particles exist)
    if !existing.is_empty() {
        return;
    }

    const GRID: i32 = 15; // 15^3 ≈ 3.3k particles
    let spacing = PARTICLE_RADIUS * 2.5; // slightly larger gap to avoid excessive overlap

    let start_x = -((GRID as f32 - 1.0) * spacing) * 0.5;
    let start_y = start_x;
    let start_z = start_x;

    // Create shared mesh & material
    let sphere_mesh = meshes.add(
        Sphere::new(PARTICLE_RADIUS)
            .mesh()
            .ico(2)
            .unwrap(),
    );
    let base_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.4, 1.0),
        perceptual_roughness: 0.8,
        ..default()
    });

    for xi in 0..GRID {
        for yi in 0..GRID {
            for zi in 0..GRID {
                let pos = Vec3::new(
                    start_x + xi as f32 * spacing,
                    start_y + yi as f32 * spacing + 100.0, // elevate a bit for nice fall
                    start_z + zi as f32 * spacing,
                );

                commands.spawn((
                    Mesh3d(sphere_mesh.clone()),
                    MeshMaterial3d(base_material.clone()),
                    Transform::from_translation(pos),
                    Particle3D {
                        velocity: Vec3::ZERO,
                        density: 0.0,
                        pressure: 0.0,
                        near_density: 0.0,
                        near_pressure: 0.0,
                    },
                    Marker3D,
                ));
            }
        }
    }
}

// =================== PHYSICS SYSTEMS =======================

pub fn apply_external_forces_3d(
    time: Res<Time>,
    mut particles: Query<&mut Particle3D>,
    sim_dim: Res<SimulationDimension>,
) {
    if *sim_dim != SimulationDimension::Dim3 {
        return;
    }

    let dt = time.delta_secs();
    for mut p in particles.iter_mut() {
        p.velocity += GRAVITY_3D * dt;
    }
}

pub fn calculate_density_pressure_3d(
    mut particles_q: Query<(Entity, &Transform, &mut Particle3D)>,
    sim_dim: Res<SimulationDimension>,
) {
    if *sim_dim != SimulationDimension::Dim3 {
        return;
    }

    let smoothing_radius: f32 = 35.0; // TODO param
    let smoothing_radius_squared = smoothing_radius * smoothing_radius;
    let math = FluidMath3D::new(smoothing_radius);
    let target_density = 1000.0;
    let pressure_mult = 200.0;

    // Cache positions and store entities order
    let mut positions: Vec<Vec3> = Vec::with_capacity(particles_q.iter().count());
    let mut entities: Vec<Entity> = Vec::with_capacity(positions.capacity());

    for (e, t, _) in particles_q.iter() {
        entities.push(e);
        positions.push(t.translation);
    }

    let count = positions.len();
    let mut densities = vec![0.0f32; count];

    for i in 0..count {
        let pos_i = positions[i];
        let mut density = 0.0;
        for j in 0..count {
            let pos_j = positions[j];
            let r2 = (pos_i - pos_j).length_squared();
            density += math.poly6(r2, smoothing_radius_squared);
        }
        densities[i] = density;
    }

    // Write back density/pressure
    for (idx, entity) in entities.iter().enumerate() {
        if let Ok((_, _, mut part)) = particles_q.get_mut(*entity) {
            let density = densities[idx];
            part.density = density;
            part.pressure = (density - target_density) * pressure_mult;
        }
    }
}

pub fn integrate_positions_3d(
    time: Res<Time>,
    mut particles: Query<(&mut Transform, &mut Particle3D)>,
    sim_dim: Res<SimulationDimension>,
) {
    if *sim_dim != SimulationDimension::Dim3 {
        return;
    }
    let dt = time.delta_secs();
    for (mut transform, mut particle) in particles.iter_mut() {
        transform.translation += particle.velocity * dt;

        // Boundary collisions simple
        let mut pos = transform.translation;
        if pos.x < BOUNDARY_MIN.x + PARTICLE_RADIUS {
            pos.x = BOUNDARY_MIN.x + PARTICLE_RADIUS;
            particle.velocity.x = -particle.velocity.x * BOUNDARY_DAMPENING;
        } else if pos.x > BOUNDARY_MAX.x - PARTICLE_RADIUS {
            pos.x = BOUNDARY_MAX.x - PARTICLE_RADIUS;
            particle.velocity.x = -particle.velocity.x * BOUNDARY_DAMPENING;
        }
        if pos.y < BOUNDARY_MIN.y + PARTICLE_RADIUS {
            pos.y = BOUNDARY_MIN.y + PARTICLE_RADIUS;
            particle.velocity.y = -particle.velocity.y * BOUNDARY_DAMPENING;
        } else if pos.y > BOUNDARY_MAX.y - PARTICLE_RADIUS {
            pos.y = BOUNDARY_MAX.y - PARTICLE_RADIUS;
            particle.velocity.y = -particle.velocity.y * BOUNDARY_DAMPENING;
        }
        if pos.z < BOUNDARY_MIN.z + PARTICLE_RADIUS {
            pos.z = BOUNDARY_MIN.z + PARTICLE_RADIUS;
            particle.velocity.z = -particle.velocity.z * BOUNDARY_DAMPENING;
        } else if pos.z > BOUNDARY_MAX.z - PARTICLE_RADIUS {
            pos.z = BOUNDARY_MAX.z - PARTICLE_RADIUS;
            particle.velocity.z = -particle.velocity.z * BOUNDARY_DAMPENING;
        }
        transform.translation = pos;
    }
} 