use bevy::prelude::*;
use crate::simulation3d::Particle3D;
use crate::simulation::Particle;
use crate::simulation::SimulationDimension;

pub struct ScreenSpaceFluidPlugin;

impl Plugin for ScreenSpaceFluidPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenSpaceFluidSettings>()
            .add_systems(Startup, setup_screen_space_resources)
            .add_systems(Update, render_screen_space_fluid_system
                .run_if(|settings: Res<ScreenSpaceFluidSettings>, sim_dim: Res<State<SimulationDimension>>| 
                    settings.enabled && *sim_dim.get() == SimulationDimension::Dim3)
            )
            .add_systems(Update, cleanup_screen_space_system
                .run_if(|settings: Res<ScreenSpaceFluidSettings>, sim_dim: Res<State<SimulationDimension>>| 
                    (!settings.enabled && *sim_dim.get() == SimulationDimension::Dim3) || 
                    *sim_dim.get() == SimulationDimension::Dim2)
            );
    }
}

#[derive(Component)]
pub struct ScreenSpaceFluid;

#[derive(Component)]
pub struct ScreenSpaceFluidMesh;

// Settings for screen space fluid rendering - completely independent
#[derive(Resource)]
pub struct ScreenSpaceFluidSettings {
    pub enabled: bool,
    pub particle_scale: f32,
    pub color: Color,
    pub alpha: f32,
    pub unlit: bool,
}

impl Default for ScreenSpaceFluidSettings {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default as requested
            particle_scale: 5.0,
            color: Color::srgb(0.3, 0.7, 1.0),
            alpha: 0.8,
            unlit: true,
        }
    }
}

// Setup resources for screen space rendering
fn setup_screen_space_resources(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // Create circle mesh for particles
    let circle_mesh = meshes.add(Circle::new(1.0));
    commands.insert_resource(CircleMesh(circle_mesh));
}

#[derive(Resource)]
pub struct CircleMesh(pub Handle<Mesh>);

/// Helper function to despawn entities from a query
fn despawn_entities<T: Component>(commands: &mut Commands, query: &Query<Entity, With<T>>) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

// System to clean up screen space entities when disabled
fn cleanup_screen_space_system(
    mut commands: Commands,
    existing_screen_space: Query<Entity, With<ScreenSpaceFluid>>,
) {
    despawn_entities(&mut commands, &existing_screen_space);
}

// Main rendering system - completely independent of ray marching
pub fn render_screen_space_fluid_system(
    particles: Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mut commands: Commands,
    existing_screen_space: Query<Entity, With<ScreenSpaceFluid>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    circle_mesh: Option<Res<CircleMesh>>,
    settings: Res<ScreenSpaceFluidSettings>,
    camera_3d: Query<&Transform, With<Camera3d>>,
    sim_dim: Res<State<SimulationDimension>>,
) {
    // Only proceed if 3D mode and screen space rendering is enabled
    if *sim_dim.get() != SimulationDimension::Dim3 || !settings.enabled {
        return;
    }

    // Remove existing screen space entities
    despawn_entities(&mut commands, &existing_screen_space);
    
    let particle_count = particles.iter().count();
    if particle_count == 0 {
        return;
    }
    
    // Get camera transform for billboarding
    let camera_transform = if let Ok(cam) = camera_3d.single() {
        cam
    } else {
        return; // No camera, can't render
    };
    
    // If we don't have the circle mesh resource, create a temporary one
    let mesh_handle = if let Some(circle) = circle_mesh {
        circle.0.clone()
    } else {
        meshes.add(Circle::new(1.0))
    };
    
    // Create material based on settings
    let material = materials.add(StandardMaterial {
        base_color: settings.color.with_alpha(settings.alpha),
        unlit: settings.unlit,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    
    // Spawn individual entities for each particle as billboards
    for particle_transform in particles.iter() {
        // Calculate billboard rotation to face camera
        let look_at = camera_transform.translation - particle_transform.translation;
        let rotation = Transform::from_translation(Vec3::ZERO)
            .looking_at(-look_at, Vec3::Y)
            .rotation;
        
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(particle_transform.translation)
                .with_rotation(rotation)
                .with_scale(Vec3::splat(settings.particle_scale)),
            ScreenSpaceFluid,
            ScreenSpaceFluidMesh,
            Name::new("Screen Space Fluid Particle"),
        ));
    }
    
    info!("Screen space rendering: {} particles as billboards", particle_count);
}

// Legacy function for backward compatibility - now just calls the system
pub fn render_screen_space_fluid(
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    commands: &mut Commands,
    existing_screen_space: &Query<Entity, With<ScreenSpaceFluid>>,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    circle_mesh: Option<&Res<CircleMesh>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_3d: &Query<&Transform, With<Camera3d>>,
) {
    // Remove existing screen space entities
    despawn_entities(commands, existing_screen_space);
    
    let particle_count = particles.iter().count();
    if particle_count == 0 {
        return;
    }
    
    // Get camera transform for billboarding
    let camera_transform = if let Ok(cam) = camera_3d.single() {
        cam
    } else {
        return; // No camera, can't render
    };
    
    // If we don't have the circle mesh resource, create a temporary one
    let mesh_handle = if let Some(circle) = circle_mesh {
        circle.0.clone()
    } else {
        meshes.add(Circle::new(1.0))
    };
    
    // Create material based on settings
    let material = materials.add(StandardMaterial {
        base_color: settings.color.with_alpha(settings.alpha),
        unlit: settings.unlit,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    
    // Spawn individual entities for each particle as billboards
    for particle_transform in particles.iter() {
        // Calculate billboard rotation to face camera
        let look_at = camera_transform.translation - particle_transform.translation;
        let rotation = Transform::from_translation(Vec3::ZERO)
            .looking_at(-look_at, Vec3::Y)
            .rotation;
        
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(particle_transform.translation)
                .with_rotation(rotation)
                .with_scale(Vec3::splat(settings.particle_scale)),
            ScreenSpaceFluid,
            ScreenSpaceFluidMesh,
            Name::new("Screen Space Fluid Particle"),
        ));
    }
    
    info!("Screen space rendering: {} particles as billboards", particle_count);
} 