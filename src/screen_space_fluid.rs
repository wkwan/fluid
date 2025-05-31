use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::pbr::{MaterialPlugin, Material};
use crate::simulation3d::Particle3D;
use crate::simulation::Particle;
use crate::simulation::SimulationDimension;

pub struct ScreenSpaceFluidPlugin;

impl Plugin for ScreenSpaceFluidPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenSpaceFluidSettings>()
            // Temporarily remove custom material plugin until we fix shader bindings
            // .add_plugins(MaterialPlugin::<ScreenSpaceFluidMaterial>::default())
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

// Enhanced screen space fluid material that supports both billboard and depth modes
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct ScreenSpaceFluidMaterial {
    #[uniform(0)]
    pub base_color: Vec4,
    #[uniform(0)]
    pub particle_scale: f32,
    #[uniform(0)]
    pub depth_mode: f32,
    #[uniform(0)]
    pub _padding1: f32,
    #[uniform(0)]
    pub _padding2: f32,
}

impl Material for ScreenSpaceFluidMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/screen_space_fluid.wgsl".into()
    }
    
    fn vertex_shader() -> ShaderRef {
        "shaders/screen_space_fluid.wgsl".into()
    }
    
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
    
    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline<Self>,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.vertex.entry_point = "vertex".into();
        descriptor.fragment.as_mut().unwrap().entry_point = "fragment".into();
        Ok(())
    }
}

// Rendering mode for screen space fluid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderingMode {
    Billboard,  // Simple billboard circles (current)
    DepthOnly,  // Depth buffer rendering (Phase 1)
    Filtered,   // With bilateral filtering (Phase 2)
    Normals,    // With surface normals (Phase 3)
    FullFluid,  // Complete screen space fluid (Phase 4)
}

// Settings for screen space fluid rendering - completely independent
#[derive(Resource)]
pub struct ScreenSpaceFluidSettings {
    pub enabled: bool,
    pub particle_scale: f32,
    pub color: Color,
    pub alpha: f32,
    pub unlit: bool,
    pub rendering_mode: RenderingMode,
}

impl Default for ScreenSpaceFluidSettings {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default as requested
            particle_scale: 5.0,
            color: Color::srgb(0.3, 0.7, 1.0),
            alpha: 0.8,
            unlit: true,
            rendering_mode: RenderingMode::Billboard, // Start with working mode
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
    
    // Simplified approach - use standard materials for now
    match settings.rendering_mode {
        RenderingMode::DepthOnly => {
            // Use grayscale to represent depth mode
            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.7, 0.7, 0.7).with_alpha(settings.alpha),
                unlit: settings.unlit,
                alpha_mode: AlphaMode::Blend,
                ..default()
            });
            
            // Spawn particles
            for particle_transform in particles.iter() {
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
                    Name::new("Screen Space Fluid Particle - Depth"),
                ));
            }
        }
        _ => {
            // Billboard mode - normal colored particles
            let material = materials.add(StandardMaterial {
                base_color: settings.color.with_alpha(settings.alpha),
                unlit: settings.unlit,
                alpha_mode: AlphaMode::Blend,
                ..default()
            });
            
            // Spawn particles
            for particle_transform in particles.iter() {
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
                    Name::new("Screen Space Fluid Particle - Billboard"),
                ));
            }
        }
    }
    
    info!("Screen space rendering: {} particles in {:?} mode", particle_count, settings.rendering_mode);
}

