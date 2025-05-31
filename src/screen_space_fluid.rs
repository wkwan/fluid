use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::pbr::Material;
use crate::simulation3d::Particle3D;
use crate::simulation::Particle;
use crate::simulation::SimulationDimension;

pub struct ScreenSpaceFluidPlugin;

impl Plugin for ScreenSpaceFluidPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenSpaceFluidSettings>()
            // Temporarily remove custom material until we can fix the shader binding issues
            // .add_plugins(MaterialPlugin::<FluidNormalsMaterial>::default())
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
        "shaders/3d/screenspace/screenspace_fluid.wgsl".into()
    }
    
    fn vertex_shader() -> ShaderRef {
        "shaders/3d/screenspace/screenspace_fluid.wgsl".into()
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

// Custom material for normal visualization and lighting
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct FluidNormalsMaterial {
    #[uniform(0)]
    pub base_color: Vec4,
    #[uniform(0)]
    pub particle_scale: f32,
    #[uniform(0)]
    pub _padding1: f32,
    #[uniform(0)]
    pub _padding2: f32,
    #[uniform(0)]
    pub _padding3: f32,
}

impl Material for FluidNormalsMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/screenspace/fluid_normals.wgsl".into()
    }
    
    fn vertex_shader() -> ShaderRef {
        "shaders/screenspace/fluid_normals.wgsl".into()
    }
    
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
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
    // Bilateral filtering parameters
    pub filter_radius: f32,
    pub depth_threshold: f32,
    pub sigma_spatial: f32,
    pub sigma_depth: f32,
    // Full fluid rendering parameters
    pub fluid_transparency: f32,
    pub internal_glow: f32,
    pub volume_scale: f32,
}

impl Default for ScreenSpaceFluidSettings {
    fn default() -> Self {
        Self {
            enabled: true, // Enabled by default as requested
            particle_scale: 5.0,
            color: Color::srgb(0.3, 0.7, 1.0),
            alpha: 0.8,
            unlit: true,
            rendering_mode: RenderingMode::FullFluid, // Show the final result
            // Bilateral filtering defaults
            filter_radius: 3.0,
            depth_threshold: 1.0,
            sigma_spatial: 2.0,
            sigma_depth: 0.5,
            // Full fluid rendering defaults
            fluid_transparency: 0.7,
            internal_glow: 0.3,
            volume_scale: 1.4,
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
    
    // Render particles based on mode
    match settings.rendering_mode {
        RenderingMode::DepthOnly => {
            render_depth_mode_particles(&mut commands, &particles, &mesh_handle, &mut materials, &settings, &camera_transform);
        }
        RenderingMode::Filtered => {
            render_filtered_mode_particles(&mut commands, &particles, &mesh_handle, &mut materials, &settings, &camera_transform);
        }
        RenderingMode::Normals => {
            render_normals_mode_particles_simple(&mut commands, &particles, &mesh_handle, &mut materials, &settings, &camera_transform);
        }
        RenderingMode::FullFluid => {
            render_full_fluid_particles(&mut commands, &particles, &mesh_handle, &mut materials, &settings, &camera_transform);
        }
        _ => {
            render_billboard_mode_particles(&mut commands, &particles, &mesh_handle, &mut materials, &settings, &camera_transform);
        }
    }
    
    info!("Screen space rendering: {} particles in {:?} mode", particle_count, settings.rendering_mode);
}

// Helper function for billboard mode particles
fn render_billboard_mode_particles(
    commands: &mut Commands,
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mesh_handle: &Handle<Mesh>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_transform: &Transform,
) {
    let material = materials.add(StandardMaterial {
        base_color: settings.color.with_alpha(settings.alpha),
        unlit: settings.unlit,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    
    spawn_particles(commands, particles, mesh_handle, material, camera_transform, settings.particle_scale, "Billboard");
}

// Helper function for depth mode particles  
fn render_depth_mode_particles(
    commands: &mut Commands,
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mesh_handle: &Handle<Mesh>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_transform: &Transform,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.7, 0.7).with_alpha(settings.alpha),
        unlit: settings.unlit,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    
    spawn_particles(commands, particles, mesh_handle, material, camera_transform, settings.particle_scale, "Depth");
}

// Helper function for filtered mode particles - simulates bilateral filtering with parameter-driven effects
fn render_filtered_mode_particles(
    commands: &mut Commands,
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mesh_handle: &Handle<Mesh>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_transform: &Transform,
) {
    // Create material with enhanced smoothness based on filter parameters
    let transparency_factor = 1.0 - (settings.filter_radius / 10.0).clamp(0.0, 0.6);
    let material = materials.add(StandardMaterial {
        base_color: settings.color.with_alpha(settings.alpha * transparency_factor),
        unlit: settings.unlit,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    
    // Scale particles based on filter radius to simulate smoothing
    let filter_scale_factor = 1.0 + (settings.filter_radius / 5.0);
    let enhanced_scale = settings.particle_scale * filter_scale_factor;
    
    spawn_particles(commands, particles, mesh_handle, material, camera_transform, enhanced_scale, "Filtered");
}

// Helper function for normals mode particles - simplified approach using StandardMaterial
fn render_normals_mode_particles_simple(
    commands: &mut Commands,
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mesh_handle: &Handle<Mesh>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_transform: &Transform,
) {
    // Create a material that simulates normal-based lighting
    // Use special material properties to create a distinct visual appearance
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 1.0).with_alpha(settings.alpha), // Light blue-white
        // Note: In Bevy 0.16, PBR properties have different names
        // We'll use a more emissive approach for visibility
        emissive: LinearRgba::new(0.1, 0.1, 0.3, 1.0), // Slight blue glow
        unlit: false,      // Enable lighting to show the difference
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    
    spawn_particles(commands, particles, mesh_handle, material, camera_transform, settings.particle_scale, "Normals");
}

// Helper function for full fluid mode - combines all screen space fluid techniques
fn render_full_fluid_particles(
    commands: &mut Commands,
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mesh_handle: &Handle<Mesh>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_transform: &Transform,
) {
    // Create a high-quality fluid material using user-configurable parameters
    let base_linear = settings.color.to_linear();
    let material = materials.add(StandardMaterial {
        // Use base color with configurable transparency
        base_color: Color::srgba(
            base_linear.red * 0.8,    // Darken for depth
            base_linear.green * 0.9, 
            base_linear.blue * 1.0,   // Keep blue strong
            settings.fluid_transparency
        ),
        
        // Configurable internal glow for volumetric effect
        emissive: LinearRgba::new(
            base_linear.red * settings.internal_glow * 0.3,
            base_linear.green * settings.internal_glow * 0.5,
            base_linear.blue * settings.internal_glow * 1.0,
            1.0
        ),
        
        // Enable lighting for realistic shading
        unlit: false,
        
        // Use blend mode for proper transparency layering
        alpha_mode: AlphaMode::Blend,
        
        ..default()
    });
    
    // Use configurable volume scale with filter radius influence
    let fluid_scale = settings.particle_scale * settings.volume_scale * (1.0 + settings.filter_radius / 10.0);
    
    spawn_particles(commands, particles, mesh_handle, material, camera_transform, fluid_scale, "FullFluid");
}

// Common particle spawning function
fn spawn_particles(
    commands: &mut Commands,
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mesh_handle: &Handle<Mesh>,
    material: Handle<StandardMaterial>,
    camera_transform: &Transform,
    particle_scale: f32,
    mode_name: &str,
) {
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
                .with_scale(Vec3::splat(particle_scale)),
            ScreenSpaceFluid,
            ScreenSpaceFluidMesh,
            Name::new(format!("Screen Space Fluid Particle - {}", mode_name)),
        ));
    }
}


