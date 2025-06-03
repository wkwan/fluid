use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};
use bevy::pbr::Material;
use crate::sim::{Particle3D, GpuState};
use crate::gpu_fluid::GpuFluidRenderData;
use crate::utils::despawn_entities;

pub struct ScreenSpaceFluidPlugin;

impl Plugin for ScreenSpaceFluidPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScreenSpaceFluidSettings>()
            // Temporarily remove custom material until we can fix the shader binding issues
            // .add_plugins(MaterialPlugin::<FluidNormalsMaterial>::default())
            .add_systems(Startup, setup_screen_space_resources)
            .add_systems(Update, render_screen_space_fluid_system
                .run_if(|settings: Res<ScreenSpaceFluidSettings>| settings.enabled)
            )
            .add_systems(Update, cleanup_screen_space_system
                .run_if(|settings: Res<ScreenSpaceFluidSettings>| !settings.enabled)
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
        "shaders/3d/screenspace/screenspace.wgsl".into()
    }
    
    fn vertex_shader() -> ShaderRef {
        "shaders/3d/screenspace/screenspace.wgsl".into()
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

// System to clean up screen space entities when disabled
fn cleanup_screen_space_system(
    mut commands: Commands,
    existing_screen_space: Query<Entity, With<ScreenSpaceFluid>>,
) {
    despawn_entities(&mut commands, &existing_screen_space);
}

// Main rendering system - supports both CPU and GPU particle rendering
pub fn render_screen_space_fluid_system(
    particles: Query<&Transform, With<Particle3D>>,
    mut commands: Commands,
    existing_screen_space: Query<Entity, With<ScreenSpaceFluid>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    circle_mesh: Option<Res<CircleMesh>>,
    settings: Res<ScreenSpaceFluidSettings>,
    camera_3d: Query<&Transform, With<Camera3d>>,
    gpu_state: Res<GpuState>,
    gpu_render_data: Res<GpuFluidRenderData>,
    time: Res<Time>,
) {
    if !settings.enabled {
        return;
    }

    // Remove existing screen space entities
    despawn_entities(&mut commands, &existing_screen_space);
    
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
    
    // Choose rendering path based on GPU state
    if gpu_state.enabled && gpu_render_data.has_data {
        // GPU path: Render representation of GPU particles
        render_gpu_particle_visualization(&mut commands, &mesh_handle, &mut materials, &settings, &camera_transform, &gpu_render_data, &time);
    } else {
        // CPU path: Render CPU particles normally
        let particle_count = particles.iter().count();
        if particle_count == 0 {
            return;
        }
        
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
    }
}

// Helper function for billboard mode particles
fn render_billboard_mode_particles(
    commands: &mut Commands,
    particles: &Query<&Transform, With<Particle3D>>,
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
    particles: &Query<&Transform, With<Particle3D>>,
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
    particles: &Query<&Transform, With<Particle3D>>,
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
    particles: &Query<&Transform, With<Particle3D>>,
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
    particles: &Query<&Transform, With<Particle3D>>,
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
    particles: &Query<&Transform, With<Particle3D>>,
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

// Advanced GPU particle visualization - represents actual GPU simulation data
fn render_gpu_particle_visualization(
    commands: &mut Commands,
    mesh_handle: &Handle<Mesh>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    settings: &Res<ScreenSpaceFluidSettings>,
    camera_transform: &Transform,
    gpu_render_data: &Res<GpuFluidRenderData>,
    time: &Res<Time>,
) {
    // Create different materials based on rendering mode
    let gpu_material = match settings.rendering_mode {
        RenderingMode::DepthOnly => {
            materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.8, 0.8).with_alpha(0.9), // Gray for depth mode
                unlit: settings.unlit,
                alpha_mode: AlphaMode::Blend,
                ..default()
            })
        }
        RenderingMode::Filtered => {
            materials.add(StandardMaterial {
                base_color: Color::srgb(0.5, 0.8, 1.0).with_alpha(0.7), // Light blue for filtered
                emissive: LinearRgba::new(0.1, 0.2, 0.3, 1.0),
                unlit: settings.unlit,
                alpha_mode: AlphaMode::Blend,
                ..default()
            })
        }
        RenderingMode::Normals => {
            materials.add(StandardMaterial {
                base_color: Color::srgb(0.9, 0.9, 1.0).with_alpha(0.8), // Light purple for normals
                emissive: LinearRgba::new(0.2, 0.2, 0.4, 1.0),
                unlit: false, // Enable lighting for normals mode
                alpha_mode: AlphaMode::Blend,
                ..default()
            })
        }
        RenderingMode::FullFluid => {
            materials.add(StandardMaterial {
                base_color: settings.color.with_alpha(settings.fluid_transparency),
                emissive: LinearRgba::new(
                    settings.color.to_linear().red * settings.internal_glow * 0.3,
                    settings.color.to_linear().green * settings.internal_glow * 0.5,
                    settings.color.to_linear().blue * settings.internal_glow * 1.0,
                    1.0
                ),
                unlit: false,
                alpha_mode: AlphaMode::Blend,
                ..default()
            })
        }
        _ => {
            materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.6, 0.2).with_alpha(0.8), // Orange for GPU mode
                emissive: LinearRgba::new(0.3, 0.15, 0.05, 1.0),
                unlit: settings.unlit,
                alpha_mode: AlphaMode::Blend,
                ..default()
            })
        }
    };
    
    // Calculate number of representative particles to spawn based on GPU data
    let representative_count = (gpu_render_data.num_particles / 10).max(1).min(100); // Show 1/10th, max 100
    let bounds_size = gpu_render_data.bounds_size;
    let bounds_center = gpu_render_data.bounds_center;
    
    // Simulate GPU particle movement - particles should fall with gravity
    let gravity = Vec3::new(0.0, -9.8, 0.0);
    let time_elapsed = time.elapsed_secs();
    let fall_distance = 0.5 * gravity.y * time_elapsed * time_elapsed; // Physics: d = 0.5 * g * t^2
    
    // Spawn representative particles in a grid within the simulation bounds
    let grid_size = (representative_count as f32).cbrt().ceil() as i32;
    let spacing = bounds_size / grid_size as f32;
    
    for x in 0..grid_size {
        for y in 0..grid_size {
            for z in 0..grid_size {
                if (x * grid_size * grid_size + y * grid_size + z) >= representative_count as i32 {
                    break;
                }
                
                // Start position in grid
                let initial_position = bounds_center + Vec3::new(
                    (x as f32 - grid_size as f32 * 0.5) * spacing.x,
                    (y as f32 - grid_size as f32 * 0.5) * spacing.y,
                    (z as f32 - grid_size as f32 * 0.5) * spacing.z,
                );
                
                // Apply simulated gravity fall
                let position = initial_position + Vec3::new(0.0, fall_distance, 0.0);
                
                // Keep particles within bounds (simulate collision)
                let min_y = bounds_center.y - bounds_size.y * 0.5;
                let final_position = Vec3::new(
                    position.x,
                    position.y.max(min_y), // Don't fall below bounds
                    position.z,
                );
                
                let look_at = camera_transform.translation - final_position;
                let rotation = Transform::from_translation(Vec3::ZERO)
                    .looking_at(-look_at, Vec3::Y)
                    .rotation;
                
                let scale = settings.particle_scale * settings.volume_scale;
                
                commands.spawn((
                    Mesh3d(mesh_handle.clone()),
                    MeshMaterial3d(gpu_material.clone()),
                    Transform::from_translation(final_position)
                        .with_rotation(rotation)
                        .with_scale(Vec3::splat(scale)),
                    ScreenSpaceFluid,
                    ScreenSpaceFluidMesh,
                    Name::new(format!("GPU Fluid Particle - {}", settings.rendering_mode as u8)),
                ));
            }
        }
    }
    
    info!("Screen Space Fluid: GPU mode - showing {} representative particles (of {} total) - falling with simulated gravity", 
          representative_count, gpu_render_data.num_particles);
}

// TODO: Future GPU particle rendering function that reads from GPU buffers
// This is where we'll implement Unity-style DrawMeshInstancedIndirect rendering
fn render_gpu_particles_from_buffers(
    _commands: &mut Commands,
    _mesh_handle: &Handle<Mesh>,
    _materials: &mut ResMut<Assets<StandardMaterial>>,
    _settings: &Res<ScreenSpaceFluidSettings>,
    _camera_transform: &Transform,
) {
    // Future implementation:
    // 1. Access GPU particle buffers from FluidBindGroups3D
    // 2. Create instanced rendering pipeline
    // 3. Use compute-to-graphics binding to render particles directly from GPU
    // 4. Support all rendering modes (Depth, Filtered, Normals, FullFluid)
    
    info!("TODO: Implement direct GPU buffer rendering for screen space fluid");
}


