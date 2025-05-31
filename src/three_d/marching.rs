use bevy::prelude::*;
use bevy::{
    render::{
        render_resource::{AsBindGroup, ShaderRef, Extent3d, TextureDimension, TextureFormat},
        render_asset::RenderAssetUsages,
    },
    reflect::TypePath,
    math::primitives::Cuboid,
};
use crate::three_d::simulation::Particle3D;
use crate::sim::Particle;
use crate::sim::SimulationDimension;
use crate::constants::{RAY_MARCH_BOUNDS_MIN, RAY_MARCH_BOUNDS_MAX};
use crate::utils::despawn_entities;

// Component to mark the free surface mesh entity
#[derive(Component)]
pub struct FreeSurfaceMesh;

// Resource to store ray marching settings
#[derive(Resource)]
pub struct RayMarchingSettings {
    pub enabled: bool,
    pub quality: f32,
    pub step_count: u32,
    pub density_multiplier: f32,
    pub density_threshold: f32,
    pub absorption: f32,
    pub scattering: f32,
    pub light_intensity: f32,
    pub shadow_steps: u32,
    pub use_shadows: bool,
    // New settings for advanced features
    pub refraction_enabled: bool,
    pub reflection_enabled: bool,
    pub environment_sampling: bool,
    pub max_bounces: u32,
    pub ior_water: f32,
    pub ior_air: f32,
    pub extinction_coefficient: Vec3,
    pub surface_smoothness: f32,
}

impl Default for RayMarchingSettings {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default
            quality: 1.0,
            step_count: 32, // Reduced for better performance
            density_multiplier: 10.0, // Much higher for visibility
            density_threshold: 0.001, // Higher threshold for smoother surfaces
            absorption: 5.0, // Much higher for better visibility
            scattering: 1.0,
            light_intensity: 5.0, // Much brighter
            shadow_steps: 8,
            use_shadows: false, // Disabled by default for performance
            refraction_enabled: false,
            reflection_enabled: false,
            environment_sampling: false,
            max_bounces: 4,
            ior_water: 1.33,
            ior_air: 1.0,
            extinction_coefficient: Vec3::new(0.45, 0.15, 0.1),
            surface_smoothness: 0.8, // Higher smoothness by default
        }
    }
}

// Main system for rendering the free surface
pub fn render_free_surface_system(
    sim_dim: Res<State<SimulationDimension>>,
    raymarching_settings: Res<RayMarchingSettings>,
    particles_3d: Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    raymarch_materials: ResMut<Assets<RayMarchMaterial>>,
    images: ResMut<Assets<Image>>,
    existing_mesh: Query<Entity, With<FreeSurfaceMesh>>,
    existing_volume: Query<Entity, With<RayMarchVolume>>,
    time: Res<Time>,
) {
    // Only proceed if 3D mode and free surface rendering is enabled
    if *sim_dim.get() != SimulationDimension::Dim3 || !raymarching_settings.enabled {
        // Clean up any existing render entities
        cleanup_all_render_entities(&mut commands, &existing_mesh, &existing_volume);
        return;
    }
    
    // Check if we have enough particles
    let particle_count = particles_3d.iter().count();
    if particle_count < 10 {
        cleanup_all_render_entities(&mut commands, &existing_mesh, &existing_volume);
        return;
    }
    
    // Call existing ray marching implementation
    render_ray_march_volume(
        sim_dim,
        raymarching_settings,
        particles_3d,
        commands,
        meshes,
        raymarch_materials,
        images,
        existing_volume,
        time,
    );
}

// Helper function to clean up render entities
fn cleanup_all_render_entities(
    commands: &mut Commands,
    existing_mesh: &Query<Entity, With<FreeSurfaceMesh>>,
    existing_volume: &Query<Entity, With<RayMarchVolume>>,
) {
    despawn_entities(commands, existing_mesh);
    despawn_entities(commands, existing_volume);
}

// System to clean up free surface entities when show_free_surface is disabled
fn cleanup_free_surface_system(
    mut commands: Commands,
    existing_mesh: Query<Entity, With<FreeSurfaceMesh>>,
    existing_volume: Query<Entity, With<RayMarchVolume>>,
) {
    cleanup_all_render_entities(&mut commands, &existing_mesh, &existing_volume);
}

// Plugin for ray marching functionality
pub struct RayMarchPlugin;

impl Plugin for RayMarchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RayMarchingSettings>()
            .add_plugins(MaterialPlugin::<RayMarchMaterial>::default())
            .add_systems(Update, update_ray_march_material)
            .add_systems(Update, render_free_surface_system
                .run_if(|settings: Res<RayMarchingSettings>, sim_dim: Res<State<SimulationDimension>>| 
                    settings.enabled && *sim_dim.get() == SimulationDimension::Dim3)
            )
            .add_systems(Update, cleanup_free_surface_system
                .run_if(|settings: Res<RayMarchingSettings>, sim_dim: Res<State<SimulationDimension>>| 
                    (!settings.enabled && *sim_dim.get() == SimulationDimension::Dim3) || 
                    *sim_dim.get() == SimulationDimension::Dim2)
            );
    }
}

// Custom material for ray marching
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct RayMarchMaterial {
    #[uniform(0)]
    pub camera_pos: Vec3,
    #[uniform(0)]
    pub bounds_min: Vec3,
    #[uniform(0)]
    pub bounds_max: Vec3,
    #[uniform(0)]
    pub step_size: f32,
    #[uniform(0)]
    pub density_multiplier: f32,
    #[uniform(0)]
    pub density_threshold: f32,
    #[uniform(0)]
    pub max_density: f32,
    #[uniform(0)]
    pub absorption: f32,
    #[uniform(0)]
    pub scattering: f32,
    #[uniform(0)]
    pub light_intensity: f32,
    #[uniform(0)]
    pub refraction_enabled: u32,
    #[uniform(0)]
    pub reflection_enabled: u32,
    #[uniform(0)]
    pub environment_sampling: u32,
    #[uniform(0)]
    pub max_bounces: u32,
    #[uniform(0)]
    pub ior_water: f32,
    #[uniform(0)]
    pub ior_air: f32,
    #[uniform(0)]
    pub extinction_coefficient: Vec3,
    #[uniform(0)]
    pub surface_smoothness: f32,
    #[texture(1, dimension = "3d")]
    #[sampler(2)]
    pub density_texture: Option<Handle<Image>>,
}

impl Default for RayMarchMaterial {
    fn default() -> Self {
        Self {
            camera_pos: Vec3::ZERO,
            bounds_min: Vec3::from(RAY_MARCH_BOUNDS_MIN),
            bounds_max: Vec3::from(RAY_MARCH_BOUNDS_MAX),
            step_size: 5.0,
            density_multiplier: 10.0,
            density_threshold: 0.00001,
            max_density: 1.0,
            absorption: 5.0,
            scattering: 1.0,
            light_intensity: 5.0,
            refraction_enabled: 0,
            reflection_enabled: 0,
            environment_sampling: 0,
            max_bounces: 4,
            ior_water: 1.33,
            ior_air: 1.0,
            extinction_coefficient: Vec3::new(0.45, 0.15, 0.1),
            surface_smoothness: 0.5,
            density_texture: None,
        }
    }
}

impl Material for RayMarchMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }
    
    fn fragment_shader() -> ShaderRef {
        "shaders/3d/raymarch/raymarch.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend // Use blend for volumetric transparency
    }
}

// System to update ray march material uniforms
fn update_ray_march_material(
    mut materials: ResMut<Assets<RayMarchMaterial>>,
    camera_query: Query<&Transform, With<Camera3d>>,
    raymarching_settings: Res<RayMarchingSettings>,
) {
    if !raymarching_settings.enabled {
        return;
    }

    // Get camera position
    let camera_pos = if let Ok(camera_transform) = camera_query.single() {
        camera_transform.translation
    } else {
        Vec3::ZERO
    };

    let material_count = materials.len();
    if material_count == 0 {
        return;
    }

    // Calculate step size based on bounds and step count
    let bounds_size = Vec3::new(300.0, 550.0, 300.0); // bounds_max - bounds_min
    let max_dimension = bounds_size.x.max(bounds_size.y).max(bounds_size.z);
    let step_size = max_dimension / raymarching_settings.step_count as f32;

    // Update all ray march materials with current settings
    for (_, material) in materials.iter_mut() {
        material.camera_pos = camera_pos;
        material.step_size = step_size;
        material.density_multiplier = raymarching_settings.density_multiplier;
        material.density_threshold = raymarching_settings.density_threshold;
        material.absorption = raymarching_settings.absorption;
        material.scattering = raymarching_settings.scattering;
        material.light_intensity = raymarching_settings.light_intensity;
        material.refraction_enabled = raymarching_settings.refraction_enabled as u32;
        material.reflection_enabled = raymarching_settings.reflection_enabled as u32;
        material.environment_sampling = raymarching_settings.environment_sampling as u32;
        material.max_bounces = raymarching_settings.max_bounces;
        material.ior_water = raymarching_settings.ior_water;
        material.ior_air = raymarching_settings.ior_air;
        material.extinction_coefficient = raymarching_settings.extinction_coefficient;
        material.surface_smoothness = raymarching_settings.surface_smoothness;
    }
}

// Component to mark ray marching volume entity
#[derive(Component)]
pub struct RayMarchVolume;

// System to create/update ray marching volume
pub fn render_ray_march_volume(
    sim_dim: Res<State<SimulationDimension>>,
    raymarching_settings: Res<RayMarchingSettings>,
    particles_3d: Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<RayMarchMaterial>>,
    mut images: ResMut<Assets<Image>>,
    existing_volume: Query<Entity, With<RayMarchVolume>>,
    time: Res<Time>,
) {
    match *sim_dim.get() {
        SimulationDimension::Dim2 => {
            // Remove existing volume in 2D mode
            despawn_entities(&mut commands, &existing_volume);
        }
        SimulationDimension::Dim3 => {
            if !raymarching_settings.enabled {
                // Remove existing volume if raymarching is disabled
                despawn_entities(&mut commands, &existing_volume);
                return;
            }

            // Only generate volume if we have enough particles
            let particle_count = particles_3d.iter().count();
            
            if particle_count < 10 {
                // Remove existing volume if not enough particles
                despawn_entities(&mut commands, &existing_volume);
                return;
            }

            // Only update every 0.5 seconds to reduce flashing and improve performance
            static mut LAST_UPDATE: f32 = 0.0;
            let current_time = time.elapsed_secs();
            unsafe {
                if current_time - LAST_UPDATE < 0.5 && !existing_volume.is_empty() {
                    return; // Skip this frame
                }
                LAST_UPDATE = current_time;
            }

            // Generate density texture from particles
            if let Some((density_texture, max_density)) = generate_density_texture(&particles_3d, &mut images) {
                spawn_ray_march_volume(&mut commands, &mut meshes, &mut materials, density_texture, max_density, &raymarching_settings, &existing_volume);
            } else {
                // Remove existing volume if no density texture could be generated
                despawn_entities(&mut commands, &existing_volume);
            }
        }
    }
}

// Generate 3D density texture from particles
fn generate_density_texture(
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    images: &mut ResMut<Assets<Image>>,
) -> Option<(Handle<Image>, f32)> {
    let resolution = 96u32; // Increased to 96 for even smoother surfaces
    let bounds_min = Vec3::from(RAY_MARCH_BOUNDS_MIN);
    let bounds_max = Vec3::from(RAY_MARCH_BOUNDS_MAX);
    let bounds_size = bounds_max - bounds_min;
    let cell_size = bounds_size / resolution as f32;
    let smoothing_radius = 50.0; // Increased for even smoother blending

    // Collect particle positions
    let particle_positions: Vec<Vec3> = particles.iter().map(|t| t.translation).collect();
    
    if particle_positions.is_empty() {
        return None;
    }

    // Generate density field with improved kernel
    let mut density_data = vec![0.0f32; (resolution * resolution * resolution) as usize];
    let mut max_density = 0.0f32;
    let mut _non_zero_cells = 0;
    
    for i in 0..resolution {
        for j in 0..resolution {
            for k in 0..resolution {
                let grid_pos = bounds_min + Vec3::new(
                    (i as f32 + 0.5) * cell_size.x, // Sample at cell center
                    (j as f32 + 0.5) * cell_size.y,
                    (k as f32 + 0.5) * cell_size.z,
                );
                
                let mut density = 0.0;
                for &particle_pos in &particle_positions {
                    let distance = (grid_pos - particle_pos).length();
                    if distance < smoothing_radius {
                        // Use an even smoother kernel function
                        let normalized_distance = distance / smoothing_radius;
                        
                        // Wendland C2 kernel for maximum smoothness
                        if normalized_distance <= 1.0 {
                            let one_minus_r = 1.0 - normalized_distance;
                            let kernel_value = one_minus_r * one_minus_r * one_minus_r * one_minus_r * (4.0 * normalized_distance + 1.0);
                            
                            // Normalize the kernel (approximate normalization for 3D)
                            let normalized_kernel = kernel_value * 21.0 / (16.0 * std::f32::consts::PI * smoothing_radius * smoothing_radius * smoothing_radius);
                            density += normalized_kernel * 12.0; // Adjusted scaling for visibility
                        }
                    }
                }
                
                let index = (i * resolution * resolution + j * resolution + k) as usize;
                density_data[index] = density;
                
                if density > 0.0 {
                    _non_zero_cells += 1;
                    max_density = max_density.max(density);
                }
            }
        }
    }

    if max_density < 0.001 {
        return None;
    }

    // Apply two passes of smoothing for ultra-smooth surfaces
    let mut smoothed_data = density_data.clone();
    
    // First smoothing pass
    for i in 1..(resolution - 1) {
        for j in 1..(resolution - 1) {
            for k in 1..(resolution - 1) {
                let index = (i * resolution * resolution + j * resolution + k) as usize;
                
                // 3x3x3 smoothing kernel
                let mut sum = 0.0;
                let mut count = 0;
                for di in -1i32..=1 {
                    for dj in -1i32..=1 {
                        for dk in -1i32..=1 {
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            let nk = (k as i32 + dk) as usize;
                            
                            if ni < resolution as usize && nj < resolution as usize && nk < resolution as usize {
                                let neighbor_index = ni * resolution as usize * resolution as usize + nj * resolution as usize + nk;
                                sum += density_data[neighbor_index];
                                count += 1;
                            }
                        }
                    }
                }
                
                // Blend original with smoothed value
                let smoothed_value = sum / count as f32;
                smoothed_data[index] = density_data[index] * 0.6 + smoothed_value * 0.4; // More smoothing
            }
        }
    }
    
    // Second smoothing pass for extra smoothness
    let mut final_data = smoothed_data.clone();
    for i in 1..(resolution - 1) {
        for j in 1..(resolution - 1) {
            for k in 1..(resolution - 1) {
                let index = (i * resolution * resolution + j * resolution + k) as usize;
                
                // Smaller 2x2x2 smoothing kernel for fine details
                let mut sum = 0.0;
                let mut count = 0;
                for di in 0i32..=1 {
                    for dj in 0i32..=1 {
                        for dk in 0i32..=1 {
                            let ni = (i as i32 + di) as usize;
                            let nj = (j as i32 + dj) as usize;
                            let nk = (k as i32 + dk) as usize;
                            
                            if ni < resolution as usize && nj < resolution as usize && nk < resolution as usize {
                                let neighbor_index = ni * resolution as usize * resolution as usize + nj * resolution as usize + nk;
                                sum += smoothed_data[neighbor_index];
                                count += 1;
                            }
                        }
                    }
                }
                
                let fine_smoothed = sum / count as f32;
                final_data[index] = smoothed_data[index] * 0.8 + fine_smoothed * 0.2;
            }
        }
    }

    // Convert to bytes for texture - store raw density values
    let mut texture_data = Vec::with_capacity(final_data.len() * 4);
    
    for density in final_data {
        // Store density directly in red channel (0-1 range)
        let normalized_density = (density / max_density).clamp(0.0, 1.0);
        let value = (normalized_density * 255.0) as u8;
        texture_data.extend_from_slice(&[value, 0, 0, 255]); // R=density, G=0, B=0, A=255
    }

    // Create 3D texture
    let image = Image::new(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: resolution,
        },
        TextureDimension::D3,
        texture_data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );

    Some((images.add(image), max_density))
}

// Spawn ray marching volume mesh
fn spawn_ray_march_volume(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<RayMarchMaterial>>,
    density_texture: Handle<Image>,
    max_density: f32,
    raymarching_settings: &RayMarchingSettings,
    existing_volume: &Query<Entity, With<RayMarchVolume>>,
) {
    // Remove existing volume
    despawn_entities(commands, existing_volume);

    // Only proceed if we have valid density data
    if max_density < 0.001 {
        return;
    }

    // Create a cube that covers the simulation bounds
    let bounds_min = Vec3::from(RAY_MARCH_BOUNDS_MIN);
    let bounds_max = Vec3::from(RAY_MARCH_BOUNDS_MAX);
    let bounds_size = bounds_max - bounds_min;
    let mesh_handle = meshes.add(Cuboid::new(bounds_size.x, bounds_size.y, bounds_size.z));

    // Calculate step size
    let max_dimension = bounds_size.x.max(bounds_size.y).max(bounds_size.z);
    let step_size = max_dimension / raymarching_settings.step_count as f32;

    // Create ray march material with density texture and current settings
    let material = RayMarchMaterial {
        camera_pos: Vec3::ZERO, // Will be updated by the system
        bounds_min,
        bounds_max,
        step_size,
        density_multiplier: raymarching_settings.density_multiplier,
        density_threshold: raymarching_settings.density_threshold,
        max_density,
        absorption: raymarching_settings.absorption,
        scattering: raymarching_settings.scattering,
        light_intensity: raymarching_settings.light_intensity,
        refraction_enabled: raymarching_settings.refraction_enabled as u32,
        reflection_enabled: raymarching_settings.reflection_enabled as u32,
        environment_sampling: raymarching_settings.environment_sampling as u32,
        max_bounces: raymarching_settings.max_bounces,
        ior_water: raymarching_settings.ior_water,
        ior_air: raymarching_settings.ior_air,
        extinction_coefficient: raymarching_settings.extinction_coefficient,
        surface_smoothness: raymarching_settings.surface_smoothness,
        density_texture: Some(density_texture),
    };
    
    let material_handle = materials.add(material);

    // Position the cube at the center of the bounds
    let center_position = (bounds_min + bounds_max) * 0.5;
    
    // Spawn the volume - visible by default, shader will handle invalid textures gracefully
    let _entity = commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material_handle),
        Transform::from_translation(center_position),
        RayMarchVolume,
    )).id();
} 
