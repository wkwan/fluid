use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::render_asset::{RenderAssets, RenderAssetUsages};
use bevy::render::renderer::RenderDevice;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::reflect::TypePath;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::math::primitives::{Sphere, Cuboid};
use crate::simulation::{Particle, SimulationDimension};
use crate::simulation3d::Particle3D;
use crate::screen_space_fluid::{render_screen_space_fluid, ScreenSpaceFluid};

// Fluid rendering mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
pub enum FluidRenderMode {
    #[default]
    ScreenSpace,
    RayMarching,
    MarchingCubes,
}

// Resource to control fluid rendering settings
#[derive(Resource, Reflect)]
pub struct FluidRenderSettings {
    pub show_free_surface: bool,
    pub render_mode: FluidRenderMode,
}

impl Default for FluidRenderSettings {
    fn default() -> Self {
        Self {
            show_free_surface: true,
            render_mode: FluidRenderMode::ScreenSpace,
        }
    }
}

// Component to mark the free surface mesh entity
#[derive(Component)]
pub struct FreeSurfaceMesh;

// Resource to store the density texture
#[derive(Resource)]
pub struct DensityTexture {
    pub texture: Handle<Image>,
    pub resolution: usize,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

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
            enabled: true,
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

// Resource to store grid settings
#[derive(Resource)]
pub struct MarchingGridSettings {
    pub grid_resolution: usize,
    pub iso_threshold: f32,
    pub grid_bounds_min: Vec3,
    pub grid_bounds_max: Vec3,
    pub smoothing_radius: f32,
    pub particle_mass: f32,
    pub update_frequency: f32, // How often to update the mesh (in seconds)
    pub last_update: f32,      // Time since last update
}

impl Default for MarchingGridSettings {
    fn default() -> Self {
        Self {
            grid_resolution: 64,  // Higher resolution for smoother surface
            iso_threshold: 0.15,  // Lower threshold for smoother surface
            grid_bounds_min: Vec3::new(-150.0, -350.0, -150.0),  // Cover entire simulation space
            grid_bounds_max: Vec3::new(150.0, 200.0, 150.0),
            smoothing_radius: 35.0,  // Larger radius for smoother density field
            particle_mass: 3.5,      // Higher mass for stronger density field
            update_frequency: 0.15,  // Reasonable update rate
            last_update: 0.0,
        }
    }
}

// Simplified marching cubes lookup table for basic cases
const EDGE_TABLE: [u16; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
];

// Edge vertex positions (where edges intersect cube faces)
const EDGE_VERTICES: [[usize; 2]; 12] = [
    [0, 1], [1, 2], [2, 3], [3, 0],  // bottom face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  // top face edges  
    [0, 4], [1, 5], [2, 6], [3, 7],  // vertical edges
];

// Cube corner offsets
const CUBE_CORNERS: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 1.0, 1.0),
];

// System to render free surface using the selected rendering method
pub fn render_free_surface(
    sim_dim: Res<State<SimulationDimension>>,
    render_settings: Res<FluidRenderSettings>,  // Add this parameter
    mut grid_settings: ResMut<MarchingGridSettings>,
    raymarching_settings: Res<RayMarchingSettings>,
    particles_3d: Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials_3d: ResMut<Assets<StandardMaterial>>,
    mut raymarch_materials: ResMut<Assets<RayMarchMaterial>>,
    mut images: ResMut<Assets<Image>>,
    existing_mesh: Query<Entity, With<FreeSurfaceMesh>>,
    existing_volume: Query<Entity, With<RayMarchVolume>>,
    existing_screen_space: Query<Entity, With<ScreenSpaceFluid>>,  // Add this parameter
    time: Res<Time>,
) {
    // Early exit if free surface rendering is disabled
    if !render_settings.show_free_surface {
        // Clean up all existing rendering entities
        cleanup_all_render_entities(
            &mut commands,
            &existing_mesh,
            &existing_volume,
            &existing_screen_space,
        );
        return;
    }

    match *sim_dim.get() {
        SimulationDimension::Dim2 => {
            // Remove all 3D rendering entities in 2D mode
            cleanup_all_render_entities(
                &mut commands,
                &existing_mesh,
                &existing_volume,
                &existing_screen_space,
            );
        }
        SimulationDimension::Dim3 => {
            // Render based on selected mode
            match render_settings.render_mode {
                FluidRenderMode::ScreenSpace => {
                    // Clean up other modes
                    for entity in existing_mesh.iter() {
                        commands.entity(entity).despawn();
                    }
                    for entity in existing_volume.iter() {
                        commands.entity(entity).despawn();
                    }
                    
                    // Screen space rendering (placeholder for now)
                    render_screen_space_fluid(
                        &particles_3d,
                        &mut commands,
                        &existing_screen_space,
                    );
                }
                FluidRenderMode::RayMarching => {
                    // Clean up other modes
                    for entity in existing_mesh.iter() {
                        commands.entity(entity).despawn();
                    }
                    for entity in existing_screen_space.iter() {
                        commands.entity(entity).despawn();
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
                FluidRenderMode::MarchingCubes => {
                    // Clean up other modes
                    for entity in existing_volume.iter() {
                        commands.entity(entity).despawn();
                    }
                    for entity in existing_screen_space.iter() {
                        commands.entity(entity).despawn();
                    }
                    
                    // Existing marching cubes implementation
                    render_marching_cubes(
                        &particles_3d,
                        &mut commands,
                        &mut meshes,
                        &mut materials_3d,
                        &mut grid_settings,
                        &existing_mesh,
                        &time,
                    );
                }
            }
        }
    }
}

// Helper function to clean up render entities
fn cleanup_all_render_entities(
    commands: &mut Commands,
    existing_mesh: &Query<Entity, With<FreeSurfaceMesh>>,
    existing_volume: &Query<Entity, With<RayMarchVolume>>,
    existing_screen_space: &Query<Entity, With<ScreenSpaceFluid>>,
) {
    for entity in existing_mesh.iter() {
        commands.entity(entity).despawn();
    }
    for entity in existing_volume.iter() {
        commands.entity(entity).despawn();
    }
    for entity in existing_screen_space.iter() {
        commands.entity(entity).despawn();
    }
}

// Extracted marching cubes implementation
fn render_marching_cubes(
    particles_3d: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials_3d: &mut ResMut<Assets<StandardMaterial>>,
    grid_settings: &mut ResMut<MarchingGridSettings>,
    existing_mesh: &Query<Entity, With<FreeSurfaceMesh>>,
    time: &Res<Time>,
) {
    // Check if enough time has passed since last update
    let current_time = time.elapsed_secs();
    if current_time - grid_settings.last_update < grid_settings.update_frequency {
        return; // Skip this frame
    }
    grid_settings.last_update = current_time;

    // Only generate surface if we have enough particles
    let particle_count = particles_3d.iter().count();
    if particle_count < 10 {
        // Remove existing mesh if not enough particles
        for entity in existing_mesh.iter() {
            commands.entity(entity).despawn();
        }
        return;
    }

    // Generate density field from particles
    if let Some(density_field) = generate_density_field(particles_3d, grid_settings) {
        // Debug: Check density field statistics
        let max_density = density_field.iter().fold(0.0f32, |a, &b| a.max(b));
        let avg_density = density_field.iter().sum::<f32>() / density_field.len() as f32;
        let above_threshold = density_field.iter().filter(|&&d| d > grid_settings.iso_threshold).count();
        
        if current_time - grid_settings.last_update > 1.0 { // Print debug info every second
            println!("Marching Cubes Debug: {} particles, max_density={:.3}, avg_density={:.6}, above_threshold={}/{}, iso_threshold={:.3}", 
                     particle_count, max_density, avg_density, above_threshold, density_field.len(), grid_settings.iso_threshold);
        }
        
        // Generate surface mesh using simplified marching cubes
        if let Some(mesh_data) = generate_surface_mesh(&density_field, grid_settings) {
            let (ref vertices, ref indices, _) = mesh_data;
            if current_time - grid_settings.last_update > 1.0 {
                println!("Generated surface mesh: {} vertices, {} triangles", vertices.len(), indices.len() / 3);
            }
            spawn_3d_surface_mesh(commands, meshes, materials_3d, mesh_data, existing_mesh);
        } else {
            if current_time - grid_settings.last_update > 1.0 {
                println!("No surface mesh generated");
            }
            // Remove existing mesh if no surface generated
            for entity in existing_mesh.iter() {
                commands.entity(entity).despawn();
            }
        }
    } else {
        if current_time - grid_settings.last_update > 1.0 {
            println!("No density field generated");
        }
        // Remove existing mesh if no surface generated
        for entity in existing_mesh.iter() {
            commands.entity(entity).despawn();
        }
    }
}

// Simplified density kernel function for better surface detection
fn poly6_kernel(distance: f32, smoothing_radius: f32) -> f32 {
    if distance >= smoothing_radius {
        return 0.0;
    }
    
    let normalized_distance = distance / smoothing_radius;
    let falloff = 1.0 - normalized_distance;
    falloff * falloff // Quadratic falloff
}

// Generate density field from particles using SPH kernel
fn generate_density_field(
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    grid_settings: &MarchingGridSettings,
) -> Option<Vec<f32>> {
    let particle_count = particles.iter().count();
    if particle_count == 0 {
        return None;
    }
    
    let resolution = grid_settings.grid_resolution;
    let bounds_min = grid_settings.grid_bounds_min;
    let bounds_max = grid_settings.grid_bounds_max;
    let grid_size = bounds_max - bounds_min;
    let cell_size = grid_size / resolution as f32;
    
    let mut density_field = vec![0.0f32; resolution * resolution * resolution];
    
    // Collect particle positions for faster access
    let particle_positions: Vec<Vec3> = particles.iter().map(|t| t.translation).collect();
    
    // Calculate density at each grid point using SPH kernel
    for i in 0..resolution {
        for j in 0..resolution {
            for k in 0..resolution {
                let grid_pos = bounds_min + Vec3::new(
                    i as f32 * cell_size.x,
                    j as f32 * cell_size.y,
                    k as f32 * cell_size.z,
                );
                
                let mut density = 0.0;
                for &particle_pos in &particle_positions {
                    let distance = (grid_pos - particle_pos).length();
                    let kernel_value = poly6_kernel(distance, grid_settings.smoothing_radius);
                    density += grid_settings.particle_mass * kernel_value;
                }
                
                let index = i * resolution * resolution + j * resolution + k;
                density_field[index] = density;
            }
        }
    }
    
    Some(density_field)
}

// Generate mesh using simplified marching cubes algorithm
fn generate_surface_mesh(
    density_field: &[f32],
    grid_settings: &MarchingGridSettings,
) -> Option<(Vec<Vec3>, Vec<u32>, Vec<Vec3>)> {
    if density_field.is_empty() {
        return None;
    }

    let resolution = grid_settings.grid_resolution;
    let bounds_min = grid_settings.grid_bounds_min;
    let grid_size = grid_settings.grid_bounds_max - bounds_min;
    let cell_size = grid_size / resolution as f32;
    let iso_level = grid_settings.iso_threshold;
    
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    
    // Process each cube in the grid
    for i in 0..(resolution - 1) {
        for j in 0..(resolution - 1) {
            for k in 0..(resolution - 1) {
                let cube_index = get_cube_configuration(density_field, resolution, i, j, k, iso_level);
                
                if cube_index == 0 || cube_index == 255 {
                    continue; // Cube is entirely inside or outside
                }
                
                // Get edge intersections for this cube
                let edge_mask = EDGE_TABLE[cube_index as usize];
                if edge_mask == 0 {
                    continue;
                }
                
                let mut edge_vertices = [Vec3::ZERO; 12];
                
                // Calculate intersection points on edges
                for edge in 0..12 {
                    if (edge_mask & (1 << edge)) != 0 {
                        let corner1 = EDGE_VERTICES[edge][0];
                        let corner2 = EDGE_VERTICES[edge][1];
                        
                        let pos1 = get_grid_position(i, j, k, corner1, bounds_min, cell_size);
                        let pos2 = get_grid_position(i, j, k, corner2, bounds_min, cell_size);
                        
                        let density1 = get_density_at_corner(density_field, resolution, i, j, k, corner1);
                        let density2 = get_density_at_corner(density_field, resolution, i, j, k, corner2);
                        
                        // Linear interpolation
                        let t = if (density2 - density1).abs() > 0.001 {
                            (iso_level - density1) / (density2 - density1)
                        } else {
                            0.5
                        };
                        edge_vertices[edge] = pos1 + t.clamp(0.0, 1.0) * (pos2 - pos1);
                    }
                }
                
                // Generate triangles using simplified approach
                generate_triangles_for_cube(
                    cube_index,
                    &edge_vertices,
                    &mut vertices,
                    &mut normals,
                    &mut indices,
                );
            }
        }
    }
    
    if vertices.is_empty() {
        None
    } else {
        Some((vertices, indices, normals))
    }
}

// Get cube configuration (8-bit value representing which corners are inside/outside)
fn get_cube_configuration(
    density_field: &[f32],
    resolution: usize,
    i: usize,
    j: usize,
    k: usize,
    iso_level: f32,
) -> u8 {
    let mut config = 0u8;
    
    for corner in 0..8 {
        let density = get_density_at_corner(density_field, resolution, i, j, k, corner);
        if density > iso_level {
            config |= 1 << corner;
        }
    }
    
    config
}

// Get density value at a specific corner of a cube
fn get_density_at_corner(
    density_field: &[f32],
    resolution: usize,
    i: usize,
    j: usize,
    k: usize,
    corner: usize,
) -> f32 {
    let corner_offset = CUBE_CORNERS[corner];
    let x = (i as f32 + corner_offset.x) as usize;
    let y = (j as f32 + corner_offset.y) as usize;
    let z = (k as f32 + corner_offset.z) as usize;
    
    if x >= resolution || y >= resolution || z >= resolution {
        return 0.0;
    }
    
    let index = x * resolution * resolution + y * resolution + z;
    density_field.get(index).copied().unwrap_or(0.0)
}

// Get world position for a corner
fn get_grid_position(
    i: usize,
    j: usize,
    k: usize,
    corner: usize,
    bounds_min: Vec3,
    cell_size: Vec3,
) -> Vec3 {
    let corner_offset = CUBE_CORNERS[corner];
    bounds_min + Vec3::new(
        (i as f32 + corner_offset.x) * cell_size.x,
        (j as f32 + corner_offset.y) * cell_size.y,
        (k as f32 + corner_offset.z) * cell_size.z,
    )
}

// Generate triangles for a cube (simplified version that actually creates visible triangles)
fn generate_triangles_for_cube(
    cube_index: u8,
    edge_vertices: &[Vec3; 12],
    vertices: &mut Vec<Vec3>,
    normals: &mut Vec<Vec3>,
    indices: &mut Vec<u32>,
) {
    let edge_mask = EDGE_TABLE[cube_index as usize];
    
    // Collect active edges
    let mut active_edges = Vec::new();
    for edge in 0..12 {
        if (edge_mask & (1 << edge)) != 0 {
            active_edges.push(edge);
        }
    }
    
    // Create triangles from groups of 3 active edges
    // This is a simplified approach - a full implementation would use proper triangle tables
    for chunk in active_edges.chunks(3) {
        if chunk.len() == 3 {
            let base_index = vertices.len() as u32;
            
            // Add vertices
            let v0 = edge_vertices[chunk[0]];
            let v1 = edge_vertices[chunk[1]];
            let v2 = edge_vertices[chunk[2]];
            
            vertices.push(v0);
            vertices.push(v1);
            vertices.push(v2);
            
            // Calculate face normal
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(edge2).normalize_or_zero();
            
            normals.push(normal);
            normals.push(normal);
            normals.push(normal);
            
            // Add triangle indices
            indices.extend_from_slice(&[base_index, base_index + 1, base_index + 2]);
        }
    }
}

// Spawn 3D surface mesh from vertices, indices, and normals
fn spawn_3d_surface_mesh(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    mesh_data: (Vec<Vec3>, Vec<u32>, Vec<Vec3>),
    existing_mesh: &Query<Entity, With<FreeSurfaceMesh>>,
) {
    // Remove existing mesh
    for entity in existing_mesh.iter() {
        commands.entity(entity).despawn();
    }
    
    let (vertices, indices, normals) = mesh_data;
    
    if vertices.is_empty() || indices.is_empty() {
        return;
    }
    
    // Create mesh with proper normals
    let mut mesh = Mesh::new(bevy::render::render_resource::PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 0.7, 1.0, 0.95),  // Slightly more opaque
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            metallic: 0.0,
            perceptual_roughness: 0.3,
            emissive: LinearRgba::rgb(0.1, 0.2, 0.3),  // Add some glow
            ..default()
        })),
        Transform::default(),
        FreeSurfaceMesh,
    ));
}

// Plugin for ray marching functionality
pub struct RayMarchPlugin;

impl Plugin for RayMarchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RayMarchingSettings>()
            .init_resource::<FluidRenderSettings>()
            .add_plugins(MaterialPlugin::<RayMarchMaterial>::default())
            .add_systems(Update, update_ray_march_material);
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
            bounds_min: Vec3::new(-150.0, -350.0, -150.0),
            bounds_max: Vec3::new(150.0, 200.0, 150.0),
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
        "shaders/raymarch.wgsl".into()
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
            for entity in existing_volume.iter() {
                commands.entity(entity).despawn();
            }
        }
        SimulationDimension::Dim3 => {
            if !raymarching_settings.enabled {
                // Remove existing volume if raymarching is disabled
                for entity in existing_volume.iter() {
                    commands.entity(entity).despawn();
                }
                return;
            }

            // Only generate volume if we have enough particles
            let particle_count = particles_3d.iter().count();
            
            if particle_count < 10 {
                // Remove existing volume if not enough particles
                for entity in existing_volume.iter() {
                    commands.entity(entity).despawn();
                }
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
                for entity in existing_volume.iter() {
                    commands.entity(entity).despawn();
                }
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
    let bounds_min = Vec3::new(-150.0, -350.0, -150.0);
    let bounds_max = Vec3::new(150.0, 200.0, 150.0);
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
    for entity in existing_volume.iter() {
        commands.entity(entity).despawn();
    }

    // Only proceed if we have valid density data
    if max_density < 0.001 {
        return;
    }

    // Create a cube that covers the simulation bounds
    let bounds_min = Vec3::new(-150.0, -350.0, -150.0);
    let bounds_max = Vec3::new(150.0, 200.0, 150.0);
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
