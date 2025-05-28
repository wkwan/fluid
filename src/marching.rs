use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use crate::simulation::{Particle, SimulationDimension};
use crate::simulation3d::Particle3D;

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

// Resource to store grid settings
#[derive(Resource)]
pub struct MarchingGridSettings {
    pub grid_resolution: usize,
    pub iso_threshold: f32,
    pub grid_bounds_min: Vec3,
    pub grid_bounds_max: Vec3,
    pub smoothing_radius: f32,
    pub particle_mass: f32,
}

impl Default for MarchingGridSettings {
    fn default() -> Self {
        Self {
            grid_resolution: 32,
            iso_threshold: 0.3,
            grid_bounds_min: Vec3::new(-300.0, -300.0, -300.0),
            grid_bounds_max: Vec3::new(300.0, 300.0, 300.0),
            smoothing_radius: 15.0,
            particle_mass: 1.0,
        }
    }
}

// System to render free surface using marching squares/cubes
pub fn render_free_surface(
    sim_dim: Res<State<SimulationDimension>>,
    grid_settings: Res<MarchingGridSettings>,
    particles_3d: Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials_3d: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    density_texture: Option<ResMut<DensityTexture>>,
    existing_mesh: Query<Entity, With<FreeSurfaceMesh>>,
) {
    match *sim_dim.get() {
        SimulationDimension::Dim2 => {
            // Remove existing mesh in 2D mode (no surface rendering)
            for entity in existing_mesh.iter() {
                commands.entity(entity).despawn();
            }
        }
        SimulationDimension::Dim3 => {
            // Generate or update density texture
            if let Some(density_data) = generate_density_texture(&particles_3d, &grid_settings) {
                // Update or create density texture resource
                if let Some(density_tex) = density_texture {
                    if let Some(image) = images.get_mut(&density_tex.texture) {
                        image.data = Some(density_data.clone());
                    }
                } else {
                    let texture_handle = create_density_texture(&mut images, &density_data, &grid_settings);
                    commands.insert_resource(DensityTexture {
                        texture: texture_handle,
                        resolution: grid_settings.grid_resolution,
                        bounds_min: grid_settings.grid_bounds_min,
                        bounds_max: grid_settings.grid_bounds_max,
                    });
                }
                
                // Generate 3D surface using marching cubes
                if let Some(mesh_data) = generate_marching_cubes_from_density(
                    &density_data,
                    &grid_settings,
                ) {
                    spawn_3d_surface_mesh(&mut commands, &mut meshes, &mut materials_3d, mesh_data, &existing_mesh);
                } else {
                    // Remove existing mesh if no surface generated
                    for entity in existing_mesh.iter() {
                        commands.entity(entity).despawn();
                    }
                }
            } else {
                // Remove existing mesh if no surface generated
                for entity in existing_mesh.iter() {
                    commands.entity(entity).despawn();
                }
            }
        }
    }
}

// SPH Poly6 kernel function for density calculation
fn poly6_kernel(distance_sq: f32, smoothing_radius: f32) -> f32 {
    let h_sq = smoothing_radius * smoothing_radius;
    if distance_sq >= h_sq {
        return 0.0;
    }
    
    let diff = h_sq - distance_sq;
    let poly6_constant = 315.0 / (64.0 * std::f32::consts::PI * smoothing_radius.powi(9));
    poly6_constant * diff.powi(3)
}

// Generate density texture from particles
fn generate_density_texture(
    particles: &Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    grid_settings: &MarchingGridSettings,
) -> Option<Vec<u8>> {
    let particle_count = particles.iter().count();
    if particle_count == 0 {
        return None;
    }
    
    let resolution = grid_settings.grid_resolution;
    let bounds_min = grid_settings.grid_bounds_min;
    let bounds_max = grid_settings.grid_bounds_max;
    let grid_size = bounds_max - bounds_min;
    let cell_size = grid_size / resolution as f32;
    
    let mut density_data = vec![0.0f32; resolution * resolution * resolution];
    
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
                for transform in particles.iter() {
                    let distance_sq = (grid_pos - transform.translation).length_squared();
                    let kernel_value = poly6_kernel(distance_sq, grid_settings.smoothing_radius);
                    density += grid_settings.particle_mass * kernel_value;
                }
                
                let index = i * resolution * resolution + j * resolution + k;
                density_data[index] = density;
            }
        }
    }
    
    // Convert to u8 texture data (normalize and scale)
    let max_density = density_data.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_density = density_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let density_range = max_density - min_density;
    
    if density_range > 0.0 {
        Some(density_data.iter()
            .map(|&density| {
                ((density - min_density) / density_range * 255.0) as u8
            })
            .collect())
    } else {
        None
    }
}

// Create density texture
fn create_density_texture(
    images: &mut ResMut<Assets<Image>>,
    density_data: &[u8],
    grid_settings: &MarchingGridSettings,
) -> Handle<Image> {
    let resolution = grid_settings.grid_resolution;
    
    let image = Image::new(
        Extent3d {
            width: resolution as u32,
            height: resolution as u32,
            depth_or_array_layers: resolution as u32,
        },
        TextureDimension::D3,
        density_data.to_vec(),
        TextureFormat::R8Unorm,
        default(),
    );
    
    images.add(image)
}

// Generate mesh data using marching cubes for 3D
fn generate_marching_cubes_from_density(
    density_data: &[u8],
    grid_settings: &MarchingGridSettings,
) -> Option<(Vec<Vec3>, Vec<u32>)> {
    if density_data.is_empty() {
        return None;
    }

    let min_pos = grid_settings.grid_bounds_min;
    let max_pos = grid_settings.grid_bounds_max;
    let resolution = grid_settings.grid_resolution;
    let grid_size = max_pos - min_pos;
    let cell_size = grid_size / resolution as f32;
    
    // Convert density threshold to u8 range
    let threshold_u8 = (grid_settings.iso_threshold * 255.0) as u8;
    
    // Generate cubes where density exceeds threshold
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut vertex_count = 0;
    
    for i in 0..(resolution - 1) {
        for j in 0..(resolution - 1) {
            for k in 0..(resolution - 1) {
                // Check if any corner of this cell exceeds threshold
                let corner_indices = [
                    i * resolution * resolution + j * resolution + k,
                    (i + 1) * resolution * resolution + j * resolution + k,
                    (i + 1) * resolution * resolution + (j + 1) * resolution + k,
                    i * resolution * resolution + (j + 1) * resolution + k,
                    i * resolution * resolution + j * resolution + (k + 1),
                    (i + 1) * resolution * resolution + j * resolution + (k + 1),
                    (i + 1) * resolution * resolution + (j + 1) * resolution + (k + 1),
                    i * resolution * resolution + (j + 1) * resolution + (k + 1),
                ];
                
                let above_threshold = corner_indices.iter()
                    .any(|&idx| idx < density_data.len() && density_data[idx] > threshold_u8);
                
                if above_threshold {
                    // Create a small cube at this grid cell
                    let cell_center = min_pos + Vec3::new(
                        (i as f32 + 0.5) * cell_size.x,
                        (j as f32 + 0.5) * cell_size.y,
                        (k as f32 + 0.5) * cell_size.z,
                    );
                    
                    let half_cell = cell_size * 0.4; // Make cubes slightly smaller than cells
                    
                    // Add 8 vertices for this cube
                    let cube_vertices = [
                        cell_center + Vec3::new(-half_cell.x, -half_cell.y, -half_cell.z),
                        cell_center + Vec3::new(half_cell.x, -half_cell.y, -half_cell.z),
                        cell_center + Vec3::new(half_cell.x, half_cell.y, -half_cell.z),
                        cell_center + Vec3::new(-half_cell.x, half_cell.y, -half_cell.z),
                        cell_center + Vec3::new(-half_cell.x, -half_cell.y, half_cell.z),
                        cell_center + Vec3::new(half_cell.x, -half_cell.y, half_cell.z),
                        cell_center + Vec3::new(half_cell.x, half_cell.y, half_cell.z),
                        cell_center + Vec3::new(-half_cell.x, half_cell.y, half_cell.z),
                    ];
                    
                    vertices.extend_from_slice(&cube_vertices);
                    
                    // Add indices for this cube (12 triangles, 6 faces)
                    let base = vertex_count;
                    let cube_indices = [
                        // Bottom face
                        base, base + 1, base + 2, base + 2, base + 3, base,
                        // Top face
                        base + 4, base + 7, base + 6, base + 6, base + 5, base + 4,
                        // Front face
                        base, base + 4, base + 5, base + 5, base + 1, base,
                        // Back face
                        base + 2, base + 6, base + 7, base + 7, base + 3, base + 2,
                        // Left face
                        base, base + 3, base + 7, base + 7, base + 4, base,
                        // Right face
                        base + 1, base + 5, base + 6, base + 6, base + 2, base + 1,
                    ];
                    
                    indices.extend_from_slice(&cube_indices);
                    vertex_count += 8;
                }
            }
        }
    }
    
    if vertices.is_empty() {
        None
    } else {
        Some((vertices, indices))
    }
}

// Spawn 3D surface mesh from vertices and indices
fn spawn_3d_surface_mesh(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    mesh_data: (Vec<Vec3>, Vec<u32>),
    existing_mesh: &Query<Entity, With<FreeSurfaceMesh>>,
) {
    // Remove existing mesh
    for entity in existing_mesh.iter() {
        commands.entity(entity).despawn();
    }
    
    let (vertices, indices) = mesh_data;
    
    // Create mesh
    let mut mesh = Mesh::new(bevy::render::render_resource::PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));
    mesh.compute_normals();
    
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.0, 0.8, 1.0, 0.3),
            alpha_mode: AlphaMode::Blend,
            cull_mode: None, // Render both sides
            unlit: true,
            ..default()
        })),
        Transform::default(),
        FreeSurfaceMesh,
    ));
} 