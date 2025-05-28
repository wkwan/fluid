use bevy::prelude::*;
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
    pub update_frequency: f32, // How often to update the mesh (in seconds)
    pub last_update: f32,      // Time since last update
}

impl Default for MarchingGridSettings {
    fn default() -> Self {
        Self {
            grid_resolution: 40,  // Slightly higher for better quality
            iso_threshold: 0.3,   // Higher threshold to capture more of the fluid volume
            grid_bounds_min: Vec3::new(-150.0, -350.0, -150.0),  // Cover entire simulation space
            grid_bounds_max: Vec3::new(150.0, 200.0, 150.0),
            smoothing_radius: 25.0,  // Balanced radius for good coverage
            particle_mass: 2.0,      // Balanced mass for stable density field
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

// System to render free surface using marching cubes
pub fn render_free_surface(
    sim_dim: Res<State<SimulationDimension>>,
    mut grid_settings: ResMut<MarchingGridSettings>,
    particles_3d: Query<&Transform, (With<Particle3D>, Without<Particle>)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials_3d: ResMut<Assets<StandardMaterial>>,
    existing_mesh: Query<Entity, With<FreeSurfaceMesh>>,
    time: Res<Time>,
) {
    match *sim_dim.get() {
        SimulationDimension::Dim2 => {
            // Remove existing mesh in 2D mode
            for entity in existing_mesh.iter() {
                commands.entity(entity).despawn();
            }
        }
        SimulationDimension::Dim3 => {
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
            if let Some(density_field) = generate_density_field(&particles_3d, &grid_settings) {
                // Debug: Check density field statistics
                let max_density = density_field.iter().fold(0.0f32, |a, &b| a.max(b));
                let avg_density = density_field.iter().sum::<f32>() / density_field.len() as f32;
                let above_threshold = density_field.iter().filter(|&&d| d > grid_settings.iso_threshold).count();
                
                if current_time - grid_settings.last_update > 1.0 { // Print debug info every second
                    println!("Marching Cubes Debug: {} particles, max_density={:.3}, avg_density={:.6}, above_threshold={}/{}, iso_threshold={:.3}", 
                             particle_count, max_density, avg_density, above_threshold, density_field.len(), grid_settings.iso_threshold);
                }
                
                // Generate surface mesh using simplified marching cubes
                if let Some(mesh_data) = generate_surface_mesh(&density_field, &grid_settings) {
                    let (ref vertices, ref indices, _) = mesh_data;
                    if current_time - grid_settings.last_update > 1.0 {
                        println!("Generated surface mesh: {} vertices, {} triangles", vertices.len(), indices.len() / 3);
                    }
                    spawn_3d_surface_mesh(&mut commands, &mut meshes, &mut materials_3d, mesh_data, &existing_mesh);
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
    }
}

// Simplified density kernel function for better surface detection
fn simple_density_kernel(distance: f32, smoothing_radius: f32) -> f32 {
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
                    let kernel_value = simple_density_kernel(distance, grid_settings.smoothing_radius);
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
            base_color: Color::srgba(0.2, 0.8, 1.0, 0.9),  // More opaque and brighter
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            metallic: 0.0,
            perceptual_roughness: 0.5,
            emissive: LinearRgba::rgb(0.1, 0.2, 0.3),  // Add some glow
            ..default()
        })),
        Transform::default(),
        FreeSurfaceMesh,
    ));
} 
