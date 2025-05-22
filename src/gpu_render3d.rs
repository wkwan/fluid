use bevy::prelude::*;
use bevy::pbr::MeshMaterial3d;
use crate::simulation3d::Particle3D;
use crate::gpu_fluid3d::GpuParticles3D;

// Plugin to handle 3D particle rendering with velocity colors
pub struct GpuRender3DPlugin;

impl Plugin for GpuRender3DPlugin {
    fn build(&self, app: &mut App) {
        // Add a system to update particle colors based on velocity
        app.add_systems(Update, update_particle_colors_per_velocity);
    }
}

// Custom system to update particle colors based on their individual velocities
fn update_particle_colors_per_velocity(
    mut materials: ResMut<Assets<StandardMaterial>>,
    particles: Query<(&Particle3D, &MeshMaterial3d<StandardMaterial>)>,
    gpu_particles: Option<Res<GpuParticles3D>>,
    time: Res<Time>,
) {
    // Use a consistent maximum velocity for normalization
    const MAX_VELOCITY: f32 = 700.0;
    
    // Debug info for velocity monitoring
    let mut total_magnitude = 0.0;
    let mut count = 0;
    let mut max_seen: f32 = 0.0;
    
    // Create a map of entity indices to velocities if we have GPU data
    let mut gpu_velocities = Vec::new();
    if let Some(gpu_data) = gpu_particles.as_ref() {
        if gpu_data.updated && !gpu_data.velocities.is_empty() {
            gpu_velocities = gpu_data.velocities.clone();
        }
    }
    
    // For each particle with a material, update its color based on velocity
    for (i, (particle, material)) in particles.iter().enumerate() {
        // Get velocity either from GPU or CPU particle
        let velocity = if !gpu_velocities.is_empty() && i < gpu_velocities.len() {
            gpu_velocities[i]
        } else {
            particle.velocity
        };
        
        // Calculate velocity magnitude and normalize
        let velocity_magnitude = velocity.length();
        total_magnitude += velocity_magnitude;
        count += 1;
        max_seen = max_seen.max(velocity_magnitude);
        
        let normalized_velocity = (velocity_magnitude / MAX_VELOCITY).clamp(0.0, 1.0);
        
        // Create a color based on velocity direction and magnitude
        // Use 3D direction for more varied coloring
        let velocity_dir = if velocity_magnitude > 0.1 {
            velocity.normalize()
        } else {
            Vec3::new(0.0, 1.0, 0.0) // Default direction if velocity is nearly zero
        };
        
        // Map 3D direction to hue using a spherical mapping
        // This creates more varied colors based on full 3D direction
        let phi = f32::atan2(velocity_dir.z, velocity_dir.x);
        let theta = f32::acos(velocity_dir.y.clamp(-1.0, 1.0));
        let hue = ((phi / std::f32::consts::PI) * 0.5 + 0.5 + (theta / std::f32::consts::PI) * 0.2) % 1.0;
        
        // Convert HSV to RGB with higher saturation for more vibrant colors
        let color = hsv_to_rgb(hue, 0.9, 0.5 + normalized_velocity * 0.5);
        
        // Update the material color without time-based pulse
        if let Some(mat) = materials.get_mut(&material.0) {
            mat.base_color = Color::srgb(
                color.0,
                color.1,
                color.2
            );
            
            // Set emission with properly converted color
            let emission_color = Color::srgb(
                color.0 * normalized_velocity * 0.5,
                color.1 * normalized_velocity * 0.5,
                color.2 * normalized_velocity * 0.5
            ).into();
            
            mat.emissive = emission_color;
        }
    }
    
    // Log debug info every 60 frames
    if count > 0 && (time.elapsed_secs() * 60.0) as i32 % 60 == 0 {
        info!(
            "3D Particles - Avg velocity: {:.2}, Max velocity: {:.2}, Using MAX_VELOCITY={:.2} for normalization",
            total_magnitude / count as f32,
            max_seen,
            MAX_VELOCITY
        );
    }
}

// Helper function to convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let h_prime = h * 6.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    
    (r + m, g + m, b + m)
} 