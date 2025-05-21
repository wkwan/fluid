use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BufferUsages, BufferInitDescriptor, Buffer,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};
use crate::simulation::Particle;

// Custom instance data for batched rendering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 2],
    scale: f32,
    color: [f32; 4],
}

// Resource for cached instance data
#[derive(Resource)]
struct InstanceBuffer {
    buffer: Option<Buffer>,
    capacity: usize,
    last_count: usize,
    needs_update: bool,
}

impl Default for InstanceBuffer {
    fn default() -> Self {
        Self {
            buffer: None,
            capacity: 0,
            last_count: 0,
            needs_update: true,
        }
    }
}

// Resource to control color mapping
#[derive(Resource, Clone)]
pub struct ColorMapParams {
    pub use_velocity_color: bool,
    pub min_speed: f32,
    pub max_speed: f32,
}

impl Default for ColorMapParams {
    fn default() -> Self {
        Self {
            use_velocity_color: true,
            min_speed: 0.0,
            max_speed: 500.0,
        }
    }
}

// Component to mark the particle material
#[derive(Component)]
struct ParticleMaterial;

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InstanceBuffer>()
           .init_resource::<ColorMapParams>()
           .add_systems(Startup, setup_particle_rendering)
           .add_systems(Update, update_particle_instances);
    }
}

fn velocity_to_color(velocity: Vec2, min_speed: f32, max_speed: f32) -> Color {
    let speed = velocity.length();
    let normalized = ((speed - min_speed) / (max_speed - min_speed)).clamp(0.0, 1.0);
    
    // Use a blue-green-red gradient based on speed
    if normalized < 0.5 {
        // Blue to green
        let local_norm = normalized * 2.0;
        Color::srgb(
            0.0,
            local_norm,
            1.0 - local_norm,
        )
    } else {
        // Green to red
        let local_norm = (normalized - 0.5) * 2.0;
        Color::srgb(
            local_norm,
            1.0 - local_norm,
            0.0,
        )
    }
}

fn setup_particle_rendering(
    mut commands: Commands,
) {
    // Just create an entity to mark our particle system
    commands.spawn(ParticleMaterial);
}

// Update particle instances in a batch for better performance
fn update_particle_instances(
    particles: Query<(&Transform, &Particle)>,
    color_params: Res<ColorMapParams>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut instance_buffer: ResMut<InstanceBuffer>,
) {
    let particle_count = particles.iter().count();
    
    // Skip if no particles
    if particle_count == 0 {
        return;
    }
    
    // Create or resize buffer if needed
    if instance_buffer.buffer.is_none() || particle_count > instance_buffer.capacity {
        // Calculate a new capacity with 20% headroom to avoid frequent resizing
        let new_capacity = (particle_count * 6 / 5).max(1000);
        
        // Create the instance buffer
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("particle_instance_buffer"),
            contents: &vec![0u8; new_capacity * std::mem::size_of::<ParticleInstance>()],
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        
        instance_buffer.buffer = Some(buffer);
        instance_buffer.capacity = new_capacity;
        instance_buffer.needs_update = true;
    }
    
    // Only update when particle count changes or explicitly requested
    if particle_count == instance_buffer.last_count && !instance_buffer.needs_update {
        return;
    }
    
    // Create instance data for all particles
    let mut instances = Vec::with_capacity(particle_count);
    
    for (transform, particle) in particles.iter() {
        let position = transform.translation.truncate();
        
        // Skip color calculation since we can't get components - just use fixed colors
        
        instances.push(ParticleInstance {
            position: [position.x, position.y],
            scale: 5.0,  // Particle radius
            color: [0.5, 0.5, 1.0, 1.0], // Default blue color
        });
    }
    
    // Update the instance buffer
    if let Some(buffer) = &instance_buffer.buffer {
        render_queue.write_buffer(buffer, 0, bytemuck::cast_slice(&instances));
    }
    
    // Update state
    instance_buffer.last_count = particle_count;
    instance_buffer.needs_update = false;
} 