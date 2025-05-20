use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BufferUsages, BufferInitDescriptor, BindGroupLayoutEntry, BindingType,
            BufferBindingType, ShaderStages, Buffer, BindGroup, BindGroupLayoutDescriptor,
            BindGroupDescriptor, BindGroupEntry,
        },
        renderer::RenderDevice,
    },
};
use bytemuck::{Pod, Zeroable};

#[derive(Resource)]
pub struct ParticleRenderResources {
    pub instance_buffer: Option<Buffer>,
    pub args_buffer: Option<Buffer>,
    pub bind_group: Option<BindGroup>,
}

impl Default for ParticleRenderResources {
    fn default() -> Self {
        Self {
            instance_buffer: None,
            args_buffer: None,
            bind_group: None,
        }
    }
}

pub fn create_particle_render_resources(
    render_device: &RenderDevice,
    particle_count: u32,
) -> ParticleRenderResources {
    // Create instance buffer for particle data
    let instance_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("particle_instance_buffer"),
        contents: &vec![0u8; (particle_count as usize * std::mem::size_of::<ParticleInstance>())],
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    });

    // Create indirect args buffer
    let args = [6u32, particle_count, 0, 0, 0]; // 6 vertices per quad, particle_count instances
    let args_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("particle_args_buffer"),
        contents: bytemuck::cast_slice(&args),
        usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = render_device.create_bind_group_layout(
        "particle_bind_group_layout",
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    // Create bind group
    let bind_group = render_device.create_bind_group(
        "particle_bind_group",
        &bind_group_layout,
        &[BindGroupEntry {
            binding: 0,
            resource: instance_buffer.as_entire_binding(),
        }],
    );

    ParticleRenderResources {
        instance_buffer: Some(instance_buffer),
        args_buffer: Some(args_buffer),
        bind_group: Some(bind_group),
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ParticleInstance {
    position: [f32; 3],
    scale: f32,
    color: [f32; 4],
} 