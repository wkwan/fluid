use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindGroupEntry,
            BindingType, Buffer, BufferBindingType, BufferUsages, BufferDescriptor,
            BufferInitDescriptor, PipelineLayoutDescriptor, RenderPipeline,
            RenderPipelineDescriptor, ShaderStages, VertexState, FragmentState,
            ColorTargetState, ColorWrite, BlendState, BlendFactor, BlendOperation,
            PrimitiveState, MultisampleState, DepthStencilState, TextureFormat,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
        view::VisibilityBundle,
    },
    asset::AssetServer,
};
use bytemuck::{Pod, Zeroable, cast_slice};
use std::borrow::Cow;

use crate::GpuState;
use crate::simulation3d::Particle3D;

#[derive(Resource, Default)]
pub struct GpuPerformanceStats {
    pub frame_time: f32,
    pub compute_time: f32,
    pub render_time: f32,
    pub particle_count: u32,
    pub draw_calls: u32,
    pub triangles: u32,
}

#[derive(Resource)]
pub struct GpuRenderPipelines3D {
    pub particle_pipeline: Option<RenderPipeline>,
    pub bind_group_layout: Option<BindGroupLayout>,
}

impl Default for GpuRenderPipelines3D {
    fn default() -> Self {
        Self {
            particle_pipeline: None,
            bind_group_layout: None,
        }
    }
}

#[derive(Resource)]
pub struct GpuRenderBindGroups3D {
    pub bind_group: Option<BindGroup>,
    pub particle_buffer: Option<Buffer>,
    pub camera_buffer: Option<Buffer>,
}

impl Default for GpuRenderBindGroups3D {
    fn default() -> Self {
        Self {
            bind_group: None,
            particle_buffer: None,
            camera_buffer: None,
        }
    }
}

// Plugin for GPU-driven rendering
pub struct GpuRender3DPlugin;

impl Plugin for GpuRender3DPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuPerformanceStats>();
        
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<GpuRenderPipelines3D>()
            .init_resource::<GpuRenderBindGroups3D>()
            .add_systems(ExtractSchedule, extract_render_data_3d)
            .add_systems(Render, prepare_render_bind_groups_3d.after(RenderSet::Prepare))
            .add_systems(Render, queue_render_3d.after(RenderSet::Queue));
    }
}

// Extract render data from the main world
fn extract_render_data_3d(
    mut commands: Commands,
    particles: Extract<Query<(&Particle3D, &Transform)>>,
    camera: Extract<Query<(&Camera, &GlobalTransform)>>,
    gpu_state: Extract<Res<GpuState>>,
) {
    if !gpu_state.enabled {
        return;
    }

    // Extract camera data
    if let Some((camera, transform)) = camera.iter().next() {
        let view = transform.compute_matrix();
        let proj = camera.projection_matrix();
        let view_proj = proj * view;
        
        commands.insert_resource(ExtractedCameraData {
            view_proj,
            view,
            position: transform.translation(),
        });
    }
}

// Prepare render pipelines and bind groups
fn prepare_render_bind_groups_3d(
    mut render_pipelines: ResMut<GpuRenderPipelines3D>,
    mut render_bind_groups: ResMut<GpuRenderBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_data: Option<Res<ExtractedCameraData>>,
    asset_server: Res<AssetServer>,
) {
    let extracted_data = match extracted_data {
        Some(data) => data,
        None => return,
    };
    
    // Create bind group layout if it doesn't exist
    if render_pipelines.bind_group_layout.is_none() {
        let layout_entries = vec![
            // Binding 0: Particle buffer
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Camera uniforms
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        
        render_pipelines.bind_group_layout = Some(
            render_device.create_bind_group_layout("particle_render_3d_bind_group_layout", &layout_entries)
        );
    }
    
    // Create render pipeline if it doesn't exist
    if render_pipelines.particle_pipeline.is_none() {
        let shader = asset_server.load("shaders/particle_render3d.wgsl");
        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("particle_render_3d_pipeline_layout"),
            bind_group_layouts: &[render_pipelines.bind_group_layout.as_ref().unwrap()],
            push_constant_ranges: &[],
        });
        
        let pipeline_descriptor = RenderPipelineDescriptor {
            label: Some("particle_render_3d_pipeline"),
            layout: Some(pipeline_layout),
            vertex: VertexState {
                shader,
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![],
            },
            fragment: Some(FragmentState {
                shader,
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![ColorTargetState {
                    format: TextureFormat::Bgra8UnormSrgb,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrite::ALL,
                }],
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        };
        
        render_pipelines.particle_pipeline = Some(render_device.create_render_pipeline(pipeline_descriptor));
    }
    
    // Create or update camera buffer
    if render_bind_groups.camera_buffer.is_none() {
        render_bind_groups.camera_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("camera_uniforms_3d_buffer"),
            contents: bytemuck::bytes_of(&extracted_data),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    } else {
        render_queue.write_buffer(
            render_bind_groups.camera_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&extracted_data),
        );
    }
    
    // Create bind group if it doesn't exist
    if render_bind_groups.bind_group.is_none() &&
       render_bind_groups.particle_buffer.is_some() &&
       render_bind_groups.camera_buffer.is_some()
    {
        let bind_group = render_device.create_bind_group("particle_render_3d_bind_group",
            render_pipelines.bind_group_layout.as_ref().unwrap(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: render_bind_groups.particle_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: render_bind_groups.camera_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ]
        );
        
        render_bind_groups.bind_group = Some(bind_group);
    }
}

// Queue render commands
fn queue_render_3d(
    render_pipelines: Res<GpuRenderPipelines3D>,
    render_bind_groups: Res<GpuRenderBindGroups3D>,
    mut render_device: ResMut<RenderDevice>,
    mut render_queue: ResMut<RenderQueue>,
    mut performance_stats: ResMut<GpuPerformanceStats>,
) {
    if render_pipelines.particle_pipeline.is_none() ||
       render_bind_groups.bind_group.is_none() {
        return;
    }
    
    let start_time = std::time::Instant::now();
    
    // Create command encoder
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particle_render_3d_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_device.create_texture_view(&wgpu::TextureViewDescriptor {
                    label: Some("particle_render_3d_color_attachment"),
                    format: Some(TextureFormat::Bgra8UnormSrgb),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                }),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_device.create_texture_view(&wgpu::TextureViewDescriptor {
                    label: Some("particle_render_3d_depth_attachment"),
                    format: Some(TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                }),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        
        render_pass.set_pipeline(render_pipelines.particle_pipeline.as_ref().unwrap());
        render_pass.set_bind_group(0, render_bind_groups.bind_group.as_ref().unwrap(), &[]);
        render_pass.draw(0..6, 0..performance_stats.particle_count);
    }
    
    render_queue.submit(std::iter::once(encoder.finish()));
    
    // Update performance stats
    let end_time = std::time::Instant::now();
    performance_stats.render_time = (end_time - start_time).as_secs_f32() * 1000.0;
}

#[derive(Resource)]
struct ExtractedCameraData {
    view_proj: Mat4,
    view: Mat4,
    position: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuCameraUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    position: [f32; 3],
    _padding: f32,
} 