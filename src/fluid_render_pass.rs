use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
    render_resource::{
        BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePassDescriptor,
        ComputePipelineDescriptor, PipelineCache, ShaderStages, StorageTextureAccess, TextureFormat,
        TextureSampleType, TextureViewDimension,
    },
    renderer::{RenderContext, RenderDevice},
};
use bevy::render::render_resource::{
    CachedComputePipelineId, Extent3d, TextureDescriptor, TextureDimension,
    TextureUsages,
};
use bevy::asset::RenderAssetUsages;

pub struct FluidRenderPassPlugin;

impl Plugin for FluidRenderPassPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FluidRenderPipelineData>();
    }
}

#[derive(Resource)]
pub struct FluidRenderPipelineData {
    pub depth_texture: Option<Handle<Image>>,
    pub filtered_texture: Option<Handle<Image>>,
    pub bind_group_layout: Option<BindGroupLayout>,
    pub pipeline_id: Option<CachedComputePipelineId>,
}

impl Default for FluidRenderPipelineData {
    fn default() -> Self {
        Self {
            depth_texture: None,
            filtered_texture: None,
            bind_group_layout: None,
            pipeline_id: None,
        }
    }
}

impl FluidRenderPipelineData {
    pub fn create_textures(
        &mut self,
        _render_device: &RenderDevice,
        images: &mut Assets<Image>,
        width: u32,
        height: u32,
    ) {
        let texture_descriptor = TextureDescriptor {
            label: Some("fluid_depth_texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        // Create depth texture
        let depth_image = Image::new_uninit(
            texture_descriptor.size,
            texture_descriptor.dimension,
            texture_descriptor.format,
            RenderAssetUsages::RENDER_WORLD,
        );
        self.depth_texture = Some(images.add(depth_image));

        // Create filtered texture
        let filtered_image = Image::new_uninit(
            texture_descriptor.size,
            texture_descriptor.dimension,
            texture_descriptor.format,
            RenderAssetUsages::RENDER_WORLD,
        );
        self.filtered_texture = Some(images.add(filtered_image));
    }

    pub fn create_bind_group_layout(&mut self, render_device: &RenderDevice) {
        let entries = vec![
            // Input depth texture
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Output filtered texture
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // Filter parameters
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        
        let layout = render_device.create_bind_group_layout(
            Some("fluid_bilateral_filter_layout"),
            &entries,
        );
        self.bind_group_layout = Some(layout);
    }

    pub fn create_compute_pipeline(&mut self, pipeline_cache: &PipelineCache, asset_server: &AssetServer) {
        if let Some(layout) = &self.bind_group_layout {
            let shader = asset_server.load("shaders/bilateral_filter.wgsl");
            let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("fluid_bilateral_filter_pipeline".into()),
                layout: vec![layout.clone()],
                push_constant_ranges: vec![],
                shader: shader,
                shader_defs: vec![],
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });
            self.pipeline_id = Some(pipeline_id);
        }
    }
}

pub struct FluidBilateralFilterNode;

impl Node for FluidBilateralFilterNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new("view", SlotType::Entity)]
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_data = world.resource::<FluidRenderPipelineData>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if let (Some(pipeline_id), Some(_depth_texture), Some(_filtered_texture)) = (
            &pipeline_data.pipeline_id,
            &pipeline_data.depth_texture,
            &pipeline_data.filtered_texture,
        ) {
            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(*pipeline_id) {
                let mut compute_pass = render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("fluid_bilateral_filter_pass"),
                        timestamp_writes: None,
                    });

                compute_pass.set_pipeline(pipeline);
                
                // TODO: Set bind groups and dispatch
                // This is where we'll bind textures and run the bilateral filter
                // For now, just a placeholder
                
                // Dispatch with appropriate workgroup sizes
                compute_pass.dispatch_workgroups(1, 1, 1);
            }
        }

        Ok(())
    }
}