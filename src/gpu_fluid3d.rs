use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindGroupEntry, 
            BindingType, Buffer, BufferBindingType, BufferUsages, BufferDescriptor,
            BufferInitDescriptor, ComputePipeline, ComputePipelineDescriptor, 
            PipelineCache, ShaderStages, MapMode, Maintain,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
};
use bytemuck::{Pod, Zeroable, cast_slice};
use std::borrow::Cow;

use crate::GpuState;
use crate::simulation3d::{Particle3D, Fluid3DParams};
use crate::simulation::{MouseInteraction};

/// Plugin for GPU-accelerated 3D Fluid Simulation
pub struct GpuSim3DPlugin;

impl Plugin for GpuSim3DPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_gpu_particles_3d);
        
        // Register the render app systems
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidComputePipelines3D>()
            .init_resource::<FluidBindGroups3D>()
            .add_systems(ExtractSchedule, extract_fluid_data_3d)
            .add_systems(Render, prepare_fluid_bind_groups_3d.after(RenderSet::Prepare))
            .add_systems(Render, queue_fluid_compute_3d.after(RenderSet::Queue));
    }
}

// Main update system that runs the GPU simulation
fn update_gpu_particles_3d(
    mut particles: Query<(&mut Transform, &mut Particle3D)>,
    gpu_state: Res<GpuState>,
    gpu_particles: Res<GpuParticles3D>,
) {
    // Only process if GPU mode is enabled and data is updated
    if !gpu_state.enabled || !gpu_particles.updated {
        return;
    }
    
    // Update particles with the latest GPU simulation results
    for (i, (mut transform, mut particle)) in particles.iter_mut().enumerate() {
        if i < gpu_particles.positions.len() {
            transform.translation = gpu_particles.positions[i];
            particle.velocity = gpu_particles.velocities[i];
            particle.density = gpu_particles.densities[i];
            particle.pressure = gpu_particles.pressures[i];
            particle.near_density = gpu_particles.near_densities[i];
            particle.near_pressure = gpu_particles.near_pressures[i];
        }
    }
}

// Resource to store GPU-computed particle data that gets synchronized back to CPU
#[derive(Resource, Default, Debug)]
pub struct GpuParticles3D {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub densities: Vec<f32>,
    pub pressures: Vec<f32>,
    pub near_densities: Vec<f32>,
    pub near_pressures: Vec<f32>,
    pub updated: bool,
    pub particle_count: usize,
}

// GPU-compatible structures with padding for alignment
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticleData3D {
    position: [f32; 3],
    padding0: f32,  // Padding for 16-byte alignment
    velocity: [f32; 3],
    padding1: f32,  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Fluid parameters with padding for GPU alignment
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct GpuFluidParams3D {
    // Vec4 aligned group 1
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    // Vec4 aligned group 2
    viscosity_strength: f32,
    boundary_dampening: f32,
    particle_radius: f32,
    dt: f32,
    
    // Vec4 aligned group 3
    boundary_min: [f32; 3],
    boundary_min_padding: f32,
    
    // Vec4 aligned group 4
    boundary_max: [f32; 3],
    boundary_max_padding: f32,
    
    // Vec4 aligned group 5
    gravity: [f32; 3],
    gravity_padding: f32,
    
    // Vec4 aligned group 6
    mouse_position: [f32; 3],
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: [u32; 2],
}

// Resources for the render app
#[derive(Resource)]
struct FluidComputePipelines3D {
    spatial_hash_pipeline: Option<ComputePipeline>,
    density_pressure_pipeline: Option<ComputePipeline>,
    pressure_force_pipeline: Option<ComputePipeline>,
    viscosity_pipeline: Option<ComputePipeline>,
    update_positions_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
    neighbor_reduction_pipeline: Option<ComputePipeline>,
}

impl Default for FluidComputePipelines3D {
    fn default() -> Self {
        Self {
            spatial_hash_pipeline: None,
            density_pressure_pipeline: None,
            pressure_force_pipeline: None,
            viscosity_pipeline: None,
            update_positions_pipeline: None,
            bind_group_layout: None,
            neighbor_reduction_pipeline: None,
        }
    }
}

#[derive(Resource)]
struct FluidBindGroups3D {
    bind_group: Option<BindGroup>,
    particle_buffer: Option<Buffer>,
    params_buffer: Option<Buffer>,
    spatial_keys_buffer: Option<Buffer>,
    spatial_offsets_buffer: Option<Buffer>,
    num_particles: u32,
    neighbor_counts_buffer: Option<Buffer>,
    neighbor_indices_buffer: Option<Buffer>,
    readback_buffer: Option<Buffer>,
}

impl Default for FluidBindGroups3D {
    fn default() -> Self {
        Self {
            bind_group: None,
            particle_buffer: None,
            params_buffer: None,
            spatial_keys_buffer: None,
            spatial_offsets_buffer: None,
            num_particles: 0,
            neighbor_counts_buffer: None,
            neighbor_indices_buffer: None,
            readback_buffer: None,
        }
    }
}

// Resource to store extracted fluid data for GPU processing
#[derive(Resource)]
struct ExtractedFluidData3D {
    params: Fluid3DParams,
    mouse: MouseInteraction,
    dt: f32,
    num_particles: usize,
    particle_positions: Vec<Vec3>,
    particle_velocities: Vec<Vec3>,
    particle_densities: Vec<f32>,
    particle_pressures: Vec<f32>,
    near_densities: Vec<f32>,
    near_pressures: Vec<f32>,
}

// Extract fluid data from the main world to the render world
fn extract_fluid_data_3d(
    mut commands: Commands,
    fluid_params: Extract<Res<Fluid3DParams>>,
    mouse_interaction: Extract<Res<MouseInteraction>>,
    time: Extract<Res<Time>>,
    particles: Extract<Query<(&Particle3D, &Transform)>>,
    gpu_state: Extract<Res<GpuState>>,
) {
    // Skip extraction if GPU is disabled
    if !gpu_state.enabled {
        return;
    }

    let mut positions = Vec::with_capacity(particles.iter().len());
    let mut velocities = Vec::with_capacity(particles.iter().len());
    let mut densities = Vec::with_capacity(particles.iter().len());
    let mut pressures = Vec::with_capacity(particles.iter().len());
    let mut near_densities = Vec::with_capacity(particles.iter().len());
    let mut near_pressures = Vec::with_capacity(particles.iter().len());

    for (particle, transform) in particles.iter() {
        positions.push(transform.translation);
        velocities.push(particle.velocity);
        densities.push(particle.density);
        pressures.push(particle.pressure);
        near_densities.push(particle.near_density);
        near_pressures.push(particle.near_pressure);
    }

    commands.insert_resource(ExtractedFluidData3D {
        params: fluid_params.clone(),
        mouse: mouse_interaction.clone(),
        dt: time.delta_secs(),
        num_particles: particles.iter().len(),
        particle_positions: positions,
        particle_velocities: velocities,
        particle_densities: densities,
        particle_pressures: pressures,
        near_densities,
        near_pressures,
    });
}

// Prepare the compute pipeline and bind group layout
fn prepare_fluid_bind_groups_3d(
    mut fluid_pipelines: ResMut<FluidComputePipelines3D>,
    mut fluid_bind_groups: ResMut<FluidBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_data: Option<Res<ExtractedFluidData3D>>,
    asset_server: Res<AssetServer>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    // Skip if no data has been extracted
    let extracted_data = match extracted_data {
        Some(data) => data,
        None => return,
    };
    
    // Create the bind group layout if it doesn't exist
    if fluid_pipelines.bind_group_layout.is_none() {
        let layout_entries = vec![
            // Binding 0: Particle buffer (read-write)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Parameters buffer (uniform)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Spatial Keys (read-write)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 3: Spatial Offsets (read-write)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 4: Neighbor counts (read-write)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 5: Neighbor indices (read-write)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        
        fluid_pipelines.bind_group_layout = Some(
            render_device.create_bind_group_layout("fluid_compute_3d_bind_group_layout", &layout_entries)
        );
    }
    
    // Create the compute pipelines if they don't exist
    if fluid_pipelines.spatial_hash_pipeline.is_none() {
        let shader = asset_server.load("shaders/fluid_sim3d.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_spatial_hash_3d_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.spatial_hash_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    if fluid_pipelines.density_pressure_pipeline.is_none() {
        let shader = asset_server.load("shaders/density_pressure3d.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_density_pressure_3d_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.density_pressure_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    if fluid_pipelines.pressure_force_pipeline.is_none() {
        let shader = asset_server.load("shaders/pressure_force3d.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_pressure_force_3d_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.pressure_force_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    if fluid_pipelines.viscosity_pipeline.is_none() {
        let shader = asset_server.load("shaders/viscosity_force3d.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_viscosity_3d_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.viscosity_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    if fluid_pipelines.update_positions_pipeline.is_none() {
        let shader = asset_server.load("shaders/update_particles3d.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_update_positions_3d_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.update_positions_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    // Create or update buffers
    let particle_count = extracted_data.num_particles;
    let mut particle_data: Vec<GpuParticleData3D> = Vec::with_capacity(particle_count);
    for i in 0..particle_count {
        particle_data.push(GpuParticleData3D {
            position: extracted_data.particle_positions[i].to_array(),
            padding0: 0.0,
            velocity: extracted_data.particle_velocities[i].to_array(),
            padding1: 0.0,
            density: extracted_data.particle_densities[i],
            pressure: extracted_data.particle_pressures[i],
            near_density: extracted_data.near_densities[i],
            near_pressure: extracted_data.near_pressures[i],
        });
    }
    
    // Build GPU params struct
    let gpu_params = GpuFluidParams3D {
        smoothing_radius: extracted_data.params.smoothing_radius,
        target_density: extracted_data.params.target_density,
        pressure_multiplier: extracted_data.params.pressure_multiplier,
        near_pressure_multiplier: 0.0,
        viscosity_strength: extracted_data.params.viscosity_strength,
        boundary_dampening: 0.3,
        particle_radius: 5.0,
        dt: extracted_data.dt,
        boundary_min: [ -300.0, -300.0, -300.0 ],
        boundary_min_padding: 0.0,
        boundary_max: [ 300.0, 300.0, 300.0 ],
        boundary_max_padding: 0.0,
        gravity: [ 0.0, -9.81, 0.0 ],
        gravity_padding: 0.0,
        mouse_position: [ 0.0, 0.0, 0.0 ],
        mouse_radius: 0.0,
        mouse_strength: 0.0,
        mouse_active: 0,
        mouse_repel: 0,
        padding: [0,0],
    };
    
    // Create or update the parameters buffer
    if fluid_bind_groups.params_buffer.is_none() {
        fluid_bind_groups.params_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid_params_3d_buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    } else {
        render_queue.write_buffer(
            fluid_bind_groups.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&gpu_params),
        );
    }
    
    // Create or update the particle buffer
    if fluid_bind_groups.particle_buffer.is_none() || fluid_bind_groups.num_particles != particle_count as u32 {
        fluid_bind_groups.particle_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid_particle_3d_buffer"),
            contents: cast_slice(&particle_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        
        // Create spatial hashing buffers with the same size
        fluid_bind_groups.spatial_keys_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_spatial_keys_3d_buffer"),
            size: (particle_count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.spatial_offsets_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_spatial_offsets_3d_buffer"),
            size: (particle_count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.num_particles = particle_count as u32;
    } else {
        // Update existing buffer data
        render_queue.write_buffer(
            fluid_bind_groups.particle_buffer.as_ref().unwrap(),
            0,
            cast_slice(&particle_data),
        );
    }
    
    // Create or resize readback buffer (positions + velocities) for debug every N frames
    let particle_buffer_size = (particle_count as u64) * std::mem::size_of::<GpuParticleData3D>() as u64;
    if fluid_bind_groups.readback_buffer.is_none() || fluid_bind_groups.num_particles != particle_count as u32 {
        fluid_bind_groups.readback_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_readback_3d_buffer"),
            size: particle_buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }
    
    // Create neighbor reduction pipeline if it doesn't exist
    if fluid_pipelines.neighbor_reduction_pipeline.is_none() {
        let shader = asset_server.load("shaders/neighbor_reduction3d.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_neighbor_reduction_3d_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.neighbor_reduction_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    // Create or resize neighbor buffers
    if fluid_bind_groups.neighbor_counts_buffer.is_none() || 
       fluid_bind_groups.neighbor_indices_buffer.is_none() ||
       fluid_bind_groups.num_particles != particle_count as u32 {
        
        // Create neighbor counts buffer
        fluid_bind_groups.neighbor_counts_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_neighbor_counts_3d_buffer"),
            size: (particle_count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        // Create neighbor indices buffer (max 128 neighbors per particle)
        let max_neighbors = 128;
        fluid_bind_groups.neighbor_indices_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_neighbor_indices_3d_buffer"),
            size: (particle_count * max_neighbors * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    
    // Create bind group
    if fluid_bind_groups.bind_group.is_none() && 
       fluid_bind_groups.particle_buffer.is_some() && 
       fluid_bind_groups.params_buffer.is_some() &&
       fluid_bind_groups.spatial_keys_buffer.is_some() &&
       fluid_bind_groups.spatial_offsets_buffer.is_some() &&
       fluid_bind_groups.neighbor_counts_buffer.is_some() &&
       fluid_bind_groups.neighbor_indices_buffer.is_some()
    {
        let bind_group = render_device.create_bind_group("fluid_compute_3d_bind_group", 
            fluid_pipelines.bind_group_layout.as_ref().unwrap(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: fluid_bind_groups.particle_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: fluid_bind_groups.params_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: fluid_bind_groups.spatial_keys_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: fluid_bind_groups.spatial_offsets_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: fluid_bind_groups.neighbor_counts_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: fluid_bind_groups.neighbor_indices_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ]
        );
        
        fluid_bind_groups.bind_group = Some(bind_group);
    }
}

// Execute the compute shaders to perform the full fluid simulation on GPU
fn queue_fluid_compute_3d(
    fluid_pipelines: Res<FluidComputePipelines3D>,
    fluid_bind_groups: Res<FluidBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // Skip if any required pipeline is missing
    if fluid_pipelines.spatial_hash_pipeline.is_none() ||
       fluid_pipelines.density_pressure_pipeline.is_none() ||
       fluid_pipelines.pressure_force_pipeline.is_none() ||
       fluid_pipelines.viscosity_pipeline.is_none() ||
       fluid_pipelines.update_positions_pipeline.is_none() ||
       fluid_bind_groups.bind_group.is_none() {
        return;
    }
    
    // Determine workgroup count based on particle count
    let particle_count = fluid_bind_groups.num_particles;
    if particle_count == 0 {
        return;
    }
    
    // First batch: spatial hash + neighbor reduction
    let mut encoder = render_device.create_command_encoder(&Default::default());

    {
        // 1. Build spatial hash
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.spatial_hash_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups((particle_count + 127) / 128, 1, 1);
    }

    {
        // 2. Neighbor reduction
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.neighbor_reduction_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups((particle_count + 127) / 128, 1, 1);
    }

    // Submit batch
    render_queue.submit(std::iter::once(encoder.finish()));

    // Second batch: density & pressure
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        // 2. Calculate densities and pressures
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.density_pressure_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups((particle_count + 127) / 128, 1, 1);
    }
    
    // Submit second batch
    render_queue.submit(std::iter::once(encoder.finish()));
    
    // Create command encoder for the third batch (forces)
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        // 3. Calculate pressure forces
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.pressure_force_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups((particle_count + 127) / 128, 1, 1);
    }
    
    {
        // 4. Calculate viscosity forces
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.viscosity_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups((particle_count + 127) / 128, 1, 1);
    }
    
    // Submit third batch
    render_queue.submit(std::iter::once(encoder.finish()));

    // Create command encoder for the fourth batch (integration)
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        // 5. Update positions and handle collisions
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.update_positions_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups((particle_count + 127) / 128, 1, 1);
    }

    // Copy data to readback buffer (debug every frame for now)
    if let Some(readback) = &fluid_bind_groups.readback_buffer {
        encoder.copy_buffer_to_buffer(
            fluid_bind_groups.particle_buffer.as_ref().unwrap(),
            0,
            readback,
            0,
            (particle_count as u64) * std::mem::size_of::<GpuParticleData3D>() as u64,
        );
    }

    render_queue.submit(std::iter::once(encoder.finish()));

    // Simple immediate mapping (could throttle to N frames)
    if let Some(readback) = &fluid_bind_groups.readback_buffer {
        // map asynchronously
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        // poll device to drive mapping
        render_device.wgpu_device().poll(Maintain::Wait);
        let _data = slice.get_mapped_range();
        // TODO: deserialize into GpuParticles3D for debugging
        readback.unmap();
    }
} 