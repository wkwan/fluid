use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindGroupEntry, 
            BindingType, Buffer, BufferBindingType, BufferUsages, BufferDescriptor,
            BufferInitDescriptor, ComputePipeline, ComputePipelineDescriptor, 
            PipelineCache, ShaderStages,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
};
use bytemuck::{Pod, Zeroable, cast_slice};
use std::borrow::Cow;

use crate::simulation::{Particle, FluidParams, MouseInteraction};
use crate::gpu_fluid::GpuState;

/// Plugin for GPU-accelerated Fluid Simulation
pub struct GpuSimPlugin;

impl Plugin for GpuSimPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_gpu_particles);
        
        // Register the render app systems
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidComputePipelines>()
            .init_resource::<FluidBindGroups>()
            .add_systems(ExtractSchedule, extract_fluid_data)
            .add_systems(Render, prepare_fluid_bind_groups.after(RenderSet::Prepare))
            .add_systems(Render, queue_fluid_compute.after(RenderSet::Queue));
    }
}

// Main update system that runs the GPU simulation
fn update_gpu_particles(
    mut particles: Query<(&mut Transform, &mut Particle)>,
    gpu_state: Res<GpuState>,
    gpu_particles: Res<GpuParticles>,
) {
    // Only process if GPU mode is enabled and data is updated
    if !gpu_state.enabled || !gpu_particles.updated {
        return;
    }
    
    // Update particles with the latest GPU simulation results
    for (i, (mut transform, mut particle)) in particles.iter_mut().enumerate() {
        if i < gpu_particles.positions.len() {
            transform.translation.x = gpu_particles.positions[i].x;
            transform.translation.y = gpu_particles.positions[i].y;
            particle.velocity.x = gpu_particles.velocities[i].x;
            particle.velocity.y = gpu_particles.velocities[i].y;
            particle.density = gpu_particles.densities[i];
            particle.pressure = gpu_particles.pressures[i];
            particle.near_density = gpu_particles.near_densities[i];
            particle.near_pressure = gpu_particles.near_pressures[i];
        }
    }
}

// Resource to store GPU-computed particle data that gets synchronized back to CPU
#[derive(Resource, Default, Debug)]
pub struct GpuParticles {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
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
struct GpuParticleData {
    position: [f32; 2],
    padding0: [f32; 2],  // Padding for 16-byte alignment
    velocity: [f32; 2],
    padding1: [f32; 2],  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Fluid parameters with padding for GPU alignment
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct GpuFluidParams {
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
    boundary_min: [f32; 2],
    boundary_min_padding: [f32; 2],
    
    // Vec4 aligned group 4
    boundary_max: [f32; 2],
    boundary_max_padding: [f32; 2],
    
    // Vec4 aligned group 5
    gravity: [f32; 2],
    gravity_padding: [f32; 2],
    
    // Vec4 aligned group 6
    mouse_position: [f32; 2],
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: [u32; 2],
}

// Resources for the render app
#[derive(Resource)]
struct FluidComputePipelines {
    external_forces_pipeline: Option<ComputePipeline>,
    spatial_hash_pipeline: Option<ComputePipeline>,
    reorder_pipeline: Option<ComputePipeline>,
    density_pressure_pipeline: Option<ComputePipeline>,
    pressure_force_pipeline: Option<ComputePipeline>,
    viscosity_pipeline: Option<ComputePipeline>,
    update_positions_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
}

impl Default for FluidComputePipelines {
    fn default() -> Self {
        Self {
            external_forces_pipeline: None,
            spatial_hash_pipeline: None,
            reorder_pipeline: None,
            density_pressure_pipeline: None,
            pressure_force_pipeline: None,
            viscosity_pipeline: None,
            update_positions_pipeline: None,
            bind_group_layout: None,
        }
    }
}

#[derive(Resource)]
struct FluidBindGroups {
    particle_buffer: Option<Buffer>,
    params_buffer: Option<Buffer>,
    bind_group: Option<BindGroup>,
    num_particles: u32,
    spatial_keys_buffer: Option<Buffer>,
    spatial_offsets_buffer: Option<Buffer>,
    target_particles_buffer: Option<Buffer>,
}

impl Default for FluidBindGroups {
    fn default() -> Self {
        Self {
            particle_buffer: None,
            params_buffer: None,
            bind_group: None,
            num_particles: 0,
            spatial_keys_buffer: None,
            spatial_offsets_buffer: None,
            target_particles_buffer: None,
        }
    }
}

// Extract data from the main app to the render app
fn extract_fluid_data(
    mut commands: Commands,
    fluid_params: Extract<Res<FluidParams>>,
    mouse_interaction: Extract<Res<MouseInteraction>>,
    time: Extract<Res<Time>>,
    particles: Extract<Query<(&Particle, &Transform)>>,
    gpu_state: Extract<Res<GpuState>>,
) {
    // Skip if GPU mode is disabled
    if !gpu_state.enabled {
        return;
    }
    
    // Convert FluidParams to GPU-compatible format
    let gpu_params = GpuFluidParams {
        smoothing_radius: fluid_params.smoothing_radius,
        target_density: fluid_params.target_density,
        pressure_multiplier: fluid_params.pressure_multiplier,
        near_pressure_multiplier: fluid_params.near_pressure_multiplier,
        viscosity_strength: fluid_params.viscosity_strength,
        boundary_dampening: 0.3, // Hardcoded for now, would be better to expose
        particle_radius: 5.0,    // Hardcoded for now, would be better to expose
        dt: time.delta_secs(),
        boundary_min: [fluid_params.boundary_min.x, fluid_params.boundary_min.y],
        boundary_min_padding: [0.0, 0.0],
        boundary_max: [fluid_params.boundary_max.x, fluid_params.boundary_max.y],
        boundary_max_padding: [0.0, 0.0],
        gravity: [0.0, -9.81],  // Hardcoded gravity, would be better to make configurable
        gravity_padding: [0.0, 0.0],
        mouse_position: [mouse_interaction.position.x, mouse_interaction.position.y],
        mouse_radius: mouse_interaction.radius,
        mouse_strength: mouse_interaction.strength,
        mouse_active: if mouse_interaction.active { 1 } else { 0 },
        mouse_repel: if mouse_interaction.repel { 1 } else { 0 },
        padding: [0, 0],
    };
    
    // Prepare particle data for GPU
    let mut gpu_particles_data = Vec::with_capacity(particles.iter().len());
    
    // Create positions and velocities vectors for synchronizing back to CPU later
    let mut positions = Vec::with_capacity(particles.iter().len());
    let mut velocities = Vec::with_capacity(particles.iter().len());
    let mut densities = Vec::with_capacity(particles.iter().len());
    let mut pressures = Vec::with_capacity(particles.iter().len());
    let mut near_densities = Vec::with_capacity(particles.iter().len());
    let mut near_pressures = Vec::with_capacity(particles.iter().len());
    
    for (particle, transform) in particles.iter() {
        let position = transform.translation.truncate();
        positions.push(position);
        velocities.push(particle.velocity);
        densities.push(particle.density);
        pressures.push(particle.pressure);
        near_densities.push(particle.near_density);
        near_pressures.push(particle.near_pressure);
        
        gpu_particles_data.push(GpuParticleData {
            position: [position.x, position.y],
            padding0: [0.0, 0.0],
            velocity: [particle.velocity.x, particle.velocity.y],
            padding1: [0.0, 0.0],
            density: particle.density,
            pressure: particle.pressure,
            near_density: particle.near_density,
            near_pressure: particle.near_pressure,
        });
    }
    
    // Store the particle data in a resource for the render world
    commands.insert_resource(ExtractedFluidData {
        params: gpu_params,
        particles: gpu_particles_data,
        particle_count: particles.iter().len(),
        positions,
        velocities,
        densities,
        pressures,
        near_densities,
        near_pressures,
    });
}

#[derive(Resource)]
struct ExtractedFluidData {
    params: GpuFluidParams,
    particles: Vec<GpuParticleData>,
    particle_count: usize,
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    densities: Vec<f32>,
    pressures: Vec<f32>,
    near_densities: Vec<f32>,
    near_pressures: Vec<f32>,
}

// Prepare GPU buffers and bind groups
fn prepare_fluid_bind_groups(
    mut fluid_pipelines: ResMut<FluidComputePipelines>,
    mut fluid_bind_groups: ResMut<FluidBindGroups>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_data: Option<Res<ExtractedFluidData>>,
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
            // Binding 4: Target Particles buffer for reordering (read-write)
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
        ];
        
        fluid_pipelines.bind_group_layout = Some(
            render_device.create_bind_group_layout("fluid_compute_bind_group_layout", &layout_entries)
        );
    }
    
    // Create the compute pipelines if they don't exist
    if fluid_pipelines.external_forces_pipeline.is_none() {
        let shader = asset_server.load("shaders/external_forces.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_external_forces_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.external_forces_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    if fluid_pipelines.spatial_hash_pipeline.is_none() {
        let shader = asset_server.load("shaders/spatial_hash.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_spatial_hash_pipeline")),
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
    
    if fluid_pipelines.reorder_pipeline.is_none() {
        let shader = asset_server.load("shaders/reorder.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_reorder_pipeline")),
            layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipelines.reorder_pipeline = pipeline_cache.get_compute_pipeline(pipeline_id).cloned();
    }
    
    if fluid_pipelines.density_pressure_pipeline.is_none() {
        let shader = asset_server.load("shaders/density_pressure.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_density_pressure_pipeline")),
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
        let shader = asset_server.load("shaders/pressure_force.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_pressure_force_pipeline")),
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
        let shader = asset_server.load("shaders/viscosity.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_viscosity_pipeline")),
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
        let shader = asset_server.load("shaders/update_positions.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("fluid_update_positions_pipeline")),
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
    let particle_count = extracted_data.particle_count;
    let particle_data = &extracted_data.particles;
    
    // Create or resize the particle buffer
    if fluid_bind_groups.particle_buffer.is_none() || fluid_bind_groups.num_particles != particle_count as u32 {
        fluid_bind_groups.particle_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid_particle_buffer"),
            contents: cast_slice(particle_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        
        // Create spatial hashing buffers with the same size
        fluid_bind_groups.spatial_keys_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_spatial_keys_buffer"),
            size: (particle_count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.spatial_offsets_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_spatial_offsets_buffer"),
            size: (particle_count * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.target_particles_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("fluid_target_particles_buffer"),
            size: (particle_count * std::mem::size_of::<GpuParticleData>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.num_particles = particle_count as u32;
    } else {
        // Update existing buffer data
        render_queue.write_buffer(
            fluid_bind_groups.particle_buffer.as_ref().unwrap(),
            0,
            cast_slice(particle_data),
        );
    }
    
    // Create or update the parameters buffer
    if fluid_bind_groups.params_buffer.is_none() {
        fluid_bind_groups.params_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid_params_buffer"),
            contents: bytemuck::bytes_of(&extracted_data.params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    } else {
        render_queue.write_buffer(
            fluid_bind_groups.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&extracted_data.params),
        );
    }
    
    // Create bind group
    if fluid_bind_groups.bind_group.is_none() && 
       fluid_bind_groups.particle_buffer.is_some() && 
       fluid_bind_groups.params_buffer.is_some() &&
       fluid_bind_groups.spatial_keys_buffer.is_some() &&
       fluid_bind_groups.spatial_offsets_buffer.is_some() &&
       fluid_bind_groups.target_particles_buffer.is_some() {
        
        let bind_group = render_device.create_bind_group("fluid_compute_bind_group", 
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
                    resource: fluid_bind_groups.target_particles_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ]
        );
        
        fluid_bind_groups.bind_group = Some(bind_group);
    }
}

// Execute the compute shaders to perform the full fluid simulation on GPU
fn queue_fluid_compute(
    fluid_pipelines: Res<FluidComputePipelines>,
    fluid_bind_groups: Res<FluidBindGroups>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
    extracted_data: Option<Res<ExtractedFluidData>>,
) {
    // Skip if any required pipeline is missing
    if fluid_pipelines.external_forces_pipeline.is_none() ||
       fluid_pipelines.spatial_hash_pipeline.is_none() ||
       fluid_pipelines.reorder_pipeline.is_none() ||
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
    
    let workgroup_size = 128;
    let workgroup_count = ((particle_count + workgroup_size - 1) / workgroup_size) as u32;
    
    // Create command encoder
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        // 1. External forces pass
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.external_forces_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    {
        // 2. Spatial hashing pass
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.spatial_hash_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    {
        // 3. Reorder pass - sorts particles by spatial hash for faster neighbor lookups
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.reorder_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    // Submit first batch of operations to avoid long compute passes
    render_queue.submit(std::iter::once(encoder.finish()));
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        // 4. Calculate densities and pressures
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.density_pressure_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    {
        // 5. Calculate pressure forces
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.pressure_force_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    // Submit second batch
    render_queue.submit(std::iter::once(encoder.finish()));
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    {
        // 6. Calculate viscosity forces
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.viscosity_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    {
        // 7. Update positions and handle collisions
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(fluid_pipelines.update_positions_pipeline.as_ref().unwrap());
        compute_pass.set_bind_group(0, fluid_bind_groups.bind_group.as_ref().unwrap(), &[]);
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    // For simplicity, we'll send the data back to the main app
    if let Some(data) = extracted_data {
        commands.insert_resource(GpuParticles {
            positions: data.positions.clone(),
            velocities: data.velocities.clone(),
            densities: data.densities.clone(),
            pressures: data.pressures.clone(),
            near_densities: data.near_densities.clone(),
            near_pressures: data.near_pressures.clone(),
            updated: true,
            particle_count: data.particle_count,
        });
    }
    
    // Submit commands
    render_queue.submit(std::iter::once(encoder.finish()));
} 