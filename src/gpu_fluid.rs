use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, 
            BindingType, Buffer, BufferBindingType,
            ComputePipeline, ComputePipelineDescriptor, 
            PipelineCache, ShaderStages,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
    log::info,
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

use crate::simulation::{Particle, FluidParams, MouseInteraction};

pub struct GpuFluidPlugin;

impl Plugin for GpuFluidPlugin {
    fn build(&self, app: &mut App) {
        // Main app systems
        app.add_systems(Update, handle_mouse_input)
           .init_resource::<GpuState>()
           .init_resource::<GpuPerformanceStats>()
           .init_resource::<GpuParticles>() // Initialize this so it's always available
           // Only update GPU performance if GPU is enabled
           .add_systems(Update, update_gpu_performance.run_if(gpu_enabled))
           // Only try to sync GPU particles if GPU is enabled
           .add_systems(Update, sync_gpu_to_cpu_particles.run_if(gpu_enabled));

        // Render app systems for GPU processing
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidPipeline>()
            .init_resource::<FluidBindGroup>()
            .init_resource::<RenderGpuState>()
            .init_resource::<GpuResultsBuffer>()
            // Extract resources from the main world
            .add_systems(ExtractSchedule, (
                extract_fluid_params, 
                extract_gpu_state,
                extract_perf_stats,
            ).run_if(render_gpu_enabled))
            // Prepare resources happens before Queue
            .add_systems(Render, prepare_bind_groups.in_set(RenderSet::Prepare).run_if(render_gpu_enabled))
            // Queue the particle buffers during the Queue phase
            .add_systems(Render, queue_particle_buffers.in_set(RenderSet::Queue).run_if(render_gpu_enabled))
            // Handle buffer readbacks after rendering is complete
            .add_systems(Render, handle_buffer_readbacks.in_set(RenderSet::Cleanup).run_if(render_gpu_enabled));
    }
}

// Main app state for fluid compute
#[derive(Resource, Default)]
struct FluidComputeState {
    particle_positions: Vec<Vec2>,
    particle_velocities: Vec<Vec2>,
    particle_densities: Vec<f32>,
    particle_pressures: Vec<f32>,
    readback_scheduled: bool,
}

// Track GPU state and errors
#[derive(Resource, Clone)]
pub struct GpuState {
    pub enabled: bool,
    pub error_count: u32,
    pub last_error: Option<String>,
}

impl Default for GpuState {
    fn default() -> Self {
        Self {
            enabled: true,
            error_count: 0,
            last_error: None,
        }
    }
}

// Render world copy of GPU state
#[derive(Resource, Clone, Default)]
struct RenderGpuState {
    enabled: bool,
    error_count: u32,
    last_error: Option<String>,
}

// Track GPU performance data
#[derive(Resource, Clone)]
pub struct GpuPerformanceStats {
    pub frame_times: Vec<f32>,
    pub avg_frame_time: f32,
    pub max_sample_count: usize,
    pub max_timestep_fps: f32,
    pub iterations_per_frame: u32,
    pub time_scale: f32,
    pub max_velocity: f32,
    pub adaptive_timestep: bool,
    pub adaptive_iterations: bool,
    pub base_iterations: u32,
    pub velocity_iteration_scale: f32,
}

impl Default for GpuPerformanceStats {
    fn default() -> Self {
        Self {
            frame_times: Vec::with_capacity(100),
            avg_frame_time: 0.0,
            max_sample_count: 100,
            max_timestep_fps: 60.0,
            iterations_per_frame: 3,
            time_scale: 1.0,
            max_velocity: 0.0,
            adaptive_timestep: true,
            adaptive_iterations: true,
            base_iterations: 3,
            velocity_iteration_scale: 1.0,
        }
    }
}

// Render app resources
#[derive(Resource, Default)]
struct FluidPipeline {
    density_pressure_pipeline: Option<ComputePipeline>,
    forces_pipeline: Option<ComputePipeline>,
    integration_pipeline: Option<ComputePipeline>,
    spatial_hash_pipeline: Option<ComputePipeline>,
    reorder_pipeline: Option<ComputePipeline>,
    reorder_copyback_pipeline: Option<ComputePipeline>,
    calculate_offsets_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
    
    // Store pipeline IDs for retrieval
    density_pressure_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    forces_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    integration_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    spatial_hash_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    reorder_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    reorder_copyback_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    calculate_offsets_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
}

#[derive(Resource, Default)]
struct FluidBindGroup {
    bind_group: Option<BindGroup>,
    particle_buffer: Option<Buffer>,
    params_buffer: Option<Buffer>,
    
    // Spatial hash buffers
    spatial_keys_buffer: Option<Buffer>,
    spatial_indices_buffer: Option<Buffer>,
    spatial_offsets_buffer: Option<Buffer>,
    target_particles_buffer: Option<Buffer>,
    
    // Pipeline state
    num_particles: u32,
    pipeline_ready: bool,
}

// Extracted params for the render world
#[derive(Resource, Clone, Default)]
struct ExtractedFluidParams {
    params: FluidParams,
    mouse: MouseInteraction,
    dt: f32,
    num_particles: usize,
    particle_positions: Vec<Vec2>,
    particle_velocities: Vec<Vec2>,
    particle_densities: Vec<f32>,
    particle_pressures: Vec<f32>,
    near_densities: Vec<f32>,
    near_pressures: Vec<f32>,
}

// Resource to handle GPU result readback
#[derive(Resource, Default)]
struct GpuResultsBuffer {
    readback_buffer: Option<Buffer>,
    readback_scheduled: bool,
    buffer_mapped: bool,
    gpu_data: Vec<GpuParticle>,
    debug_log: bool,
}

// Resource to hold GPU particle data in the main world
#[derive(Resource, Default)]
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

// Extract GPU state to render world
fn extract_gpu_state(
    mut commands: Commands,
    gpu_state: Extract<Res<GpuState>>,
) {
    commands.insert_resource(RenderGpuState {
        enabled: gpu_state.enabled,
        error_count: gpu_state.error_count,
        last_error: gpu_state.last_error.clone(),
    });
}

// Extract performance stats to render world
fn extract_perf_stats(
    mut commands: Commands,
    perf_stats: Extract<Res<GpuPerformanceStats>>,
) {
    commands.insert_resource(perf_stats.clone());
}

// GPU-compatible structures
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticle {
    position: [f32; 2],
    padding0: [f32; 2],  // Padding for 16-byte alignment
    velocity: [f32; 2],
    padding1: [f32; 2],  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Update GPU performance stats with adaptive timestep
fn update_gpu_performance(
    time: Res<Time>,
    gpu_state: Res<GpuState>,
    mut perf_stats: ResMut<GpuPerformanceStats>,
    particles: Query<&Particle>,
) {
    if !gpu_state.enabled {
        perf_stats.frame_times.clear();
        perf_stats.avg_frame_time = 0.0;
        return;
    }

    let frame_time = time.delta_secs() * 1000.0;
    perf_stats.frame_times.push(frame_time);
    
    if perf_stats.frame_times.len() > perf_stats.max_sample_count {
        perf_stats.frame_times.remove(0);
    }
    
    // Calculate average frame time
    let sum: f32 = perf_stats.frame_times.iter().sum();
    perf_stats.avg_frame_time = sum / perf_stats.frame_times.len() as f32;
    
    // Find maximum particle velocity for adaptive iterations
    let mut max_velocity = 0.0;
    for particle in particles.iter() {
        let velocity_magnitude = particle.velocity.length();
        if velocity_magnitude > max_velocity {
            max_velocity = velocity_magnitude;
        }
    }
    perf_stats.max_velocity = max_velocity;
    
    // Adjust iterations based on performance and velocity
    if perf_stats.adaptive_iterations {
        // If frame time is very high (low FPS), use minimal iterations
        if perf_stats.avg_frame_time > 33.33 { // Below 30 FPS
            perf_stats.iterations_per_frame = 1; // Minimum for performance
        } else if perf_stats.avg_frame_time > 16.67 { // Below 60 FPS
            // Use 1 or 2 iterations based on velocity
            let iterations = if max_velocity > 300.0 { 2 } else { 1 };
            perf_stats.iterations_per_frame = iterations;
        } else {
            // Higher FPS - can afford more iterations
            let base_iterations = perf_stats.base_iterations.min(3); // Cap at 3 for better performance
            
            // Scale iterations based on maximum velocity
            // Only scale up for high velocities to maintain stability
            let velocity_scale = if max_velocity > 200.0 {
                let normalized_velocity = ((max_velocity - 200.0) / 300.0).clamp(0.0, 1.0);
                1.0 + normalized_velocity // Max scale of 2.0
            } else {
                1.0 // No scaling for normal velocities
            };
            
            perf_stats.iterations_per_frame = (base_iterations as f32 * velocity_scale).round() as u32;
            perf_stats.iterations_per_frame = perf_stats.iterations_per_frame.max(1).min(3);
        }
    } else {
        // When not adaptive, use a conservative number of iterations (1-3)
        perf_stats.iterations_per_frame = perf_stats.base_iterations.min(3);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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
    boundary_min_padding: [f32; 2],  // Padding to align vec2
    
    // Vec4 aligned group 4
    boundary_max: [f32; 2],
    boundary_max_padding: [f32; 2],  // Padding to align vec2
    
    // Vec4 aligned group 5
    gravity: [f32; 2],
    gravity_padding: [f32; 2],  // Padding to align vec2
    
    // Vec4 aligned group 6
    mouse_position: [f32; 2],
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: [u32; 2],  // Padding to ensure alignment
}

// Handle mouse input for interaction with the fluid
fn handle_mouse_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut gpu_state: ResMut<GpuState>,
) {
    // Handle mouse interaction
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
                    mouse_interaction.position = world_position;
                    mouse_interaction.active = mouse_buttons.pressed(MouseButton::Left) || 
                                               mouse_buttons.pressed(MouseButton::Right);
                    mouse_interaction.repel = mouse_buttons.pressed(MouseButton::Right);
                }
            }
        }
    }

    // Toggle force strength with number keys
    if keys.just_pressed(KeyCode::Digit1) {
        mouse_interaction.strength = 1000.0;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = 2000.0;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = 3000.0;
    }
    
    // Toggle GPU acceleration with G key
    if keys.just_pressed(KeyCode::KeyG) {
        gpu_state.enabled = !gpu_state.enabled;
        if gpu_state.enabled {
            info!("GPU acceleration enabled");
        } else {
            info!("GPU acceleration disabled (using CPU fallback)");
        }
    }
}

// Extract data from the main world to the render world
fn extract_fluid_params(
    mut commands: Commands,
    fluid_params: Extract<Res<FluidParams>>,
    mouse_interaction: Extract<Res<MouseInteraction>>,
    time: Extract<Res<Time>>,
    particles: Extract<Query<(&Particle, &Transform)>>,
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
        positions.push(transform.translation.truncate());
        velocities.push(particle.velocity);
        densities.push(particle.density);
        pressures.push(particle.pressure);
        near_densities.push(particle.near_density);
        near_pressures.push(particle.near_pressure);
    }

    commands.insert_resource(ExtractedFluidParams {
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
fn prepare_bind_groups(
    mut fluid_pipeline: ResMut<FluidPipeline>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: ResMut<PipelineCache>,
    mut bind_group: ResMut<FluidBindGroup>,
    gpu_state: ResMut<RenderGpuState>,
    mut gpu_results: ResMut<GpuResultsBuffer>,
    extracted_params: Option<Res<ExtractedFluidParams>>,
) {
    // Skip if GPU is disabled
    if !gpu_state.enabled || extracted_params.is_none() {
        return;
    }
    
    let extracted_data = extracted_params.as_ref().unwrap();
    let num_particles = extracted_data.num_particles;
    
    // Create bind group layout if it doesn't exist
    if fluid_pipeline.bind_group_layout.is_none() {
        let entries = vec![
            // Particle data (read/write)
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
            // Simulation parameters (read-only)
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
            // Spatial hash keys (read/write)
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
            // Spatial hash indices (read/write)
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
            // Spatial hash offsets (read/write)
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
        
        let layout = render_device.create_bind_group_layout("fluid_simulation_bind_group_layout", &entries);
        fluid_pipeline.bind_group_layout = Some(layout);
    }

    // Create compute pipelines if they don't exist
    
    // 1. Spatial hash pipeline
    if fluid_pipeline.spatial_hash_pipeline.is_none() && fluid_pipeline.spatial_hash_id.is_none() {
        let shader = asset_server.load("shaders/spatial_hash.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("spatial_hash_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.spatial_hash_id = Some(id);
    }
    
    // 2. Calculate offsets pipeline
    if fluid_pipeline.calculate_offsets_pipeline.is_none() && fluid_pipeline.calculate_offsets_id.is_none() {
        let shader = asset_server.load("shaders/calculate_offsets.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("calculate_offsets_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.calculate_offsets_id = Some(id);
    }
    
    // 3. Reorder pipeline
    if fluid_pipeline.reorder_pipeline.is_none() && fluid_pipeline.reorder_id.is_none() {
        let shader = asset_server.load("shaders/reorder.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("reorder_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.reorder_id = Some(id);
    }
    
    // 4. Reorder copyback pipeline
    if fluid_pipeline.reorder_copyback_pipeline.is_none() && fluid_pipeline.reorder_copyback_id.is_none() {
        let shader = asset_server.load("shaders/reorder_copyback.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("reorder_copyback_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.reorder_copyback_id = Some(id);
    }

    // 5. Density pressure pipeline
    if fluid_pipeline.density_pressure_pipeline.is_none() && fluid_pipeline.density_pressure_id.is_none() {
        let shader = asset_server.load("shaders/density_pressure.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("density_pressure_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.density_pressure_id = Some(id);
    }

    // 6. Forces pipeline
    if fluid_pipeline.forces_pipeline.is_none() && fluid_pipeline.forces_id.is_none() {
        let shader = asset_server.load("shaders/forces.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("forces_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.forces_id = Some(id);
    }

    // 7. Integration pipeline
    if fluid_pipeline.integration_pipeline.is_none() && fluid_pipeline.integration_id.is_none() {
        let shader = asset_server.load("shaders/integration.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("integration_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.integration_id = Some(id);
    }

    // Check if all pipelines are available
    let all_pipelines_ready = 
        fluid_pipeline.spatial_hash_pipeline.is_some() &&
        fluid_pipeline.calculate_offsets_pipeline.is_some() &&
        fluid_pipeline.reorder_pipeline.is_some() &&
        fluid_pipeline.reorder_copyback_pipeline.is_some() &&
        fluid_pipeline.density_pressure_pipeline.is_some() &&
        fluid_pipeline.forces_pipeline.is_some() &&
        fluid_pipeline.integration_pipeline.is_some();

    bind_group.pipeline_ready = all_pipelines_ready;

    // Create particle buffer for GPU simulation
    if bind_group.particle_buffer.is_none() || bind_group.num_particles != num_particles as u32 {
        // Create the GPU buffer for particles 
        // ...existing buffer creation code...
        
        // Create readback buffer for returning results to CPU
        if gpu_results.readback_buffer.is_none() || bind_group.num_particles != num_particles as u32 {
            let buffer_size = std::mem::size_of::<GpuParticle>() * num_particles;
            
            // Create a buffer suitable for readback (mapping)
            let buffer = render_device.create_buffer(&bevy::render::render_resource::BufferDescriptor {
                label: Some("particle_readback_buffer"),
                size: buffer_size as u64,
                usage: bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            gpu_results.readback_buffer = Some(buffer);
            gpu_results.readback_scheduled = false;
            gpu_results.gpu_data.clear();
            
            if gpu_results.debug_log {
                info!("Created GPU readback buffer for {} particles ({} bytes)", num_particles, buffer_size);
            }
        }
    }
}

// Queue particle buffers for GPU computation with optimized workgroups
fn queue_particle_buffers(
    fluid_bind_group: ResMut<FluidBindGroup>,
    mut fluid_pipeline: ResMut<FluidPipeline>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_params: Option<Res<ExtractedFluidParams>>,
    pipeline_cache: Res<PipelineCache>,
    mut gpu_state: ResMut<RenderGpuState>,
    perf_stats: Res<GpuPerformanceStats>,
    mut gpu_results: ResMut<GpuResultsBuffer>,
) {
    // Skip if no data or GPU is disabled
    if extracted_params.is_none() || !gpu_state.enabled {
        return;
    }
    let extracted_data = extracted_params.as_ref().unwrap();
    
    // Check if pipelines are ready
    if !fluid_bind_group.pipeline_ready {
        // First check if pipelines are ready in cache
        let spatial_hash_pipeline = fluid_pipeline.spatial_hash_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let calculate_offsets_pipeline = fluid_pipeline.calculate_offsets_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let reorder_pipeline = fluid_pipeline.reorder_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let reorder_copyback_pipeline = fluid_pipeline.reorder_copyback_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let density_pressure_pipeline = fluid_pipeline.density_pressure_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let forces_pipeline = fluid_pipeline.forces_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let integration_pipeline = fluid_pipeline.integration_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));

        if let (
            Some(spatial_hash_pipeline),
            Some(calculate_offsets_pipeline),
            Some(reorder_pipeline),
            Some(reorder_copyback_pipeline),
            Some(density_pressure_pipeline),
            Some(forces_pipeline),
            Some(integration_pipeline),
        ) = (
            spatial_hash_pipeline,
            calculate_offsets_pipeline,
            reorder_pipeline,
            reorder_copyback_pipeline,
            density_pressure_pipeline,
            forces_pipeline,
            integration_pipeline,
        ) {
            // Set pipelines from cache
            fluid_pipeline.spatial_hash_pipeline = Some(spatial_hash_pipeline.clone());
            fluid_pipeline.calculate_offsets_pipeline = Some(calculate_offsets_pipeline.clone());
            fluid_pipeline.reorder_pipeline = Some(reorder_pipeline.clone());
            fluid_pipeline.reorder_copyback_pipeline = Some(reorder_copyback_pipeline.clone());
            fluid_pipeline.density_pressure_pipeline = Some(density_pressure_pipeline.clone());
            fluid_pipeline.forces_pipeline = Some(forces_pipeline.clone());
            fluid_pipeline.integration_pipeline = Some(integration_pipeline.clone());
        } else {
            // Pipelines not ready yet
            return;
        }
    }

    // Prepare parameters for shader
    let fluid_params = &extracted_data.params;
    let mouse = &extracted_data.mouse;
    
    // Calculate optimal timestep based on iterations 
    let dt = extracted_data.dt;
    // Don't divide by iterations here - we'll run the simulation multiple times with full dt
    // This matches how the CPU simulation works, using the full dt each frame
    let sub_step_dt = dt;
    
    // Prepare GPU fluid params
    let gpu_params = GpuFluidParams {
        smoothing_radius: fluid_params.smoothing_radius,
        target_density: fluid_params.target_density,
        pressure_multiplier: fluid_params.pressure_multiplier,
        near_pressure_multiplier: fluid_params.near_pressure_multiplier,
        viscosity_strength: fluid_params.viscosity_strength,
        boundary_dampening: 0.3, // Use constant as this field doesn't exist in FluidParams
        particle_radius: 5.0, // Hard-coded for now
        dt: sub_step_dt,
        boundary_min: [fluid_params.boundary_min.x, fluid_params.boundary_min.y],
        boundary_min_padding: [0.0, 0.0],
        boundary_max: [fluid_params.boundary_max.x, fluid_params.boundary_max.y],
        boundary_max_padding: [0.0, 0.0],
        gravity: [0.0, -9.81], // Hard-coded gravity value for now
        gravity_padding: [0.0, 0.0],
        mouse_position: [mouse.position.x, mouse.position.y],
        mouse_radius: mouse.radius,
        mouse_strength: mouse.strength,
        mouse_active: if mouse.active { 1 } else { 0 },
        mouse_repel: if mouse.repel { 1 } else { 0 },
        padding: [0, 0],
    };

    // Write parameters to buffer
    if let Some(params_buffer) = &fluid_bind_group.params_buffer {
        render_queue.write_buffer(params_buffer, 0, bytemuck::cast_slice(&[gpu_params]));
    }

    // Calculate optimal workgroup size to match Unity's implementation
    let workgroup_size = 64; // Match Unity's approach of using 64 threads per workgroup
    let num_workgroups = (extracted_data.num_particles as u32 + workgroup_size - 1) / workgroup_size;

    // Use empty compute pass descriptor
    let compute_pass_descriptor = bevy::render::render_resource::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    };

    if let (
        Some(spatial_hash_pipeline),
        Some(calculate_offsets_pipeline),
        Some(reorder_pipeline),
        Some(reorder_copyback_pipeline),
        Some(density_pressure_pipeline),
        Some(forces_pipeline),
        Some(integration_pipeline),
        Some(bind_group),
    ) = (
        &fluid_pipeline.spatial_hash_pipeline,
        &fluid_pipeline.calculate_offsets_pipeline,
        &fluid_pipeline.reorder_pipeline,
        &fluid_pipeline.reorder_copyback_pipeline,
        &fluid_pipeline.density_pressure_pipeline,
        &fluid_pipeline.forces_pipeline,
        &fluid_pipeline.integration_pipeline,
        &fluid_bind_group.bind_group,
    ) {
        // Get number of iterations to perform
        let iterations = perf_stats.iterations_per_frame as usize;
        
        // Process physics multiple times per frame for better stability
        for _ in 0..iterations {
            // Create a new command encoder for each iteration
            let mut command_encoder = render_device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
                label: Some("fluid_simulation_pass_1"),
            });
            
            {
                // 1. Spatial hash pass
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(spatial_hash_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            {
                // 2. Calculate offsets pass
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(calculate_offsets_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                // This needs to consider all keys so it stays with full dispatch size
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            {
                // 3. Reorder pass
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(reorder_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            {
                // 4. Reorder copy back pass
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(reorder_copyback_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Submit this set of operations and create a new encoder for the next phase
            render_queue.submit(std::iter::once(command_encoder.finish()));
            let mut command_encoder = render_device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
                label: Some("fluid_simulation_pass_2"),
            });

            {
                // 5. Density and pressure calculation
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(density_pressure_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Submit and create new encoder for next phase
            render_queue.submit(std::iter::once(command_encoder.finish()));
            let mut command_encoder = render_device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
                label: Some("fluid_simulation_pass_3"),
            });

            {
                // 6. Force calculation
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(forces_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            {
                // 7. Final integration and collision
                let mut pass = command_encoder.begin_compute_pass(&compute_pass_descriptor);
                pass.set_pipeline(integration_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            // Submit final encoder for this iteration
            render_queue.submit(std::iter::once(command_encoder.finish()));
        }
    } else {
        // Error - this shouldn't happen
        gpu_state.error_count += 1;
        gpu_state.last_error = Some("Failed to get compute pipelines or bind group.".to_string());
    }

    // After simulation is complete, schedule a readback of results
    // Only schedule readbacks every other frame to reduce overhead
    // Use extracted_data.dt to create a timer-like behavior (approximately 30Hz readbacks)
    static mut READBACK_TIMER: f32 = 0.0;
    unsafe {
        READBACK_TIMER += extracted_data.dt;
        
        // Only do readback approximately 30 times per second (every ~33ms)
        if READBACK_TIMER >= 0.033 && !gpu_results.readback_scheduled && gpu_results.readback_buffer.is_some() {
            READBACK_TIMER = 0.0;
            
            // In Bevy 0.16, we need to use the render queue to copy data
            if let (Some(particle_buffer), Some(readback_buffer)) = 
                (&fluid_bind_group.particle_buffer, &gpu_results.readback_buffer) {
                
                // Create a command encoder for the copy operation
                let mut encoder = render_device.create_command_encoder(&Default::default());
                
                // Copy from particle buffer to readback buffer
                let size = (std::mem::size_of::<GpuParticle>() * extracted_data.num_particles) as u64;
                encoder.copy_buffer_to_buffer(particle_buffer, 0, readback_buffer, 0, size);
                
                // Submit the copy command
                render_queue.submit(std::iter::once(encoder.finish()));
                
                // In Bevy 0.16, map_buffer requires a callback
                let buffer_slice = readback_buffer.slice(..);
                
                // Define a callback that does nothing - we'll check for mapping in handle_buffer_readbacks
                let callback = move |_: Result<(), bevy::render::render_resource::BufferAsyncError>| {
                    // This is a no-op callback since we'll check mapping status later
                };
                
                // Start buffer mapping
                render_device.map_buffer(&buffer_slice, bevy::render::render_resource::MapMode::Read, callback);
                
                // Mark readback as scheduled
                gpu_results.readback_scheduled = true;
                gpu_results.buffer_mapped = true;
                
                // Only poll if absolutely necessary - polling can be expensive
                // render_device.poll(bevy::render::render_resource::Maintain::Wait);
                
                // Log readback for debugging
                if gpu_results.debug_log {
                    info!("Scheduled GPU->CPU readback for {} particles", extracted_data.num_particles);
                }
            }
        }
    }
}

// Handle buffer readbacks from GPU to main world
fn handle_buffer_readbacks(
    render_device: Res<RenderDevice>,
    mut gpu_results: ResMut<GpuResultsBuffer>,
    mut commands: Commands,
    fluid_bind_group: Res<FluidBindGroup>,
) {
    // Skip if no readback is scheduled or buffer unavailable
    if !gpu_results.readback_scheduled || gpu_results.readback_buffer.is_none() {
        return;
    }
    
    // First clone/get what we need to avoid borrow issues
    let buffer = gpu_results.readback_buffer.clone().unwrap();
    let particle_count = fluid_bind_group.num_particles as usize;
    let buffer_mapped = gpu_results.buffer_mapped;
    let debug_log = gpu_results.debug_log;
    
    // Poll the device to check for completed operations (non-blocking poll)
    render_device.poll(bevy::render::render_resource::Maintain::Poll);
    
    // Bevy 0.16 doesn't have a direct way to check if a buffer is mapped
    // We'll use our own tracking in gpu_results.buffer_mapped
    
    // If the buffer is mapped (we assume it is if our tracking says so), try to read its data
    if buffer_mapped {
        // Get the buffer size and create a slice
        let buffer_slice = buffer.slice(..);
        
        // Try to get the mapped range safely
        // In Bevy 0.16/wgpu 0.19.1, this function will panic if the buffer isn't mapped
        // We'll use a safer approach
        let maybe_data = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            buffer_slice.get_mapped_range()
        }));
        
        if let Ok(data) = maybe_data {
            // Convert the raw bytes to GpuParticle structs
            let particles = bytemuck::cast_slice::<u8, GpuParticle>(&data);
            
            // Only allocate new vectors if we need to (optimization)
            let mut positions = Vec::with_capacity(particles.len());
            let mut velocities = Vec::with_capacity(particles.len());
            let mut densities = Vec::with_capacity(particles.len());
            let mut pressures = Vec::with_capacity(particles.len());
            let mut near_densities = Vec::with_capacity(particles.len());
            let mut near_pressures = Vec::with_capacity(particles.len());
            
            // Extract particle data more efficiently by reducing function calls
            // This tight loop is faster than individual function calls 
            for particle in particles.iter() {
                positions.push(Vec2::new(particle.position[0], particle.position[1]));
                velocities.push(Vec2::new(particle.velocity[0], particle.velocity[1]));
                densities.push(particle.density);
                pressures.push(particle.pressure);
                near_densities.push(particle.near_density);
                near_pressures.push(particle.near_pressure);
            }
            
            // Store a copy of the data for fallback (only if needed)
            if gpu_results.gpu_data.is_empty() {
                gpu_results.gpu_data = particles.to_vec();
            } else {
                // Update the existing data only if significantly different
                // This reduces memory allocations
                let should_update = particles.len() != gpu_results.gpu_data.len() ||
                    (particles.len() > 0 && particles[0].position[0] != gpu_results.gpu_data[0].position[0]);
                    
                if should_update {
                    gpu_results.gpu_data = particles.to_vec();
                }
            }
            
            // Drop the mapped range (necessary before unmapping)
            drop(data);
            
            // Unmap the buffer
            buffer.unmap();
            
            // Update buffer state
            gpu_results.buffer_mapped = false;
            
            // Get the position count for logging before we move it
            let positions_count = positions.len();

            // Pass the data to the main world for rendering
            commands.insert_resource(GpuParticles {
                positions,
                velocities,
                densities,
                pressures,
                near_densities,
                near_pressures,
                updated: true,
                particle_count,
            });
            
            if debug_log {
                info!("Successfully read {} particles from GPU", positions_count);
            }
        } else {
            // Buffer mapping failed or isn't ready - use fallback data
            if !gpu_results.gpu_data.is_empty() {
                // Use the cached data as fallback
                let cached_data = &gpu_results.gpu_data;
                let particle_count = cached_data.len();
                
                // Reuse the existing data without creating new allocations when possible
                let mut positions = Vec::with_capacity(particle_count);
                let mut velocities = Vec::with_capacity(particle_count);
                let mut densities = Vec::with_capacity(particle_count);
                let mut pressures = Vec::with_capacity(particle_count);
                let mut near_densities = Vec::with_capacity(particle_count);
                let mut near_pressures = Vec::with_capacity(particle_count);
                
                // Extract particle data from previous read (efficient tight loop)
                for particle in cached_data {
                    positions.push(Vec2::new(particle.position[0], particle.position[1]));
                    velocities.push(Vec2::new(particle.velocity[0], particle.velocity[1]));
                    densities.push(particle.density);
                    pressures.push(particle.pressure);
                    near_densities.push(particle.near_density);
                    near_pressures.push(particle.near_pressure);
                }
                
                // Get the actual size before moving
                let positions_count = positions.len();
                
                // Pass the cached data to the main world
                commands.insert_resource(GpuParticles {
                    positions,
                    velocities,
                    densities,
                    pressures,
                    near_densities,
                    near_pressures,
                    updated: true,
                    particle_count,
                });
                
                if debug_log {
                    info!("Using cached data for {} particles until mapping completes", positions_count);
                }
            }
        }
    } else {
        // Buffer not mapped yet - try initiating a mapping
        let buffer_slice = buffer.slice(..);
        
        // Map the buffer - In Bevy 0.16, map_buffer requires a callback
        let callback = move |result: Result<(), bevy::render::render_resource::BufferAsyncError>| {
            if result.is_err() {
                error!("Failed to map buffer for reading");
            }
        };
        
        // Start buffer mapping
        render_device.map_buffer(&buffer_slice, bevy::render::render_resource::MapMode::Read, callback);
        
        // Mark the buffer as mapped
        gpu_results.buffer_mapped = true;
        
        // Use the fallback in sync_gpu_to_cpu_particles for this frame
    }
    
    // Reset readback state
    gpu_results.readback_scheduled = false;
}

// Sync GPU particle data back to CPU entities
fn sync_gpu_to_cpu_particles(
    gpu_state: Res<GpuState>,
    mut particles: Query<(Entity, &mut Transform, &mut Particle)>,
    gpu_particles: Option<Res<GpuParticles>>,
    fluid_params: Res<FluidParams>,
    mouse_interaction: Res<MouseInteraction>,
    time: Res<Time>,
    mut commands: Commands,
) {
    // Skip if GPU is disabled
    if !gpu_state.enabled {
        return;
    }
    
    // If we don't have GPU data, need to use CPU-compatible simulation as fallback
    if gpu_particles.is_none() || 
       gpu_particles.as_ref().unwrap().positions.is_empty() {
        
        // These parameters match the CPU simulation exactly
        let dt = time.delta_secs();
        let boundary_min = fluid_params.boundary_min;
        let boundary_max = fluid_params.boundary_max;
        let boundary_dampening = 0.3;
        let particle_radius = 5.0;
        
        // Apply simplified CPU-like physics to keep particles moving
        for (_, mut transform, mut particle) in particles.iter_mut() {
            // Apply gravity (matches CPU approach)
            particle.velocity += Vec2::new(0.0, -9.81) * dt;
            
            // Apply mouse force if active (matches CPU approach)
            if mouse_interaction.active {
                let direction = mouse_interaction.position - transform.translation.truncate();
                let distance = direction.length();
                
                if distance < mouse_interaction.radius {
                    let force_direction = if mouse_interaction.repel { -direction } else { direction };
                    let force_strength = mouse_interaction.strength * (1.0 - distance / mouse_interaction.radius);
                    particle.velocity += force_direction.normalize() * force_strength * dt;
                }
            }
            
            // Dampen velocity (matches CPU approach)
            particle.velocity *= 0.98;
            
            // Update position
            transform.translation += Vec3::new(particle.velocity.x, particle.velocity.y, 0.0) * dt;
            
            // Handle boundary collisions (matches CPU approach)
            if transform.translation.x < boundary_min.x + particle_radius {
                transform.translation.x = boundary_min.x + particle_radius;
                particle.velocity.x = -particle.velocity.x * boundary_dampening;
            } else if transform.translation.x > boundary_max.x - particle_radius {
                transform.translation.x = boundary_max.x - particle_radius;
                particle.velocity.x = -particle.velocity.x * boundary_dampening;
            }
            
            if transform.translation.y < boundary_min.y + particle_radius {
                transform.translation.y = boundary_min.y + particle_radius;
                particle.velocity.y = -particle.velocity.y * boundary_dampening;
            } else if transform.translation.y > boundary_max.y - particle_radius {
                transform.translation.y = boundary_max.y - particle_radius;
                particle.velocity.y = -particle.velocity.y * boundary_dampening;
            }
            
            // Set simplified density for visualization
            particle.density = 1000.0 + particle.velocity.length() * 10.0;
        }
        
        return;
    }
    
    let gpu_data = gpu_particles.unwrap();
    
    // Skip if no update has occurred
    if !gpu_data.updated || gpu_data.positions.is_empty() {
        return;
    }
    
    // Update CPU entities with GPU data
    for (i, (_, mut transform, mut particle)) in particles.iter_mut().enumerate() {
        if i < gpu_data.positions.len() {
            // Update transform
            transform.translation.x = gpu_data.positions[i].x;
            transform.translation.y = gpu_data.positions[i].y;
            
            // Update particle properties
            particle.velocity = gpu_data.velocities[i];
            particle.density = gpu_data.densities[i];
            particle.pressure = gpu_data.pressures[i];
            particle.near_density = gpu_data.near_densities[i];
            particle.near_pressure = gpu_data.near_pressures[i];
        }
    }
    
    // Mark the resource as processed
    commands.insert_resource(GpuParticles {
        positions: gpu_data.positions.clone(),
        velocities: gpu_data.velocities.clone(),
        densities: gpu_data.densities.clone(),
        pressures: gpu_data.pressures.clone(),
        near_densities: gpu_data.near_densities.clone(),
        near_pressures: gpu_data.near_pressures.clone(),
        updated: false,
        particle_count: gpu_data.particle_count,
    });
}

// Run condition to only run if GPU is enabled
fn gpu_enabled(gpu_state: Res<GpuState>) -> bool {
    gpu_state.enabled
}

// Run condition to only run render systems if GPU is enabled
fn render_gpu_enabled(gpu_state: Res<RenderGpuState>) -> bool {
    gpu_state.enabled
} 