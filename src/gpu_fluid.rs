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
           .add_systems(Update, update_gpu_performance);

        // Render app systems for GPU processing
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidPipeline>()
            .init_resource::<FluidBindGroup>()
            .init_resource::<RenderGpuState>()
            // Extract resources from the main world
            .add_systems(ExtractSchedule, (
                extract_fluid_params, 
                extract_gpu_state,
                extract_perf_stats,
            ))
            .add_systems(Render, prepare_bind_groups.after(RenderSet::Prepare))
            .add_systems(Render, queue_particle_buffers.after(RenderSet::Queue));
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
        // Scale iterations based on performance
        let base_iterations = if perf_stats.avg_frame_time > 16.67 { // Below 60 FPS
            perf_stats.base_iterations.min(2) // Cap at 2 if performance is low
        } else {
            perf_stats.base_iterations
        };
        
        // Scale iterations based on maximum velocity
        // Higher velocities need more iterations for stability
        let velocity_scale = if max_velocity > 0.0 {
            let normalized_velocity = (max_velocity / 500.0).clamp(1.0, 4.0);
            perf_stats.velocity_iteration_scale * normalized_velocity
        } else {
            1.0
        };
        
        perf_stats.iterations_per_frame = (base_iterations as f32 * velocity_scale).max(1.0) as u32;
    } else {
        perf_stats.iterations_per_frame = perf_stats.base_iterations;
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
    _gpu_state: ResMut<RenderGpuState>,
) {
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
) {
    if !gpu_state.enabled {
        return;
    }

    // Check if we have extracted data
    let Some(extracted_params) = extracted_params else {
        return;
    };

    // Get number of particles
    let num_particles = extracted_params.num_particles;
    if num_particles == 0 {
        return;
    }

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
    let fluid_params = &extracted_params.params;
    let mouse = &extracted_params.mouse;
    
    // Calculate optimal timestep based on iterations 
    let dt = extracted_params.dt;
    let sub_step_dt = dt / perf_stats.iterations_per_frame as f32;
    
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

    // Calculate optimal workgroup count for RTX 4090
    let workgroup_size = 128; // RTX 4090 optimal workgroup size
    let num_workgroups = (num_particles as u32 + workgroup_size - 1) / workgroup_size;

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
} 