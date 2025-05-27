use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindGroupEntry, 
            BindingType, Buffer, BufferBindingType, BufferUsages, BufferDescriptor,
            BufferInitDescriptor, ComputePipeline, ComputePipelineDescriptor, 
            PipelineCache, ShaderStages, CachedComputePipelineId,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
    log::info,
};
use bytemuck::{Pod, Zeroable, cast_slice};
use std::borrow::Cow;

use crate::simulation::{Particle, FluidParams, MouseInteraction};
use crate::constants::{PARTICLE_RADIUS, BOUNDARY_DAMPENING, GRAVITY_2D};

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
    position_correction_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
    
    // Store pipeline IDs for retrieval
    density_pressure_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    forces_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    integration_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    spatial_hash_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    reorder_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    reorder_copyback_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    calculate_offsets_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    position_correction_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
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

// Resource to handle GPU result readback - improved with triple buffering
#[derive(Resource)]
struct GpuResultsBuffer {
    readback_buffers: [Option<Buffer>; 3], // Triple buffering for even better async performance
    active_buffer_index: usize,
    readback_scheduled: bool,
    is_mapped: bool,
    gpu_data: Vec<GpuParticle>,
    debug_log: bool,
    frame_counter: u32, // For controlling readback frequency
    last_readback_time: Option<std::time::Instant>,
}

impl Default for GpuResultsBuffer {
    fn default() -> Self {
        Self {
            readback_buffers: [None, None, None],
            active_buffer_index: 0,
            readback_scheduled: false,
            is_mapped: false,
            gpu_data: Vec::new(),
            debug_log: false,
            frame_counter: 0,
            last_readback_time: None,
        }
    }
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
    draw_lake_mode: Res<crate::simulation::DrawLakeMode>,
) {
    // Handle mouse interaction (disabled when Draw Lake mode is active)
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
                    mouse_interaction.position = world_position;
                    // Disable mouse forces when Draw Lake mode is active
                    if !draw_lake_mode.enabled {
                    mouse_interaction.active = mouse_buttons.pressed(MouseButton::Left) || 
                                               mouse_buttons.pressed(MouseButton::Right);
                    mouse_interaction.repel = mouse_buttons.pressed(MouseButton::Right);
                    } else {
                        mouse_interaction.active = false;
                        mouse_interaction.repel = false;
                    }
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

    // Debug logging for GPU parameters
    static mut EXTRACT_FRAME_COUNTER: u32 = 0;
    unsafe {
        EXTRACT_FRAME_COUNTER += 1;
        if EXTRACT_FRAME_COUNTER % 120 == 0 { // Log every 2 seconds at 60fps
            println!("GPU 2D EXTRACT: Frame {}: Extracting {} particles", EXTRACT_FRAME_COUNTER, particles.iter().len());
            println!("GPU 2D PARAMS: smoothing_radius={:.2}, target_density={:.2}, pressure_mult={:.2}, near_pressure_mult={:.2}", 
                     fluid_params.smoothing_radius, fluid_params.target_density, 
                     fluid_params.pressure_multiplier, fluid_params.near_pressure_multiplier);
            println!("GPU 2D PARAMS: particle_radius={:.2}, dt={:.4}, boundary_min=({:.1}, {:.1}), boundary_max=({:.1}, {:.1})", 
                     PARTICLE_RADIUS, time.delta_secs(), 
                     fluid_params.boundary_min.x, fluid_params.boundary_min.y,
                     fluid_params.boundary_max.x, fluid_params.boundary_max.y);
        }
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
    if !gpu_state.enabled {
        return;
    }
    
    // DEBUG: Log pipeline preparation start
    info!("GPU DEBUG: Starting pipeline preparation");
    
    // Create bind group layout if needed
    if fluid_pipeline.bind_group_layout.is_none() {
        info!("GPU DEBUG: Creating bind group layout");
        let layout = render_device.create_bind_group_layout(
            "fluid_bind_group_layout",
            &[
                // Particle buffer
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
                // Params buffer
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
                // Spatial hash keys buffer
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
                // Spatial hash indices buffer
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
                // Spatial hash offsets buffer
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
                // Target particles buffer
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
            ],
        );
        fluid_pipeline.bind_group_layout = Some(layout);
        info!("GPU DEBUG: Bind group layout created successfully");
    }

    // Create pipelines
    let pipeline_cache = pipeline_cache.into_inner();

    // 1. Density pressure pipeline
    if fluid_pipeline.density_pressure_pipeline.is_none() && fluid_pipeline.density_pressure_id.is_none() {
        info!("GPU DEBUG: Creating density pressure pipeline");
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
        info!("GPU DEBUG: Density pressure pipeline queued with ID: {:?}", id);
    }

    // 2. Forces pipeline
    if fluid_pipeline.forces_pipeline.is_none() && fluid_pipeline.forces_id.is_none() {
        info!("GPU DEBUG: Creating forces pipeline");
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
        info!("GPU DEBUG: Forces pipeline queued with ID: {:?}", id);
    }

    // 3. Integration pipeline
    if fluid_pipeline.integration_pipeline.is_none() && fluid_pipeline.integration_id.is_none() {
        info!("GPU DEBUG: Creating integration pipeline");
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
        info!("GPU DEBUG: Integration pipeline queued with ID: {:?}", id);
    }

    // 4. Spatial hash pipeline
    if fluid_pipeline.spatial_hash_pipeline.is_none() && fluid_pipeline.spatial_hash_id.is_none() {
        info!("GPU DEBUG: Creating spatial hash pipeline");
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
        info!("GPU DEBUG: Spatial hash pipeline queued with ID: {:?}", id);
    }

    // 5. Reorder pipeline
    if fluid_pipeline.reorder_pipeline.is_none() && fluid_pipeline.reorder_id.is_none() {
        info!("GPU DEBUG: Creating reorder pipeline");
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
        info!("GPU DEBUG: Reorder pipeline queued with ID: {:?}", id);
    }

    // 6. Reorder copyback pipeline
    if fluid_pipeline.reorder_copyback_pipeline.is_none() && fluid_pipeline.reorder_copyback_id.is_none() {
        info!("GPU DEBUG: Creating reorder copyback pipeline");
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
        info!("GPU DEBUG: Reorder copyback pipeline queued with ID: {:?}", id);
    }

    // 7. Calculate offsets pipeline
    if fluid_pipeline.calculate_offsets_pipeline.is_none() && fluid_pipeline.calculate_offsets_id.is_none() {
        info!("GPU DEBUG: Creating calculate offsets pipeline");
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
        info!("GPU DEBUG: Calculate offsets pipeline queued with ID: {:?}", id);
    }

    // 8. Position correction pipeline - CRITICAL FIX for particle overlapping
    if fluid_pipeline.position_correction_pipeline.is_none() && fluid_pipeline.position_correction_id.is_none() {
        info!("GPU DEBUG: Creating position correction pipeline - THIS IS CRITICAL FOR OVERLAP PREVENTION");
        let shader = asset_server.load("shaders/position_correction.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("position_correction_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.position_correction_id = Some(id);
        info!("GPU DEBUG: Position correction pipeline queued with ID: {:?}", id);
    }

    // CRITICAL FIX: Retrieve compiled pipelines from cache
    if let Some(id) = fluid_pipeline.density_pressure_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.density_pressure_pipeline.is_none() {
                fluid_pipeline.density_pressure_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Density pressure pipeline retrieved and cached");
            }
        }
    }

    if let Some(id) = fluid_pipeline.forces_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.forces_pipeline.is_none() {
                fluid_pipeline.forces_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Forces pipeline retrieved and cached");
            }
        }
    }

    if let Some(id) = fluid_pipeline.integration_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.integration_pipeline.is_none() {
                fluid_pipeline.integration_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Integration pipeline retrieved and cached");
            }
        }
    }

    if let Some(id) = fluid_pipeline.spatial_hash_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.spatial_hash_pipeline.is_none() {
                fluid_pipeline.spatial_hash_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Spatial hash pipeline retrieved and cached");
            }
        }
    }

    if let Some(id) = fluid_pipeline.reorder_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.reorder_pipeline.is_none() {
                fluid_pipeline.reorder_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Reorder pipeline retrieved and cached");
            }
        }
    }

    if let Some(id) = fluid_pipeline.reorder_copyback_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.reorder_copyback_pipeline.is_none() {
                fluid_pipeline.reorder_copyback_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Reorder copyback pipeline retrieved and cached");
            }
        }
    }

    if let Some(id) = fluid_pipeline.calculate_offsets_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.calculate_offsets_pipeline.is_none() {
                fluid_pipeline.calculate_offsets_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: Calculate offsets pipeline retrieved and cached");
            }
        }
    }

    // CRITICAL: Position correction pipeline retrieval
    if let Some(id) = fluid_pipeline.position_correction_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            if fluid_pipeline.position_correction_pipeline.is_none() {
                fluid_pipeline.position_correction_pipeline = Some(pipeline.clone());
                info!("GPU DEBUG: *** POSITION CORRECTION PIPELINE RETRIEVED AND CACHED ***");
            }
        } else {
            info!("GPU DEBUG: Position correction pipeline not yet compiled, ID: {:?}", id);
        }
    }

    // Check if all pipelines are ready
    let all_pipelines_ready = fluid_pipeline.density_pressure_pipeline.is_some() &&
                             fluid_pipeline.forces_pipeline.is_some() &&
                             fluid_pipeline.integration_pipeline.is_some() &&
        fluid_pipeline.spatial_hash_pipeline.is_some() &&
        fluid_pipeline.reorder_pipeline.is_some() &&
        fluid_pipeline.reorder_copyback_pipeline.is_some() &&
                             fluid_pipeline.calculate_offsets_pipeline.is_some() &&
                             fluid_pipeline.position_correction_pipeline.is_some();

    info!("GPU DEBUG: Pipeline readiness check - All ready: {}", all_pipelines_ready);
    info!("GPU DEBUG: - Density pressure: {}", fluid_pipeline.density_pressure_pipeline.is_some());
    info!("GPU DEBUG: - Forces: {}", fluid_pipeline.forces_pipeline.is_some());
    info!("GPU DEBUG: - Integration: {}", fluid_pipeline.integration_pipeline.is_some());
    info!("GPU DEBUG: - Spatial hash: {}", fluid_pipeline.spatial_hash_pipeline.is_some());
    info!("GPU DEBUG: - Reorder: {}", fluid_pipeline.reorder_pipeline.is_some());
    info!("GPU DEBUG: - Reorder copyback: {}", fluid_pipeline.reorder_copyback_pipeline.is_some());
    info!("GPU DEBUG: - Calculate offsets: {}", fluid_pipeline.calculate_offsets_pipeline.is_some());
    info!("GPU DEBUG: - Position correction: {}", fluid_pipeline.position_correction_pipeline.is_some());

    // Create buffers and bind group if we have extracted params and all pipelines are ready
    if let Some(params) = extracted_params {
        if all_pipelines_ready {
            info!("GPU DEBUG: Creating buffers for {} particles", params.num_particles);
            
            // Create particle buffer
            let particle_data: Vec<GpuParticle> = (0..params.num_particles)
                .map(|i| GpuParticle {
                    position: [params.particle_positions[i].x, params.particle_positions[i].y],
                    padding0: [0.0, 0.0],
                    velocity: [params.particle_velocities[i].x, params.particle_velocities[i].y],
                    padding1: [0.0, 0.0],
                    density: params.particle_densities[i],
                    pressure: params.particle_pressures[i],
                    near_density: params.near_densities[i],
                    near_pressure: params.near_pressures[i],
                })
                .collect();

            let particle_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("particle_buffer"),
                contents: bytemuck::cast_slice(&particle_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

            // Create params buffer
            let gpu_params = GpuFluidParams {
                smoothing_radius: params.params.smoothing_radius,
                target_density: params.params.target_density,
                pressure_multiplier: params.params.pressure_multiplier,
                near_pressure_multiplier: params.params.near_pressure_multiplier,
                viscosity_strength: params.params.viscosity_strength,
                boundary_dampening: BOUNDARY_DAMPENING,
                particle_radius: PARTICLE_RADIUS,
                dt: params.dt,
                boundary_min: [params.params.boundary_min.x, params.params.boundary_min.y],
                boundary_min_padding: [0.0, 0.0],
                boundary_max: [params.params.boundary_max.x, params.params.boundary_max.y],
                boundary_max_padding: [0.0, 0.0],
                gravity: [GRAVITY_2D[0], GRAVITY_2D[1]],
                gravity_padding: [0.0, 0.0],
                mouse_position: [params.mouse.position.x, params.mouse.position.y],
                mouse_radius: params.mouse.radius,
                mouse_strength: params.mouse.strength,
                mouse_active: if params.mouse.active { 1 } else { 0 },
                mouse_repel: if params.mouse.repel { 1 } else { 0 },
                padding: [0, 0],
            };

            let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("params_buffer"),
                contents: bytemuck::cast_slice(&[gpu_params]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });

            // Create spatial hash buffers
            let spatial_keys_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("spatial_keys_buffer"),
                size: (params.num_particles * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let spatial_indices_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("spatial_indices_buffer"),
                size: (params.num_particles * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let spatial_offsets_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("spatial_offsets_buffer"),
                size: (params.num_particles * std::mem::size_of::<u32>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let target_particles_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("target_particles_buffer"),
                size: (params.num_particles * std::mem::size_of::<GpuParticle>()) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create bind group
            let bind_group_inner = render_device.create_bind_group(
                "fluid_bind_group",
                fluid_pipeline.bind_group_layout.as_ref().unwrap(),
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: particle_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: spatial_keys_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: spatial_indices_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: spatial_offsets_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: target_particles_buffer.as_entire_binding(),
                    },
                ],
            );

            // Update bind group resource
            bind_group.bind_group = Some(bind_group_inner);
            bind_group.particle_buffer = Some(particle_buffer);
            bind_group.params_buffer = Some(params_buffer);
            bind_group.spatial_keys_buffer = Some(spatial_keys_buffer);
            bind_group.spatial_indices_buffer = Some(spatial_indices_buffer);
            bind_group.spatial_offsets_buffer = Some(spatial_offsets_buffer);
            bind_group.target_particles_buffer = Some(target_particles_buffer);
            bind_group.num_particles = params.num_particles as u32;
            bind_group.pipeline_ready = true;

            info!("GPU DEBUG: All buffers and bind group created successfully");
            info!("GPU DEBUG: Pipeline is now ready for compute shader execution");
        } else {
            info!("GPU DEBUG: Waiting for all pipelines to compile before creating buffers");
        }
    } else {
        info!("GPU DEBUG: No extracted params available, skipping buffer creation");
    }
}

// Prepare readback buffers at the right moment - key for triple buffering performance
fn queue_particle_buffers(
    _commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    fluid_bind_group: Res<FluidBindGroup>,
    mut gpu_results: ResMut<GpuResultsBuffer>,
    fluid_pipeline: Res<FluidPipeline>,  // ADD PIPELINE ACCESS
) {
    // Skip if readback is already scheduled
    if gpu_results.readback_scheduled {
        return;
    }
    
    // CRITICAL FIX: Actually run the GPU compute shaders!
    // The previous code was only doing readback without any computation
    if fluid_bind_group.pipeline_ready && fluid_bind_group.particle_buffer.is_some() {
        let particle_count = fluid_bind_group.num_particles;
        info!("GPU DEBUG: Starting compute shader execution for {} particles", particle_count);
        
        // Check if all required pipelines are available
        let all_pipelines_available = 
            fluid_pipeline.density_pressure_pipeline.is_some() &&
            fluid_pipeline.forces_pipeline.is_some() &&
            fluid_pipeline.integration_pipeline.is_some() &&
            fluid_pipeline.spatial_hash_pipeline.is_some() &&
            fluid_pipeline.reorder_pipeline.is_some() &&
            fluid_pipeline.reorder_copyback_pipeline.is_some() &&
            fluid_pipeline.calculate_offsets_pipeline.is_some() &&
            fluid_pipeline.position_correction_pipeline.is_some();
            
        if !all_pipelines_available {
            info!("GPU DEBUG: Not all pipelines available yet, skipping compute execution");
            info!("GPU DEBUG: - Density pressure: {}", fluid_pipeline.density_pressure_pipeline.is_some());
            info!("GPU DEBUG: - Forces: {}", fluid_pipeline.forces_pipeline.is_some());
            info!("GPU DEBUG: - Integration: {}", fluid_pipeline.integration_pipeline.is_some());
            info!("GPU DEBUG: - Spatial hash: {}", fluid_pipeline.spatial_hash_pipeline.is_some());
            info!("GPU DEBUG: - Reorder: {}", fluid_pipeline.reorder_pipeline.is_some());
            info!("GPU DEBUG: - Reorder copyback: {}", fluid_pipeline.reorder_copyback_pipeline.is_some());
            info!("GPU DEBUG: - Calculate offsets: {}", fluid_pipeline.calculate_offsets_pipeline.is_some());
            info!("GPU DEBUG: - Position correction: {}", fluid_pipeline.position_correction_pipeline.is_some());
        return;
    }
    
        info!("GPU DEBUG: All pipelines available, creating command encoder");
        let mut encoder = render_device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
            label: Some("fluid_compute_encoder"),
        });
        
        let workgroup_size = 64;
        let dispatch_size = (particle_count + workgroup_size - 1) / workgroup_size;
        
        info!("GPU DEBUG: Dispatch size: {} workgroups for {} particles", dispatch_size, particle_count);
        
        // Get bind group
        if let Some(bind_group) = &fluid_bind_group.bind_group {
            info!("GPU DEBUG: Starting compute pass");
            let mut compute_pass = encoder.begin_compute_pass(&bevy::render::render_resource::ComputePassDescriptor {
                label: Some("fluid_simulation_pass"),
                timestamp_writes: None,
            });
            
            // Step 1: Spatial hash
            if let Some(spatial_hash_pipeline) = &fluid_pipeline.spatial_hash_pipeline {
                info!("GPU DEBUG: Step 1 - Executing spatial hash");
                compute_pass.set_pipeline(spatial_hash_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Spatial hash pipeline not available!");
            }
            
            // Step 2: Calculate offsets
            if let Some(calculate_offsets_pipeline) = &fluid_pipeline.calculate_offsets_pipeline {
                info!("GPU DEBUG: Step 2 - Executing calculate offsets");
                compute_pass.set_pipeline(calculate_offsets_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Calculate offsets pipeline not available!");
            }
            
            // Step 3: Reorder
            if let Some(reorder_pipeline) = &fluid_pipeline.reorder_pipeline {
                info!("GPU DEBUG: Step 3 - Executing reorder");
                compute_pass.set_pipeline(reorder_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Reorder pipeline not available!");
            }
            
            // Step 4: Reorder copyback
            if let Some(reorder_copyback_pipeline) = &fluid_pipeline.reorder_copyback_pipeline {
                info!("GPU DEBUG: Step 4 - Executing reorder copyback");
                compute_pass.set_pipeline(reorder_copyback_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Reorder copyback pipeline not available!");
            }
            
            // Step 5: Density pressure
            if let Some(density_pressure_pipeline) = &fluid_pipeline.density_pressure_pipeline {
                info!("GPU DEBUG: Step 5 - Executing density pressure");
                compute_pass.set_pipeline(density_pressure_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Density pressure pipeline not available!");
            }
            
            // Step 6: Forces
            if let Some(forces_pipeline) = &fluid_pipeline.forces_pipeline {
                info!("GPU DEBUG: Step 6 - Executing forces");
                compute_pass.set_pipeline(forces_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Forces pipeline not available!");
            }
            
            // Step 7: Position correction - CRITICAL FOR PREVENTING OVERLAPS
            if let Some(position_correction_pipeline) = &fluid_pipeline.position_correction_pipeline {
                info!("GPU DEBUG: Step 7 - *** EXECUTING POSITION CORRECTION - CRITICAL FOR OVERLAP PREVENTION ***");
                compute_pass.set_pipeline(position_correction_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
                info!("GPU DEBUG: Position correction dispatched successfully!");
            } else {
                info!("GPU DEBUG: *** ERROR - POSITION CORRECTION PIPELINE NOT AVAILABLE! THIS IS WHY PARTICLES OVERLAP! ***");
            }
            
            // Step 8: Integration
            if let Some(integration_pipeline) = &fluid_pipeline.integration_pipeline {
                info!("GPU DEBUG: Step 8 - Executing integration");
                compute_pass.set_pipeline(integration_pipeline);
                compute_pass.set_bind_group(0, bind_group, &[]);
                compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
            } else {
                info!("GPU DEBUG: ERROR - Integration pipeline not available!");
            }
            
            drop(compute_pass);
            info!("GPU DEBUG: Compute pass completed, submitting commands");
        } else {
            info!("GPU DEBUG: ERROR - Bind group not available!");
            return;
        }
        
        // CRITICAL FIX: Copy GPU particle buffer to readback buffer BEFORE submitting
        // Create readback buffer for getting results back to CPU
        if gpu_results.readback_buffers[0].is_none() {
            let buffer_size = std::mem::size_of::<GpuParticle>() * particle_count as usize;
            
            let readback_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("particle_readback_buffer"),
                size: buffer_size as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            gpu_results.readback_buffers[0] = Some(readback_buffer);
            info!("GPU DEBUG: Created readback buffer ({} bytes)", buffer_size);
        }
        
        // CRITICAL: Copy the computed particle data from GPU buffer to readback buffer
        if let Some(readback_buffer) = &gpu_results.readback_buffers[0] {
            if let Some(particle_buffer) = &fluid_bind_group.particle_buffer {
                let buffer_size = std::mem::size_of::<GpuParticle>() * particle_count as usize;
                encoder.copy_buffer_to_buffer(
                    particle_buffer,
                    0,
                    readback_buffer,
                    0,
                    buffer_size as u64,
                );
                info!("GPU DEBUG: *** CRITICAL FIX - Copying {} bytes from GPU particle buffer to readback buffer ***", buffer_size);
            } else {
                info!("GPU DEBUG: ERROR - Particle buffer not available for copy!");
            }
        } else {
            info!("GPU DEBUG: ERROR - Readback buffer not available for copy!");
        }
        
        // Submit the command buffer
        render_queue.submit(std::iter::once(encoder.finish()));
        info!("GPU DEBUG: Commands submitted to GPU with copy operation");
        
        // Schedule readback
        gpu_results.readback_scheduled = true;
        gpu_results.frame_counter += 1;
        info!("GPU DEBUG: Readback scheduled for frame {}", gpu_results.frame_counter);
    } else {
        if !fluid_bind_group.pipeline_ready {
            info!("GPU DEBUG: Pipeline not ready yet");
        }
        if fluid_bind_group.particle_buffer.is_none() {
            info!("GPU DEBUG: Particle buffer not available");
        }
    }
}

// Handle buffer readbacks from GPU to main world - optimized with triple buffering
fn handle_buffer_readbacks(
    _render_device: Res<RenderDevice>,
    mut gpu_results: ResMut<GpuResultsBuffer>,
    mut commands: Commands,
    _render_queue: Res<RenderQueue>,
    fluid_bind_group: Res<FluidBindGroup>,
    _time: Res<Time>,
) {
    // Update frame counter to manage readback frequency
    gpu_results.frame_counter = gpu_results.frame_counter.wrapping_add(1);
    
    // Skip if no readback is scheduled or buffers unavailable
    if !gpu_results.readback_scheduled || gpu_results.readback_buffers[gpu_results.active_buffer_index].is_none() {
        return;
    }
    
    // Store values we need to avoid borrowing issues
    let active_buffer_index = gpu_results.active_buffer_index;
    let active_buffer = gpu_results.readback_buffers[active_buffer_index].clone().unwrap();
    let particles_copy = std::mem::take(&mut gpu_results.gpu_data);
    let particle_count = fluid_bind_group.num_particles as usize;
    let debug_log = gpu_results.debug_log;
    
    // Poll the active buffer - non-blocking
    let buffer_slice = active_buffer.slice(..);
    
    // If we already started mapping, check if it's ready
    if gpu_results.is_mapped {
        // Get the mapped buffer
        match buffer_slice.get_mapped_range().len() {
            0 => {
                // Mapping not complete yet, try again next frame
                gpu_results.gpu_data = particles_copy;
                return;
            }
            _ => {
                // Mapping complete, process the data
                let buffer_range = buffer_slice.get_mapped_range();
                let particle_bytes = bytemuck::cast_slice(&buffer_range);

                // Create new collections
                let mut positions = Vec::with_capacity(particle_count);
                let mut velocities = Vec::with_capacity(particle_count);
                let mut densities = Vec::with_capacity(particle_count);
                let mut pressures = Vec::with_capacity(particle_count);
                let mut near_densities = Vec::with_capacity(particle_count);
                let mut near_pressures = Vec::with_capacity(particle_count);
                
                // Use a temporary variable to store particles
                let particles = bytemuck::cast_slice::<u8, GpuParticle>(particle_bytes);
                let mut particles_copy = Vec::with_capacity(particles.len());
                particles_copy.extend_from_slice(particles);
                
                // Extract data from GPU particles
                for particle in &particles_copy {
                    positions.push(Vec2::new(particle.position[0], particle.position[1]));
                    velocities.push(Vec2::new(particle.velocity[0], particle.velocity[1]));
                    densities.push(particle.density);
                    pressures.push(particle.pressure);
                    near_densities.push(particle.near_density);
                    near_pressures.push(particle.near_pressure);
                }
                
                // Get count for logging
                let particle_count_processed = positions.len();
                
                // Submit data to the main world
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
                
                // Finish working with this buffer
                drop(buffer_range);
                active_buffer.unmap();
                
                // Capture timing data
                if let Some(last_time) = gpu_results.last_readback_time {
                    let elapsed = last_time.elapsed();
                    if debug_log {
                        info!("GPU readback completed: {}/{} particles, took {:?}", 
                              particle_count_processed, particle_count, elapsed);
                    }
                }
                gpu_results.last_readback_time = Some(std::time::Instant::now());
                
                // Reset the readback flags
                gpu_results.readback_scheduled = false;
                gpu_results.is_mapped = false;
                
                // Rotate to the next buffer
                gpu_results.active_buffer_index = (gpu_results.active_buffer_index + 1) % 3;
                
                // Store the particles for reuse
                gpu_results.gpu_data = particles_copy;
            }
        }
    } else {
        // Start mapping this buffer
        buffer_slice.map_async(bevy::render::render_resource::MapMode::Read, |_| {});
        gpu_results.is_mapped = true;
        gpu_results.gpu_data = particles_copy;
    }
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
        let boundary_dampening = BOUNDARY_DAMPENING;
        let particle_radius = PARTICLE_RADIUS;
        
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
    
    // Debug: Check for overlapping particles and log GPU debug info
    let mut overlap_count = 0;
    let mut total_particles_checked = 0;
    let min_distance = PARTICLE_RADIUS * 2.0;
    
    // Update CPU entities with GPU data
    for (i, (entity, mut transform, mut particle)) in particles.iter_mut().enumerate() {
        if i < gpu_data.positions.len() {
            let current_pos = gpu_data.positions[i];
            
            // Update transform
            transform.translation.x = current_pos.x;
            transform.translation.y = current_pos.y;
            
            // Update particle properties
            particle.velocity = gpu_data.velocities[i];
            particle.density = gpu_data.densities[i];
            particle.pressure = gpu_data.pressures[i];
            particle.near_density = gpu_data.near_densities[i];
            particle.near_pressure = gpu_data.near_pressures[i];
            
            // Check for overlaps with other particles (debug logging)
            for (j, other_pos) in gpu_data.positions.iter().enumerate() {
                if i == j { continue; }
                
                let distance = current_pos.distance(*other_pos);
                if distance < min_distance {
                    overlap_count += 1;
                    if overlap_count <= 5 { // Limit spam
                        println!("GPU 2D OVERLAP: Particle {} overlapping with particle {}! Distance: {:.3}, Min required: {:.3}", 
                                 i, j, distance, min_distance);
                    }
                }
            }
            total_particles_checked += 1;
        }
    }
    
    // Log summary every few frames
    static mut FRAME_COUNTER: u32 = 0;
    unsafe {
        FRAME_COUNTER += 1;
        if FRAME_COUNTER % 60 == 0 { // Log every 60 frames
            if overlap_count > 0 {
                println!("GPU 2D DEBUG: Frame {}: Found {} overlaps out of {} particles checked", 
                         FRAME_COUNTER, overlap_count / 2, total_particles_checked); // Divide by 2 since we count each pair twice
            }
            
            // Log GPU shader debug info from first few particles
            for i in 0..3.min(gpu_data.positions.len()) {
                // Note: We can't directly access padding fields from GPU data here
                // The debug info would need to be extracted during GPU readback
                println!("GPU 2D DEBUG: Particle {}: pos=({:.2}, {:.2}), vel=({:.2}, {:.2}), density={:.2}", 
                         i, gpu_data.positions[i].x, gpu_data.positions[i].y,
                         gpu_data.velocities[i].x, gpu_data.velocities[i].y,
                         gpu_data.densities[i]);
            }
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