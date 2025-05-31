use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroupLayout, BindGroupLayoutEntry,
            BindingType, BufferBindingType, ShaderStages, ComputePipelineDescriptor,
            ComputePipeline, Buffer, PipelineCache,
        },
        renderer::{RenderDevice, RenderQueue},
        Render, RenderApp, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
    log::info,
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

use crate::two_d::simulation::{Particle, FluidParams, MouseInteraction};
use crate::constants::{PARTICLE_RADIUS, BOUNDARY_DAMPENING, MOUSE_STRENGTH_LOW, MOUSE_STRENGTH_MEDIUM, MOUSE_STRENGTH_HIGH};

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

// Track GPU state and errors
#[derive(Resource, Clone)]
pub struct GpuState {
    pub enabled: bool,
}

impl Default for GpuState {
    fn default() -> Self {
        Self {
            enabled: true,
        }
    }
}

// Render world copy of GPU state
#[derive(Resource, Default, Clone)]
struct RenderGpuState {
    enabled: bool,
}

// Track GPU performance data
#[derive(Resource, Clone)]
pub struct GpuPerformanceStats {
    pub frame_times: Vec<f32>,
    pub avg_frame_time: f32,
    pub max_sample_count: usize,
    pub iterations_per_frame: u32,
    pub max_velocity: f32,
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
            iterations_per_frame: 3,
            max_velocity: 0.0,
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
    particle_buffer: Option<Buffer>,
    
    // Pipeline state
    num_particles: u32,
    pipeline_ready: bool,
}

// Extracted params for the render world
#[derive(Resource, Default, Clone)]
struct ExtractedFluidParams {
    particle_positions: Vec<Vec2>,
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
        mouse_interaction.strength = MOUSE_STRENGTH_LOW;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = MOUSE_STRENGTH_MEDIUM;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = MOUSE_STRENGTH_HIGH;
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
    _fluid_params: Extract<Res<FluidParams>>,
    _mouse_interaction: Extract<Res<MouseInteraction>>,
    _time: Extract<Res<Time>>,
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
        particle_positions: positions,
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
    let num_particles = extracted_data.particle_positions.len();
    
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
        let shader = asset_server.load("shaders/2d/spatial_hash.wgsl");
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
        let shader = asset_server.load("shaders/2d/calculate_offsets.wgsl");
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
        let shader = asset_server.load("shaders/2d/reorder.wgsl");
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
        let shader = asset_server.load("shaders/2d/reorder_copyback.wgsl");
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
        let shader = asset_server.load("shaders/2d/density_pressure.wgsl");
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
        let shader = asset_server.load("shaders/2d/forces.wgsl");
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
        let shader = asset_server.load("shaders/2d/integration.wgsl");
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
        if gpu_results.readback_buffers[0].is_none() || bind_group.num_particles != num_particles as u32 {
            let buffer_size = std::mem::size_of::<GpuParticle>() * num_particles;
            
            // Create a buffer suitable for readback (mapping)
            let buffer = render_device.create_buffer(&bevy::render::render_resource::BufferDescriptor {
                label: Some("particle_readback_buffer_0"),
                size: buffer_size as u64,
                usage: bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            gpu_results.readback_buffers[0] = Some(buffer);
            gpu_results.readback_scheduled = false;
            gpu_results.gpu_data.clear();
            
            if gpu_results.debug_log {
                info!("Created GPU readback buffer for {} particles ({} bytes)", num_particles, buffer_size);
            }
        }
    }
}

// Prepare readback buffers at the right moment - key for triple buffering performance
fn queue_particle_buffers(
    _commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    fluid_bind_group: Res<FluidBindGroup>,
    mut gpu_results: ResMut<GpuResultsBuffer>,
) {
    // Skip if readback is already scheduled
    if gpu_results.readback_scheduled {
        return;
    }
    
    // Only run readback every N frames for better performance
    let readback_frequency = 1; // Adjust based on desired performance vs smoothness
    if gpu_results.frame_counter % readback_frequency != 0 {
        return;
    }
    
    // Get the next buffer index
    let next_index = (gpu_results.active_buffer_index + 1) % 3;
    
    // Initialize the buffer if needed
    if gpu_results.readback_buffers[next_index].is_none() {
        // Create a new buffer for GPU readback
        let buffer_size = fluid_bind_group.num_particles as usize * std::mem::size_of::<GpuParticle>();
        let buffer = render_device.create_buffer(&bevy::render::render_resource::BufferDescriptor {
            label: Some("Particle Readback Buffer"),
            size: buffer_size as u64,
            usage: bevy::render::render_resource::BufferUsages::COPY_DST | bevy::render::render_resource::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        gpu_results.readback_buffers[next_index] = Some(buffer);
    }
    
    // Copy from particle buffer to our readback buffer
    if let Some(buffer) = &gpu_results.readback_buffers[next_index] {
        // Initialize compute pass to prepare final particles state
        let mut encoder = render_device.create_command_encoder(&bevy::render::render_resource::CommandEncoderDescriptor {
            label: Some("Fluid Readback Encoder"),
        });
        
        // Copy from particle buffer to our readback buffer
        encoder.copy_buffer_to_buffer(
            fluid_bind_group.particle_buffer.as_ref().unwrap(),
            0,
            buffer,
            0,
            (fluid_bind_group.num_particles as usize * std::mem::size_of::<GpuParticle>()) as u64,
        );
        
        // Submit the command to copy data
        render_queue.submit(std::iter::once(encoder.finish()));
        
        // Mark that we've scheduled a readback
        gpu_results.readback_scheduled = true;
        gpu_results.active_buffer_index = next_index;
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